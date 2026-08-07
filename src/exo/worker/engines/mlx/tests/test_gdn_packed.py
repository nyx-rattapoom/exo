# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportArgumentType=false
"""Pin the packed gated-delta kernel to the kernel it replaces.

The packed kernel (port of mlx-lm#1559) must be bit-identical to the explicit-tree
comparator by construction, and on Apple GPUs also to mlx-lm's simd_sum kernel.
Anything the packed path does not support must fall back to the original.
Uses random inputs — no model download required.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_kernel as original_kernel

from exo.worker.engines.mlx.patches import gdn_packed

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="gated delta kernels are GPU only",
)

# B, Hk, Hv, Dk, Dv, dtype
PACKED_CASES = [
    (1, 16, 32, 128, 128, mx.bfloat16),  # Qwen3.5/3.6 shape
    (2, 16, 32, 128, 128, mx.bfloat16),  # batched
    (1, 4, 8, 128, 128, mx.bfloat16),  # fewer heads
    (1, 8, 8, 128, 128, mx.bfloat16),  # Hv == Hk
    (1, 16, 32, 128, 128, mx.float16),
    (1, 16, 32, 128, 128, mx.float32),
    (1, 8, 16, 128, 64, mx.bfloat16),  # Dv != Dk
    (3, 2, 8, 128, 256, mx.bfloat16),  # larger Dv
]


def make_inputs(batch, seq, hk, hv, dk, dv, dtype, seed=3):
    mx.random.seed(seed)

    def normed(shape, dim):
        x = mx.random.normal(shape)
        return (mx.fast.rms_norm(x, None, 1e-6) * dim**-0.5).astype(dtype)

    q = normed((batch, seq, hk, dk), dk)
    k = normed((batch, seq, hk, dk), dk)
    v = mx.random.normal((batch, seq, hv, dv)).astype(dtype)
    a = (mx.random.normal((batch, seq, hv)) * 0.5).astype(dtype)
    b = mx.random.normal((batch, seq, hv)).astype(dtype)
    a_log = mx.log(mx.random.uniform(low=0.5, high=16.0, shape=(hv,)))
    dt_bias = mx.ones((hv,))
    state = (mx.random.normal((batch, hv, dv, dk)) * 0.3).astype(mx.float32)
    mx.eval(q, k, v, a, b, a_log, dt_bias, state)
    return q, k, v, a, b, a_log, dt_bias, state


@pytest.mark.parametrize(
    "case", PACKED_CASES, ids=lambda c: f"{c[5]}-Hv{c[2]}-Dv{c[4]}"
)
@pytest.mark.parametrize("seq", [1, 7, 64, 257])
def test_packed_matches_explicit_tree(case, seq):
    """Bitwise contract vs the comparator — holds on any device by construction."""
    batch, hk, hv, dk, dv, dtype = case
    args = make_inputs(batch, seq, hk, hv, dk, dv, dtype)

    y_p, s_p = gdn_packed.gated_delta_kernel_packed(*args)
    y_x, s_x = gdn_packed.gated_delta_kernel_xtree(*args)
    mx.eval(y_p, s_p, y_x, s_x)

    assert mx.array_equal(y_p, y_x)
    assert mx.array_equal(s_p, s_x)


@pytest.mark.parametrize(
    "case", PACKED_CASES, ids=lambda c: f"{c[5]}-Hv{c[2]}-Dv{c[4]}"
)
@pytest.mark.parametrize("seq", [1, 64, 257])
def test_packed_matches_simd_sum_kernel(case, seq):
    """Bitwise vs the kernel we actually replace. Measured to hold on M4.

    If this ever fails on a new device or toolchain the packed default should be
    revisited; the contract against the comparator above still holds.
    """
    batch, hk, hv, dk, dv, dtype = case
    args = make_inputs(batch, seq, hk, hv, dk, dv, dtype)

    y_p, s_p = gdn_packed.gated_delta_kernel_packed(*args)
    y_o, s_o = original_kernel(*args, None)
    mx.eval(y_p, s_p, y_o, s_o)

    assert mx.array_equal(y_p, y_o)
    assert mx.array_equal(s_p, s_o)


def test_patch_routes_eligible_calls_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(gdn_packed, "_original_gated_delta_kernel", None)
    monkeypatch.delenv(gdn_packed._ENV_VAR, raising=False)

    from mlx_lm.models import gated_delta

    monkeypatch.setattr(gated_delta, "gated_delta_kernel", original_kernel)

    gdn_packed.patch_gated_delta_packed()
    assert gated_delta.gated_delta_kernel is not original_kernel
    patched = gated_delta.gated_delta_kernel

    gdn_packed.patch_gated_delta_packed()
    assert gated_delta.gated_delta_kernel is patched

    args = make_inputs(1, 64, 16, 32, 128, 128, mx.bfloat16)
    y_new, s_new = patched(*args, None)
    y_packed, s_packed = gdn_packed.gated_delta_kernel_packed(*args)
    mx.eval(y_new, s_new, y_packed, s_packed)
    assert mx.array_equal(y_new, y_packed)
    assert mx.array_equal(s_new, s_packed)


def test_kill_switch_leaves_original_kernel_in_place(monkeypatch):
    monkeypatch.setattr(gdn_packed, "_original_gated_delta_kernel", None)
    monkeypatch.setenv(gdn_packed._ENV_VAR, "0")

    from mlx_lm.models import gated_delta

    monkeypatch.setattr(gated_delta, "gated_delta_kernel", original_kernel)

    gdn_packed.patch_gated_delta_packed()
    assert gated_delta.gated_delta_kernel is original_kernel


@pytest.mark.parametrize(
    ("dk", "dv", "use_mask"),
    [
        (64, 64, False),  # Dk != 128 -> generic kernel
        (128, 128, True),  # padding mask -> generic kernel
    ],
)
def test_unsupported_shapes_fall_back(monkeypatch, dk, dv, use_mask):
    monkeypatch.setattr(gdn_packed, "_original_gated_delta_kernel", original_kernel)

    args = make_inputs(1, 32, 4, 8, dk, dv, mx.bfloat16)
    mask = mx.ones((1, 32), dtype=mx.bool_) if use_mask else None

    y_r, s_r = gdn_packed._patched_gated_delta_kernel(*args, mask)
    y_o, s_o = original_kernel(*args, mask)
    mx.eval(y_r, s_r, y_o, s_o)

    assert mx.array_equal(y_r, y_o)
    assert mx.array_equal(s_r, s_o)


def test_vector_gate_falls_back():
    """A [B, T, Hv, Dk] gate must keep the vectorized kernel."""
    q, k, v, _, b, a_log, dt_bias, state = make_inputs(
        1, 65, 4, 8, 128, 128, mx.bfloat16
    )
    a_vec = (mx.random.normal((1, 65, 8, 128)) * 0.5).astype(mx.bfloat16)
    mx.eval(a_vec)
    assert not gdn_packed._eligible(a_vec, None, 128, 128, state)
