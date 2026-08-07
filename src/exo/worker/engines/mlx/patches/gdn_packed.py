# .typings/mlx has no stub for mx.fast, so metal_kernel and its return value are
# untyped here. Everything else in this module is fully typed.
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportAny=false
"""Route gated-delta prefill through a packed Metal kernel (port of mlx-lm#1559).

The generic mlx-lm kernel gives one 32-lane SIMD-group to each value row. At
Dk=128 that leaves every lane holding only four state elements and costs two
full-SIMD reductions per row and token. The packed kernel instead puts eight
value rows on a SIMD-group: four lanes own a row and each lane keeps 32
contiguous state elements in registers.

This is a port, not a cherry-pick. Upstream mlx-lm#1559 targets
`gated_delta_kernel(q, k, v, g, beta, state, mask)`, but exo pins mlx-lm to
rltakashige/mlx-lm@leo/deepseek-v4, which fuses the g/beta computation into the
kernel and therefore takes `(q, k, v, a, b, A_log, dt_bias, state, mask)`. The
fused prologue below is lifted verbatim from that fork's
`_make_gated_delta_kernel(vectorized=False)`, so the arithmetic is unchanged.

Numerics: the reduction is the explicitly-written ascending butterfly
(shuffle_xor 1,2,4,8,16) rather than `simd_sum`. Its first three levels combine
partials living in a single packed lane and the last two become the four-lane
row-group shuffles, so each 4-element partial keeps the generic kernel's
sequential order. Measured on Apple M4 (mlx 0.32.0.dev20260522): bit-identical
to the generic kernel, max|delta| = 0.0 on both y and state, across
T = 1/64/257/2048, B = 1/2 and bfloat16/float16/float32. Per-layer speedup
2.3-2.6x; decode (T=1) is neutral.

Only scalar-gate, unmasked, Dk=128, Dv%8==0, float32-state calls are routed.
Vector gates, padding masks and every other shape keep the original kernel.
Set EXO_GDN_PACKED=0 to disable the patch entirely.
"""

import os
from collections.abc import Callable

import mlx.core as mx
from mlx_lm.models import gated_delta as _gated_delta

from exo.worker.runner.bootstrap import logger

_ENV_VAR = "EXO_GDN_PACKED"

# mx.fast.metal_kernel returns a nanobind function that takes keyword-only
# arguments and returns one array per output name. .typings/mlx has no stub for
# mx.fast, hence the pragma at the top of this file.
_Kernel = Callable[..., list[mx.array]]

_INPUT_NAMES = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in", "T"]
_TEMPLATE_NAMES = ("InT", "StT", "Dk", "Dv", "Hk", "Hv")

# Lifted verbatim from the fork's _make_gated_delta_kernel(vectorized=False).
_FUSED_G = """
          float a_val = static_cast<float>(a_[hv_idx]);
          float dt_val = static_cast<float>(dt_bias[hv_idx]);
          float x_g = a_val + dt_val;
          float sp = (x_g > 20.0f) ? x_g : log(1.0f + exp(x_g));
          float g_val = exp(-exp(static_cast<float>(A_log[hv_idx])) * sp);
          float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));
"""

_PACKED_SRC = (
    """
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // a, b: [B, T, Hv]
        auto a_ = a + b_idx * T * Hv;
        auto b_ = b + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
"""
    + _FUSED_G
    + """
          // Each 4-element chain reproduces one original lane's sequential
          // accumulation, so the butterfly below matches the generic kernel.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * g_val;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          // Butterfly levels xor 1,2,4 stay inside this lane; xor 8,16 become
          // the two row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_val;

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          a_ += Hv;
          b_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
"""
)

# Unpacked layout with the same explicit reduction tree. Not used in serving;
# it is the bitwise comparator the tests pin the packed kernel against, so the
# reduction order is a contract of this file rather than of the simd_sum
# lowering.
_XTREE_SRC = (
    """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto a_ = a + b_idx * T * Hv;
        auto b_ = b + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
"""
    + _FUSED_G
    + """
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_val;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);
          kv_mem += simd_shuffle_xor(kv_mem, 4);
          kv_mem += simd_shuffle_xor(kv_mem, 8);
          kv_mem += simd_shuffle_xor(kv_mem, 16);

          auto delta = (v_[dv_idx] - kv_mem) * beta_val;

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          out += simd_shuffle_xor(out, 4);
          out += simd_shuffle_xor(out, 8);
          out += simd_shuffle_xor(out, 16);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          a_ += Hv;
          b_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
"""
)


def _build_kernel(name: str, source: str) -> _Kernel | None:
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name=name,
        input_names=_INPUT_NAMES,
        output_names=["y", "state_out"],
        source=source,
    )


_packed_kernel = _build_kernel("exo_gdn_step_packed_btree", _PACKED_SRC)
_xtree_kernel = _build_kernel("exo_gdn_step_xtree", _XTREE_SRC)

_original_gated_delta_kernel = None
_logged_dispatch = False


def is_enabled() -> bool:
    """Whether the packed kernel is enabled. EXO_GDN_PACKED=0 turns it off."""
    return os.environ.get(_ENV_VAR, "1") != "0"


def _eligible(
    a: mx.array, mask: mx.array | None, dk: int, dv: int, state: mx.array
) -> bool:
    # The packed kernel hands each lane Dk/4 state elements and packs 32/4 value
    # rows into a SIMD-group, so it needs Dk == 128 and Dv divisible by 8. It is
    # otherwise generic in B, Hk, Hv and the input element type.
    return (
        mask is None
        and a.ndim == 3
        and dk == 128
        and dv % 8 == 0
        and state.dtype == mx.float32
    )


def _dispatch(
    kernel: _Kernel,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803
    dt_bias: mx.array,
    state: mx.array,
    *,
    packed: bool,
) -> tuple[mx.array, mx.array]:
    batch, seq, hk, dk = k.shape
    hv, dv = v.shape[2:]
    grid = (32, dv // 8, batch * hv) if packed else (32, dv, batch * hv)
    threadgroup = (32, 2, 1) if packed else (32, 4, 1)
    y, state_out = kernel(
        inputs=[q, k, v, a, b, A_log, dt_bias, state, seq],
        template=list(
            zip(
                _TEMPLATE_NAMES,
                [q.dtype, state.dtype, dk, dv, hk, hv],
                strict=True,
            )
        ),
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(batch, seq, hv, dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
    return y, state_out


def gated_delta_kernel_xtree(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803
    dt_bias: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Explicit-tree comparator for the packed kernel. Test use only."""
    assert mask is None, "the comparator has no masked variant"
    assert _xtree_kernel is not None
    return _dispatch(_xtree_kernel, q, k, v, a, b, A_log, dt_bias, state, packed=False)


def gated_delta_kernel_packed(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803
    dt_bias: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Force the packed kernel regardless of the env switch. Test use only."""
    assert mask is None, "the packed kernel has no masked variant"
    assert _packed_kernel is not None
    return _dispatch(_packed_kernel, q, k, v, a, b, A_log, dt_bias, state, packed=True)


def _patched_gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803
    dt_bias: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    assert _original_gated_delta_kernel is not None
    _, _, _, dk = k.shape
    hv, dv = v.shape[2:]

    if (
        _packed_kernel is None
        or not is_enabled()
        or not _eligible(a, mask, dk, dv, state)
    ):
        return _original_gated_delta_kernel(q, k, v, a, b, A_log, dt_bias, state, mask)

    global _logged_dispatch
    if not _logged_dispatch:
        _logged_dispatch = True
        # Cheap verification signal: proves the packed kernel actually ran in
        # this runner process, not merely that the patch was installed.
        logger.info(
            f"GDN packed kernel dispatched: Hk={k.shape[2]} Hv={hv} "
            f"Dk={dk} Dv={dv} in={q.dtype} state={state.dtype}"
        )

    return _dispatch(_packed_kernel, q, k, v, a, b, A_log, dt_bias, state, packed=True)


def patch_gated_delta_packed() -> None:
    """Rebind mlx_lm's gated_delta_kernel to the packed implementation.

    `gated_delta_update` resolves `gated_delta_kernel` from module globals at
    call time, so rebinding the module attribute covers every caller
    (qwen3_5, qwen3_next, kimi_linear and mlx_vlm's qwen3_5).
    """
    global _original_gated_delta_kernel

    if _original_gated_delta_kernel is not None:
        return
    if not is_enabled():
        logger.info(f"GDN packed kernel disabled ({_ENV_VAR}=0)")
        return
    if _packed_kernel is None:
        logger.info("GDN packed kernel unavailable (no Metal device)")
        return

    _original_gated_delta_kernel = _gated_delta.gated_delta_kernel
    _gated_delta.gated_delta_kernel = _patched_gated_delta_kernel
    logger.info(
        f"GDN packed kernel installed (mlx-lm#1559 port; {_ENV_VAR}=0 to disable)"
    )
