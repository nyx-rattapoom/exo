"""The GDN capture/rollback must reproduce a plain forward over the accepted prefix.

Runs on CPU or GPU (gated_delta_update picks the ops path when Metal is absent).
"""

from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from exo.worker.engines.mlx.mtp.gdn_rollback import (
    CapturingGatedDeltaNet,
    install_gdn_capture,
)


def _args() -> TextModelArgs:
    return TextModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 100,
            # Production-shaped GDN heads: the Metal kernel needs Dk >= 32 and the
            # packed kernel (the one production dispatches) needs Dk == 128.
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
        }
    )


def _layers(seed: int) -> list[DecoderLayer]:
    mx.random.seed(seed)
    args = _args()
    layers = [DecoderLayer(args, layer_idx=i) for i in range(3)]  # all linear
    for layer in layers:
        layer.eval()
    mx.eval([layer.parameters() for layer in layers])
    return layers


def _run(layers: list[DecoderLayer], caches: list[ArraysCache], x: mx.array) -> mx.array:
    h = x
    for layer, c in zip(layers, caches, strict=True):
        h = layer(h, mask=None, cache=c)
    return h


def _eval_caches(caches: list[ArraysCache]) -> None:
    arrays: list[mx.array] = []
    for c in caches:
        for i in range(2):
            a = cast(mx.array | None, c[i])
            if a is not None:
                arrays.append(a)
    mx.eval(*arrays)


def _assert_state_equal(a: ArraysCache, b: ArraysCache) -> None:
    for i in range(2):
        ca = cast(mx.array | None, a[i])
        cb = cast(mx.array | None, b[i])
        assert (ca is None) == (cb is None)
        if ca is not None and cb is not None:
            assert ca.shape == cb.shape
            assert mx.allclose(ca, cb, atol=1e-5, rtol=1e-4).item(), f"slot {i} differs"


@pytest.mark.parametrize("n_keep", [0, 1, 2, 3])
def test_rollback_matches_prefix_forward(n_keep: int) -> None:
    layers = _layers(0)
    capture = install_gdn_capture(list(layers))
    assert len(capture.layers) == 3
    assert all(isinstance(layer.linear_attn, CapturingGatedDeltaNet) for layer in layers)

    args = _args()
    mx.random.seed(1)
    prefix = mx.random.normal((1, 5, args.hidden_size))
    chunk = mx.random.normal((1, 3, args.hidden_size))

    # Reference: the same prefix, then only the first n_keep chunk tokens.
    ref_caches = [ArraysCache(size=2) for _ in layers]
    _run(layers, ref_caches, prefix)
    if n_keep > 0:
        _run(layers, ref_caches, chunk[:, :n_keep])
    _eval_caches(ref_caches)

    # Captured: the prefix, then the whole 3-token chunk with capture armed,
    # then roll back to n_keep.
    caches = [ArraysCache(size=2) for _ in layers]
    _run(layers, caches, prefix)
    capture.arm()
    out_full = _run(layers, caches, chunk)
    capture.disarm()
    mx.eval(out_full)
    capture.rollback(n_keep)
    _eval_caches(caches)

    for a, b in zip(caches, ref_caches, strict=True):
        _assert_state_equal(a, b)

    # And the next decode step agrees too.
    nxt = mx.random.normal((1, 1, args.hidden_size))
    y_ref = _run(layers, ref_caches, nxt)
    y = _run(layers, caches, nxt)
    assert mx.allclose(y, y_ref, atol=1e-5, rtol=1e-4).item()


def test_capture_is_passthrough_when_disarmed() -> None:
    layers = _layers(2)
    plain = _layers(2)
    install_gdn_capture(list(layers))
    args = _args()
    x = mx.random.normal((1, 4, args.hidden_size))
    c1 = [ArraysCache(size=2) for _ in layers]
    c2 = [ArraysCache(size=2) for _ in plain]
    y1 = _run(layers, c1, x)
    y2 = _run(plain, c2, x)
    assert mx.array_equal(y1, y2).item()
    for a, b in zip(c1, c2, strict=True):
        _assert_state_equal(a, b)


def test_rollback_requires_one_record_per_layer() -> None:
    layers = _layers(3)
    capture = install_gdn_capture(list(layers))
    args = _args()
    caches = [ArraysCache(size=2) for _ in layers]
    capture.arm()
    _run(layers, caches, mx.random.normal((1, 2, args.hidden_size)))
    _run(layers, caches, mx.random.normal((1, 2, args.hidden_size)))
    capture.disarm()
    with pytest.raises(RuntimeError):
        capture.rollback(1)
