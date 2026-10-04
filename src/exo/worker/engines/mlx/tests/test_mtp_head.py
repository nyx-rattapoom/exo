"""MTP head: sidecar key normalisation, per-module quantisation inference, strict load."""

from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModelArgs

from exo.worker.engines.mlx.mtp.head import (
    Qwen35MtpHead,
    build_head,
    heal_delta_norms,
    load_sidecar,
    quantize_from_weights,
    strip_mtp_prefix,
)


def _args(num_experts: int = 4) -> TextModelArgs:
    return TextModelArgs.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 100,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
        }
    )


def test_strip_prefix() -> None:
    assert strip_mtp_prefix("language_model.mtp.fc.weight") == "fc.weight"
    assert strip_mtp_prefix("mtp.layers.0.norm.weight") == "layers.0.norm.weight"
    assert strip_mtp_prefix("language_model.model.layers.0.mlp.gate.weight") is None


def test_head_is_full_attention_moe_block() -> None:
    head = Qwen35MtpHead(_args())
    layer = head.layers[0]
    assert not layer.is_linear
    assert hasattr(layer, "self_attn")
    assert hasattr(layer.mlp, "switch_mlp")


def _roundtrip(bits_by_path: dict[str, tuple[int, int]]) -> None:
    """Quantise a fresh head per ``bits_by_path`` (module path -> (bits, group)),
    export its parameters as a 'sidecar', and check build_head reproduces the
    same quantisation and loads strictly."""
    args = _args()
    mx.random.seed(0)
    src = Qwen35MtpHead(args)

    def predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
        if not hasattr(module, "to_quantized"):
            return False
        spec = bits_by_path.get(path)
        if spec is None:
            return False
        return {"bits": spec[0], "group_size": spec[1]}

    nn.quantize(src, class_predicate=predicate, mode="affine")
    weights = cast(dict[str, mx.array], dict(tree_flatten(src.parameters())))
    mx.eval(list(weights.values()))

    head, specs, nbytes = build_head(args, weights)
    assert nbytes > 0
    assert {k: (v["bits"], v["group_size"]) for k, v in specs.items()} == bits_by_path
    got = cast(dict[str, mx.array], dict(tree_flatten(head.parameters())))
    assert set(got) == set(weights)
    for k, v in weights.items():
        assert mx.array_equal(got[k], v).item(), k


def test_build_head_mixed_precision_sidecar() -> None:
    _roundtrip(
        {
            "fc": (4, 32),
            "layers.0.self_attn.q_proj": (4, 32),
            "layers.0.self_attn.k_proj": (4, 32),
            "layers.0.self_attn.v_proj": (4, 32),
            "layers.0.self_attn.o_proj": (4, 32),
            "layers.0.mlp.gate": (8, 32),
            "layers.0.mlp.switch_mlp.gate_proj": (4, 32),
            "layers.0.mlp.switch_mlp.up_proj": (4, 32),
            "layers.0.mlp.switch_mlp.down_proj": (4, 32),
        }
    )


def test_build_head_bf16_sidecar() -> None:
    _roundtrip({})


def test_build_head_rejects_mismatch() -> None:
    args = _args()
    src = Qwen35MtpHead(args)
    weights = cast(dict[str, mx.array], dict(tree_flatten(src.parameters())))
    weights.pop("fc.weight")
    with pytest.raises(ValueError, match="missing"):
        build_head(args, weights)


def test_quantize_from_weights_ignores_bf16_modules() -> None:
    args = _args()
    head = Qwen35MtpHead(args)
    specs = quantize_from_weights(
        head, cast(dict[str, mx.array], dict(tree_flatten(head.parameters())))
    )
    assert specs == {}


def test_heal_delta_norms_shifts_only_zero_centred_gains() -> None:
    absolute_small = mx.array([0.15, 0.3, 0.5, 0.2])  # pre_fc style: small but positive
    absolute = mx.array([0.9, 1.2, 1.8, 0.7])
    absolute_qk = mx.array([1.7, -0.008, 2.5, 1.9])  # tiny negative entry, mean well above 1
    delta = mx.array([-0.1, 0.2, 0.8, -0.3])  # unshifted (w - 1)
    w = {
        "pre_fc_norm_embedding.weight": absolute_small,
        "layers.0.input_layernorm.weight": absolute,
        "layers.0.self_attn.q_norm.weight": absolute_qk,
        "layers.0.post_attention_layernorm.weight": delta,
        "fc.weight": mx.zeros((4, 8)),
    }
    healed, shifted = heal_delta_norms(w)
    assert shifted == ["layers.0.post_attention_layernorm.weight"]
    assert mx.array_equal(healed["layers.0.post_attention_layernorm.weight"], delta + 1.0).item()
    for k in ("pre_fc_norm_embedding.weight", "layers.0.input_layernorm.weight", "layers.0.self_attn.q_norm.weight", "fc.weight"):
        assert healed[k] is w[k]


def test_load_sidecar_accepts_unprefixed_head_only_repo(tmp_path: Path) -> None:
    """mlx-community/*-MTP-Nbit repos store the head's local tree without an mtp. prefix."""
    args = _args()
    src = Qwen35MtpHead(args)
    weights = cast(dict[str, mx.array], dict(tree_flatten(src.parameters())))
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), weights)  # pyright: ignore[reportUnknownMemberType]
    loaded = load_sidecar(path)
    assert set(loaded) == set(weights)
    prefixed = tmp_path / "mtp.safetensors"
    mx.save_safetensors(  # pyright: ignore[reportUnknownMemberType]
        str(prefixed), {"language_model.mtp." + k: v for k, v in weights.items()}
    )
    assert set(load_sidecar(prefixed)) == set(weights)
