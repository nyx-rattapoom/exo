"""Qwen3.5/3.6 MoE native MTP head: module, sidecar loader, quantisation.

The head is one FULL-attention Qwen3.5 MoE ``DecoderLayer`` plus

    pre_fc_norm_embedding  RMSNorm on embed_tokens(next_token)
    pre_fc_norm_hidden     RMSNorm on the trunk's final hidden state
    fc                     Linear 2H -> H over concat[e, h]
    norm                   RMSNorm before the (shared) lm_head

Sidecar conventions handled by ``load_sidecar``:
  * keys ``language_model.mtp.*`` / ``mtp.*`` -> local tree (``layers.0...``)
  * numbered experts ``mlp.experts.N.{gate,up,down}_proj`` -> stacked ``switch_mlp``
  * pre-stacked ``mlp.{gate,up,down}_proj`` (bf16 sidecars) -> ``switch_mlp``
Mixed-precision sidecars are handled by inferring (bits, group_size) per module
from the stored ``weight``/``scales`` shapes (``quantize_from_weights``).

Measured 2026-10-04 (single process, 16 trajectories x 256 tokens, K=1 argmax
match): TensorFold's naive affine-4bit g64 head 0.883 greedy / 0.877 under exo's
default sampler, bf16 head 0.886 / 0.878 - so the small head loses nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from exo.worker.runner.bootstrap import logger

SIDECAR_CANDIDATES: tuple[str, ...] = (
    "mtp.safetensors",
    "mtp-4bit.safetensors",
    "mtp-8bit.safetensors",
    "model-mtp-head.safetensors",
)

_EXPERT_RE = re.compile(
    r"^(layers\.\d+\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_STACKED_RE = re.compile(
    r"^(layers\.\d+\.mlp\.)(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


class Qwen35MtpHead(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        interval = int(args.full_attention_interval or 4)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        # (layer_idx + 1) % interval == 0 selects DecoderLayer's full-attention branch.
        self.layers = [DecoderLayer(args=args, layer_idx=interval - 1)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def make_cache(self) -> KVCache:
        return KVCache()

    def forward(
        self,
        hidden: mx.array,
        embeds: mx.array,
        cache: KVCache | None = None,
    ) -> tuple[mx.array, mx.array]:
        """``hidden``: [B, T, H] trunk hidden at positions i (post final norm);
        ``embeds``: [B, T, H] embed_tokens(token at position i+1).

        Returns ``(normed, raw)``: ``normed`` feeds lm_head and predicts the
        token at position i+2; ``raw`` is the block output, the hidden input for
        a chained (depth > 1) draft.
        """
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden)
        x = self.fc(mx.concatenate([e, h], axis=-1))
        t = x.shape[1]
        mask: mx.array | str | None
        if cache is not None:
            mask = cache.make_mask(t, return_array=False, window_size=None)
        else:
            mask = "causal" if t > 1 else None
        out = self.layers[0](x, mask=mask, cache=cache)  # pyright: ignore[reportArgumentType]
        return self.norm(out), out


def find_sidecar(model_path: Path, explicit: str | None = None) -> Path | None:
    if explicit is not None:
        p = model_path / explicit
        return p if p.exists() else None
    for name in SIDECAR_CANDIDATES:
        p = model_path / name
        if p.exists():
            return p
    return None


def strip_mtp_prefix(key: str) -> str | None:
    i = key.find("mtp.")
    if i < 0:
        return None
    return key[i + len("mtp.") :]


def load_sidecar(path: str | Path) -> dict[str, mx.array]:
    """Load an MTP sidecar and normalise its keys to ``Qwen35MtpHead``'s tree."""
    raw = cast(dict[str, mx.array], mx.load(str(path)))
    out: dict[str, mx.array] = {}
    experts: dict[tuple[str, str, str], dict[int, mx.array]] = {}
    for k, v in raw.items():
        local = strip_mtp_prefix(k)
        if local is None:
            continue
        m = _EXPERT_RE.match(local)
        if m:
            prefix, idx, proj, part = m.groups()
            experts.setdefault((prefix, proj, part), {})[int(idx)] = v
            continue
        m = _STACKED_RE.match(local)
        if m:
            prefix, proj, part = m.groups()
            local = f"{prefix}switch_mlp.{proj}.{part}"
        out[local] = v
    for (prefix, proj, part), items in experts.items():
        n = max(items) + 1
        if sorted(items) != list(range(n)):
            raise ValueError(f"MTP sidecar is missing experts for {prefix}{proj}.{part}")
        out[f"{prefix}switch_mlp.{proj}.{part}"] = mx.stack(
            [items[i] for i in range(n)], axis=0
        )
    return out


def _module_quant_spec(
    weights: dict[str, mx.array], path: str, in_features: int
) -> dict[str, int] | None:
    """Infer (bits, group_size) for a linear-like module from its stored tensors."""
    w = weights.get(f"{path}.weight")
    s = weights.get(f"{path}.scales")
    if w is None or s is None or w.dtype != mx.uint32:
        return None
    packed_in = w.shape[-1]
    n_groups = s.shape[-1]
    bits = 32 * packed_in // in_features
    group_size = in_features // n_groups
    if bits not in (2, 3, 4, 5, 6, 8) or group_size * n_groups != in_features:
        raise ValueError(
            f"cannot infer quantisation for {path}: weight {w.shape}, scales {s.shape}, in={in_features}"
        )
    return {"bits": bits, "group_size": group_size}


def quantize_from_weights(
    head: nn.Module, weights: dict[str, mx.array]
) -> dict[str, dict[str, int]]:
    """Quantise exactly the modules whose sidecar tensors are quantised, with the
    bits/group_size those tensors imply. Modules stored in bf16 stay bf16."""
    specs: dict[str, dict[str, int]] = {}

    def predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
        if not hasattr(module, "to_quantized"):
            return False
        weight = cast(mx.array, getattr(module, "weight"))  # noqa: B009
        spec = _module_quant_spec(weights, path, int(weight.shape[-1]))
        if spec is None:
            return False
        specs[path] = spec
        return spec

    nn.quantize(head, class_predicate=predicate, mode="affine")
    return specs


def build_head(
    args: TextModelArgs, weights: dict[str, mx.array]
) -> tuple[Qwen35MtpHead, dict[str, dict[str, int]], int]:
    """Build, quantise and strictly load the head. Returns (head, specs, nbytes)."""
    head = Qwen35MtpHead(args)
    specs = quantize_from_weights(head, weights)
    current = cast(dict[str, mx.array], dict(tree_flatten(head.parameters())))
    missing = sorted(set(current) - set(weights))
    extra = sorted(set(weights) - set(current))
    mismatched = [
        (k, current[k].shape, weights[k].shape)
        for k in sorted(set(current) & set(weights))
        if tuple(current[k].shape) != tuple(weights[k].shape)
    ]
    if missing or extra or mismatched:
        raise ValueError(
            "MTP head weights do not match the module tree: "
            f"missing={missing[:8]} extra={extra[:8]} mismatched={mismatched[:4]}"
        )
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    nbytes = sum(
        v.nbytes for _, v in cast(list[tuple[str, mx.array]], tree_flatten(head.parameters()))
    )
    return head, specs, nbytes


def load_head(
    model_path: Path, args: TextModelArgs, explicit_file: str | None = None
) -> Qwen35MtpHead | None:
    sidecar = find_sidecar(model_path, explicit_file)
    if sidecar is None:
        logger.warning(
            f"MTP requested but no sidecar found in {model_path} "
            f"(tried {explicit_file or SIDECAR_CANDIDATES}); MTP disabled for this runner"
        )
        return None
    weights = load_sidecar(sidecar)
    head, specs, nbytes = build_head(args, weights)
    quant = sorted({(s["bits"], s["group_size"]) for s in specs.values()})
    logger.info(
        f"MTP head loaded: {sidecar.name}, {len(weights)} tensors, {nbytes / 2**20:.0f} MiB, "
        f"quantised modules={len(specs)} (bits,group)={quant}"
    )
    return head
