"""Exact conv/SSM-state rollback for GatedDeltaNet after a multi-token verify.

A full-attention layer rolls a rejected draft back with ``KVCache.trim``; a
GatedDeltaNet layer cannot - its conv window and recurrent state were advanced
destructively over the whole chunk. ``CapturingGatedDeltaNet`` is the fork's
``GatedDeltaNet`` with one addition: while a ``GdnCaptureSink`` is armed, each
call records the pre-call state plus the per-token kernel inputs it computed,
and ``rollback(n_keep)`` re-derives the state after exactly ``n_keep`` of those
tokens by re-running the SAME kernel over the accepted prefix (zero steps when
nothing was accepted). The recurrence is strictly sequential inside the
kernel, so the prefix re-run reproduces the state a plain forward over just
those tokens would have left.

Installation is a per-instance ``__class__`` swap (``install_gdn_capture``) so
parameters and the pipeline wrappers around the layer are untouched. With no
sink armed the override is a plain pass-through, so prefill and ordinary
decode are unaffected.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import DecoderLayer, GatedDeltaNet

from exo.worker.engines.mlx.auto_parallel import CustomMlxLayer


@dataclass
class GdnRecord:
    cache: ArraysCache
    n_tokens: int
    conv_input: mx.array  # [B, kernel-1 + T, conv_dim]
    q: mx.array
    k: mx.array
    v: mx.array
    a: mx.array
    b: mx.array
    state0: mx.array | None
    conv_keep: int


@dataclass
class GdnCaptureSink:
    armed: bool = False
    records: list[GdnRecord] = field(default_factory=list)

    def arm(self) -> None:
        self.records.clear()
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def clear(self) -> None:
        self.records.clear()
        self.armed = False


class CapturingGatedDeltaNet(GatedDeltaNet):
    """``GatedDeltaNet`` whose multi-token calls can be rolled back.

    Only ever instantiated via ``__class__`` swap on a loaded ``GatedDeltaNet``.
    """

    def _sink(self) -> GdnCaptureSink | None:
        return cast(GdnCaptureSink | None, self.__dict__.get("_mtp_sink"))

    def _use_kernel(self) -> bool:
        return not bool(getattr(self, "training", False))

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        sink = self._sink()
        if sink is None or not sink.armed or cache is None:
            return super().__call__(inputs, mask, cache)
        if mask is not None:
            raise RuntimeError("MTP verify does not support padded/masked SSM input")
        if getattr(self, "sharding_group", None) is not None:
            raise RuntimeError("MTP verify does not support tensor-sharded GatedDeltaNet")

        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        conv_state = cast(mx.array | None, cache[0])
        if conv_state is None:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        n_keep = self.conv_kernel_size - 1
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        kd = self.key_dim
        q = conv_out[..., :kd].reshape(B, S, self.num_k_heads, self.head_k_dim)
        k = conv_out[..., kd : 2 * kd].reshape(B, S, self.num_k_heads, self.head_k_dim)
        v = conv_out[..., 2 * kd :].reshape(B, S, self.num_v_heads, self.head_v_dim)

        state0 = cast(mx.array | None, cache[1])
        inv_scale = 1.0 / math.sqrt(self.head_k_dim)
        q = inv_scale * q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state0,
            None,
            use_kernel=self._use_kernel(),
        )
        cache[1] = state
        _advance(cache, S)

        sink.records.append(
            GdnRecord(
                cache=cache,
                n_tokens=S,
                conv_input=conv_input,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                state0=state0,
                conv_keep=n_keep,
            )
        )

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    def rollback_record(self, rec: GdnRecord, n_keep_tokens: int) -> None:
        """Reset ``rec.cache`` to the state after the first ``n_keep_tokens`` of
        the recorded call (0 <= n_keep_tokens <= rec.n_tokens)."""
        if n_keep_tokens < 0 or n_keep_tokens > rec.n_tokens:
            raise ValueError(f"n_keep_tokens={n_keep_tokens} outside [0, {rec.n_tokens}]")
        if n_keep_tokens == rec.n_tokens:
            return
        cache = rec.cache
        cache[0] = mx.contiguous(
            rec.conv_input[:, n_keep_tokens : n_keep_tokens + rec.conv_keep, :]
        )
        if n_keep_tokens == 0:
            cache[1] = rec.state0
        else:
            _, state = gated_delta_update(
                rec.q[:, :n_keep_tokens],
                rec.k[:, :n_keep_tokens],
                rec.v[:, :n_keep_tokens],
                rec.a[:, :n_keep_tokens],
                rec.b[:, :n_keep_tokens],
                self.A_log,
                self.dt_bias,
                rec.state0,
                None,
                use_kernel=self._use_kernel(),
            )
            cache[1] = state
        undo = rec.n_tokens - n_keep_tokens
        for name in ("lengths", "left_padding"):
            arr = cast(mx.array | None, getattr(cache, name, None))
            if arr is not None:
                setattr(cache, name, arr + undo)


def _advance(cache: ArraysCache, n: int) -> None:
    advance = cast(Callable[[int], object], getattr(cache, "advance"))  # noqa: B009
    advance(n)


@dataclass
class GdnCaptureSet:
    """All capturing GDN layers of one model shard, driven as a unit."""

    layers: list[CapturingGatedDeltaNet]
    sinks: list[GdnCaptureSink]

    def arm(self) -> None:
        for s in self.sinks:
            s.arm()

    def disarm(self) -> None:
        for s in self.sinks:
            s.disarm()

    def rollback(self, n_keep_tokens: int) -> None:
        """Roll every captured layer back to ``n_keep_tokens`` of its last call."""
        for layer, sink in zip(self.layers, self.sinks, strict=True):
            if len(sink.records) != 1:
                raise RuntimeError(
                    f"expected exactly one captured GDN call per layer, got {len(sink.records)}"
                )
            layer.rollback_record(sink.records[0], n_keep_tokens)
            sink.records.clear()

    def clear(self) -> None:
        for s in self.sinks:
            s.clear()


def _unwrap(layer: nn.Module) -> nn.Module:
    if isinstance(layer, CustomMlxLayer):
        return cast(nn.Module, layer.original_layer)
    return layer


def install_gdn_capture(layers: list[nn.Module]) -> GdnCaptureSet:
    """Swap every ``GatedDeltaNet`` in ``layers`` (possibly pipeline-wrapped) to
    ``CapturingGatedDeltaNet`` and return the set that drives them."""
    gdns: list[CapturingGatedDeltaNet] = []
    sinks: list[GdnCaptureSink] = []
    for layer in layers:
        base = _unwrap(layer)
        if not isinstance(base, DecoderLayer) or not base.is_linear:
            continue
        gdn = base.linear_attn
        if not isinstance(gdn, CapturingGatedDeltaNet):
            if type(gdn) is not GatedDeltaNet:
                raise RuntimeError(
                    f"cannot install MTP capture on {type(gdn).__name__}; expected GatedDeltaNet"
                )
            gdn.__class__ = CapturingGatedDeltaNet
        sink = GdnCaptureSink()
        setattr(gdn, "_mtp_sink", sink)  # noqa: B010 - plain attribute, not a parameter
        gdns.append(cast(CapturingGatedDeltaNet, gdn))
        sinks.append(sink)
    return GdnCaptureSet(layers=gdns, sinks=sinks)
