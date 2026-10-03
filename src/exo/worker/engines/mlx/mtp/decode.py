"""MTP draft / verify / commit decode loop for exo's pipeline (or single) runner.

Round structure (K drafts, batch size 1), identical control flow on every rank:

  1. draft    - the drafting rank (last pipeline stage) feeds the head every
                committed-but-unseen (token_{i+1}, hidden_i) pair, takes the
                argmax at the last position as d_1, then chains d_2..d_K from the
                head's own output hidden. Draft ids are broadcast with an all_sum
                (zeros from every other rank) so all ranks see the same list.
  2. verify   - ONE trunk forward over [last_token, d_1, ..., d_K] with the GDN
                capture armed. The pipeline wrappers fire as usual, so the
                final hidden/logits land on every rank via the existing
                all_gather. Position j is sampled with the task's real sampler
                (seeded identically on every rank); draft j+1 is accepted iff
                the sample equals it. The first mismatch (or the bonus sample
                after K accepts) ends the round.
  3. commit   - KV layers trim the rejected tail, GDN layers roll back to the
                accepted prefix by re-running the kernel over it. The accepted
                positions' hiddens are paired with the committed tokens and
                queued for the head.

Output distribution equals plain sampling: every emitted token is a sample
from the trunk's own (processed) logits at its position. Greedy runs are
byte-identical to non-speculative greedy runs.

The head's context is primed during prefill (``prime_during_prefill``) by
tapping the trunk's final-norm output chunk by chunk on the drafting rank, so
the head attends to the whole prompt. Measured 2026-10-04: priming is worth
~+1.3 points of acceptance over an unprimed head.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.worker.engines.mlx.mtp.config import MTP_LOG_EVERY
from exo.worker.engines.mlx.mtp.gdn_rollback import GdnCaptureSet
from exo.worker.engines.mlx.mtp.head import Qwen35MtpHead
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

LogitsProcessor = Callable[[mx.array, mx.array], mx.array]
Sampler = Callable[[mx.array], mx.array]


@dataclass
class MtpRuntime:
    """Per-runner MTP state, attached to the loaded model at load time."""

    draft_tokens: int
    inner: Qwen3_5TextModel
    lm_head: Callable[[mx.array], mx.array]
    gdn: GdnCaptureSet
    group: mx.distributed.Group | None
    is_drafting_rank: bool
    head: Qwen35MtpHead | None  # only on the drafting rank

    def __post_init__(self) -> None:
        if self.is_drafting_rank and self.head is None:
            raise ValueError("drafting rank needs an MTP head")


def get_runtime(model: nn.Module) -> MtpRuntime | None:
    rt = cast(MtpRuntime | None, model.__dict__.get("_exo_mtp_runtime"))
    return rt


def attach_runtime(model: nn.Module, rt: MtpRuntime) -> None:
    # Plain attribute (not an mx.array / dict / list), so nn.Module keeps it out
    # of the parameter tree.
    setattr(model, "_exo_mtp_runtime", rt)  # noqa: B010


class HiddenTap:
    """Swap ``inner.norm`` for a recorder of its OUTPUT (the post-final-norm
    hidden that feeds lm_head), forwarding to the real norm."""

    def __init__(self, inner: Qwen3_5TextModel, on_chunk: Callable[[mx.array], None]):
        self._inner = inner
        self._real_norm = inner.norm
        self._on_chunk = on_chunk

    def __enter__(self) -> HiddenTap:
        real_norm = self._real_norm
        on_chunk = self._on_chunk

        def capture(x: mx.array) -> mx.array:
            y: mx.array = real_norm(x)
            on_chunk(y)
            return y

        self._inner.norm = capture  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.norm = self._real_norm


class _LastHidden:
    def __init__(self) -> None:
        self.value: mx.array | None = None

    def __call__(self, h: mx.array) -> None:
        self.value = h

    def take(self) -> mx.array:
        if self.value is None:
            raise RuntimeError("no hidden captured by the final-norm tap")
        v = self.value
        self.value = None
        return v


class HeadPrimer:
    """Feeds the head with (prompt[i+1], hidden_i) for every chunk prefill
    produces, for positions i <= limit_pos (exclusive upper bound ``limit``)."""

    def __init__(
        self,
        rt: MtpRuntime,
        prompt_tokens: mx.array,
        head_cache: KVCache,
        limit: int,
    ):
        assert rt.head is not None
        self._head = rt.head
        self._inner = rt.inner
        self._prompt = prompt_tokens
        self._cache = head_cache
        self._limit = limit
        self.offset = 0
        self.primed = 0
        self.ignored_chunks = 0

    def __call__(self, hidden: mx.array) -> None:
        t = hidden.shape[1]
        if self.offset + t > self._limit:
            # prefill's trailing re-feeds of the last token (trimmed afterwards)
            self.ignored_chunks += 1
            return
        nxt = self._prompt[self.offset + 1 : self.offset + t + 1][None]
        emb: mx.array = self._inner.embed_tokens(nxt)
        _normed, _raw = self._head.forward(hidden, emb, self._cache)
        keys, values = self._cache.state
        mx.eval([a for a in (keys, values) if a is not None])
        self.offset += t
        self.primed += t


@contextlib.contextmanager
def prime_during_prefill(
    rt: MtpRuntime | None,
    prompt_tokens: mx.array,
    head_cache: KVCache | None,
) -> Iterator[HeadPrimer | None]:
    """Context for exo's prefill. On the drafting rank the head is primed over
    prompt positions [0, N-2) - the positions prefill leaves in the trunk cache.
    Elsewhere this is a no-op."""
    if rt is None or not rt.is_drafting_rank or head_cache is None:
        yield None
        return
    n = int(prompt_tokens.shape[0])
    primer = HeadPrimer(rt, prompt_tokens, head_cache, limit=max(0, n - 2))
    with HiddenTap(rt.inner, primer):
        yield primer
    logger.debug(
        f"MTP head primed over {primer.primed} prompt positions "
        f"({primer.ignored_chunks} trailing chunks ignored)"
    )


def broadcast_drafts(
    drafts: list[int], contribute: bool, k: int, group: mx.distributed.Group | None
) -> list[int]:
    if group is None:
        return drafts
    payload = mx.array(drafts if contribute else [0] * k, dtype=mx.int32)
    summed = mx.distributed.all_sum(
        payload, group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(summed)
    return cast(list[int], summed.tolist())


@dataclass
class MtpStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    emitted: int = 0
    draft_time: float = 0.0
    verify_time: float = 0.0
    rollback_time: float = 0.0
    accepted_hist: list[int] = field(default_factory=list)

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def summary(self) -> str:
        tpr = self.emitted / self.rounds if self.rounds else 0.0
        return (
            f"MTP stats: rounds={self.rounds} emitted={self.emitted} drafted={self.drafted} "
            f"accepted={self.accepted} acceptance={self.acceptance:.3f} tokens/round={tpr:.3f} "
            f"draft={self.draft_time:.2f}s verify={self.verify_time:.2f}s rollback={self.rollback_time:.2f}s"
        )


def _draft(
    rt: MtpRuntime,
    pending: list[tuple[int, mx.array]],
    head_cache: KVCache,
) -> list[int]:
    head = rt.head
    assert head is not None
    k = rt.draft_tokens
    hidden = mx.concatenate([h for _, h in pending], axis=1)
    toks = mx.array([t for t, _ in pending])[None]
    emb: mx.array = rt.inner.embed_tokens(toks)
    normed, raw = head.forward(hidden, emb, head_cache)
    draft = mx.argmax(rt.lm_head(normed[:, -1:, :]), axis=-1)  # [1, 1]
    drafts = [draft]
    for _ in range(1, k):
        emb = rt.inner.embed_tokens(draft)
        normed, raw = head.forward(raw[:, -1:, :], emb, head_cache)
        draft = mx.argmax(rt.lm_head(normed[:, -1:, :]), axis=-1)
        drafts.append(draft)
    if k > 1:
        # chained positions were drafted from head hiddens; they are re-added
        # from trunk hiddens after verification
        head_cache.trim(k - 1)
    ids = mx.concatenate(drafts, axis=1)[0]
    return cast(list[int], ids.tolist())


def mtp_stream_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    rt: MtpRuntime,
    last_tokens: mx.array,
    max_tokens: int,
    sampler: Sampler,
    logits_processors: list[LogitsProcessor],
    cache: KVCacheType,
    head_cache: KVCache | None,
    prompt_tokens_total: int,
) -> Generator[GenerationResponse, None, None]:
    """Drop-in for ``stream_generate(model, tokenizer, prompt=last_tokens, ...)``
    after exo's prefill. ``last_tokens`` are the final two prompt tokens: the
    first is fed to the trunk to recover its hidden, the second is the first
    token the trunk has not seen."""
    if last_tokens.shape[0] != 2:
        raise ValueError("mtp_stream_generate expects the last two prompt tokens")
    if rt.is_drafting_rank and head_cache is None:
        raise ValueError("drafting rank needs a head cache")
    eos_ids = set(cast(set[int], tokenizer.eos_token_ids))
    detokenizer: StreamingDetokenizer = tokenizer.detokenizer
    detokenizer.reset()  # pyright: ignore[reportUnknownMemberType]
    kv_caches = [c for c in cache if isinstance(c, KVCache)]
    stats = MtpStats()
    last_hidden = _LastHidden()

    def response(token: int, logprobs: mx.array, n: int, tic: float, finish: str | None) -> GenerationResponse:
        return GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=False,
            prompt_tokens=int(last_tokens.shape[0]),
            prompt_tps=0.0,
            generation_tokens=n,
            generation_tps=n / max(time.perf_counter() - tic, 1e-9),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish,
        )

    tap: contextlib.AbstractContextManager[object] = (
        HiddenTap(rt.inner, last_hidden) if rt.is_drafting_rank else contextlib.nullcontext()
    )
    try:
        yield from _mtp_rounds(
            model, rt, tap, last_tokens, max_tokens, sampler, logits_processors,
            cache, head_cache, kv_caches, eos_ids, detokenizer, last_hidden, stats, response,
        )
    finally:
        # mlx_generate breaks out of the stream on the final token, so the
        # generator is closed rather than exhausted; log either way.
        rt.gdn.clear()
        logger.info(stats.summary())


def _mtp_rounds(
    model: Model,
    rt: MtpRuntime,
    tap: contextlib.AbstractContextManager[object],
    last_tokens: mx.array,
    max_tokens: int,
    sampler: Sampler,
    logits_processors: list[LogitsProcessor],
    cache: KVCacheType,
    head_cache: KVCache | None,
    kv_caches: list[KVCache],
    eos_ids: set[int],
    detokenizer: StreamingDetokenizer,
    last_hidden: _LastHidden,
    stats: MtpStats,
    response: Callable[[int, mx.array, int, float, str | None], GenerationResponse],
) -> Generator[GenerationResponse, None, None]:
    k = rt.draft_tokens

    def trunk(inputs: list[int]) -> tuple[mx.array, mx.array]:
        logits: mx.array = model(mx.array(inputs)[None], cache=cache)
        hidden = last_hidden.take()
        return logits, hidden

    with tap, wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        t0 = int(last_tokens[0].item())
        t1 = int(last_tokens[1].item())
        # Recover hidden_{N-2} without sampling; its logits predict t1, which we know.
        if rt.is_drafting_rank:
            _, h0 = trunk([t0])
            pending: list[tuple[int, mx.array]] = [(t1, h0[:, -1:, :])]
        else:
            model(mx.array([t0])[None], cache=cache)
            pending = []
        last_token = t1
        fed: list[int] = []  # inputs seen by _step in generate_step terms
        n_emitted = 0
        tic = time.perf_counter()
        finished = False

        while not finished:
            # ---- 1. draft -------------------------------------------------
            td = time.perf_counter()
            if rt.is_drafting_rank:
                assert head_cache is not None
                local = _draft(rt, pending, head_cache)
            else:
                local = [0] * k
            drafts = broadcast_drafts(local, rt.is_drafting_rank, k, rt.group)
            stats.draft_time += time.perf_counter() - td

            # ---- 2. verify ------------------------------------------------
            tv = time.perf_counter()
            inputs = [last_token] + drafts
            rt.gdn.arm()
            logits_all: mx.array = model(mx.array(inputs)[None], cache=cache)
            rt.gdn.disarm()
            hidden_all = last_hidden.take() if rt.is_drafting_rank else None
            # Process every position's logits with its own history, then sample
            # all K+1 positions in ONE sampler call: a single host sync per round.
            rows: list[mx.array] = []
            for j in range(k + 1):
                logits = logits_all[:, j, :]
                if logits_processors:
                    history = mx.array(fed + inputs[: j + 1])
                    for proc in logits_processors:
                        logits = proc(history, logits)
                rows.append(logits)
            logits_rows = mx.concatenate(rows, axis=0)  # [K+1, V]
            logprobs_rows = logits_rows - mx.logsumexp(logits_rows, axis=-1, keepdims=True)
            ys = cast(list[int], sampler(logprobs_rows).tolist())
            committed: list[tuple[int, mx.array]] = []
            n_acc = 0
            for j in range(k + 1):
                committed.append((ys[j], logprobs_rows[j]))
                if j < k and ys[j] == drafts[j]:
                    n_acc += 1
                    continue
                break
            stats.verify_time += time.perf_counter() - tv

            # ---- 3. commit / roll back ---------------------------------
            tr = time.perf_counter()
            n_consumed = n_acc + 1  # last_token + accepted drafts
            if n_consumed < k + 1:
                for c in kv_caches:
                    c.trim(k + 1 - n_consumed)
                rt.gdn.rollback(n_consumed)
            else:
                rt.gdn.clear()
            fed = fed + inputs[:n_consumed]
            if hidden_all is not None:
                pending = [
                    (committed[p][0], hidden_all[:, p : p + 1, :]) for p in range(n_consumed)
                ]
            last_token = committed[-1][0]
            stats.rollback_time += time.perf_counter() - tr
            stats.rounds += 1
            stats.drafted += k
            stats.accepted += n_acc
            stats.accepted_hist.append(n_acc)

            # ---- 4. emit --------------------------------------------------
            for token, logprobs in committed:
                if token in eos_ids:
                    detokenizer.finalize()
                    yield response(token, logprobs, n_emitted + 1, tic, "stop")
                    finished = True
                    break
                detokenizer.add_token(token)
                n_emitted += 1
                stats.emitted += 1
                if n_emitted >= max_tokens:
                    detokenizer.finalize()
                    yield response(token, logprobs, n_emitted, tic, "length")
                    finished = True
                    break
                yield response(token, logprobs, n_emitted, tic, None)
            if MTP_LOG_EVERY and stats.rounds % MTP_LOG_EVERY == 0:
                logger.info(stats.summary())
