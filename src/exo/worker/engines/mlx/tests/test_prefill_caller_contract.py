# type: ignore
"""prefill(prompt[:-1]) must leave every cache entry holding prompt[:-2].

The production contract, as used by both callers (batch_generate.py and
mlx_generate): hand ``prefill()`` the prompt minus its last token, then restart
decoding from ``prompt[-2:]``. So ``prefill(X)`` must leave the KV entries at
``len(X) - 1`` (via ``trim(2)`` after mlx-lm's stream_generate adds the final
prompt token plus one generated token) and must restore the recurrent
(SSM/conv) entries from the snapshot taken at ``len(X) - 1``
(``snapshots[-2]``). ``pipeline_parallel_prefill`` reproduces the same
end state by design ("add +1 entry to match stream_generate").

2026-09-15 lesson: a harness that calls ``prefill(prompt)`` with the WHOLE
prompt and compares against a ``prompt[-2:]`` restart is off by one by
construction, on every code version. Exactly such harnesses "confirmed" an
SSM/KV off-by-one on 2026-09-04 and again on 2026-09-15, and the resulting
"fix" (dropped from the branch) made the pipeline branch one token SHORT in
production. These tests pin the harness to the real caller convention, drive
the real ``prefill()`` on a 2-rank ring above and below the 4096-token branch
point and single-process across several chunks, assert the KV offset AND the
restored snapshot position, and require the decode to be bit-identical to a
single-process reference that was given ``prompt[:-2]`` directly.
"""

import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import numpy as np
import pytest

from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.models.model_cards import ModelCard, ModelTask

N_PIPELINE_BRANCH = 4200  # >= 4096 -> pipeline_parallel_prefill
N_STREAM_BRANCH = 2000  # < 4096 -> stream_generate branch on a pipeline model
N_SINGLE_MULTI_CHUNK = 6000  # single process, group=None -> stream_generate, 2 chunks
N_STEPS = 8  # greedy decode steps to compare


def _prefill_input(mx, p):
    """What production hands prefill(): the prompt MINUS its last token.

    batch_generate.py (`prefill(..., prompt_tokens[:-1], ...)`, then
    `last_tokens = prompt_tokens[-2:]`) and mlx_generate do exactly this. A test
    that hands prefill() the whole prompt tests a contract nothing calls.
    """
    return mx.array(p[:-1])

VOCAB = 512

CFG = dict(
    model_type="qwen3_5_moe",
    text_config=dict(
        model_type="qwen3_5_moe",
        vocab_size=VOCAB,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        full_attention_interval=int(os.environ.get("FAI", "2")),
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=256,
        moe_intermediate_size=256,
        norm_topk_prob=True,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    ),
)


class _FakeTokenizer:
    """The minimum mlx-lm's TokenizerWrapper + NaiveStreamingDetokenizer touch when
    stream_generate is given an array prompt and stopped after its first yield."""

    eos_token_id = 0
    bos_token = None
    chat_template = None

    def get_vocab(self):
        return {}

    def encode(self, text, **_kwargs):
        return [ord(c) % VOCAB for c in text]

    def decode(self, ids, **_kwargs):
        return " ".join(str(int(i)) for i in ids)


def _prompt_tokens(n):
    rng = np.random.default_rng(1234)
    return rng.integers(1, VOCAB, size=n).astype(np.int32)


def _build():
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
    import exo.worker.engines.mlx.auto_parallel  # noqa: F401

    module = __import__("mlx_lm.models.qwen3_5_moe", fromlist=["Model", "ModelArgs"])
    mx.random.seed(0)
    m = module.Model(module.ModelArgs(**CFG))

    def _to_bf16(_p, v):
        if hasattr(v, "dtype") and v.dtype in (mx.float16, mx.float32, mx.bfloat16):
            return v.astype(mx.bfloat16)
        return v

    m.update(tree_map_with_path(_to_bf16, m.parameters()))
    mx.eval(m.parameters())
    return mx, m


def _greedy(mx, m, cache, first_input, steps):
    """Feed first_input (2-D array), then greedily decode `steps` tokens."""
    logits = m(first_input, cache=cache)
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    out = [int(tok.item())]
    last_logits = np.asarray(logits[:, -1, :].astype(mx.float32))
    for _ in range(steps - 1):
        logits = m(tok, cache=cache)
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        out.append(int(tok.item()))
    return out, last_logits


def _chunked(mx, m, arr, cache, step=2048):
    i = 0
    while i < arr.shape[1]:
        n = min(step, arr.shape[1] - i)
        mx.eval(m(arr[:, i : i + n], cache=cache))
        i += n


def _reference(mx, m, p):
    """Three single-process hypotheses for what the cache holds at the [-2:]
    restart. H0 is the contract; H1/H2 are the two off-by-one failure modes.

      H0 (correct): every cache entry holds prompt[:-2].
      H1 (SSM one token ahead): KV holds prompt[:-2], SSM/conv holds prompt[:-1].
      H2 (SSM two tokens ahead): KV holds prompt[:-2], SSM/conv holds prompt.
    """
    from exo.worker.engines.mlx.generator.generate import is_non_trimmable_cache_entry

    out = {}
    c0 = m.make_cache()
    _chunked(mx, m, mx.array(p[:-2][None]), c0)
    out["toks"], out["last"] = _greedy(mx, m, c0, mx.array(p[-2:][None]), N_STEPS)

    c1 = m.make_cache()
    _chunked(mx, m, mx.array(p[:-1][None]), c1)
    for c in c1:
        if not is_non_trimmable_cache_entry(c):
            c.trim(1)
    out["toks_h1"], out["last_h1"] = _greedy(mx, m, c1, mx.array(p[-2:][None]), N_STEPS)

    c2 = m.make_cache()
    _chunked(mx, m, mx.array(p[None]), c2)
    for c in c2:
        if not is_non_trimmable_cache_entry(c):
            c.trim(2)
    out["toks_h2"], out["last_h2"] = _greedy(mx, m, c2, mx.array(p[-2:][None]), N_STEPS)
    return out


def _check_cache_position(cache, snaps, want, label):
    from exo.worker.engines.mlx.generator.generate import is_non_trimmable_cache_entry

    kv = [(i, c.offset) for i, c in enumerate(cache) if not is_non_trimmable_cache_entry(c)]
    snap_count = snaps[-1].token_count if snaps else None
    print(f"[{label}] KV offsets (want {want}): {kv}", flush=True)
    print(
        f"[{label}] restored SSM snapshot token_count (want {want}): {snap_count}; "
        f"all snapshots={[s.token_count for s in snaps]}",
        flush=True,
    )
    bad = [(i, o) for i, o in kv if o != want]
    assert not bad, f"[{label}] KV offset must be {want} after prefill, got {bad}"
    assert snap_count == want, (
        f"[{label}] prefill() must hand back an SSM snapshot at {want} "
        f"(what it restored into the cache), got {snap_count}"
    )


def _ref_worker(out_path, n_prompt, q):
    try:
        mx, m = _build()
        ref = _reference(mx, m, _prompt_tokens(n_prompt))
        np.savez(out_path, **{k: np.asarray(v) for k, v in ref.items()})
        q.put((-1, True, "ok"))
    except Exception:
        import traceback

        q.put((-1, False, traceback.format_exc()[-1500:]))


def _pipe_worker(out_path, n_prompt, rank, world_size, layer_splits, q):
    os.environ["MLX_RANK"] = str(rank)
    try:
        import mlx.core as mx
        import exo.worker.engines.mlx.auto_parallel as ap
        from exo.worker.engines.mlx.generator.generate import prefill
        from exo.shared.types.worker.shards import PipelineShardMetadata

        g = mx.distributed.init(backend="ring", strict=True)
        mx_, m = _build()

        start_layer, end_layer = layer_splits[rank]
        total_layers = layer_splits[-1][1]
        shard_meta = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId("test/qwen3_5_moe"),
                storage_size=Memory.from_gb(1),
                n_layers=total_layers,
                hidden_size=512,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
                backends=[Backend.MlxMetal],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=total_layers,
        )
        gen = ap.pipeline_auto_parallel(m, g, shard_meta)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            m = stop.value

        p = _prompt_tokens(n_prompt)
        cache = m.make_cache()
        _tps, _n, snaps = prefill(
            model=m,
            tokenizer=_FakeTokenizer(),
            sampler=lambda x: mx_.argmax(x, axis=-1),
            prompt_tokens=_prefill_input(mx_, p),
            cache=cache,
            group=g,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )
        _check_cache_position(cache, snaps, len(p) - 2, f"rank {rank} N={n_prompt}")

        toks, last = _greedy(mx_, m, cache, mx_.array(p[-2:][None]), N_STEPS)
        if rank == world_size - 1:
            np.savez(out_path, toks=np.asarray(toks), last=last)
        q.put((rank, True, "ok"))
    except Exception:
        import traceback

        q.put((rank, False, traceback.format_exc()[-1500:]))


def _single_worker(out_path, n_prompt, q):
    """Single process, group=None: the stream_generate branch at any length."""
    try:
        mx, m = _build()
        from exo.worker.engines.mlx.generator.generate import prefill

        p = _prompt_tokens(n_prompt)
        ref = _reference(mx, m, p)

        cache = m.make_cache()
        _tps, _n, snaps = prefill(
            model=m,
            tokenizer=_FakeTokenizer(),
            sampler=lambda x: mx.argmax(x, axis=-1),
            prompt_tokens=_prefill_input(mx, p),
            cache=cache,
            group=None,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )
        _check_cache_position(cache, snaps, len(p) - 2, f"single N={n_prompt}")
        toks, last = _greedy(mx, m, cache, mx.array(p[-2:][None]), N_STEPS)
        np.savez(
            out_path,
            got_toks=np.asarray(toks),
            got_last=last,
            **{k: np.asarray(v) for k, v in ref.items()},
        )
        q.put((0, True, "ok"))
    except Exception:
        import traceback

        q.put((0, False, traceback.format_exc()[-1500:]))


def _hostfile(world_size, base_port):
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
    return f.name


def _assert_bit_exact(ref, got, label):
    h0, h1, h2 = list(ref["toks"]), list(ref["toks_h1"]), list(ref["toks_h2"])
    pp = list(got["toks"])
    d0 = float(np.abs(ref["last"] - got["last"]).max())
    d1 = float(np.abs(ref["last_h1"] - got["last"]).max())
    print(f"\n[{label}] H0 (correct)      : {h0}")
    print(f"[{label}] H1 (SSM +1 ahead) : {h1}")
    print(f"[{label}] H2 (SSM +2 ahead) : {h2}")
    print(f"[{label}] prefill()         : {pp}")
    print(f"[{label}] matches: H0={h0 == pp} H1={h1 == pp} H2={h2 == pp}")
    print(f"[{label}] max|logit diff| vs H0={d0:.6g} vs H1={d1:.6g}")
    assert h0 == pp, (
        f"[{label}] prefill is not bit-exact; matches H1 (SSM +1): {h1 == pp}, "
        f"matches H2 (SSM +2): {h2 == pp}"
    )
    assert d0 == 0.0


@pytest.mark.skipif(os.sys.platform != "darwin", reason="MLX distributed requires Metal")
@pytest.mark.parametrize(
    "n_prompt,base_port",
    [(N_PIPELINE_BRANCH, 32500), (N_STREAM_BRANCH, 32510)],
    ids=["pipeline-branch-4200", "stream-generate-branch-2000"],
)
def test_pipeline_prefill_hybrid_ssm(tmp_path, n_prompt, base_port):
    ctx = mp.get_context("spawn")
    world_size = int(os.environ.get("WS", "2"))
    os.environ["MLX_HOSTFILE"] = _hostfile(world_size, base_port)
    per = 4 // world_size
    layer_splits = [(r * per, (r + 1) * per) for r in range(world_size)]

    ref_path = str(tmp_path / "ref.npz")
    pipe_path = str(tmp_path / "pipe.npz")

    q: Any = ctx.Queue()
    r = ctx.Process(target=_ref_worker, args=(ref_path, n_prompt, q))
    r.start()
    r.join(600)
    ref_res = q.get(timeout=600)

    procs = [
        ctx.Process(
            target=_pipe_worker,
            args=(pipe_path, n_prompt, rank, world_size, layer_splits, q),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(600)
    results = [ref_res] + [q.get(timeout=600) for _ in range(world_size)]
    for rank, ok, payload in results:
        if not ok:
            pytest.fail(f"[rank {rank}] FAIL:\n{payload}")

    _assert_bit_exact(np.load(ref_path), np.load(pipe_path), f"pipeline ws={world_size} N={n_prompt}")


@pytest.mark.skipif(os.sys.platform != "darwin", reason="MLX requires Metal")
def test_single_process_prefill_multi_chunk(tmp_path):
    ctx = mp.get_context("spawn")
    out_path = str(tmp_path / "single.npz")
    q: Any = ctx.Queue()
    w = ctx.Process(target=_single_worker, args=(out_path, N_SINGLE_MULTI_CHUNK, q))
    w.start()
    w.join(900)
    rank, ok, payload = q.get(timeout=900)
    if not ok:
        pytest.fail(f"[single] FAIL:\n{payload}")
    saved = np.load(out_path)
    got = {"toks": saved["got_toks"], "last": saved["got_last"]}
    _assert_bit_exact(saved, got, f"single N={N_SINGLE_MULTI_CHUNK}")
