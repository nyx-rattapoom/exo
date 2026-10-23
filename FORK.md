# About this branch

`internal-use` is a fork of [`exo-explore/exo`](https://github.com/exo-explore/exo) `main`,
run on a two-node Apple M4 cluster as a pipeline-parallel inference server. It is the
**default branch** of this repository and the lineage the cluster is built from.

| | |
|---|---|
| Base | **rebased** onto `exo-explore/exo` `main` @ `bc412e2230f395e652c23426a5eb88dfa515769b` (2026-10-13) on 2026-10-14; the previous lineage sat on `21a54c5e` (2026-08-25, merged in `b20b4adc`) |
| Upstream commits not yet merged | 0 as of 2026-10-14 (`git rev-list --count internal-use..upstream/main`) |
| mlx-lm | [`nyx-rattapoom/mlx-lm`](https://github.com/nyx-rattapoom/mlx-lm) **branch `internal-use`**, `uv.lock` at `60f738d924436556d0b2f0359757a94c9e46085c` |
| mlx (darwin) | **stock PyPI `mlx==0.32.2`** + `mlx-metal==0.32.2` (upstream: builds `ml-explore/mlx` `main` @ `0e3ff364` from source — not taken, see below) |
| mlx (linux) | unchanged from upstream: `rltakashige` CUDA wheels, `mlx-cuda-1{2,3}==0.32.0` |
| mlx-vlm | PyPI, **`>=0.7.4` on darwin** (locked 0.7.4), upstream's `>=0.3.11` elsewhere (locked 0.4.4) |

Upstream exo pins `mlx-lm` at `rltakashige/mlx-lm@leo/deepseek-v4` and, on darwin, builds `mlx`
from `ml-explore/mlx` `main` source. This branch replaces both: `mlx-lm` with our own fork of that same base branch, kept
merged with `ml-explore/mlx-lm` `main` (see its [`FORK.md`](https://github.com/nyx-rattapoom/mlx-lm/blob/internal-use/FORK.md)),
and `mlx` with the PyPI release. Everything else here is a fix upstream did not have when
we needed it.

## The mlx-lm pin is load-bearing

`pyproject.toml:97` reads

```toml
mlx-lm = { git = "https://github.com/nyx-rattapoom/mlx-lm", branch = "internal-use" }
```

The pin is **by branch name**, on purpose: a bare rev on our own fork can be garbage-collected
out from under a future build the moment no ref points at it. That happened on 2026-08-09,
when the pinned rev `6c5c60f6` was left unreachable by an amend-and-force-push (see `6eedd008`).
The consequences:

- **Renaming or deleting mlx-lm's `internal-use` breaks every from-scratch exo build.**
- The pin *follows* that branch. `uv.lock` records a concrete SHA (`60f738d9…`), so an existing
  lock stays reproducible, but any `uv lock --upgrade-package mlx-lm` picks up whatever is at
  the branch head. Do not treat mlx-lm `internal-use` as a scratch branch.
- Repinning to plain `ml-explore/mlx-lm` `main` builds and starts cleanly, then **kills every
  runner at model load**: exo imports `mlx_lm.models.deepseek_v4` at module scope, and that
  module exists only on the `leo/deepseek-v4` lineage. The relationship with upstream mlx-lm
  is a merge into our fork, never a repin.

## What this branch diverges from upstream on

21 non-merge commits over `upstream/main` `bc412e22`, 30 files net (as of the 2026-10-14 rebase).
SHAs below are the rebased ones; the pre-rebase SHA follows in parentheses where it differs.

- **Dependencies** (`pyproject.toml`, `uv.lock`, `python/parts.nix`; `0012375c`, `0a3bac20`, `1611a885`)
  — mlx-lm from our fork, darwin mlx from PyPI (measured 2026-08-30: decode tie, 32k prefill +1.8 %,
  128k fits). `0012375c` squashes the 12-commit pin history into one commit whose `pyproject.toml` and
  `uv.lock` are byte-identical to the pre-rebase head `14e2a298`. Upstream's switch to a source build of
  `ml-explore/mlx` `main` `0e3ff364` is **deliberately not taken**: it is a different numeric library from
  the one every 32k/128k number here was measured on, so adopting it is a separate experiment.
  `parts.nix` skips the source build for a wheel and merges `mlx-metal`'s dylibs into `mlx`.
  `0a3bac20`/`1611a885` (`645a9f04`/`504c672e`) still say "EXPERIMENT BRANCH ONLY — never merge"; the
  messages are stale, the code is production.
- **`ArraysCache.cache`, not `.state`** (`engines/mlx/cache.py`, `disaggregated/adapter.py`, `11e14123`,
  `3e6a6498`, `7f90c930` (`7d78c1e6`, `371cf15e`, `d3db334b`)) — mlx-lm #1632 made `.state` a 3-tuple;
  two exo sites raised, one silently sent the wrong arrays. Upstream still pins pre-#1632 mlx-lm, so it
  has no such fix. Typecheck cannot see it; `test_cache_state_compat.py` is the gate.
- **Earlier tool calls survive an unparseable last one** (`runner/llm_inference/model_output_parsers.py`,
  `137911e1`, the remainder of `f609a765`) — upstream #2394 (`75d8ad2b`) now keeps the model's
  `finish_reason` for truncated/unparseable calls, which was the core of `f609a765`. One case was still
  lost there: a parsed call followed by an unparseable terminal call dropped the parsed one. `f609a765`'s
  full 10-case test file passes on this code; the edge case fails on plain upstream.
- **Streamed usage in its own `choices: []` event** (`api/adapters/chat_completions.py`, `api/main.py`,
  `3c11bf39` (`14e2a298`)) — upstream has the `stream_options.include_usage` request field but not the
  event. With `stream_options.include_usage`, the finish chunk no longer carries `usage`; a trailing
  `choices: []` event does, per the OpenAI spec. Upstream puts it on the finish chunk, which LiteLLM
  ignores: it falls back to its own tokenizer, logs roughly half the real `prompt_tokens` and no
  `cached_tokens`, so every streamed request read as a prompt-cache miss. Without `include_usage` the
  stream is unchanged. Gated by `TestStreamUsageChunk` in `test_chat_completions_stream.py`.
- **mlx-vlm ≥ 0.7.4 on darwin** (`pyproject.toml`, `uv.lock`; upstream locks 0.4.4) — 0.4.4's vision
  towers pass an `mx.array` as `repeats` to `mx.repeat`; the rltakashige mlx fork accepted that, stock
  mlx 0.32.2 raises `TypeError`, and `batch_generate.submit` catches it and silently serves the request
  text-only (HTTP 200, `prompt_tokens` without image tokens, a hallucinated description). Vision was
  therefore broken from the 2026-08-30 stock-mlx move until 2026-09-30, first fixed by a nix patch of the
  0.4.4 wheel (`e961c4c6`, since removed) and then by this bump, which carries upstream's own fix
  (`Blaizzy/mlx-vlm@1249c7db`). The constraint is **split by platform marker** because upstream's Linux
  CUDA split pins `mlx` at 0.32.0 wheels and 0.7.4 needs ≥ 0.32.2; linux keeps upstream's `>=0.3.11`.
  Darwin runtime additions: `mlx-audio`, `llguidance`, `sounddevice`, `websockets`, `scipy`, and the forced
  `starlette` 0.50 → 1.7 / `fastapi` 0.128 → 0.142 bump under exo's own API server; `datasets`/`pandas`/
  `pyarrow` drop out. `httpx2` is in the `dev` group only so starlette 1.x's `TestClient` stays typed.
  Measured: vision output byte-identical to the patched 0.4.4, 32k throughput and peak a tie.
- **Native MTP speculative decoding** (`engines/mlx/mtp/`, `generator/generate.py`, `utils_mlx.py`,
  `builder.py`; `67d4d4c7`…`fb0f5bc9` (`7ee12061`…`ad48d3eb`), replayed without conflicts) — opt-in with `EXO_MTP_DRAFT=K` (default 0 = inert, byte-identical behaviour). Loads the
  checkpoint's Qwen3.5/3.6 MTP head from an `mtp*.safetensors` sidecar on the LAST pipeline rank, primes
  it from the trunk's final-norm output during prefill, drafts K tokens per round (broadcast by
  `all_sum`), verifies them in one multi-token trunk forward through the normal pipeline wrappers, and
  rolls GatedDeltaNet conv/SSM state back exactly by re-running the kernel over the accepted prefix
  (`gdn_rollback.py`). Forces `SequentialGenerator` (batch size 1). Tokens are sampled from the trunk's
  own logits, so the output distribution is unchanged. Gated by `test_mtp_head.py` /
  `test_mtp_gdn_rollback.py`. Head: TensorFold's naive 4-bit g64 sidecar measured 0.883 K=1 acceptance
  vs 0.886 for bf16 (2026-10-04). **Per-card since 2026-10-05**: `ModelCard.mtp = {draft_tokens, head_file}`
  turns MTP on for ONE card's instances (the runtime attaches to that instance's model; `builder.py` picks
  `SequentialGenerator` only when a runtime is attached, so every other card keeps `BatchGenerator`), and
  `ModelCard.weights_repo` lets such a card serve another card's weights directory without a second
  download or any Hugging Face fetch for the alias id (`impl_shard_downloader.weights_shard` runs
  download/status against the target repo and re-keys progress to the alias; `utils_mlx` loads/tokenises
  from `weights_model_id()`). `EXO_MTP_DRAFT` still overrides per runner for trials; `EXO_MTP_DRAFT=0` is
  the kill switch. Sidecar loader heals zero-centred RMSNorm gains per tensor (entries below −0.05) and
  casts fp16 sidecars to bf16; handles numbered-expert, pre-stacked and unprefixed head-only layouts.
  Cards with `mtp = None` and no `weights_repo` take the unchanged code path. Gated additionally by
  `test_model_card_mtp_alias.py`, `test_weights_alias.py`, `test_mtp_per_card.py`.
- **Discovery diagnostics** (`rust/networking/`, `a97df660`, `a88d44bb` (`c6d816b6`, `a1497c07`)) —
  `AddrInUse` re-join keeps the interface (upstream's `discovery.rs` still drops it into `_ => {}`), and
  `EXO_ZENOH_LOG` exposes zenoh logs. Inert unset.
- **Prefix-cache recency only for continuations** (`engines/mlx/cache.py` `get_kv_cache`,
  `test_prefix_cache_recency.py`, `4a49c396`, 2026-10-24) — upstream marks the matched entry as used on
  every lookup, even one that only shares a system prompt and then adds its own entry. A stale entry every
  new session matches stays most-recently-used, so when the byte budget binds, two live sessions evict each
  other every turn (measured: 86k vs 36k tokens prefilled, +81 % time to first token). Only a hit that
  reaches the entry's end (minus the 2 tokens prefill leaves out) refreshes it now. The rule depends only on
  tokens and entries, so every rank evicts the same entry. A no-op under our default memory threshold.

## Dropped on the 2026-10-14 rebase (upstream already fixed them)

- `2bbbdc1c` custom cards adopted on cold start → upstream `e0cf3429` (#2336) re-announces saved cards
  every 10 s until the state has them, deletes a card only after it was seen and then removed, and fixes
  the origin-node wart (the node a card was added on now writes its own TOML).
- `37a3138e` "no announce left the host" warning → upstream `3445d6ad` (#2320), same macOS Local Network hint.
- `f609a765` core → upstream `75d8ad2b` (#2394); the remainder is kept as `137911e1` (above).
- `04409021` + `92400d09` packed GDN kernel added to exo, then moved into mlx-lm → net zero. Dropping both
  also removes the stale fused-kernel `.typings/mlx_lm/models/gated_delta.pyi`; `.typings/` now equals upstream's.
- `e961c4c6` mlx-vlm 0.4.4 nix patch → already removed by the mlx-vlm 0.7.4 bump.
- Merge commit `b20b4adc` → a rebase drops merges.

## Branches in this repository

| Branch | Role |
|---|---|
| `internal-use` | 🟢 live; default; what the cluster is built from |
| `internal-use-legacy` | the fused-GDN-kernel lineage, `d3db334b65e295ae014594bd60d12a78ea4af105`; pairs with mlx-lm `internal-use-legacy`. Kept as the rollback source, not developed |
| `main` | mirror of the upstream fork point |
| `fix/tool-call-truncation-not-error` | topic branch for `f609a765`; already cherry-picked onto `internal-use` |
| `feat/mtp-pipeline` | reference: the native MTP speculative-decoding implementation as developed (2026-10-04/05); fast-forwarded into `internal-use` once the per-card version passed its gates. Kept, not rewritten |

Anything else is new work or a keep-alive someone re-added. The matching mlx-lm fork has
`internal-use`, `internal-use-legacy`, `leo/deepseek-v4` (the base upstream exo pins) and `main`.

## Working in a checkout

**`app/EXO/uninstall-exo.sh` is deleted in the working tree of every cluster checkout, not in
this branch.** The expected `git status` is exactly

```
 D app/EXO/uninstall-exo.sh
```

The script `rm -rf`s `~/.exo` — the model weights and every model card — and destroys the
networking layer that the deleted EXO.app used to maintain. **Never run it.** Two build-procedure
consequences follow from it being a working-tree deletion:

- The flake builds the **dirty working tree**, so a `git reset --hard` (or `git checkout .`,
  `git stash pop`, …) restores the file and the node then produces a *different* store path from
  a node where it is still deleted. Both nodes must show the ` D` line before building; if they
  disagree, re-delete the file rather than trusting a green build.
- The flake also **excludes untracked files**. A new source file must be `git add`ed before
  `nix build`, or the closure silently omits it and every runner crashes at import — long after
  the build reported success.

Other things that bit us: `nix build` leaves a `./result` symlink that is an indirect GC root,
so delete it after each build or retired closures never get collected; `nix build` on its own
does **not** run the type gate — `nix build .#checks.aarch64-darwin.typecheck` does, and it is a
real pass/fail (`0 errors` expected); `$?` after `nix build … | tail` is `tail`'s exit code.

## Deployed state

Stamped 2026-10-24. This section goes stale first; the cluster's operator notes are the
source of truth for anything operational.

| | |
|---|---|
| Running closure | built from this branch's head (the prefix-cache recency fix over the 2026-10-14 rebase), independently on both nodes to the same store path. Store paths live in the operator notes, not here: this file is flake source, so naming a hash here would change it |
| One-step rollback | `~/exo-rollback-jrr9jfs5` → the 2026-10-15 build from `8d5f5226` (the rebase, without the recency fix). `~/exo-rollback-*`, `~/exo-prev` and `~/exo-archive-*` roots hold the older builds |
| Deploy step | re-point the `~/exo-current` GC root and restart the `exo` tmux session on both nodes; the supervisor resolves the binary through that symlink |

## Installing / building

```sh
uv sync --extra mlx                                  # checkout venv (single-process work only)
nix build                                            # darwin closure
nix build .#checks.aarch64-darwin.typecheck          # the type gate
```

A checkout venv can run one process and the unit tests, but two checkout-venv nodes do **not**
peer with each other — every end-to-end cluster test needs a `nix build` + GC-root deploy.
Everything in the [upstream README](./README.md) otherwise applies.
