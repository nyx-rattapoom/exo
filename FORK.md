# About this branch

`internal-use` is a fork of [`exo-explore/exo`](https://github.com/exo-explore/exo) `main`,
run on a two-node Apple M4 cluster as a pipeline-parallel inference server. It is the
**default branch** of this repository and the lineage the cluster is built from.

| | |
|---|---|
| Fork point / upstream merged | `exo-explore/exo` `main` @ `21a54c5ea0230a3bec1e1a786d200126c7e34ec6` (2026-08-25; merged 2026-09-03 in `b20b4adc`, docs and `justfile` only) |
| Upstream commits not yet merged | 0 as of 2026-09-12 (`git rev-list --count internal-use..upstream/main`) |
| mlx-lm | [`nyx-rattapoom/mlx-lm`](https://github.com/nyx-rattapoom/mlx-lm) **branch `internal-use`**, `uv.lock` at `b523a61bf8a253af854fa1ec11e2f423dd8c9cc4` |
| mlx (darwin) | **stock PyPI `mlx==0.32.2`** + `mlx-metal==0.32.2` (upstream: `mlx==0.32.0` from the `rltakashige/mlx-jaccl-fix-small-recv` git fork) |
| mlx (linux) | unchanged from upstream: `rltakashige` CUDA wheels, `mlx-cuda-1{2,3}==0.32.0` |

Upstream exo pins `mlx-lm` at `rltakashige/mlx-lm@leo/deepseek-v4` and `mlx` at a private
fork. This branch replaces both: `mlx-lm` with our own fork of that same base branch, kept
merged with `ml-explore/mlx-lm` `main` (see its [`FORK.md`](https://github.com/nyx-rattapoom/mlx-lm/blob/internal-use/FORK.md)),
and `mlx` with the PyPI release. Everything else here is a fix upstream did not have when
we needed it.

## The mlx-lm pin is load-bearing

`pyproject.toml:95` reads

```toml
mlx-lm = { git = "https://github.com/nyx-rattapoom/mlx-lm", branch = "internal-use" }
```

The pin is **by branch name**, on purpose: a bare rev on our own fork can be garbage-collected
out from under a future build the moment no ref points at it. That happened on 2026-08-09,
when the pinned rev `6c5c60f6` was left unreachable by an amend-and-force-push (see `6eedd008`).
The consequences:

- **Renaming or deleting mlx-lm's `internal-use` breaks every from-scratch exo build.**
- The pin *follows* that branch. `uv.lock` records a concrete SHA (`b523a61…`), so an existing
  lock stays reproducible, but any `uv lock --upgrade-package mlx-lm` picks up whatever is at
  the branch head. Do not treat mlx-lm `internal-use` as a scratch branch.
- Repinning to plain `ml-explore/mlx-lm` `main` builds and starts cleanly, then **kills every
  runner at model load**: exo imports `mlx_lm.models.deepseek_v4` at module scope, and that
  module exists only on the `leo/deepseek-v4` lineage. The relationship with upstream mlx-lm
  is a merge into our fork, never a repin.

## What this branch diverges from upstream on

`git log --no-merges upstream/main..internal-use` is 21 commits; the net diff touches 12 files.
Grouped by why each exists:

- **Dependencies** (`pyproject.toml`, `uv.lock`, `python/parts.nix`) — `mlx-lm` and darwin `mlx`
  as in the table above. Stock mlx was adopted on 2026-08-30 after measurement: on 2× M4 it
  ties the fork mlx on decode and is +1.8 % on 32k prefill, and it fits 128k context at a
  byte-identical peak *once mlx-lm forces the fused SDPA kernel for head_dim 192/256*
  (that routing lives in the mlx-lm fork, not here). `python/parts.nix` only builds mlx from
  C++ source when uv2nix resolves a source tree; for a PyPI wheel it merges the separate
  `mlx-metal` wheel's `mlx/lib/` payload into the `mlx` output, because nix gives each wheel its
  own store path and `core.*.so`'s `@rpath/libmlx.dylib` would otherwise never resolve.
  ⚠️ The three commits that made this change (`4f08fcbf`, `645a9f04`, `504c672e`) still carry
  "EXPERIMENT BRANCH ONLY — never merge to internal-use" in their messages. That was true when
  written on 2026-08-29; the experiment succeeded and was adopted the next day. The messages
  are stale, the code is production.
- **Custom model cards survive a restart** (`src/exo/worker/main.py`, `2bbbdc1c`). Upstream's
  `_reconcile_custom_cards` treats "on disk but absent from cluster state" as a deletion, and
  state is in-memory and empty after a restart, so pristine `main` deletes every card in
  `EXO_CUSTOM_MODEL_CARDS_DIR` about one second after startup (observed: 16 cards wiped on both
  nodes). This branch remembers which cards state has advertised, adopts never-advertised
  local cards into the cluster, and deletes only cards that were advertised and then removed.
- **`ArraysCache.cache`, not `.state`** (`engines/mlx/cache.py`, `engines/mlx/disaggregated/adapter.py`,
  `7d78c1e6` + `371cf15e`, tests in `test_cache_state_compat.py`, `d3db334b`). mlx-lm #1632 made
  `ArraysCache.state` a `(cache, left_padding, lengths)` tuple; three exo sites used it as a
  plain list of arrays. One raised in `trim_cache`, one raised on inject, and one *silently
  serialised the wrong arrays* onto the disaggregated-prefill wire. `.state` is loosely typed,
  so the strict typecheck passed with all three bugs present — the regression tests are the only
  gate that catches this class.
- **Truncated or unparseable tool calls keep the model's `finish_reason`**
  (`runner/llm_inference/model_output_parsers.py`, `f609a765`). Upstream relabels both as
  `finish_reason="error"`; the non-streaming chat-completions and Responses adapters then raise
  inside the response generator after headers are sent, so the client gets HTTP 200 with an
  empty body. Same shape as upstream PR #2184 for the truncation branch, extended to the
  unparseable branch, which also no longer breaks out of the stream.
- **Discovery diagnostics** (`rust/networking/src/discovery.rs`, `lib.rs`; `c6d816b6`,
  `37a3138e`, `a1497c07`, 2026-08-03). A multicast re-join returning `AddrInUse` no longer
  drops the interface from discovery for the life of the process; a node whose announces reach
  no interface at all now warns (once after five ticks, then once a minute, with the macOS
  local-network-permission explanation) instead of logging at `debug!` that pyo3-log never
  forwards; and `EXO_ZENOH_LOG=<filter>` installs a tracing subscriber so zenoh's own retry
  logs are visible. Unset, behaviour is identical to upstream.
- **`.typings/mlx_lm/models/gated_delta.pyi`** — a leftover. It declares the *fused*
  `gated_delta_kernel(q, k, v, a, b, A_log, dt_bias, state)` signature from the period
  (2026-08-07 → 2026-08-29) when this branch carried its own packed GDN Metal kernel
  (`04409021`, moved into mlx-lm in `92400d09`). The pinned mlx-lm now ships upstream's
  unfused `(q, k, v, g, beta, state)` kernel, exo never references `gated_delta` anywhere, and
  the typecheck is indifferent. Nothing else of the kernel work remains in this tree. This
  stub should be reverted to upstream's; it is listed here so nobody mistakes it for intent.

## Branches in this repository

| Branch | Role |
|---|---|
| `internal-use` | 🟢 live; default; what the cluster is built from |
| `internal-use-legacy` | the fused-GDN-kernel lineage, `d3db334b65e295ae014594bd60d12a78ea4af105`; pairs with mlx-lm `internal-use-legacy`. Kept as the rollback source, not developed |
| `main` | mirror of the upstream fork point |
| `fix/tool-call-truncation-not-error` | topic branch for `f609a765`; already cherry-picked onto `internal-use` |

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

Stamped 2026-09-12. This section goes stale first; the cluster's operator notes are the
source of truth for anything operational.

| | |
|---|---|
| Running closure | `/nix/store/akp9idwgdhjalr9qdz9i6pn791v5wsvl-exo` (venv `4vsjmpm0r8c26gziag6gi0zay1xwmzvi-exo-venv`), built from `f609a765` independently on both nodes to the same store path |
| One-step rollback | `~/exo-prev` → `/nix/store/b3im3sylf477kvchxg508bx47h2icdlv-exo` (the 2026-08-30 build: same dependencies, without `f609a765`) |
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
