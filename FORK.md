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

21 non-merge commits over `upstream/main`, 12 files net.

- **Dependencies** (`pyproject.toml`, `uv.lock`, `python/parts.nix`) — mlx-lm from our fork, darwin mlx
  from PyPI (measured 2026-08-30: decode tie, 32k prefill +1.8 %, 128k fits). `parts.nix` skips the
  source build for a wheel and merges `mlx-metal`'s dylibs into `mlx`. Commits `4f08fcbf`/`645a9f04`/
  `504c672e` still say "EXPERIMENT ONLY — never merge"; the messages are stale, the code is production.
- **Custom model cards survive a restart** (`worker/main.py`, `2bbbdc1c`) — upstream deletes every card
  that empty post-restart state has not advertised; we adopt them instead.
- **`ArraysCache.cache`, not `.state`** (`engines/mlx/cache.py`, `disaggregated/adapter.py`, `7d78c1e6`,
  `371cf15e`, `d3db334b`) — mlx-lm #1632 made `.state` a 3-tuple; two exo sites raised, one silently sent
  the wrong arrays. Typecheck cannot see it; `test_cache_state_compat.py` is the gate.
- **Tool calls keep the model's `finish_reason`** (`runner/llm_inference/model_output_parsers.py`,
  `f609a765`) — upstream relabels truncated/unparseable calls `"error"`, which becomes an empty HTTP 200.
  Extends upstream PR #2184 to the unparseable branch.
- **Discovery diagnostics** (`rust/networking/`, `c6d816b6`, `37a3138e`, `a1497c07`) — `AddrInUse` re-join
  keeps the interface, "no announce left the host" warns, `EXO_ZENOH_LOG` exposes zenoh logs. Inert unset.
- **`.typings/mlx_lm/models/gated_delta.pyi`** — stale fused-kernel signature from the packed-GDN period
  (`04409021`/`92400d09`); exo never references it. Should be reverted to upstream's stub.

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
