"""Attach the MTP runtime (head + GDN capture) to a loaded, sharded model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel, TextModel, TextModelArgs

from exo.worker.engines.mlx.mtp.config import MTP_DRAFT_TOKENS, MTP_HEAD_FILE, mtp_enabled
from exo.worker.engines.mlx.mtp.decode import MtpRuntime, attach_runtime
from exo.worker.engines.mlx.mtp.gdn_rollback import install_gdn_capture
from exo.worker.engines.mlx.mtp.head import load_head
from exo.worker.engines.mlx.auto_parallel import get_inner_model
from exo.worker.runner.bootstrap import logger


def _text_model(model: nn.Module) -> TextModel | None:
    lm = getattr(model, "language_model", None)
    if isinstance(lm, TextModel):
        return lm
    if isinstance(model, TextModel):
        return model
    return None


def _text_args(model_path: Path) -> TextModelArgs:
    with open(model_path / "config.json") as f:
        config = cast(dict[str, object], json.load(f))
    tc = config.get("text_config", config)
    return TextModelArgs.from_dict(cast(dict[str, object], tc))


def maybe_attach_mtp(
    model: nn.Module,
    model_path: Path,
    group: mx.distributed.Group | None,
    is_last_rank: bool,
) -> bool:
    """Wire MTP into ``model`` when ``EXO_MTP_DRAFT`` > 0 and the model is a
    Qwen3.5/3.6 text model with an MTP sidecar. Returns True when attached.

    The head is loaded on the last pipeline rank only (every rank already sees
    the final hidden through the pipeline all_gather; the drafts are broadcast).
    """
    if not mtp_enabled():
        return False
    text_model = _text_model(model)
    inner = get_inner_model(model)
    if text_model is None or not isinstance(inner, Qwen3_5TextModel):
        logger.warning(
            f"EXO_MTP_DRAFT={MTP_DRAFT_TOKENS} but {type(model).__name__} is not a "
            "Qwen3.5/3.6 text model; MTP disabled for this runner"
        )
        return False

    head = None
    if is_last_rank:
        head = load_head(model_path, _text_args(model_path), MTP_HEAD_FILE)
        if head is None:
            return False

    layers = cast(list[nn.Module], model.layers)
    gdn = install_gdn_capture(layers)
    lm_head = text_model.lm_head

    rt = MtpRuntime(
        draft_tokens=MTP_DRAFT_TOKENS,
        inner=inner,
        lm_head=lm_head,
        gdn=gdn,
        group=group,
        is_drafting_rank=is_last_rank,
        head=head,
    )
    attach_runtime(model, rt)
    logger.info(
        f"MTP config: {{'draft_tokens': {MTP_DRAFT_TOKENS}, 'drafting_rank': {is_last_rank}, "
        f"'gdn_layers': {len(gdn.layers)}, 'head_loaded': {head is not None}}}"
    )
    return True
