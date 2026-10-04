"""Native multi-token-prediction (MTP) speculative decoding for Qwen3.5/3.6 MoE.

Everything in this package is inert unless the instance's model card carries an
``mtp`` block or ``EXO_MTP_DRAFT`` is set to a positive integer (the number of
draft tokens per round, K; ``0`` forces plain decode). With K > 0 the runner loads the checkpoint's MTP head (an ``mtp*.safetensors`` sidecar) on the
LAST pipeline rank only, and ``mlx_generate`` decodes with
``mtp_generate_step`` instead of mlx-lm's ``stream_generate``.

Modules:
    head          - the head module, sidecar loader and quantisation inference
    gdn_rollback  - a capturing GatedDeltaNet that can roll its conv/SSM state
                    back to any prefix of the last multi-token call
    decode        - prefill-time head priming and the draft/verify/commit loop
"""

from exo.worker.engines.mlx.mtp.config import (
    MTP_DRAFT_TOKENS,
    mtp_enabled,
    resolve_mtp_settings,
)

__all__ = ["MTP_DRAFT_TOKENS", "mtp_enabled", "resolve_mtp_settings"]
