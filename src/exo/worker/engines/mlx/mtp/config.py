import os


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


# Number of MTP draft tokens per decode round. 0 (default) disables MTP entirely.
MTP_DRAFT_TOKENS: int = _int_env("EXO_MTP_DRAFT", 0)

# Optional explicit sidecar filename inside the model directory. When unset the
# loader tries the known names in order (see head.find_sidecar).
MTP_HEAD_FILE: str | None = os.environ.get("EXO_MTP_HEAD_FILE") or None

# Log one line of acceptance statistics every N rounds (0 = only at the end).
MTP_LOG_EVERY: int = _int_env("EXO_MTP_LOG_EVERY", 0)

if MTP_DRAFT_TOKENS < 0:
    raise ValueError(f"EXO_MTP_DRAFT must be >= 0, got {MTP_DRAFT_TOKENS}")


def mtp_enabled() -> bool:
    return MTP_DRAFT_TOKENS > 0


def resolve_mtp_settings(
    card_draft_tokens: int | None, card_head_file: str | None
) -> tuple[int, str | None]:
    """Effective (draft_tokens, head_file) for one instance.

    The card decides (``ModelCard.mtp``); a non-empty ``EXO_MTP_DRAFT`` in the
    runner environment overrides it for trials, and ``EXO_MTP_DRAFT=0`` is the
    kill switch that forces plain decode even for an MTP card. Read live so a
    test can monkeypatch the environment.
    """
    env_raw = os.environ.get("EXO_MTP_DRAFT", "").strip()
    env_head = os.environ.get("EXO_MTP_HEAD_FILE") or None
    if env_raw != "":
        draft = _int_env("EXO_MTP_DRAFT", 0)
        return (draft, env_head or card_head_file) if draft > 0 else (0, None)
    if card_draft_tokens is not None and card_draft_tokens > 0:
        return card_draft_tokens, card_head_file
    return 0, None
