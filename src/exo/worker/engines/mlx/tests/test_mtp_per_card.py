"""MTP enablement is decided per instance (card block, env override) and drives the generator choice."""

import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.builder import use_sequential_generator
from exo.worker.engines.mlx.mtp.config import resolve_mtp_settings
from exo.worker.engines.mlx.mtp.decode import MtpRuntime, attach_runtime, get_runtime


@pytest.fixture(autouse=True)
def _clean_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("EXO_MTP_DRAFT", "EXO_MTP_HEAD_FILE", "EXO_NO_BATCH"):
        monkeypatch.delenv(var, raising=False)


def test_card_block_enables_mtp() -> None:
    assert resolve_mtp_settings(1, "mtp.safetensors") == (1, "mtp.safetensors")
    assert resolve_mtp_settings(2, None) == (2, None)


def test_plain_card_stays_off() -> None:
    assert resolve_mtp_settings(None, None) == (0, None)


def test_env_overrides_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_MTP_DRAFT", "3")
    monkeypatch.setenv("EXO_MTP_HEAD_FILE", "mtp-4bit.safetensors")
    assert resolve_mtp_settings(None, None) == (3, "mtp-4bit.safetensors")
    assert resolve_mtp_settings(1, "mtp.safetensors") == (3, "mtp-4bit.safetensors")
    monkeypatch.delenv("EXO_MTP_HEAD_FILE")
    # env K with no env head file keeps the card's head file
    assert resolve_mtp_settings(1, "mtp.safetensors") == (3, "mtp.safetensors")


def test_env_zero_is_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_MTP_DRAFT", "0")
    assert resolve_mtp_settings(1, "mtp.safetensors") == (0, None)


def test_bad_env_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_MTP_DRAFT", "two")
    with pytest.raises(ValueError):
        resolve_mtp_settings(1, None)


def _runtime(model: nn.Module) -> MtpRuntime:
    return MtpRuntime(
        draft_tokens=1,
        inner=model,  # type: ignore[arg-type]
        lm_head=nn.Linear(4, 4),
        gdn=None,  # type: ignore[arg-type]
        group=None,
        is_drafting_rank=False,
        head=None,
    )


def test_plain_model_uses_batch_generator() -> None:
    model = nn.Linear(4, 4)
    assert get_runtime(model) is None
    assert use_sequential_generator(model) is False


def test_mtp_instance_uses_sequential_generator() -> None:
    model = nn.Linear(4, 4)
    attach_runtime(model, _runtime(model))
    assert get_runtime(model) is not None
    assert use_sequential_generator(model) is True
    # another instance's model in the same process is unaffected
    assert use_sequential_generator(nn.Linear(4, 4)) is False


def test_no_batch_env_forces_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_NO_BATCH", "1")
    assert use_sequential_generator(nn.Linear(4, 4)) is True
