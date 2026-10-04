"""ModelCard: optional per-card MTP block and the weights_repo alias."""

import tomlkit

from exo.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    MtpCardConfig,
    VisionCardConfig,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory

UD = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
LOCAL = "local/Qwen3.6-35B-A3B-UD-MLX-4bit-MTP"


def _card(**overrides: object) -> ModelCard:
    base: dict[str, object] = dict(
        model_id=ModelId(UD),
        storage_size=Memory.from_bytes(21634738912),
        n_layers=40,
        hidden_size=2048,
        supports_tensor=True,
        num_key_value_heads=2,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
        context_length=262144,
    )
    base.update(overrides)
    return ModelCard.model_validate(base)


def test_plain_card_has_no_mtp_and_resolves_to_itself() -> None:
    card = _card()
    assert card.mtp is None
    assert card.weights_repo == ""
    assert card.weights_model_id() == ModelId(UD)


def test_mtp_block_defaults_and_toml_roundtrip() -> None:
    card = _card(
        model_id=ModelId(LOCAL),
        weights_repo=UD,
        mtp={"head_file": "mtp.safetensors"},
        vision={"image_token_id": 248056, "model_type": "qwen3_5_moe", "weights_repo": UD},
    )
    assert card.mtp == MtpCardConfig(draft_tokens=1, head_file="mtp.safetensors")
    assert card.weights_model_id() == ModelId(UD)
    assert card.model_id == ModelId(LOCAL)
    # the same serialisation ModelCard.save() uses
    dumped = tomlkit.dumps(  # pyright: ignore[reportUnknownMemberType]
        card.model_dump(exclude_none=True, exclude={"is_custom"})
    )
    assert 'weights_repo = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"' in dumped
    assert "[mtp]" in dumped
    back = ModelCard.model_validate(tomlkit.loads(dumped))
    assert back == card.model_copy(update={"is_custom": back.is_custom})
    assert back.mtp is not None and back.mtp.draft_tokens == 1
    assert back.weights_model_id() == ModelId(UD)


def test_plain_card_toml_has_no_mtp_table() -> None:
    dumped = tomlkit.dumps(  # pyright: ignore[reportUnknownMemberType]
        _card().model_dump(exclude_none=True, exclude={"is_custom"})
    )
    assert "[mtp]" not in dumped


def test_vision_weights_repo_defaults_to_alias_target() -> None:
    card = _card(
        model_id=ModelId(LOCAL),
        weights_repo=UD,
        vision=VisionCardConfig(image_token_id=248056, model_type="qwen3_5_moe"),
    )
    assert card.vision is not None
    assert card.vision.weights_repo == UD


def test_local_namespace_is_a_valid_model_id() -> None:
    mid = ModelId(LOCAL)
    assert mid.normalize() == "local--Qwen3.6-35B-A3B-UD-MLX-4bit-MTP"
    assert mid.short() == "Qwen3.6-35B-A3B-UD-MLX-4bit-MTP"
