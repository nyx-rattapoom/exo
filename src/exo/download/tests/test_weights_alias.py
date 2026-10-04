"""weights_repo alias: downloads and status run against the target repo, progress is reported under the alias card."""

from datetime import timedelta

from exo.download.impl_shard_downloader import reported_progress, weights_shard
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import RepoDownloadProgress
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

UD = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
LOCAL = "local/Qwen3.6-35B-A3B-UD-MLX-4bit-MTP"


def _shard(model_id: str, **card_overrides: object) -> ShardMetadata:
    card: dict[str, object] = dict(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=40,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
        vision={"image_token_id": 248056, "model_type": "qwen3_5_moe", "weights_repo": UD},
    )
    card.update(card_overrides)
    return PipelineShardMetadata(
        model_card=ModelCard.model_validate(card),
        device_rank=1,
        world_size=2,
        start_layer=20,
        end_layer=40,
        n_layers=40,
    )


def test_plain_shard_is_returned_unchanged() -> None:
    shard = _shard(UD)
    assert weights_shard(shard) is shard


def test_alias_shard_downloads_as_target_repo_and_keeps_layout() -> None:
    shard = _shard(LOCAL, weights_repo=UD, mtp={"draft_tokens": 1, "head_file": "mtp.safetensors"})
    dl = weights_shard(shard)
    assert dl.model_card.model_id == ModelId(UD)
    assert dl.model_card.weights_repo == ""
    assert dl.model_card.mtp is None
    # not a vision "sibling" any more: the target repo carries its own tower
    assert dl.model_card.vision is not None
    assert dl.model_card.vision.weights_repo == str(dl.model_card.model_id)
    assert (dl.start_layer, dl.end_layer, dl.device_rank, dl.world_size) == (20, 40, 1, 2)
    # the alias card itself is untouched
    assert shard.model_card.model_id == ModelId(LOCAL)
    assert shard.model_card.mtp is not None


def test_progress_is_rekeyed_to_the_alias_shard() -> None:
    shard = _shard(LOCAL, weights_repo=UD)
    dl = weights_shard(shard)
    progress = RepoDownloadProgress(
        repo_id=UD,
        repo_revision="main",
        shard=dl,
        completed_files=4,
        total_files=4,
        downloaded=Memory.from_mb(100),
        downloaded_this_session=Memory.from_mb(0),
        total=Memory.from_mb(100),
        overall_speed=0.0,
        overall_eta=timedelta(0),
        status="complete",
        file_progress={},
    )
    rekeyed = reported_progress(shard, progress)
    assert rekeyed.shard is shard
    assert rekeyed.shard.model_card.model_id == ModelId(LOCAL)
    assert rekeyed.status == "complete" and rekeyed.repo_id == UD
    assert reported_progress(shard, rekeyed) is rekeyed
