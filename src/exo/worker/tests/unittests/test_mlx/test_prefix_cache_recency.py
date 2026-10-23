"""A prefix-cache hit on a shared prefix (e.g. a system prompt) must not keep a stale entry alive."""

from collections.abc import Iterator
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import exo.worker.engines.mlx.cache as cache_module
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.types import KVCacheType

SYSTEM = list(range(100))


def _kv_entry(tokens: int) -> KVCacheType:
    entry = KVCache()
    keys = mx.zeros((1, 1, tokens, 8), dtype=mx.float16)
    entry.update_and_fetch(keys, keys)  # pyright: ignore[reportUnknownMemberType]
    return [entry]


def _prompt(*parts: list[int]) -> mx.array:
    return mx.array([t for part in parts for t in part], dtype=mx.int32)


@pytest.fixture(autouse=True)
def no_memory_pressure() -> Iterator[None]:
    with (
        patch.object(cache_module, "get_memory_used_percentage", return_value=0.1),
        patch.object(cache_module, "make_kv_cache", return_value=[]),
    ):
        yield


def _lookup_then_add(prefix_cache: KVPrefixCache, prompt: mx.array) -> None:
    prefix_cache.get_kv_cache(None, prompt)  # pyright: ignore[reportArgumentType]
    prefix_cache.add_kv_cache(prompt, _kv_entry(len(prompt)))


def test_a_shared_prefix_hit_does_not_keep_a_stale_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _prompt(SYSTEM, list(range(1000, 1300)))
    a0 = _prompt(SYSTEM, list(range(2000, 2010)))
    b0 = _prompt(SYSTEM, list(range(3000, 3010)))
    sizes = [
        cache_module._entry_nbytes(_kv_entry(len(p)), None)  # pyright: ignore[reportPrivateUsage]
        for p in (stale, a0, b0)
    ]
    # Room for any two of the three entries, not all three.
    monkeypatch.setattr(cache_module, "_MAX_BYTES", sum(sizes) - 1)
    prefix_cache = KVPrefixCache(None)
    prefix_cache.add_kv_cache(stale, _kv_entry(len(stale)))

    # Two sessions start; each first turn only shares the system prompt with the stale entry.
    _lookup_then_add(prefix_cache, a0)
    _lookup_then_add(prefix_cache, b0)

    # Session A's next turn must still find its own history.
    a1 = _prompt(SYSTEM, list(range(2000, 2010)), list(range(2100, 2110)))
    _, remaining, index, _ = prefix_cache.get_kv_cache(None, a1)  # pyright: ignore[reportArgumentType]
    assert index is not None
    assert prefix_cache.prompts[index].tolist() == a0.tolist()
    assert len(remaining) == len(a1) - len(a0)


def test_continuing_an_entry_still_marks_it_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_MAX_ENTRIES", 2)
    prefix_cache = KVPrefixCache(None)
    a0 = _prompt(SYSTEM, list(range(2000, 2010)))
    b0 = _prompt(list(range(5000, 5100)))
    prefix_cache.add_kv_cache(a0, _kv_entry(len(a0)))
    prefix_cache.add_kv_cache(b0, _kv_entry(len(b0)))

    # A's next turn continues its entry, so A is now the most recently used.
    a1 = _prompt(SYSTEM, list(range(2000, 2010)), list(range(2100, 2110)))
    prefix_cache.get_kv_cache(None, a1)  # pyright: ignore[reportArgumentType]
    prefix_cache.add_kv_cache(_prompt(list(range(7000, 7050))), _kv_entry(50))

    assert a0.tolist() in [p.tolist() for p in prefix_cache.prompts]
    assert b0.tolist() not in [p.tolist() for p in prefix_cache.prompts]
