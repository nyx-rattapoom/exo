"""Pin exo's cache plumbing against mlx-lm's ArraysCache.state shape.

mlx-lm #1632 changed `ArraysCache.state` from "literally `self.cache`" to a
`(cache, left_padding, lengths)` tuple whose setter unpacks exactly three values
and then reads `.size` on each. Every exo site that used `.state` as a plain list
of arrays broke, and `checks.typecheck` stayed green through all of it because
`.state` is loosely typed.

These tests touch only `.cache`, so they pass on both mlx-lm 0.31.3 and 0.32.0 and
will keep failing loudly if anyone routes this plumbing back through `.state`.
"""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, RotatingKVCache

from exo.worker.engines.mlx.cache import _reset_non_trimmable, trim_cache
from exo.worker.engines.mlx.disaggregated.adapter import (
    TensorBlob,
    inject_arrays_cache,
    mx_dtype_to_str,
)


def _arrays_cache(n: int = 2) -> ArraysCache:
    c = ArraysCache(size=n)
    for i in range(n):
        c[i] = mx.zeros((1, 4, 4))
    return c


def test_reset_non_trimmable_zeroes_arrays_cache() -> None:
    """The trim_cache reset path must not go through ArraysCache.state."""
    c = _arrays_cache()
    _reset_non_trimmable(c)
    assert c.cache == [None, None]


def test_reset_non_trimmable_zeroes_rotating_kv_cache() -> None:
    """RotatingKVCache.state is still (keys, values); it must keep working."""
    c = RotatingKVCache(max_size=8)
    c.update_and_fetch(mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8)))
    _reset_non_trimmable(c)
    assert c.keys is None and c.values is None
    assert c.offset == 0 and c._idx == 0


def test_trim_cache_resets_arrays_cache_without_snapshot() -> None:
    """End-to-end: the snapshot-less branch of trim_cache on an SSM entry."""
    cache = [_arrays_cache()]
    trim_cache(cache, num_tokens=1, snapshot=None)
    assert cache[0].cache == [None, None]


def test_inject_arrays_cache_round_trips_two_blobs() -> None:
    """Injecting N blobs must set N entries, not unpack into a 3-tuple."""
    a = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4)
    blob = TensorBlob(
        dtype=mx_dtype_to_str(a.dtype),
        shape=(1, 4, 4),
        data=bytes(memoryview(a)),  # type: ignore[arg-type]
    )
    c = ArraysCache(size=2)
    inject_arrays_cache(c, [blob, blob])
    assert len(c.cache) == 2
    assert all(x is not None for x in c.cache)
    assert mx.array_equal(c.cache[0], a)


def test_arrays_cache_state_is_the_three_tuple_we_expect() -> None:
    """Document the upstream shape this module is defending against.

    If this ever fails, mlx-lm changed `.state` again and every `.cache` comment
    in cache.py / adapter.py needs re-checking.
    """
    c = _arrays_cache()
    state = c.state
    assert isinstance(state, tuple) and len(state) == 3
    cached, left_padding, lengths = state
    assert isinstance(cached, list) and len(cached) == 2
    assert left_padding.size == 0 and lengths.size == 0
