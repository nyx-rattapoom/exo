"""Pin exo's cache plumbing against mlx-lm's ArraysCache.state shape.

mlx-lm #1632 changed `ArraysCache.state` from being literally `self.cache` to a
`(cache, left_padding, lengths)` tuple whose setter unpacks exactly three values
and then reads `.size` on each. Every exo site that treated `.state` as a plain
list of arrays broke, and `checks.typecheck` stayed green through all of it
because `.state` is loosely typed.

These tests exercise the public entry points only (`trim_cache`,
`inject_arrays_cache`) and touch `.cache`, so they pass on both mlx-lm 0.31.3 and
0.32.0. Verified to FAIL on the unpatched code with
`AttributeError: 'NoneType' object has no attribute 'size'` and
`ValueError: not enough values to unpack (expected 3, got 2)` respectively.
"""

from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, RotatingKVCache

from exo.worker.engines.mlx.cache import trim_cache
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


def _entries(c: ArraysCache) -> list[mx.array | None]:
    """Read ArraysCache.cache without dragging Unknown through the test."""
    return cast(
        list[mx.array | None],
        c.cache,  # type: ignore[reportUnknownMemberType]
    )


def test_trim_cache_resets_arrays_cache_without_snapshot() -> None:
    """The snapshot-less reset branch must not go through ArraysCache.state."""
    cache = [_arrays_cache()]
    trim_cache(cast(list[object], cache), num_tokens=1, snapshot=None)  # type: ignore[arg-type]
    assert _entries(cache[0]) == [None, None]


def test_trim_cache_resets_rotating_kv_cache_without_snapshot() -> None:
    """RotatingKVCache.state is still (keys, values); it must keep working."""
    c = RotatingKVCache(max_size=8)
    _ = c.update_and_fetch(  # type: ignore[reportUnknownMemberType]
        mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8))
    )
    cache = [c]
    trim_cache(cast(list[object], cache), num_tokens=1, snapshot=None)  # type: ignore[arg-type]
    assert c.keys is None
    assert c.values is None
    assert c.offset == 0


def test_inject_arrays_cache_round_trips_two_blobs() -> None:
    """Injecting N blobs must set N entries, not unpack into a 3-tuple."""
    a = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4)
    blob = TensorBlob(
        dtype=mx_dtype_to_str(a.dtype),
        shape=(1, 4, 4),
        data=bytes(memoryview(a)),
    )
    c = ArraysCache(size=2)
    inject_arrays_cache(c, [blob, blob])
    entries = _entries(c)
    assert len(entries) == 2
    first = entries[0]
    assert first is not None
    assert bool(mx.array_equal(first, a))


def test_arrays_cache_state_is_the_three_tuple_we_expect() -> None:
    """Document the upstream shape this module defends against.

    If this fails, mlx-lm changed `.state` again and every `.cache` comment in
    cache.py / adapter.py needs re-checking.
    """
    c = _arrays_cache()
    state = cast(object, c.state)
    assert isinstance(state, tuple)
    parts = cast(tuple[object, ...], state)
    assert len(parts) == 3
    assert isinstance(parts[0], list)
    assert len(cast(list[object], parts[0])) == 2
    assert isinstance(parts[1], mx.array)
    assert isinstance(parts[2], mx.array)
