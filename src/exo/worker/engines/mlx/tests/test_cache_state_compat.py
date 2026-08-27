"""Pin exo's cache plumbing against mlx-lm's ArraysCache.state shape.

mlx-lm #1632 changed `ArraysCache.state` from being literally `self.cache` to a
`(cache, left_padding, lengths)` tuple whose setter unpacks exactly three values
and then reads `.size` on each. Every exo site that treated `.state` as a plain
list of arrays broke, and `checks.typecheck` stayed green through all of it
because `.state` is loosely typed.

These tests exercise the public entry points and touch `.cache`, so they pass on
both mlx-lm 0.31.3 and 0.32.0. Verified against the pre-fix source: the trim and
the wire round-trip both FAIL there with, respectively,
`AttributeError: 'NoneType' object has no attribute 'size'` and
`ValueError: not enough values to unpack (expected 3, got 2)`.

No model download, no slow marker; runs under `--extra mlx -m ""`.
"""

import io
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, RotatingKVCache

from exo.worker.disaggregated.protocol import ArraysState, read_message
from exo.worker.engines.mlx.cache import trim_cache
from exo.worker.engines.mlx.disaggregated.adapter import (
    inject_arrays_cache,
    mx_dtype_to_str,
    send_mlx_kv_cache,
)


def _arrays_cache(n: int = 2) -> ArraysCache:
    c = ArraysCache(size=n)
    for i in range(n):
        c[i] = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4) + float(i)
    return c


def _entries(c: ArraysCache) -> list[mx.array | None]:
    """Read ArraysCache.cache without dragging Unknown through the test."""
    return cast(
        list[mx.array | None],
        c.cache,  # type: ignore[reportUnknownMemberType]
    )


def test_trim_cache_reset_clears_every_slot_and_leaves_state_readable() -> None:
    """The snapshot-less reset must not go through ArraysCache.state.

    Asserts the post-conditions the old `.state = [None] * len(.state)` had:
    every slot None, and the cache still usable (`.state` readable) afterwards.
    """
    c = _arrays_cache()
    cache = [c]
    trim_cache(cast(list[object], cache), num_tokens=1, snapshot=None)  # type: ignore[arg-type]

    assert _entries(c) == [None, None]

    # `.state` must still be readable after the reset -- the setter is what
    # blew up before, and a half-reset object would poison the next request.
    state = cast(object, c.state)
    assert isinstance(state, tuple)
    parts = cast(tuple[object, ...], state)
    assert len(parts) == 3
    assert parts[0] == [None, None]


def test_trim_cache_reset_still_handles_rotating_kv_cache() -> None:
    """RotatingKVCache.state is still (keys, values) and must keep working."""
    c = RotatingKVCache(max_size=8)
    _ = c.update_and_fetch(  # type: ignore[reportUnknownMemberType]
        mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8))
    )
    cache = [c]
    trim_cache(cast(list[object], cache), num_tokens=1, snapshot=None)  # type: ignore[arg-type]
    assert c.keys is None
    assert c.values is None
    assert c.offset == 0


def test_arrays_cache_round_trips_through_the_wire() -> None:
    """serialize -> frame -> inject must return the same arrays.

    This is the full disaggregated path: `send_mlx_kv_cache` reads the cache,
    `inject_arrays_cache` writes it back. Neither is covered by any benchmark or
    by a live cluster run in the Pipeline/MlxRing deployment.
    """
    src = _arrays_cache()
    originals = [a for a in _entries(src) if a is not None]
    assert len(originals) == 2

    stream = io.BytesIO()
    _ = send_mlx_kv_cache(
        stream,
        cast(list[object], [src]),  # type: ignore[arg-type]
        dtype=mx_dtype_to_str(originals[0].dtype),
    )
    _ = stream.seek(0)

    msg = read_message(stream)
    assert isinstance(msg, ArraysState)
    assert len(msg.arrays) == 2, "both cached arrays must reach the wire"

    dst = ArraysCache(size=2)
    inject_arrays_cache(dst, msg.arrays)

    got = _entries(dst)
    assert len(got) == 2
    for original, received in zip(originals, got):
        assert received is not None
        assert bool(mx.array_equal(received, original))


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
