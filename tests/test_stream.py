from __future__ import annotations

import gzip
import io
import zlib
from pathlib import Path
from typing import BinaryIO

import pytest

import dissect.btrfs.stream as stream_module
from dissect.btrfs.btrfs import Btrfs
from dissect.btrfs.c_btrfs import c_btrfs
from dissect.btrfs.stream import Extent, ExtentCache, ExtentStream

SECTOR_SIZE = 4096
DATA = (b"A" * 4096) + (b"B" * 4096)
BLOB = zlib.compress(DATA)


@pytest.fixture
def decode_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    calls = []
    original = stream_module.decode_extent

    def counting_decode_extent(buf: bytes, compression: int, encryption: int, sector_size: int) -> bytes:
        calls.append((compression, len(buf)))
        return original(buf, compression, encryption, sector_size)

    monkeypatch.setattr(stream_module, "decode_extent", counting_decode_extent)
    return calls


def _view(cache: ExtentCache | None, offset: int, length: int, size: int | None = None) -> ExtentStream:
    """Return a file view on a slice of the shared compressed object ``BLOB``."""
    extent = Extent(c_btrfs.BTRFS_COMPRESS_ZLIB, 0, 0, len(BLOB), offset, length)
    return ExtentStream(io.BytesIO(BLOB), [extent], size if size is not None else length, SECTOR_SIZE, cache)


def test_extent_cache_capacity() -> None:
    cache = ExtentCache(max_size=10)
    key = (0, 1, 1, 0)

    # Entries larger than the capacity are never cached
    cache.set(key, b"x" * 11)
    assert len(cache) == 0
    assert cache.get(key) is None

    cache.set(key, b"x" * 10)
    assert cache.size == 10

    # Adding a new entry evicts the least recently used one to stay within capacity
    cache.set((1, 1, 1, 0), b"y")
    assert cache.size <= 10
    assert cache.get(key) is None


def test_extent_cache_shared_extent_slices(decode_calls: list) -> None:
    # Two files referencing different slices of the same compressed extent on disk
    cache = ExtentCache()
    view_a = _view(cache, 0, 4096)
    view_b = _view(cache, 4096, 4096)

    assert view_a.read() == b"A" * 4096
    assert view_b.read() == b"B" * 4096
    assert len(decode_calls) == 1


def test_extent_cache_reuse_small_reads(decode_calls: list) -> None:
    cache = ExtentCache()
    view = _view(cache, 0, len(DATA), size=len(DATA))

    parts = []
    for offset in range(0, len(DATA), 512):
        view.seek(offset)
        parts.append(view.read(512))

    assert b"".join(parts) == DATA
    assert len(decode_calls) == 1


def test_extent_stream_without_cache_decodes_per_read(decode_calls: list) -> None:
    view = _view(None, 0, len(DATA), size=len(DATA))

    for offset in range(0, len(DATA), 512):
        view.seek(offset)
        view.read(512)

    assert len(decode_calls) > 1


def test_extent_cache_eviction(decode_calls: list) -> None:
    other = zlib.compress(b"C" * 8192)
    disk = io.BytesIO(BLOB + other)
    # Fits exactly one decoded object
    cache = ExtentCache(max_size=len(DATA))

    extent_a = Extent(c_btrfs.BTRFS_COMPRESS_ZLIB, 0, 0, len(BLOB), 0, len(DATA))
    extent_b = Extent(c_btrfs.BTRFS_COMPRESS_ZLIB, 0, len(BLOB), len(other), 0, 8192)
    view_a = ExtentStream(disk, [extent_a], len(DATA), SECTOR_SIZE, cache)
    view_b = ExtentStream(disk, [extent_b], 8192, SECTOR_SIZE, cache)

    assert view_a.read() == DATA
    assert len(decode_calls) == 1

    assert view_b.read() == b"C" * 8192
    assert len(decode_calls) == 2
    assert len(cache) == 1

    # The evicted object is decoded again, but the read result is unchanged
    view_a.seek(0)
    assert view_a.read() == DATA
    assert len(decode_calls) == 3


def test_extent_cache_close_view(decode_calls: list) -> None:
    cache = ExtentCache()
    view_a = _view(cache, 0, 4096)
    view_b = _view(cache, 4096, 4096)

    assert view_a.read() == b"A" * 4096
    view_a.close()

    # Closing one view does not affect other users of the shared cache
    assert view_b.read() == b"B" * 4096
    assert len(decode_calls) == 1

    view_c = _view(cache, 0, 4096)
    assert view_c.read() == b"A" * 4096
    assert len(decode_calls) == 1


def test_extent_cache_failed_decode_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ExtentCache()
    original = stream_module.decode_extent
    calls = []

    def flaky_decode_extent(buf: bytes, compression: int, encryption: int, sector_size: int) -> bytes:
        calls.append(len(buf))
        if len(calls) == 1:
            raise zlib.error("simulated corrupt extent")
        return original(buf, compression, encryption, sector_size)

    monkeypatch.setattr(stream_module, "decode_extent", flaky_decode_extent)

    view = _view(cache, 0, 4096)
    with pytest.raises(zlib.error, match="simulated corrupt extent"):
        view.read()

    # The failed decode was not stored as a valid cache entry
    assert len(cache) == 0

    view.seek(0)
    assert view.read() == b"A" * 4096
    assert len(calls) == 2


def test_btrfs_extent_cache_small_reads(btrfs_compression: BinaryIO, decode_calls: list) -> None:
    fs = Btrfs(btrfs_compression)
    fh = fs.get("zlib.txt").open()

    parts = []
    for offset in range(0, 16384, 1024):
        fh.seek(offset)
        parts.append(fh.read(1024))

    assert b"".join(parts) == b"zlib" * 4096
    assert len(decode_calls) == 1


def test_btrfs_extent_cache_shared_between_views(btrfs_compression: BinaryIO, decode_calls: list) -> None:
    fs = Btrfs(btrfs_compression)
    entry = fs.get("zlib.txt")

    fh1 = entry.open()
    fh2 = entry.open()
    assert fh1.read(4096) == b"zlib" * 1024
    assert fh2.read(4096) == b"zlib" * 1024
    assert len(decode_calls) == 1

    # Closing one view does not invalidate the cached extent for the other
    fh1.close()
    fh2.seek(0)
    assert fh2.read(4096) == b"zlib" * 1024
    assert len(decode_calls) == 1


def test_btrfs_extent_cache_eviction(btrfs_compression: BinaryIO, decode_calls: list) -> None:
    # Each zlib extent in this image decodes to 128 KiB, so this fits exactly one
    fs = Btrfs(btrfs_compression, extent_cache_size=131072)
    fh = fs.get("zlib.txt").open()

    first = fh.read(4096)
    assert first == b"zlib" * 1024
    assert len(decode_calls) == 1

    fh.seek(131072)
    assert fh.read(4096) == b"zlib" * 1024
    assert len(decode_calls) == 2
    assert len(fs._extent_cache) == 1

    # The first extent was evicted and is decoded again, with an unchanged result
    fh.seek(0)
    assert fh.read(4096) == first
    assert len(decode_calls) == 3


def test_btrfs_extent_cache_not_shared_between_images(decode_calls: list) -> None:
    path = Path(__file__).parent / "_data" / "btrfs-compression.bin.gz"
    with gzip.GzipFile(path, "rb") as fh1, gzip.GzipFile(path, "rb") as fh2:
        fs1 = Btrfs(fh1)
        fs2 = Btrfs(fh2)

        # The same offset in two different images must not share decoded data
        assert fs1.get("zlib.txt").open().read(4096) == b"zlib" * 1024
        assert fs2.get("zlib.txt").open().read(4096) == b"zlib" * 1024
        assert len(decode_calls) == 2
