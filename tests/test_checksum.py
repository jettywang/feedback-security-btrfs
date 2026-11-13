from __future__ import annotations

import gzip
import io
from typing import TYPE_CHECKING

import pytest

from dissect.btrfs import Btrfs, ChecksumError, ChecksumStatus
from dissect.btrfs.checksum import ChecksumStream

from tests.conftest import absolute_path

if TYPE_CHECKING:
    from dissect.btrfs.stream import Extent


def logical_to_physical(btrfs: Btrfs, address: int) -> int:
    """Map a logical address to a physical offset in a single-device image."""
    for chunk in btrfs._logical_fh.chunks:
        if chunk.offset <= address < chunk.offset + chunk.length:
            return chunk.stripes[0].offset + (address - chunk.offset)
    raise ValueError(f"No chunk covers logical address {address:#x}")


def test_verified_regular_file(btrfs_default: io.BytesIO) -> None:
    """A regular file with checksum records verifies, including small partial reads."""
    btrfs = Btrfs(btrfs_default)
    assert btrfs.checksum_tree is not None

    node = btrfs.get("large.txt")
    plain = node.open().read()

    # A small read verifies the complete sector(s) it depends on, not just the requested slice
    with node.open_verified() as fh:
        fh.seek(1_000_003)
        assert fh.read(512) == plain[1_000_003 : 1_000_003 + 512]
        assert fh.status == ChecksumStatus.VERIFIED
        # The recorded result covers the full aligned checksum unit, not just 512 bytes
        assert all(result.length % btrfs.sector_size == 0 for result in fh.results)

    assert node.verify() == ChecksumStatus.VERIFIED


def test_verified_shared_extent(btrfs_subvolume_snapshot: io.BytesIO) -> None:
    """An extent shared between a subvolume and its snapshot verifies from both paths."""
    btrfs = Btrfs(btrfs_subvolume_snapshot)

    subvol_node = btrfs.get("subvol/large.txt")
    snapshot_node = btrfs.get("subvol-snapshot/large.txt")

    # Both files reference the exact same on-disk extent
    assert subvol_node.extents()[0].disk_offset == snapshot_node.extents()[0].disk_offset

    assert subvol_node.verify() == ChecksumStatus.VERIFIED
    assert snapshot_node.verify() == ChecksumStatus.VERIFIED
    assert subvol_node.open().read() == snapshot_node.open().read()


def test_verified_compressed_file(btrfs_compression: io.BytesIO) -> None:
    """A compressed file verifies over its on-disk (compressed) checksum units."""
    btrfs = Btrfs(btrfs_compression)

    node = btrfs.get("zstd_lorem_ipsum.txt")
    plain = node.open().read()

    # A small read inside a multi-sector compressed extent verifies the entire on-disk extent
    with node.open_verified() as fh:
        fh.seek(140_000)
        assert fh.read(100) == plain[140_000:140_100]
        assert fh.status == ChecksumStatus.VERIFIED

    assert node.verify() == ChecksumStatus.VERIFIED


def test_failed_corrupted_compressed_extent() -> None:
    """Corrupted compressed bytes fail verification, even for a small read of unaffected output.

    The original image is only ever opened read-only; corruption is applied to an in-memory copy.
    """
    with gzip.GzipFile(absolute_path("_data/btrfs-compression.bin.gz"), "rb") as fh:
        image = bytearray(fh.read())

    btrfs = Btrfs(io.BytesIO(bytes(image)))
    node = btrfs.get("zstd_lorem_ipsum.txt")

    # Corrupt a single byte near the end of the second (compressed) extent
    extent = node.extents()[1]
    assert extent.compression != 0 and extent.disk_length > 20_000
    image[logical_to_physical(btrfs, extent.disk_offset + 20_000)] ^= 0xFF

    corrupted = Btrfs(io.BytesIO(bytes(image)))
    corrupted_node = corrupted.get("zstd_lorem_ipsum.txt")

    # A small read at the start of the extent still verifies the complete on-disk checksum unit,
    # so the corruption at the end of the unit is detected instead of silently passing
    with corrupted_node.open_verified() as fh:
        fh.seek(131_072)
        with pytest.raises(ChecksumError):
            fh.read(512)
        assert fh.status == ChecksumStatus.FAILED

    assert corrupted_node.verify() == ChecksumStatus.FAILED


def test_not_covered_inline_file(btrfs_compression: io.BytesIO) -> None:
    """Inline files have no data checksum records: readable, but not covered."""
    btrfs = Btrfs(btrfs_compression)

    node = btrfs.get("zlib_inline.txt")
    with node.open_verified() as fh:
        assert fh.read() == node.open().read()
        assert fh.status == ChecksumStatus.NOT_COVERED

    assert node.verify() == ChecksumStatus.NOT_COVERED


def test_not_covered_nodatasum(btrfs_default: io.BytesIO) -> None:
    """A NODATASUM file has no checksum records for its extents: absence of records is not a pass.

    Simulated by streaming a real extent layout without a checksum tree, which is exactly what
    the checksum tree looks like for a NODATASUM file: no records for its disk ranges.
    """
    btrfs = Btrfs(btrfs_default)

    node = btrfs.get("large.txt")
    extent: Extent = node.extents()[0]

    fh = ChecksumStream(
        btrfs._logical_fh,
        [extent],
        extent.length,
        btrfs.sector_size,
        checksum_tree=None,
    )
    # The data is still readable...
    assert fh.read(4096) == node.open().read(4096)
    # ...but nothing was verified, and "no checksum records" must not be reported as verified
    assert fh.status == ChecksumStatus.NOT_COVERED
    assert fh.status != ChecksumStatus.VERIFIED
