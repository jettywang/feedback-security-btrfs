# References:
# - https://btrfs.readthedocs.io/en/latest/Checksums.html
# - https://github.com/torvalds/linux/tree/master/fs/btrfs/check.c
from __future__ import annotations

import enum
from bisect import bisect_right
from typing import TYPE_CHECKING, BinaryIO, NamedTuple

from dissect.util.hash import crc32c

from dissect.btrfs.c_btrfs import c_btrfs
from dissect.btrfs.exceptions import ChecksumError
from dissect.btrfs.stream import Extent, ExtentStream, decode_extent

if TYPE_CHECKING:
    from dissect.btrfs.btrfs import Btrfs


class ChecksumStatus(enum.Enum):
    """Verification status of a read file range."""

    #: All checksum units matched the recorded checksums.
    VERIFIED = "verified"
    #: At least one checksum unit mismatched the recorded checksum.
    FAILED = "failed"
    #: No checksum records cover (part of) the read data, e.g. NODATASUM files, inline data or holes.
    #: Absence of checksum records is not a successful verification.
    NOT_COVERED = "not_covered"


class ChecksumResult(NamedTuple):
    """Verification result for a range of the file, as returned by :class:`ChecksumStream`."""

    offset: int
    length: int
    status: ChecksumStatus


# Size in bytes of a single checksum, per checksum type
CSUM_TYPE_SIZES = {
    c_btrfs.BTRFS_CSUM_TYPE_CRC32C: 4,
    c_btrfs.BTRFS_CSUM_TYPE_XXHASH: 8,
    c_btrfs.BTRFS_CSUM_TYPE_SHA256: 32,
    c_btrfs.BTRFS_CSUM_TYPE_BLAKE2B: 32,
}


class ChecksumTree:
    """Queryable view over the checksum tree of a Btrfs filesystem.

    Data checksums are recorded per sector of *on-disk* (logical) data. For compressed extents the
    checksums cover the compressed bytes as stored on disk, not the decompressed file content.

    Args:
        btrfs: The filesystem to open the checksum tree of.
    """

    def __init__(self, btrfs: Btrfs):
        self.btrfs = btrfs
        self.csum_type = btrfs.sb.csum_type
        self.sector_size = btrfs.sector_size

        try:
            self.csum_size = CSUM_TYPE_SIZES[self.csum_type]
        except KeyError:
            raise NotImplementedError(f"Unsupported checksum type: {self.csum_type}") from None

        self._offsets: list[int] = []
        self._items: list[bytes] = []

        tree = btrfs._open_tree(c_btrfs.BTRFS_CSUM_TREE_OBJECTID)
        objectid = c_btrfs.BTRFS_EXTENT_CSUM_OBJECTID & 0xFFFFFFFFFFFFFFFF
        for item, data in tree.cursor().iter(
            objectid=objectid, type=c_btrfs.BTRFS_EXTENT_CSUM_KEY, offset=0, ignore_offset=True
        ):
            self._offsets.append(item.key.offset)
            self._items.append(bytes(data))

    def __len__(self) -> int:
        return len(self._offsets)

    def lookup(self, address: int) -> bytes | None:
        """Return the recorded checksum for the sector at the logical ``address``, or ``None`` if there is none.

        Args:
            address: The logical address of the sector to look up.
        """
        idx = bisect_right(self._offsets, address) - 1
        if idx < 0:
            return None

        item_idx = (address - self._offsets[idx]) // self.sector_size
        data = self._items[idx]

        offset = item_idx * self.csum_size
        if offset + self.csum_size > len(data):
            return None

        return data[offset : offset + self.csum_size]

    def checksum(self, data: bytes) -> bytes:
        """Calculate the checksum of ``data`` using the checksum type of the filesystem.

        Args:
            data: The data to calculate the checksum of.
        """
        if self.csum_type == c_btrfs.BTRFS_CSUM_TYPE_CRC32C:
            return crc32c.crc32c(data).to_bytes(4, "little")

        raise NotImplementedError(f"Unsupported checksum type: {self.csum_type}")

    def verify(self, address: int, data: bytes) -> ChecksumStatus:
        """Verify a single sector of on-disk data against its recorded checksum.

        Args:
            address: The logical address of the sector.
            data: The sector data as read from disk.
        """
        expected = self.lookup(address)
        if expected is None:
            return ChecksumStatus.NOT_COVERED

        return ChecksumStatus.VERIFIED if self.checksum(data) == expected else ChecksumStatus.FAILED


class ChecksumStream(ExtentStream):
    """Implement a stream over Btrfs file extents that verifies data checksums while reading.

    Every read verifies the complete checksum units it depends on:

    - uncompressed extents are verified per filesystem sector covering the requested range;
    - compressed extents are verified over their entire on-disk (compressed) size *before* decoding,
      so a small read still verifies the full checksum unit it depends on;
    - ranges without checksum records (holes, inline data, NODATASUM files) are reported as
      :attr:`ChecksumStatus.NOT_COVERED` -- no checksum records is not the same as verified.

    A checksum mismatch raises :class:`ChecksumError` before any unverified data is returned.
    The results of everything read so far are queryable through ``results`` and ``status``.

    Args:
        fh: The file-like object to read the data from, usually the :class:`ChunkStream` of the filesystem.
        extents: The list of extents to open the stream on.
        size: The size of the file.
        sector_size: The sector size of the filesystem, which is also the checksum unit size.
        checksum_tree: Optional :class:`ChecksumTree` to verify against. Without one, nothing is covered.
        inline_data: Optional inline file data, for files that have no extents on disk.
    """

    def __init__(
        self,
        fh: BinaryIO,
        extents: list[Extent],
        size: int,
        sector_size: int,
        checksum_tree: ChecksumTree | None = None,
        inline_data: bytes | None = None,
    ):
        self.checksum_tree = checksum_tree
        self.results: list[ChecksumResult] = []
        self._inline_data = inline_data

        super().__init__(fh, extents, size, sector_size)

    @property
    def status(self) -> ChecksumStatus:
        """Return the aggregated verification status of everything read so far."""
        statuses = {result.status for result in self.results}
        if ChecksumStatus.FAILED in statuses:
            return ChecksumStatus.FAILED
        if statuses == {ChecksumStatus.VERIFIED}:
            return ChecksumStatus.VERIFIED
        return ChecksumStatus.NOT_COVERED

    def _read(self, offset: int, length: int) -> bytes:
        if self._inline_data is not None:
            # Inline file data is stored in the B-tree and has no data checksum records
            buf = self._inline_data[offset : offset + length]
            if buf:
                self._record(offset, len(buf), ChecksumStatus.NOT_COVERED)
            return buf

        result = []

        extent_idx = bisect_right(self._extent_offsets, offset)
        extents_len = len(self.extents)
        size = self.size

        while length > 0:
            if extent_idx >= extents_len:
                # We somehow requested more data than we have runs for
                break

            # If run_idx == 0, we only have a single run
            extent_pos = 0 if extent_idx == 0 else self._extent_offsets[extent_idx - 1]
            extent = self.extents[extent_idx]
            extent_pos = offset - extent_pos
            extent_remaining = extent.length - extent_pos

            # The relative extent offset is only relevant for knowing where to actually start reading on the disk
            # Add it to the current extent_pos because that's where this variable is only going to be used for
            extent_pos += extent.offset

            # Sometimes the self.size is way larger than what we actually have runs for?
            # Stop reading if we reach a negative run_remaining
            if extent_remaining < 0:
                break

            read_count = min(size - offset, min(extent_remaining, length))

            # Sparse run
            if (extent.disk_offset, extent.disk_length) == (0, 0):
                # Holes have no checksum records
                result.append(b"\x00" * read_count)
                self._record(offset, read_count, ChecksumStatus.NOT_COVERED)
            elif (extent.compression, extent.encryption) == (c_btrfs.BTRFS_COMPRESS_NONE, 0):
                # Verify the complete sectors that this read depends on
                disk_start = extent.disk_offset + extent_pos
                aligned_start = disk_start - (disk_start % self.sector_size)
                aligned_end = -(-(disk_start + read_count) // self.sector_size) * self.sector_size

                self._fh.seek(aligned_start)
                buf = self._fh.read(aligned_end - aligned_start)
                self._verify(offset, read_count, aligned_start, buf)

                cut = disk_start - aligned_start
                result.append(buf[cut : cut + read_count])
            else:
                # Verify the entire on-disk (compressed) extent before decoding it
                self._fh.seek(extent.disk_offset)
                buf = self._fh.read(extent.disk_length)
                self._verify(offset, read_count, extent.disk_offset, buf)

                buf = decode_extent(buf, extent.compression, extent.encryption, self.sector_size)
                if extent_pos or read_count != len(buf):
                    buf = buf[extent_pos : extent_pos + read_count]
                result.append(buf)

            offset += read_count
            length -= read_count
            extent_idx += 1

        return b"".join(result)

    def _verify(self, offset: int, length: int, disk_offset: int, data: bytes) -> None:
        """Verify all checksum units in ``data``, as read from ``disk_offset``, and record the result.

        Raises:
            ChecksumError: If a checksum unit mismatches its recorded checksum.
        """
        status = ChecksumStatus.VERIFIED

        for unit_offset in range(0, len(data), self.sector_size):
            unit = data[unit_offset : unit_offset + self.sector_size]

            if self.checksum_tree is None:
                unit_status = ChecksumStatus.NOT_COVERED
            else:
                unit_status = self.checksum_tree.verify(disk_offset + unit_offset, unit)

            if unit_status == ChecksumStatus.FAILED:
                self._record(offset, length, ChecksumStatus.FAILED)
                raise ChecksumError(
                    f"Checksum mismatch for sector at logical address {disk_offset + unit_offset:#x}"
                )
            if unit_status == ChecksumStatus.NOT_COVERED:
                status = ChecksumStatus.NOT_COVERED

        self._record(offset, length, status)

    def _record(self, offset: int, length: int, status: ChecksumStatus) -> None:
        self.results.append(ChecksumResult(offset, length, status))
