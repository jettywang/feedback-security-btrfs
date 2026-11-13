from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import BinaryIO

import pytest

from dissect.btrfs.btrfs import Btrfs, read_superblock_copies
from dissect.btrfs.consistency import (
    ChunkStatus,
    Verdict,
    examine,
    verify_file_data,
)
from dissect.btrfs.exceptions import (
    ConflictingChunkError,
    ConflictingDeviceError,
    InvalidTreeBlockError,
    MissingDeviceError,
)

DATA = Path(__file__).parent / "_data"


def load(name: str) -> io.BytesIO:
    return io.BytesIO(gzip.open(DATA / f"{name}.bin.gz", "rb").read())


def load_pair(*names: str) -> list[BinaryIO]:
    return [load(name) for name in names]


def test_superblock_mirrors_are_self_identifying() -> None:
    copies = read_superblock_copies(load("btrfs-default"))

    assert [copy.offset for copy in copies] == [0x10000, 0x4000000]
    assert all(copy.checksum_valid for copy in copies)
    assert all(copy.sb.bytenr == copy.offset for copy in copies)
    assert {copy.generation for copy in copies} == {8}


def test_mixed_generation_selection_was_previously_spliced() -> None:
    """Two same-fsid, same-devid copies at generations 8 and 14 must not be mixed.

    The old implementation selected the highest superblock generation but bound
    devices by last-write-wins, so the selected superblock could describe one
    transaction while every tree block came from the other image.
    """
    with pytest.raises(ConflictingDeviceError, match="devid 1"):
        Btrfs(load_pair("btrfs-default", "btrfs-subvolume"))

    with pytest.raises(ConflictingDeviceError, match="devid 1"):
        Btrfs(load_pair("btrfs-subvolume", "btrfs-default"))


def test_same_generation_duplicate_devid_is_also_rejected() -> None:
    with pytest.raises(ConflictingDeviceError):
        Btrfs(load_pair("btrfs-raid1-1", "btrfs-raid1-1"))


def test_missing_dup_member_is_not_silently_repaired() -> None:
    # Both stripes of the system chunk live on devid 2 in this fixture, so the
    # supplied devid 1 image cannot satisfy it and must not pretend to.
    with pytest.raises(MissingDeviceError, match="devid 2"):
        Btrfs(load_pair("btrfs-dup-1"))


def test_raid1_degrades_but_never_claims_proven() -> None:
    report = examine(load_pair("btrfs-raid1-1"))

    assert report.verdict is Verdict.UNVERIFIED
    assert report.missing_devids == [2]
    assert [(chunk.status, chunk.missing_devids) for chunk in report.chunks] == [
        (ChunkStatus.DEGRADED, (2,))
    ]

    # The parser can still read it, which is exactly why parsing is not proof.
    fs = Btrfs(load_pair("btrfs-raid1-1"))
    assert fs.get("small.txt").open().read() == b"small file content goes here\n"


def test_raid1c4_with_two_of_four_members_is_unverified() -> None:
    report = examine(load_pair("btrfs-raid1c4-1", "btrfs-raid1c4-2"))

    assert report.verdict is Verdict.UNVERIFIED
    assert report.missing_devids == [3, 4]
    assert all(chunk.status is ChunkStatus.DEGRADED for chunk in report.chunks)

    fs = Btrfs(load_pair("btrfs-raid1c4-1", "btrfs-raid1c4-2"))
    assert fs.get("large.txt").open().read() == (b"a" * 5242880) + b"\n"


def test_complete_mirrored_set_is_proven() -> None:
    report = examine(load_pair("btrfs-raid1-1", "btrfs-raid1-2"))

    assert report.verdict is Verdict.PROVEN
    assert report.missing_devids == []
    assert all(chunk.status is ChunkStatus.READABLE for chunk in report.chunks)


def test_foreign_fsid_is_rejected() -> None:
    report = examine(load_pair("btrfs-default", "btrfs-sparse"))

    assert report.verdict is Verdict.REJECTED
    assert any("different filesystems" in finding for finding in report.findings)
    with pytest.raises(ValueError, match="same filesystem UUID"):
        Btrfs(load_pair("btrfs-default", "btrfs-sparse"))


def test_duplicate_devid_report_keeps_both_collection_times() -> None:
    report = examine(load_pair("btrfs-default", "btrfs-subvolume"))

    assert report.verdict is Verdict.REJECTED
    assert report.duplicate_devids == [1]
    assert {member.generation for member in report.members} == {8, 14}
    assert report.newest_generation == 14
    assert any("not a single-point-in-time" in finding for finding in report.findings)


def test_member_selection_cannot_override_duplicate_devid() -> None:
    with pytest.raises(ConflictingDeviceError):
        Btrfs(load_pair("btrfs-default", "btrfs-subvolume"), member=0)


def test_member_selection_pins_a_single_image_superblock() -> None:
    fs = Btrfs(load_pair("btrfs-raid1-1", "btrfs-raid1-2"), member=1)

    assert fs.sb.generation == 8
    assert fs.member_generations == {1: 8, 2: 8}


def test_stripe_device_uuid_mismatch_is_rejected() -> None:
    """A chunk stripe naming devid 1 must also name devid 1's device UUID."""
    from dissect.util.hash import crc32c

    image = bytearray(gzip.open(DATA / "btrfs-default.bin.gz", "rb").read())
    # First system chunk stripe starts at superblock + sys_chunk_array + key(17) + header(48).
    stripe = 0x10000 + 811 + 17 + 48
    image[stripe + 16] ^= 0xFF
    for mirror in (0x10000, 0x4000000):
        image[mirror : mirror + 4] = (
            crc32c.update(0, bytes(image[mirror + 32 : mirror + 4096])) & 0xFFFFFFFF
        ).to_bytes(4, "little")

    with pytest.raises(ConflictingDeviceError, match="dev_uuid"):
        Btrfs(io.BytesIO(image))


def test_chunk_map_conflict_is_rejected() -> None:
    from dissect.btrfs.c_btrfs import c_btrfs

    fs = Btrfs(load("btrfs-default"))
    chunk = fs._logical_fh.chunks[1]

    def copy_chunk(delta: int) -> c_btrfs.btrfs_chunk:
        duplicate = c_btrfs.btrfs_chunk()
        duplicate.length = chunk.length
        duplicate.stripe_len = chunk.stripe_length
        duplicate.type = chunk.type
        duplicate.num_stripes = chunk.num_stripes
        duplicate.sub_stripes = chunk.sub_stripes
        duplicate.stripe = []
        for stripe in chunk.stripes:
            item = c_btrfs.btrfs_stripe()
            item.devid = stripe.devid
            item.offset = stripe.offset + delta
            item.dev_uuid = stripe.dev_uuid.bytes
            duplicate.stripe.append(item)
        return duplicate

    fs._logical_fh.add(chunk.offset, copy_chunk(0))
    with pytest.raises(ConflictingChunkError):
        fs._logical_fh.add(chunk.offset, copy_chunk(0x1000))


def test_data_verification_covers_disk_bytes_and_metadata() -> None:
    fs = Btrfs(load("btrfs-default"))

    result = verify_file_data(fs, "large.txt")
    assert result.fully_verified
    assert result.checked_sectors == 1281
    assert result.proven_bytes == 5242881

    # Inline data has no separate checksum item; the leaf block checksum covers it.
    inline = verify_file_data(fs, "small.txt")
    assert inline.fully_verified
    assert "tree block" in inline.reason

    # Compressed extents are checksummed over their compressed on-disk bytes.
    compressed_fs = Btrfs(load("btrfs-compression"))
    compressed = verify_file_data(compressed_fs, "zlib.txt")
    assert compressed.fully_verified
    assert compressed.checked_sectors == 161

    # Holes are proven by the checksum-verified extent metadata, not by data checksums.
    sparse_fs = Btrfs(load("btrfs-sparse"))
    holes = verify_file_data(sparse_fs, "sparse_all")
    assert holes.fully_verified
    assert holes.checked_sectors == 0


def test_data_verification_reports_a_corrupted_sector() -> None:
    image = bytearray(gzip.open(DATA / "btrfs-default.bin.gz", "rb").read())
    image[0xD00000 + 0x10] ^= 0xFF

    fs = Btrfs(io.BytesIO(image))
    result = verify_file_data(fs, "large.txt")

    assert not result.fully_verified
    assert "checksum mismatch" in result.reason

    # The directory still parses, which is precisely why "it lists files" is not
    # evidence of a consistent or correct snapshot.
    assert "large.txt" in fs.get("/").listdir()


def test_data_verification_uses_all_supplied_mirrors() -> None:
    first = bytearray(gzip.open(DATA / "btrfs-raid1-1.bin.gz", "rb").read())
    second = bytearray(gzip.open(DATA / "btrfs-raid1-2.bin.gz", "rb").read())

    single = Btrfs(io.BytesIO(first))
    extent = single.get("large.txt").extents()[0]
    chunk = next(
        chunk for chunk in single._logical_fh.chunks if chunk.offset <= extent.disk_offset < chunk.offset + chunk.length
    )
    offset = extent.disk_offset - chunk.offset
    stripe_num, stripe_offset = divmod(offset, chunk.stripe_length)

    physical = []
    for index, image in enumerate((first, second)):
        stripe = chunk.stripes[index]
        position = stripe.offset + stripe_offset + stripe_num * chunk.stripe_length
        physical.append(position)
        image[position + 0x10] ^= 0xFF

        result = verify_file_data(Btrfs([io.BytesIO(first), io.BytesIO(second)]), "large.txt")
        if index == 0:
            # One corrupted copy still leaves the other mirror to prove the range.
            assert result.fully_verified
        else:
            assert not result.fully_verified


def test_tree_blocks_are_checksum_verified_on_read() -> None:
    image = bytearray(gzip.open(DATA / "btrfs-default.bin.gz", "rb").read())
    fs = Btrfs(io.BytesIO(image))

    # The root block is stored on the second metadata stripe, not at the
    # logical address itself, so corrupt the physical copy.
    root = fs.sb.root
    chunk = next(chunk for chunk in fs._logical_fh.chunks if chunk.offset <= root < chunk.offset + chunk.length)
    stripe = chunk.stripes[0]
    physical = stripe.offset + (root - chunk.offset)
    image[physical] ^= 0xFF

    with pytest.raises(InvalidTreeBlockError, match="checksum"):
        Btrfs(io.BytesIO(image), strict=True)


def test_strict_mode_can_be_disabled_for_damaged_media() -> None:
    image = bytearray(gzip.open(DATA / "btrfs-default.bin.gz", "rb").read())
    fs = Btrfs(io.BytesIO(image))
    root = fs.sb.root
    chunk = next(chunk for chunk in fs._logical_fh.chunks if chunk.offset <= root < chunk.offset + chunk.length)
    physical = chunk.stripes[0].offset + (root - chunk.offset)
    image[physical] ^= 0xFF

    fs = Btrfs(io.BytesIO(image), strict=False)
    # Deliberately unverified: the point is that this mode is explicit.
    assert fs.label == "btrfs-default"


def test_report_is_independent_of_image_order() -> None:
    forward = examine(load_pair("btrfs-default", "btrfs-subvolume"))
    reverse = examine(load_pair("btrfs-subvolume", "btrfs-default"))

    assert forward.verdict is reverse.verdict is Verdict.REJECTED
    assert forward.duplicate_devids == reverse.duplicate_devids == [1]
    assert forward.newest_generation == reverse.newest_generation == 14
    assert sorted(member.generation for member in forward.members) == [8, 14]
    assert sorted(member.generation for member in reverse.members) == [8, 14]


def test_report_describe_is_stable_for_receipts() -> None:
    report = examine(load_pair("btrfs-raid1-1", "btrfs-raid1-2"))
    text = report.describe()

    assert text.startswith("verdict: proven")
    assert "devid=1" in text and "devid=2" in text
    assert "chunk 0x1500000" in text
