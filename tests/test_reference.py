from __future__ import annotations

import gzip
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from dissect.btrfs import Btrfs
from dissect.btrfs.reference import find_extent_references, map_overlap
from dissect.btrfs.tree import Cursor


def open_fixture(name: str) -> Btrfs:
    with gzip.open(Path(__file__).parent / "_data" / name, "rb") as fh:
        return Btrfs(io.BytesIO(fh.read()))


def test_shared_extent_across_snapshot_roots() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")
    report = find_extent_references(fs, 0x4500000, 0x1000, [256, 257])
    assert report == fs.find_extent_references(0x4500000, 0x1000, [256, 257])

    assert report.complete
    assert not report.exhaustive  # subvolume 5 exists but was not selected
    assert [scope.objectid for scope in report.selected] == [256, 257]
    assert [scope.objectid for scope in report.unselected] == [5]

    assert len(report.references) == 2
    left, right = report.references

    assert (left.subvolume_objectid, left.inode, left.paths) == (256, 259, ("subvol/large.txt",))
    assert (right.subvolume_objectid, right.inode, right.paths) == (257, 259, ("subvol-snapshot/large.txt",))

    for reference in report.references:
        assert reference.precision == "byte"
        assert (reference.disk_start, reference.disk_end) == (0x4500000, 0x501000 + 0x4500000)
        assert (reference.overlap_start, reference.overlap_end) == (0x4500000, 0x4501000)
        assert (reference.file_start, reference.file_end) == (0, 0x1000)
        assert reference.extent_key_leaf is not None
        assert reference.root_bytenr is not None
        assert reference.root_generation is not None

    # Both records point at the very same physical extent.
    assert (left.disk_start, left.disk_num_bytes) == (right.disk_start, right.disk_num_bytes)
    assert left.extent_generation == right.extent_generation


def test_shared_extent_matches_forward_reads() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")
    report = fs.find_extent_references(0x4500000, 0x1000, [256, 257])

    for reference in report.references:
        inode = fs.open_subvolume(reference.subvolume_objectid).inode(reference.inode)
        with inode.open() as fh:
            fh.seek(reference.file_start)
            data = fh.read(reference.file_end - reference.file_start)

        assert data == (b"b" * 0x1000)[: len(data)]

    # And the same bytes come from the queried disk range.
    fs._logical_fh.seek(0x4500000)
    assert fs._logical_fh.read(0x1000) == (b"b" * 0x1000)


def test_partial_exposure_in_snapshot_reports_smaller_file_range() -> None:
    fs = open_fixture("btrfs-sparse.bin.gz")
    report = fs.find_extent_references(0xD00000, 0x14000, [5, 256])

    assert report.complete
    assert report.exhaustive
    assert len(report.references) == 2

    original, snapshot = report.references
    assert original.paths == ("sparse_hole",)
    assert (original.exposed_start, original.exposed_end) == (0, 0x14000)
    assert (original.file_start, original.file_end) == (0, 0x14000)

    assert snapshot.paths == ("snapshot/sparse_hole",)
    assert (snapshot.exposed_start, snapshot.exposed_end) == (0, 0xA000)
    assert (snapshot.file_start, snapshot.file_end) == (0, 0xA000)
    assert snapshot.notes == ("record exposes only part of the physical extent",)

    # The snapshot file is longer than what this extent exposes.
    inode = fs.open_subvolume(256).inode(snapshot.inode)
    assert inode.size == 0x3C000


def test_extent_tail_only_reference_has_no_file_range() -> None:
    fs = open_fixture("btrfs-sparse.bin.gz")
    report = fs.find_extent_references(0xD3C000, 0x1000, [256])

    tail_records = [reference for reference in report.references if reference.extent_key_offset == 0x2000]
    assert len(tail_records) == 1
    tail = tail_records[0]
    assert (tail.overlap_start, tail.overlap_end) == (0xD3C000, 0xD3D000)
    assert (tail.exposed_start, tail.exposed_end) == (0x2000, 0x14000)
    assert (tail.file_start, tail.file_end) == (None, None)
    assert "disk overlap lies outside the slice exposed by this record" in tail.notes


def test_identical_content_with_separate_allocation_is_not_shared() -> None:
    fs = open_fixture("btrfs-sparse.bin.gz")

    # sparse_hole, sparse_start and sparse_end all contain the same byte pattern but live at
    # different logical offsets.
    bytenrs = {0xD00000, 0xD14000, 0xD28000, 0xD3C000}
    allocations = set()
    for bytenr in bytenrs:
        report = fs.find_extent_references(bytenr, 0x1000, [5])
        assert len(report.references) == 1
        allocations.add((report.references[0].disk_start, report.references[0].disk_num_bytes))

    assert allocations == {(bytenr, 0x14000) for bytenr in bytenrs}

    # Forward reads prove the content is identical while the reverse query proves the
    # allocations are distinct.
    contents = set()
    for bytenr in bytenrs:
        fs._logical_fh.seek(bytenr)
        contents.add(fs._logical_fh.read(0x1000))
    assert contents == {b"\x01" * 0x1000}

    for reference in fs.find_extent_references(0xD00000, 0x1000, [5]).references:
        assert reference.disk_start == 0xD00000


def test_unselected_roots_are_reported_not_silently_skipped() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")

    partial = fs.find_extent_references(0x4500000, 0x1000, [256])
    assert [reference.subvolume_objectid for reference in partial.references] == [256]
    assert [scope.objectid for scope in partial.unselected] == [5, 257]
    assert not partial.exhaustive

    complete = fs.find_extent_references(0x4500000, 0x1000, [5, 256, 257])
    assert complete.exhaustive
    assert [reference.subvolume_objectid for reference in complete.references] == [256, 257]


def test_no_references_outside_extent() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")
    report = fs.find_extent_references(0x4400000, 0x1000, [256, 257])

    assert report.references == ()
    assert report.complete
    assert not report.exhaustive
    assert report.stats.extent_items_scanned == 8


def test_compressed_extent_reports_extent_precision() -> None:
    fs = open_fixture("btrfs-compression.bin.gz")
    report = fs.find_extent_references(0x5700000, 0x1000, [5])

    assert len(report.references) == 1
    reference = report.references[0]
    assert reference.paths == ("zlib.txt",)
    assert reference.precision == "extent"
    assert (reference.exposed_start, reference.exposed_end) == (0, 0x20000)
    assert reference.compression == 1
    assert "compressed extent; byte-level attribution within the extent is unresolved" in reference.notes

    inode = fs.get("zlib.txt")
    with inode.open() as fh:
        fh.seek(reference.file_start)
        data = fh.read(reference.file_end - reference.file_start)
    assert data == b"zlib" * 0x8000


def test_coverage_gap_for_unreadable_root(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")

    def broken_tree(self: object) -> None:
        raise OSError("simulated damaged root")

    monkeypatch.setattr("dissect.btrfs.btrfs.Subvolume.tree", property(broken_tree))

    report = fs.find_extent_references(0x4500000, 0x1000, [256])
    assert not report.complete
    assert report.references == ()
    assert len(report.coverage_gaps) == 1
    gap = report.coverage_gaps[0]
    assert gap.kind in ("root_unreadable", "partial_scan")
    assert gap.subvolume_objectid == 256
    assert "damaged root" in gap.reason


def test_invalid_arguments_are_rejected() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")

    with pytest.raises(ValueError, match="length"):
        fs.find_extent_references(0x4500000, 0, [256])

    with pytest.raises(ValueError, match="bytenr"):
        fs.find_extent_references(-1, 0x1000, [256])

    with pytest.raises(ValueError, match="explicitly selected"):
        fs.find_extent_references(0x4500000, 0x1000, [])

    other = open_fixture("btrfs-default.bin.gz")
    with pytest.raises(ValueError, match="different filesystem"):
        fs.find_extent_references(0x4500000, 0x1000, [other.default_subvolume])


@pytest.mark.parametrize(
    ("query", "extent", "expected"),
    [
        # Whole uncompressed extent, full file slice.
        ((0x1000, 0x2000), (0x1000, 0x1000, 0, 0x1000, 0x1000, 0, 0), (0, 0x1000, 0x1000, "byte")),
        # Query in the middle of the extent: the file range shifts by the same amount.
        ((0x1400, 0x1800), (0x1000, 0x1000, 0, 0x1000, 0x1000, 0, 0), (0x400, 0x800, 0x1000, "byte")),
        # Extent record exposes only half of a larger physical extent.
        ((0x1000, 0x2000), (0x1000, 0x2000, 0, 0x1000, 0x1000, 0, 0), (0, 0x1000, 0x1000, "byte")),
        # Query only hits the half this record does not expose.
        ((0x2000, 0x2400), (0x1000, 0x2000, 0, 0x1000, 0x1000, 0, 0), (None, None, 0x1000, "byte")),
        # Record starts at a non-zero file offset (0x5000) while its data starts 0x1000 into the extent.
        ((0x2000, 0x2800), (0x1000, 0x2000, 0x1000, 0x1000, 0x1000, 0x5000, 0), (0x5000, 0x5800, 0x1000, "byte")),
        # Compressed extent: only extent precision, full exposed slice.
        ((0x1000, 0x1400), (0x1000, 0x1000, 0, 0x20000, 0x20000, 0, 1), (0, 0x20000, 0x20000, "extent")),
    ],
)
def test_map_overlap(
    query: tuple[int, int],
    extent: tuple[int, int, int, int, int, int, int],
    expected: tuple[int | None, int | None, int, str],
) -> None:
    disk_start, disk_num_bytes, extent_offset, num_bytes, ram_bytes, key_offset, compression = extent
    mapping = map_overlap(
        query[0],
        query[1],
        disk_start=disk_start,
        disk_num_bytes=disk_num_bytes,
        extent_offset=extent_offset,
        num_bytes=num_bytes,
        ram_bytes=ram_bytes,
        key_offset=key_offset,
        compression=compression,
    )

    file_start, file_end, exposed_length, precision = expected
    assert mapping.precision == precision
    assert (mapping.exposed_start, mapping.exposed_end) == (key_offset, key_offset + exposed_length)
    assert (mapping.file_start, mapping.file_end) == (file_start, file_end)


def test_compressed_snapshot_like_reference_exposes_extent_slice_only() -> None:
    fs = open_fixture("btrfs-compression.bin.gz")
    report = fs.find_extent_references(0x4D6E000, 0x1000, [5])

    assert len(report.references) == 1
    reference = report.references[0]
    assert reference.paths == ("zstd_lorem_ipsum.txt",)
    assert (reference.disk_start, reference.disk_end) == (0x4D6E000, 0x4D6E000 + 0x7000)
    assert (reference.exposed_start, reference.exposed_end) == (0x20000, 0x40000)
    assert reference.precision == "extent"
    assert reference.compressed
    assert not reference.partial_exposure

    # The forward mapping can still be used to read the exposed slice.
    inode = fs.get("zstd_lorem_ipsum.txt")
    with inode.open() as fh:
        fh.seek(reference.file_start)
        assert fh.read(16) == b"in. Vivamus temp"


def test_extent_tree_reference_counts_corroborate_shared_and_private_extents() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")
    report = fs.find_extent_references(0x4500000, 0x1000, [256, 257])

    assert len(report.allocations) == 1
    allocation = report.allocations[0]
    assert (allocation.bytenr, allocation.length) == (0x4500000, 0x501000)
    assert allocation.refs == len(report.references) == 2
    assert allocation.flags & 1  # BTRFS_EXTENT_FLAG_DATA

    sparse = open_fixture("btrfs-sparse.bin.gz")
    first = sparse.find_extent_references(0xD00000, 0x1000, [5])
    second = sparse.find_extent_references(0xD14000, 0x1000, [5])

    assert first.allocations[0].refs == 2  # shared with the snapshot root
    assert second.allocations[0].refs == 1  # private, despite identical content


def test_allocation_probe_handles_ranges_spanning_multiple_extents() -> None:
    fs = open_fixture("btrfs-sparse.bin.gz")

    # 0xd00000..0xd14000 and 0xd14000..0xd28000 are adjacent allocations with different refcounts.
    report = fs.find_extent_references(0xD10000, 0x5000, [5])
    assert [(item.bytenr, item.refs) for item in report.allocations] == [(0xD00000, 2), (0xD14000, 1)]

    # Query starting inside an allocation must still find that allocation via the backward probe.
    inner = fs.find_extent_references(0xD0F000, 0x1000, [5])
    assert [(item.bytenr, item.length) for item in inner.allocations] == [(0xD00000, 0x14000)]

    # A hole between allocations has no extent tree record.
    hole = fs.find_extent_references(0xD51800, 0x100, [5])
    assert hole.allocations == ()
    assert hole.references == ()


def test_reverse_lookup_works_on_raid1_and_reports_multi_device_scope() -> None:
    halves = [
        io.BytesIO(gzip.open(Path(__file__).parent / "_data" / f"btrfs-raid1-{index}.bin.gz", "rb").read())
        for index in (1, 2)
    ]
    fs = Btrfs(halves)
    report = fs.find_extent_references(0x5410000, 0x1000, [subvolume.objectid for subvolume in fs.subvolumes()])

    assert report.complete
    assert report.exhaustive
    assert len(report.references) == 1
    reference = report.references[0]
    assert (reference.inode, reference.paths) == (263, ("large.txt",))
    assert (reference.file_start, reference.file_end) == (0, 0x1000)
    assert [(item.bytenr, item.refs) for item in report.allocations] == [(0x5410000, 1)]


def test_partial_scan_mid_walk_reports_gap_and_no_false_completeness() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")
    original_walk = Cursor.walk

    def truncated_walk(self: Cursor, *args: object, **kwargs: object) -> object:
        for index, item in enumerate(original_walk(self, *args, **kwargs)):
            if index == 24:
                raise OSError("simulated corrupt leaf")
            yield item

    with patch.object(Cursor, "walk", truncated_walk):
        report = fs.find_extent_references(0x4500000, 0x1000, [256])

    assert not report.complete
    assert not report.exhaustive
    scan_gaps = [gap for gap in report.coverage_gaps if gap.kind == "partial_scan"]
    assert len(scan_gaps) == 1
    assert scan_gaps[0].subvolume_objectid == 256
    assert scan_gaps[0].items_scanned == 24
    assert scan_gaps[0].extent_items_scanned == 3
    # References found before the gap are still returned, but never presented as complete.
    assert len(report.references) == 1
    assert "simulated corrupt leaf" in scan_gaps[0].reason


def test_map_overlap_without_disk_overlap_returns_no_file_range() -> None:
    for compression in (0, 1):
        mapping = map_overlap(
            0x2000,
            0x3000,
            disk_start=0x1000,
            disk_num_bytes=0x1000,
            extent_offset=0,
            num_bytes=0x1000,
            ram_bytes=0x1000,
            key_offset=0,
            compression=compression,
        )
        assert mapping.file_start is None
        assert mapping.file_end is None


def test_inline_only_inode_yields_no_disk_reference_but_counts_it() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")

    # small.txt is stored inline, so it has no disk extent to match.
    report = fs.find_extent_references(0x4500000, 0x1000, [256])
    assert report.stats.inline_extents_skipped >= 1
    assert all(reference.inode != 258 for reference in report.references)


def test_unknown_selected_root_is_a_coverage_gap() -> None:
    fs = open_fixture("btrfs-subvolume-snapshot.bin.gz")
    report = fs.find_extent_references(0x4500000, 0x1000, [999])

    assert not report.complete
    assert len(report.coverage_gaps) == 1
    gap = report.coverage_gaps[0]
    assert gap.kind == "root_not_found"
    assert gap.subvolume_objectid == 999
    assert report.selected[0].objectid == 999
    assert report.selected[0].root_bytenr is None
