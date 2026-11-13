from __future__ import annotations

import datetime
import hashlib
import io
import struct
from typing import BinaryIO

import pytest
from dissect.util.hash import crc32c
from dissect.util.stream import BufferedStream

from dissect.btrfs.btrfs import Btrfs
from dissect.btrfs.c_btrfs import c_btrfs
from dissect.btrfs.exceptions import Error, FileNotFoundError


def assert_test_data(fs: Btrfs) -> None:
    entry = fs.get("")
    assert entry.is_dir()
    assert entry.path == ""

    entry = fs.get("path")
    assert entry.is_dir()
    assert not entry.is_file()
    assert not entry.is_symlink()
    assert not entry.is_device()
    assert not entry.is_ipc()
    assert entry.size == 4
    assert entry.path == "path"
    assert entry.uid == 1000
    assert entry.gid == 1000

    entry = fs.get("path/to/a/file.txt")
    assert entry.is_file()
    assert entry.size == 12
    assert entry.path == "path/to/a/file.txt"
    assert entry.open().read() == b"file in dir\n"

    entry = fs.get("small.txt")
    assert entry.is_file()
    assert entry.size == 29
    assert entry.path == "small.txt"
    assert entry.open().read() == b"small file content goes here\n"

    entry = fs.get("large.txt")
    assert entry.is_file()
    assert entry.size == 5242881
    assert entry.path == "large.txt"
    assert entry.open().read() == (b"a" * 5242880) + b"\n"

    entry = fs.get("link.txt")
    assert entry.is_symlink()
    assert entry.path == "link.txt"
    assert entry.link == "path/to/a/file.txt"


def test_btrfs(btrfs_default: BinaryIO) -> None:
    fs = Btrfs(btrfs_default)

    assert fs.label == "btrfs-default"
    assert str(fs.uuid) == "74387226-fa97-4f42-a276-9bb07ce5e62d"

    root = fs.get("/")
    assert list(root.listdir().keys()) == [".", "..", "path", "link.txt", "small.txt", "large.txt"]

    assert_test_data(fs)

    entry = fs.get("path")
    assert entry.mode == 0o40755
    assert entry.atime == datetime.datetime(2023, 6, 28, 3, 4, 16, tzinfo=datetime.timezone.utc)
    assert entry.ctime == datetime.datetime(2023, 6, 28, 3, 4, 12, tzinfo=datetime.timezone.utc)
    assert entry.mtime == datetime.datetime(2023, 6, 28, 3, 4, 12, tzinfo=datetime.timezone.utc)
    assert entry.otime == datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


def test_btrfs_subvolume(btrfs_subvolume: BinaryIO) -> None:
    fs = Btrfs(btrfs_subvolume)

    assert fs.label == "btrfs-subvolume"
    assert str(fs.uuid) == "74387226-fa97-4f42-a276-9bb07ce5e62d"

    subvol = fs.get("subvol")
    assert subvol.path == ""
    assert subvol.subvolume.path == "subvol"
    assert subvol.full_path == "subvol"
    assert list(subvol.listdir().keys()) == [".", "..", "cross-volume-link.txt", "small.txt", "large.txt", "some"]

    assert_test_data(fs)

    entry = fs.get("subvol/some/more/dirs/empty.txt")
    assert entry.is_file()
    assert entry.size == 0
    assert entry.path == "some/more/dirs/empty.txt"
    assert entry.full_path == "subvol/some/more/dirs/empty.txt"
    assert entry.open().read() == b""

    entry = fs.get("subvol/small.txt")
    assert entry.is_file()
    assert entry.size == 18
    assert entry.open().read() == b"file in subvolume\n"

    entry = fs.get("subvol/large.txt")
    assert entry.is_file()
    assert entry.size == 5242881
    assert entry.open().read() == (b"b" * 5242880) + b"\n"

    entry = fs.get("subvol/cross-volume-link.txt")
    assert entry.is_symlink()
    assert entry.link == "../link.txt"


def test_btrfs_subvolume_custom_default(btrfs_subvolume_custom_default: BinaryIO) -> None:
    fs = Btrfs(btrfs_subvolume_custom_default)

    assert fs.label == "btrfs-subvolume-custom-default"
    assert str(fs.uuid) == "74387226-fa97-4f42-a276-9bb07ce5e62d"


def test_btrfs_subvolume_nested(btrfs_subvolume_nested: BinaryIO) -> None:
    fs = Btrfs(btrfs_subvolume_nested)

    assert fs.label == "btrfs-subvolume-nested"
    assert str(fs.uuid) == "983bda53-2928-4182-a53d-041eade66473"

    subvolumes = list(fs.subvolumes())

    assert len(subvolumes) == 4
    assert sorted((subvol.objectid, subvol.path) for subvol in subvolumes) == [
        (5, ""),
        (256, "dir/volume"),
        (257, "default"),
        (258, "default/volume"),
    ]

    assert fs.find_subvolume("default/volume").objectid == 258


def test_btrfs_subvolume_dotdot_context(btrfs_subvolume_nested: BinaryIO) -> None:
    """Entering a subvolume and returning with ".." must switch back to the parent tree.

    The parent and child subvolumes deliberately share inode numbers (all roots are 256)
    and directory names ("volume" exists in multiple subvolumes).
    """
    fs = Btrfs(btrfs_subvolume_nested)

    root = fs.get("")
    assert (root.subvolume.objectid, root.inum) == (257, 256)

    entry = fs.get("volume")
    assert (entry.subvolume.objectid, entry.inum) == (258, 256)

    # ".." from the subvolume root returns to the parent directory in the parent subvolume
    parent = fs.get("volume/..")
    assert (parent.subvolume.objectid, parent.inum) == (257, 256)

    # re-entering after returning resolves in the parent tree again
    again = fs.get("volume/../volume")
    assert (again.subvolume.objectid, again.inum) == (258, 256)

    # full path lookup, segment-wise lookup and directory enumeration must agree
    segment = fs.get("volume").get("..")
    assert (segment.subvolume.objectid, segment.inum) == (257, 256)

    listing = fs.get("volume/..").listdir()
    assert (listing["volume"].subvolume.objectid, listing["volume"].inum) == (258, 256)
    assert (listing[".."].subvolume.objectid, listing[".."].inum) == (257, 256)

    # ".." at the root of a separately opened subvolume stays at that root
    subvolume = fs.open_subvolume(258)
    assert (subvolume.get("..").subvolume.objectid, subvolume.get("..").inum) == (258, 256)
    assert (subvolume.get("../..").subvolume.objectid, subvolume.get("../..").inum) == (258, 256)


def test_btrfs_subvolume_dotdot_content(btrfs_subvolume: BinaryIO) -> None:
    """Reading after crossing into a subvolume and back must return parent volume data."""
    fs = Btrfs(btrfs_subvolume)

    parent_small = fs.get("small.txt")
    subvol_small = fs.get("subvol/small.txt")
    assert parent_small.open().read() == b"small file content goes here\n"
    assert subvol_small.open().read() == b"file in subvolume\n"

    # cross into the subvolume and return to the parent volume
    entry = fs.get("subvol/../small.txt")
    assert (entry.subvolume.objectid, entry.inum) == (parent_small.subvolume.objectid, parent_small.inum)
    assert entry.open().read() == b"small file content goes here\n"

    # ".." stays within the subvolume until it crosses the subvolume root
    entry = fs.get("subvol/some/more/dirs/../../../small.txt")
    assert (entry.subvolume.objectid, entry.inum) == (subvol_small.subvolume.objectid, subvol_small.inum)
    assert entry.open().read() == b"file in subvolume\n"

    entry = fs.get("subvol/some/more/dirs/../../../../small.txt")
    assert (entry.subvolume.objectid, entry.inum) == (parent_small.subvolume.objectid, parent_small.inum)
    assert entry.open().read() == b"small file content goes here\n"

    # segment-wise lookup and enumeration agree with the full path lookup
    entry = fs.get("subvol").get("..").get("small.txt")
    assert (entry.subvolume.objectid, entry.inum) == (parent_small.subvolume.objectid, parent_small.inum)

    listing = fs.get("subvol/..").listdir()
    assert (listing["small.txt"].subvolume.objectid, listing["small.txt"].inum) == (
        parent_small.subvolume.objectid,
        parent_small.inum,
    )


def test_btrfs_get_not_found_diagnostics(btrfs_default: BinaryIO) -> None:
    """A failed lookup must report the path, component, directory inode and subvolume."""
    fs = Btrfs(btrfs_default)

    with pytest.raises(
        FileNotFoundError,
        match=r"File not found: nonexistent \(no entry b'nonexistent' in directory inode 256 of subvolume 5\)",
    ):
        fs.get("nonexistent")


def test_btrfs_subvolume_snapshot(btrfs_subvolume_snapshot: BinaryIO) -> None:
    fs = Btrfs(btrfs_subvolume_snapshot)

    assert fs.label == "btrfs-subvolume-snapshot"
    assert str(fs.uuid) == "74387226-fa97-4f42-a276-9bb07ce5e62d"

    assert_test_data(fs)


def test_btrfs_compression(btrfs_compression: BinaryIO) -> None:
    fs = Btrfs(btrfs_compression)

    fh = fs.get("zlib.txt").open()
    assert len([extent for extent in fh.extents if extent.compression == c_btrfs.BTRFS_COMPRESS_ZLIB]) == 160
    assert fh.read() == (b"zlib" * 1024 * 1024 * 5) + b"\n"

    fh = fs.get("zlib_inline.txt").open()
    assert isinstance(fh, BufferedStream)
    assert fh.read() == (b"zlib" * 256) + b"\n"

    fh = fs.get("lzo.txt").open()
    assert len([extent for extent in fh.extents if extent.compression == c_btrfs.BTRFS_COMPRESS_LZO]) == 120
    assert fh.read() == b"lzo" * 1024 * 1024 * 5 + b"\n"

    fh = fs.get("lzo_inline.txt").open()
    assert isinstance(fh, BufferedStream)
    assert fh.read() == (b"lzo" * 256) + b"\n"

    fh = fs.get("zstd.txt").open()
    assert len([extent for extent in fh.extents if extent.compression == c_btrfs.BTRFS_COMPRESS_ZSTD]) == 160
    assert fh.read() == (b"zstd" * 1024 * 1024 * 5) + b"\n"

    fh = fs.get("zstd_inline.txt").open()
    assert isinstance(fh, BufferedStream)
    assert fh.read() == (b"zstd" * 256) + b"\n"

    fh = fs.get("zstd_lorem_ipsum.txt").open()
    assert len([extent for extent in fh.extents if extent.compression == c_btrfs.BTRFS_COMPRESS_ZSTD]) == 4
    assert hashlib.sha256(fh.read()).hexdigest() == "ac9db55e11e804c58cb5ac8baded462cd3e7a570e4f3c980298ef2b417467b1d"


@pytest.mark.parametrize(
    "fixture",
    [
        "btrfs_profile_dup",
        "btrfs_profile_raid0",
        "btrfs_profile_raid1",
        "btrfs_profile_raid1c3",
        "btrfs_profile_raid1c4",
        "btrfs_profile_raid5",
        "btrfs_profile_raid6",
        "btrfs_profile_raid10",
    ],
)
def test_btrfs_profiles(fixture: str, request: pytest.FixtureRequest) -> None:
    fs = Btrfs(request.getfixturevalue(fixture))
    assert_test_data(fs)


@pytest.mark.parametrize(
    "fixture",
    [
        "btrfs_profile_dup",
        "btrfs_profile_raid0",
        "btrfs_profile_raid1",
        "btrfs_profile_raid1c3",
        "btrfs_profile_raid1c4",
        "btrfs_profile_raid5",
        "btrfs_profile_raid6",
        "btrfs_profile_raid10",
    ],
)
def test_btrfs_profiles_partial(fixture: str, request: pytest.FixtureRequest) -> None:
    fhs = request.getfixturevalue(fixture)
    # The btrfs_profile_dup test data just so happens to have all chunks on the second device
    fhs = fhs[:-1] if fixture == "btrfs_profile_dup" else fhs[1:]

    if fixture in ("btrfs_profile_dup", "btrfs_profile_raid0"):
        with pytest.raises(Error, match=r"Missing stripe disk for chunk offset .+"):
            fs = Btrfs(fhs)

    elif fixture in ("btrfs_profile_raid5", "btrfs_profile_raid6"):
        pass

    else:
        fs = Btrfs(fhs)
        assert_test_data(fs)


def test_btrfs_sparse(btrfs_sparse: BinaryIO) -> None:
    fs = Btrfs(btrfs_sparse)

    entry = fs.get("sparse_start")
    assert entry.size == 0x3C000
    assert entry.extents() == [
        (0, 0, 0, 0, 0, 40 * 4096),
        (0, 0, 0xD28000, 20 * 4096, 0, 20 * 4096),
    ]
    assert entry.open().read() == ((b"\x00" * 4096) * 40) + ((b"\x01" * 4096) * 20)

    entry = fs.get("sparse_hole")
    assert entry.size == 0x3C000
    assert entry.extents() == [
        (0, 0, 0xD00000, 20 * 4096, 0, 20 * 4096),
        (0, 0, 0, 0, 0, 20 * 4096),
        (0, 0, 0xD14000, 20 * 4096, 0, 20 * 4096),
    ]
    assert entry.open().read() == ((b"\x01" * 4096) * 20) + ((b"\x00" * 4096) * 20) + ((b"\x01" * 4096) * 20)

    entry = fs.get("sparse_end")
    assert entry.size == 0x3C000
    assert entry.extents() == [
        (0, 0, 0xD3C000, 20 * 4096, 0, 20 * 4096),
        (0, 0, 0, 0, 0, 40 * 4096),
    ]
    assert entry.open().read() == ((b"\x01" * 4096) * 20) + ((b"\x00" * 4096) * 40)

    entry = fs.get("sparse_all")
    assert entry.size == 0x500000
    assert entry.extents() == [
        (0, 0, 0, 0, 0, 1280 * 4096),
    ]
    assert entry.open().read() == (b"\x00" * 4096) * 1280

    entry = fs.get("snapshot/sparse_hole")
    assert entry.size == 0x3C000
    assert entry.extents() == [
        (0, 0, 0xD00000, 20 * 4096, 0, 10 * 4096),
        (0, 0, 0xD50000, 1 * 4096, 0, 1 * 4096),
        (0, 0, 0, 0, 0, 39 * 4096),
        (0, 0, 0xD52000, 10 * 4096, 0, 10 * 4096),
    ]
    assert entry.open().read() == ((b"\x01" * 4096) * 10) + ((b"\x00" * 4096) * 50)

    entry = fs.get("snapshot/sparse_end")
    assert entry.size == 0x3C000
    assert entry.extents() == [
        (0, 0, 0xD3C000, 20 * 4096, 0, 1 * 4096),
        (0, 0, 0xD5C000, 1 * 4096, 0, 1 * 4096),
        (0, 0, 0xD3C000, 20 * 4096, 2 * 4096, 18 * 4096),
        (0, 0, 0, 0, 0, 40 * 4096),
    ]
    assert entry.open().read() == (b"\x01" * (4096 + 123)) + (b"\x02") + (b"\x01" * ((4096 * 19) - 124)) + (
        (b"\x00" * 4096) * 40
    )


def _build_dir_item(objectid: int, name: bytes, type: int = c_btrfs.BTRFS_FT_REG_FILE) -> bytes:
    """Build a raw ``btrfs_dir_item`` record."""
    location = struct.pack("<QBQ", objectid, c_btrfs.BTRFS_INODE_ITEM_KEY, 0)
    return location + struct.pack("<QHHB", 5, 0, len(name), type) + name


def _pack_dir_item_records(image: bytes, name: bytes, records: list[bytes]) -> BinaryIO:
    """Return a copy of a filesystem image where the ``BTRFS_DIR_ITEM_KEY`` item for ``name``
    in the root directory of the default subvolume is replaced by the given packed records.

    This emulates how Btrfs stores directory entries with colliding name hashes:
    as consecutive ``btrfs_dir_item`` records in the data of a single tree item.
    Node checksums are not verified by the parser, so the image remains readable.
    """
    fs = Btrfs(io.BytesIO(image))
    root = fs.root
    name_hash = (crc32c.update(1, name) ^ 0xFFFFFFFF) & 0xFFFFFFFF

    cursor = fs.default_subvolume.tree.cursor()
    if not next(cursor.iter(root.inum, c_btrfs.BTRFS_DIR_ITEM_KEY, name_hash), None):
        raise ValueError(f"No dir_item for {name!r} in image")

    # Locate the tree node in the raw image by its fsid + logical address header fields
    raw = bytearray(image)
    needle = fs.uuid.bytes + struct.pack("<Q", cursor._header.bytenr)
    node_offset = raw.find(needle) - c_btrfs.BTRFS_CSUM_SIZE
    if node_offset < 0:
        raise ValueError("Can't locate tree node in image")

    header_size = len(c_btrfs.btrfs_header)
    item_size = len(c_btrfs.btrfs_item)
    node = raw[node_offset : node_offset + fs.node_size]
    header = c_btrfs.btrfs_header(node)

    packed = b"".join(records)

    # Relocate the item data to the free space between the item array and the item data area
    # Item data offsets are relative to the end of the node header
    data_offsets = []
    for index in range(header.nritems):
        item_offset = header_size + (index * item_size)
        item = c_btrfs.btrfs_item(node[item_offset : item_offset + item_size])
        data_offsets.append(item.offset)

        if (item.key.objectid, item.key.type, item.key.offset) == (root.inum, c_btrfs.BTRFS_DIR_ITEM_KEY, name_hash):
            target_item_offset = item_offset

    new_data_offset = min(data_offsets) - len(packed)
    if new_data_offset < header.nritems * item_size:
        raise ValueError("Not enough free space in tree node")

    raw[node_offset + header_size + new_data_offset : node_offset + header_size + new_data_offset + len(packed)] = (
        packed
    )
    struct.pack_into("<I", raw, node_offset + target_item_offset + len(c_btrfs.btrfs_disk_key), new_data_offset)
    struct.pack_into("<I", raw, node_offset + target_item_offset + len(c_btrfs.btrfs_disk_key) + 4, len(packed))

    return io.BytesIO(bytes(raw))


def test_btrfs_dir_item_hash_collision(btrfs_default: BinaryIO) -> None:
    """Multiple name records under one DIR_ITEM hash must be matched by name, not by position."""
    image = btrfs_default.read()
    fs = Btrfs(io.BytesIO(image))
    small = fs.get("small.txt")
    large = fs.get("large.txt")

    # An impostor record for a different name is stored first under the hash of "small.txt",
    # followed by the real record for "small.txt"
    records = [
        _build_dir_item(large.inum, b"impostor"),
        _build_dir_item(small.inum, b"small.txt"),
    ]
    fs = Btrfs(_pack_dir_item_records(image, b"small.txt", records))

    # full path lookup must skip the impostor and find the real record
    entry = fs.get("small.txt")
    assert (entry.subvolume.objectid, entry.inum) == (small.subvolume.objectid, small.inum)
    assert entry.open().read() == b"small file content goes here\n"

    # segment-wise lookup and directory enumeration must agree with the full path lookup
    segment = fs.get("").get("small.txt")
    assert (segment.subvolume.objectid, segment.inum) == (entry.subvolume.objectid, entry.inum)

    listing = fs.get("").listdir()
    assert (listing["small.txt"].subvolume.objectid, listing["small.txt"].inum) == (
        entry.subvolume.objectid,
        entry.inum,
    )

    # the impostor name itself does not resolve
    with pytest.raises(FileNotFoundError, match=r"no entry b'impostor'"):
        fs.get("impostor")


def test_btrfs_dir_item_corrupt_collision(btrfs_default: BinaryIO) -> None:
    """A corrupt record under a shared DIR_ITEM hash must not let another record impersonate the target."""
    image = btrfs_default.read()
    fs = Btrfs(io.BytesIO(image))
    large = fs.get("large.txt")

    # The first record is a valid impostor pointing to a different inode,
    # the second record is corrupt and can not be parsed
    records = [
        _build_dir_item(large.inum, b"impostor"),
        b"\xde\xad\xbe\xef",
    ]
    fs = Btrfs(_pack_dir_item_records(image, b"small.txt", records))

    # the lookup must fail with a locatable diagnostic instead of returning the impostor
    with pytest.raises(
        Error,
        match=r"Corrupt dir_item in directory inode 256 of subvolume 5 "
        r"while resolving component b'small\.txt' of path 'small\.txt'",
    ):
        fs.get("small.txt")
