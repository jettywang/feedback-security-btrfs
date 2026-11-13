"""Build small, self-contained Btrfs images for a single extent sample.

Only the structures reachable by ``dissect.btrfs``'s read path are emitted:
superblock, chunk tree, root tree and one FS-tree leaf holding the target
file's inode, directory entry and file extent item.  The target file payload
is placed in a plain DATA chunk so both entry points exercise the real
:class:`~dissect.btrfs.stream.ExtentStream` and
:meth:`~dissect.btrfs.btrfs.INode.open` code.
"""

from __future__ import annotations

import struct

FSID = bytes(range(16))
CHUNK_TREE_UUID = bytes([2] * 16)
NODE_SIZE = 16384
SECTOR_SIZE = 4096

CHUNK_LOGICAL = 0x100000
CHUNK_LENGTH = 0x300000
CHUNK_TREE = 0x100000
ROOT_TREE = 0x104000
FS_TREE = 0x108000
DATA_BASE = 0x200000
DATA_SLOT = 0x10000

ROOT_DIR = 0x100
TARGET_INUM = 0x101
TARGET_NAME = b"extent.bin"

BTRFS_MAGIC = 0x4D5F53665248425F
BTRFS_INODE_ITEM_KEY = 0x1
BTRFS_DIR_ITEM_KEY = 0x54
BTRFS_EXTENT_DATA_KEY = 0x6C
BTRFS_ROOT_ITEM_KEY = 0x84
BTRFS_CHUNK_ITEM_KEY = 0xE4
BTRFS_FS_TREE_OBJECTID = 0x5
BTRFS_ROOT_TREE_DIR_OBJECTID = 0x6
BTRFS_FT_REG_FILE = 0x1
BTRFS_FT_DIR = 0x2
BTRFS_BLOCK_GROUP_DATA = 0x1


def _key(objectid: int, type_: int, offset: int) -> bytes:
    return struct.pack("<QBQ", objectid, type_, offset)


def _dir_hash(name: bytes) -> int:
    from dissect.util.hash import crc32c

    return (crc32c.update(1, name) ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _header(bytenr: int, owner: int, nritems: int, level: int = 0) -> bytes:
    return struct.pack("<32s16sQQ16sQQIB", b"\0" * 32, FSID, bytenr, 0, b"\0" * 16, 1, owner, nritems, level)


def _timespec() -> bytes:
    return struct.pack("<QI", 0, 0)


def _inode(size: int, mode: int) -> bytes:
    return (
        struct.pack("<QQQQQIIIIQQQ", 1, 1, size, size, 0, 1, 0, 0, mode, 0, 0, 0)
        + struct.pack("<4Q", 0, 0, 0, 0)
        + _timespec() * 4
    )


def _dirent(name: bytes, inum: int, type_: int) -> bytes:
    return _key(inum, BTRFS_INODE_ITEM_KEY, 0) + struct.pack("<QHHB", 1, 0, len(name), type_) + name


def inline_item(
    payload: bytes,
    compression: int,
    declared: int,
    encryption: int = 0,
    extent_type: int = 0,
    generation: int = 1,
    other_encoding: int = 0,
) -> bytes:
    """Serialize a ``btrfs_file_extent_item`` with an inline payload."""
    return (
        struct.pack("<QQBBHB", generation, declared, compression, encryption, other_encoding, extent_type) + payload
    )


def regular_item(
    disk_bytenr: int,
    disk_num_bytes: int,
    declared: int,
    num_bytes: int,
    compression: int,
    offset: int = 0,
    encryption: int = 0,
    extent_type: int = 1,
    generation: int = 1,
    other_encoding: int = 0,
) -> bytes:
    """Serialize a ``btrfs_file_extent_item`` with a regular (non-inline) extent."""
    return struct.pack(
        "<QQBBHBQQQQ",
        generation,
        declared,
        compression,
        encryption,
        other_encoding,
        extent_type,
        disk_bytenr,
        disk_num_bytes,
        offset,
        num_bytes,
    )


def extent_item(sample) -> bytes:
    """Build the extent item for a :class:`~btrfs_extent_fuzz.corpus.Sample`."""
    from btrfs_extent_fuzz.corpus import Entrypoint

    compression = sample.codec.compression_id
    if sample.entry is Entrypoint.INLINE:
        return inline_item(sample.payload, compression, sample.declared)
    return regular_item(
        disk_bytenr=extent_disk_bytenr(sample),
        disk_num_bytes=sample.disk_num_bytes,
        declared=sample.declared,
        num_bytes=sample.declared,
        compression=compression,
        offset=sample.disk_offset,
    )


def extent_disk_bytenr(sample) -> int:
    return DATA_BASE


def _leaf(bytenr: int, owner: int, items: list[tuple[tuple[int, int, int], bytes]]) -> bytes:
    ordered = sorted(items, key=lambda kv: kv[0])
    header = _header(bytenr, owner, len(ordered))
    capacity = NODE_SIZE - len(header) - sum(len(_key(*key)) + 8 for key, _ in ordered)
    if sum(len(data) for _, data in ordered) > capacity:
        raise ValueError("extent item does not fit in a single B-tree leaf")
    buf = bytearray(NODE_SIZE)
    buf[0 : len(header)] = header
    pos = NODE_SIZE
    for index, ((objectid, type_, offset), data) in enumerate(ordered):
        pos -= len(data)
        buf[pos : pos + len(data)] = data
        item = _key(objectid, type_, offset) + struct.pack("<II", pos - len(header), len(data))
        buf[len(header) + index * len(item) : len(header) + (index + 1) * len(item)] = item
    return bytes(buf)


def _root_item(bytenr: int) -> bytes:
    buf = bytearray(439)
    buf[0:160] = _inode(0, 0o40755)
    struct.pack_into("<QQQQQQQI", buf, 160, 1, ROOT_DIR, bytenr, 0, 0, 0, 0, 1)
    buf[220:237] = _key(0, 0, 0)
    struct.pack_into("<Q", buf, 239, 1)
    buf[247:263] = bytes([1] * 16)
    struct.pack_into("<QQQQ", buf, 295, 1, 1, 0, 0)
    buf[327:375] = _timespec() * 4
    return bytes(buf)


def _chunk(length: int) -> bytes:
    body = struct.pack(
        "<QQQQIIIHH", length, 2, 0x10000, BTRFS_BLOCK_GROUP_DATA, SECTOR_SIZE, SECTOR_SIZE, SECTOR_SIZE, 1, 1
    )
    return body + struct.pack("<QQ16s", 1, CHUNK_LOGICAL, CHUNK_TREE_UUID)


def _superblock(total_bytes: int, sys_chunk_array: bytes) -> bytes:
    sb = bytearray(4096)
    struct.pack_into("<32s16sQQQ", sb, 0, b"\0" * 32, FSID, 0x10000, 0, BTRFS_MAGIC)
    struct.pack_into("<QQQQQQQQ", sb, 72, 1, ROOT_TREE, CHUNK_TREE, 0, 0, 0, 0, 0)
    struct.pack_into("<Q", sb, 136, 1)
    struct.pack_into("<IIIIIQ", sb, 144, SECTOR_SIZE, NODE_SIZE, NODE_SIZE, SECTOR_SIZE, len(sys_chunk_array), 1)
    struct.pack_into(
        "<QQQIIIQQQI BB 16s16s".replace(" ", ""),
        sb,
        201,
        1,
        total_bytes,
        0,
        SECTOR_SIZE,
        SECTOR_SIZE,
        SECTOR_SIZE,
        0,
        1,
        0,
        0,
        0,
        0,
        CHUNK_TREE_UUID,
        FSID,
    )
    sb[299:307] = b"fuzzseed"
    sb[811 : 811 + len(sys_chunk_array)] = sys_chunk_array
    sb[571:587] = FSID
    struct.pack_into("<Q", sb, 587, 1)
    return bytes(sb)


def build_image(sample) -> bytes:
    """Return a complete Btrfs volume image holding ``sample`` as one file."""
    from btrfs_extent_fuzz.corpus import Entrypoint

    chunk_bytes = _chunk(CHUNK_LENGTH)
    image = bytearray(DATA_BASE + 4 * DATA_SLOT)

    fs_items: list[tuple[tuple[int, int, int], bytes]] = [
        ((ROOT_DIR, BTRFS_INODE_ITEM_KEY, 0), _inode(0, 0o40755)),
        ((ROOT_DIR, BTRFS_DIR_ITEM_KEY, _dir_hash(TARGET_NAME)), _dirent(TARGET_NAME, TARGET_INUM, BTRFS_FT_REG_FILE)),
        ((TARGET_INUM, BTRFS_INODE_ITEM_KEY, 0), _inode(sample.inode_size, 0o100644)),
    ]
    fs_items.append(((TARGET_INUM, BTRFS_EXTENT_DATA_KEY, 0), extent_item(sample)))

    if sample.entry is Entrypoint.REGULAR:
        disk_bytenr = extent_disk_bytenr(sample)
        # Only the four reserved data slots exist; a payload that overflows
        # them would be a harness artefact, not a decoder input.
        if len(sample.payload) > 4 * DATA_SLOT:
            raise ValueError("payload exceeds the reserved data slots")
        stored = sample.payload + b"\0" * (-len(sample.payload) % SECTOR_SIZE)
        image[disk_bytenr : disk_bytenr + len(stored)] = stored

    image[FS_TREE : FS_TREE + NODE_SIZE] = _leaf(FS_TREE, BTRFS_FS_TREE_OBJECTID, fs_items)
    image[ROOT_TREE : ROOT_TREE + NODE_SIZE] = _leaf(
        ROOT_TREE,
        1,
        [
            ((BTRFS_FS_TREE_OBJECTID, BTRFS_ROOT_ITEM_KEY, 0), _root_item(FS_TREE)),
            (
                (BTRFS_ROOT_TREE_DIR_OBJECTID, BTRFS_DIR_ITEM_KEY, _dir_hash(b"default")),
                _dirent(b"default", BTRFS_FS_TREE_OBJECTID, BTRFS_FT_DIR),
            ),
        ],
    )
    image[CHUNK_TREE : CHUNK_TREE + NODE_SIZE] = _leaf(
        CHUNK_TREE, 1, [((256, BTRFS_CHUNK_ITEM_KEY, CHUNK_LOGICAL), chunk_bytes)]
    )

    sys_chunk_array = _key(256, BTRFS_CHUNK_ITEM_KEY, CHUNK_LOGICAL) + chunk_bytes
    sb = _superblock(len(image), sys_chunk_array)
    image[0x10000 : 0x10000 + len(sb)] = sb
    return bytes(image)


def read_target(image: bytes) -> tuple[str, bytes]:
    """Open the built image and read the target file through the public API."""
    import io

    from dissect.btrfs import Btrfs

    fs = Btrfs(io.BytesIO(image))
    inode = fs.get(TARGET_NAME.decode())
    fh = inode.open()
    return type(fh).__name__, fh.read()


__all__ = [
    "BTRFS_MAGIC",
    "TARGET_INUM",
    "TARGET_NAME",
    "build_image",
    "extent_item",
    "inline_item",
    "read_target",
    "regular_item",
]
