from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from dissect.util.hash import crc32c

from dissect.btrfs.btrfs import SuperblockCopy, metadata_uuid_of, read_superblock_copies
from dissect.btrfs.c_btrfs import (
    BTRFS_BLOCK_GROUP,
    BTRFS_BLOCK_GROUP_PROFILE_MASK,
    BTRFS_RAID_ATTRIBUTES,
    c_btrfs,
)
from dissect.btrfs.exceptions import InconsistentFilesystemError

if TYPE_CHECKING:
    from collections.abc import Iterator


class Verdict(str, Enum):
    """How far a statement about a multi-device collection can be trusted."""

    PROVEN = "proven"
    """Every referenced member is present and every compared mapping agrees."""

    UNVERIFIED = "unverified"
    """A member or mapping is missing, so the collection cannot be shown to be one state."""

    REJECTED = "rejected"
    """Members actively contradict each other, or no valid superblock was supplied."""


class ChunkStatus(str, Enum):
    """Whether one logical chunk can be read from the supplied members."""

    READABLE = "readable"
    """Every stripe of this chunk is backed by a supplied, identity-matching member."""

    DEGRADED = "degraded"
    """At least one stripe is missing, but the RAID profile can reconstruct or skip it."""

    UNREADABLE = "unreadable"
    """More stripes are missing than the RAID profile tolerates."""

    CONFLICTING = "conflicting"
    """Two members describe this logical chunk with different geometry or stripes."""


@dataclass(frozen=True)
class MemberReport:
    """Identity and collection generation of one supplied image."""

    index: int
    devid: int
    dev_uuid: UUID
    generation: int
    superblock_offset: int
    superblock_checksum_valid: bool
    label: str


@dataclass(frozen=True)
class ChunkReport:
    """A logical chunk and the members needed to read it."""

    offset: int
    length: int
    profile: int
    status: ChunkStatus
    required_devids: tuple[int, ...]
    present_devids: tuple[int, ...]
    missing_devids: tuple[int, ...]


@dataclass
class IntakeReport:
    """Evidence-oriented summary of a candidate multi-device collection."""

    verdict: Verdict
    members: list[MemberReport] = field(default_factory=list)
    newest_generation: int = 0
    chunks: list[ChunkReport] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    conflicting_devids: list[int] = field(default_factory=list)
    duplicate_devids: list[int] = field(default_factory=list)
    chunk_scope: str = "system chunk array only"

    @property
    def snapshot_proven(self) -> bool:
        """Return whether a consistent point-in-time tree was established.

        Intake examination alone never establishes this: it compares superblocks
        and bootstrap chunk mappings, not the directory tree.  Opening the
        filesystem with ``Btrfs(..., strict=True)`` verifies each reachable tree
        block and each file's data separately.
        """
        return False

    @property
    def readable_chunks(self) -> list[ChunkReport]:
        return [chunk for chunk in self.chunks if chunk.status is ChunkStatus.READABLE]

    @property
    def unreadable_chunks(self) -> list[ChunkReport]:
        return [
            chunk for chunk in self.chunks if chunk.status in (ChunkStatus.DEGRADED, ChunkStatus.UNREADABLE)
        ]

    @property
    def missing_devids(self) -> list[int]:
        return sorted({devid for chunk in self.chunks for devid in chunk.missing_devids})

    def describe(self) -> str:
        """Return a compact, stable summary for a collection receipt."""
        lines = [
            f"verdict: {self.verdict.value}",
            f"newest generation: {self.newest_generation}",
            f"chunk scope: {self.chunk_scope}",
            "snapshot proven: no (intake did not verify the directory tree)",
        ]
        for member in self.members:
            checksum = "ok" if member.superblock_checksum_valid else "BAD"
            lines.append(
                f"image {member.index}: devid={member.devid} dev_uuid={member.dev_uuid} "
                f"generation={member.generation} superblock={checksum}"
            )
        for chunk in self.chunks:
            lines.append(
                f"chunk {chunk.offset:#x}+{chunk.length:#x}: {chunk.status.value} "
                f"required={list(chunk.required_devids)} missing={list(chunk.missing_devids)}"
            )
        lines.extend(f"finding: {finding}" for finding in self.findings)
        return "\n".join(lines)


def examine(fhs: list[BinaryIO]) -> IntakeReport:
    """Classify a set of device images before opening the filesystem.

    The examination is deliberately limited to superblocks and system chunk
    arrays, so it can describe collections that :class:`~dissect.btrfs.btrfs.Btrfs`
    would refuse.  It answers three questions:

    * Do the images claim the same filesystem (fsid and metadata UUID)?
    * Is any devid or dev_uuid claimed twice, possibly at different generations?
    * Which logical chunks are fully backed, degraded, unreadable or contradictory?

    It cannot prove that a directory tree is a consistent snapshot: that requires
    verifying every reachable tree block against its checksum and the generation
    recorded by its parent, which the parser does while reading.

    The chunk list is limited to the system chunk array, the bootstrap view every
    superblock carries.  Reading the chunk tree would extend it, but only for sets
    that are complete enough to open; this function therefore never implies more
    coverage than it examined.

    Args:
        fhs: Device images to examine.
    """
    report = IntakeReport(verdict=Verdict.UNVERIFIED)
    usable: list[SuperblockCopy] = []
    all_copies: list[list[SuperblockCopy]] = []

    for index, fh in enumerate(fhs):
        copies = read_superblock_copies(fh)
        all_copies.append(copies)
        valid = [copy for copy in copies if copy.checksum_valid]

        if not copies:
            report.findings.append(f"image {index}: no Btrfs superblock at a standard mirror")
            report.verdict = Verdict.REJECTED
            continue
        if not valid:
            report.findings.append(f"image {index}: all superblock mirrors failed checksum validation")
            report.verdict = Verdict.REJECTED
            continue

        identities = {(item.sb.dev_item.devid, bytes(item.sb.dev_item.uuid)) for item in valid}
        if len(identities) > 1:
            report.findings.append(f"image {index}: superblock mirrors disagree on devid/dev_uuid: {identities}")
            report.verdict = Verdict.REJECTED

        copy = max(valid, key=lambda item: item.generation)
        usable.append(copy)
        report.members.append(
            MemberReport(
                index=index,
                devid=copy.sb.dev_item.devid,
                dev_uuid=UUID(bytes=bytes(copy.sb.dev_item.uuid)),
                generation=copy.generation,
                superblock_offset=copy.offset,
                superblock_checksum_valid=copy.checksum_valid,
                label=copy.sb.label.split(b"\x00")[0].decode(errors="surrogateescape"),
            )
        )

        generations = sorted({item.generation for item in copies})
        if len(generations) > 1:
            report.findings.append(f"image {index}: superblock mirrors disagree on generation: {generations}")

    if not usable:
        report.verdict = Verdict.REJECTED
        return report

    fsids = {bytes(copy.sb.fsid) for copy in usable}
    if len(fsids) > 1:
        report.findings.append(f"images belong to {len(fsids)} different filesystems (fsid mismatch)")
        report.verdict = Verdict.REJECTED
        return report

    metadata_uuids = {metadata_uuid_of(copy.sb) for copy in usable}
    if len(metadata_uuids) > 1:
        report.findings.append(f"same fsid but different metadata UUIDs: {sorted(map(str, metadata_uuids))}")
        report.verdict = Verdict.REJECTED

    for copy in usable:
        sb = copy.sb
        if bytes(sb.dev_item.fsid) != bytes(sb.fsid):
            report.findings.append(f"devid {sb.dev_item.devid}: device fsid does not match superblock fsid")
            report.verdict = Verdict.REJECTED

    report.newest_generation = max(copy.generation for copy in usable)
    _check_members(usable, report)
    _check_chunks(usable, report)

    if report.verdict is Verdict.REJECTED:
        return report

    if report.unreadable_chunks:
        report.verdict = Verdict.UNVERIFIED
    elif any(copy.generation < report.newest_generation for copy in usable):
        report.verdict = Verdict.UNVERIFIED
    elif report.readable_chunks:
        report.verdict = Verdict.PROVEN
    else:
        report.verdict = Verdict.UNVERIFIED

    return report


def _check_members(usable: list[SuperblockCopy], report: IntakeReport) -> None:
    devids: dict[int, UUID] = {}
    uuids: dict[UUID, int] = {}

    for copy in usable:
        devid = copy.sb.dev_item.devid
        dev_uuid = UUID(bytes=bytes(copy.sb.dev_item.uuid))

        if devid in devids:
            report.duplicate_devids.append(devid)
            same_identity = devids[devid] == dev_uuid
            report.findings.append(
                f"devid {devid} appears on more than one image (dev_uuid {devids[devid]} and {dev_uuid}); "
                f"identities are {'identical' if same_identity else 'different'}"
            )
            report.verdict = Verdict.REJECTED
            continue

        if dev_uuid in uuids and uuids[dev_uuid] != devid:
            report.conflicting_devids.append(devid)
            report.findings.append(f"dev_uuid {dev_uuid} is claimed with devid {uuids[dev_uuid]} and {devid}")
            report.verdict = Verdict.REJECTED

        devids[devid] = dev_uuid
        uuids[dev_uuid] = devid

    older = [copy.sb.dev_item.devid for copy in usable if copy.generation < report.newest_generation]
    if older:
        report.findings.append(
            f"devids {sorted(older)} have older superblocks than generation {report.newest_generation}; "
            "the collection is not a single-point-in-time capture"
        )


def _iter_system_chunks(copy: SuperblockCopy) -> Iterator[tuple[int, c_btrfs.btrfs_chunk]]:
    sb = copy.sb
    stream = io.BytesIO(bytes(sb.sys_chunk_array)[: sb.sys_chunk_array_size])

    while stream.tell() < sb.sys_chunk_array_size:
        key = c_btrfs.btrfs_disk_key(stream)
        if key.type != c_btrfs.BTRFS_CHUNK_ITEM_KEY:
            raise InconsistentFilesystemError(f"Unexpected item type in system chunk array: {key}")
        yield key.offset, c_btrfs.btrfs_chunk(stream)


def _signature(chunk: c_btrfs.btrfs_chunk) -> tuple:
    return (
        chunk.length,
        chunk.stripe_len,
        int(chunk.type),
        chunk.num_stripes,
        chunk.sub_stripes,
        tuple((stripe.devid, stripe.offset, bytes(stripe.dev_uuid)) for stripe in chunk.stripe),
    )


def _check_chunks(usable: list[SuperblockCopy], report: IntakeReport) -> None:
    mappings: dict[int, tuple] = {}
    conflicting: set[int] = set()

    for copy in usable:
        try:
            for offset, chunk in _iter_system_chunks(copy):
                signature = _signature(chunk)
                previous = mappings.setdefault(offset, signature)
                if previous != signature:
                    conflicting.add(offset)
                    report.findings.append(
                        f"devid {copy.sb.dev_item.devid} maps system chunk {offset:#x} differently "
                        "from another member"
                    )
        except (InconsistentFilesystemError, EOFError, ValueError) as error:
            report.findings.append(
                f"devid {copy.sb.dev_item.devid}: unreadable system chunk array ({error})"
            )
            report.verdict = Verdict.REJECTED

    if conflicting:
        report.verdict = Verdict.REJECTED

    present = {copy.sb.dev_item.devid for copy in usable}
    for offset, signature in sorted(mappings.items()):
        length, _, chunk_type, num_stripes, _, stripes = signature
        required = tuple(sorted({stripe[0] for stripe in stripes}))
        missing = tuple(devid for devid in required if devid not in present)

        if offset in conflicting:
            status = ChunkStatus.CONFLICTING
        elif missing:
            _, _, tolerated = BTRFS_RAID_ATTRIBUTES[
                c_btrfs.BTRFS_BLOCK_GROUP(chunk_type) & BTRFS_BLOCK_GROUP_PROFILE_MASK
            ]
            missing_stripes = sum(1 for stripe in stripes if stripe[0] not in present)
            status = ChunkStatus.DEGRADED if missing_stripes <= tolerated else ChunkStatus.UNREADABLE
        else:
            status = ChunkStatus.READABLE

        report.chunks.append(
            ChunkReport(
                offset=offset,
                length=length,
                profile=chunk_type & int(BTRFS_BLOCK_GROUP_PROFILE_MASK),
                status=status,
                required_devids=required,
                present_devids=tuple(devid for devid in required if devid in present),
                missing_devids=missing,
            )
        )

        if num_stripes > len(required):
            report.findings.append(
                f"chunk {offset:#x}: {num_stripes} stripes share {len(required)} device(s); "
                "the loss of one image can remove more than one copy"
            )

    if report.verdict is not Verdict.REJECTED and any(
        chunk.status is not ChunkStatus.READABLE for chunk in report.chunks
    ):
        report.verdict = Verdict.UNVERIFIED


@dataclass(frozen=True)
class DataVerification:
    """Result of checking one file's extents against the checksum tree."""

    path: str
    checked_sectors: int
    unverified_sectors: int
    proven_bytes: int
    total_bytes: int
    reason: str = ""

    @property
    def fully_verified(self) -> bool:
        """Return whether every data sector of the file was checksum-verified."""
        return self.unverified_sectors == 0 and self.proven_bytes == self.total_bytes


def verify_file_data(fs: object, path: str) -> DataVerification:
    """Verify a file's data sectors against the Btrfs checksum tree.

    Checksums cover the bytes as stored on disk.  A verified compressed extent
    therefore proves the logical bytes it decompresses to, and inline data is
    covered by the checksum of the leaf block that contains it.  The result
    describes what the collection *proves*, not whether the file can be parsed.

    Args:
        fs: An opened :class:`~dissect.btrfs.btrfs.Btrfs` instance.
        path: Path of the file to verify.
    """
    node = fs.get(path)
    if not node.is_file():
        raise InconsistentFilesystemError(f"Not a regular file: {path}")

    total_bytes = node.size
    checksums = _ChecksumIndex(fs)
    checked = 0
    unverified = 0
    proven = 0
    reason = ""

    with node.open() as fh:
        extents = getattr(fh, "extents", None)

    if not extents:
        # Inline data is stored inside the leaf block itself, so the block checksum
        # covers it.  Emptiness is likewise established by metadata.
        reason = "inline data is covered by its enclosing tree block checksum" if total_bytes else "empty file"
        return DataVerification(path, 0, 0, total_bytes, total_bytes, reason)

    for extent in extents:
        if (extent.disk_offset, extent.disk_length) == (0, 0):
            # A hole is defined to read as zeroes.  The extent item describing it
            # lives in a checksum-verified tree block, so the zero range is proven
            # by metadata rather than by a data checksum.
            proven += extent.length
            continue
        # Checksums cover the bytes as stored on disk: for compressed extents that
        # is the compressed form at disk_offset, not the decompressed output.  A
        # verified compressed extent proves the logical bytes it decompresses to.
        sectors = extent.disk_length // fs.sector_size
        bad_sector = None
        for sector in range(sectors):
            logical = extent.disk_offset + sector * fs.sector_size
            if checksums.verify(fs, logical):
                checked += 1
            else:
                unverified += 1
                bad_sector = logical
                break

        if bad_sector is not None:
            reason = f"checksum mismatch or missing item at {bad_sector:#x}"
            continue

        proven += extent.length

    if total_bytes and proven < total_bytes and not reason:
        reason = "file has sectors without checksum coverage"

    return DataVerification(
        path=path,
        checked_sectors=checked,
        unverified_sectors=unverified,
        proven_bytes=min(proven, total_bytes),
        total_bytes=total_bytes,
        reason=reason,
    )


class _ChecksumIndex:
    """Lazily index the checksum tree by physical sector."""

    def __init__(self, fs: object):
        self._fs = fs
        self._items: list[tuple[int, bytes]] | None = None

    def _load(self) -> list[tuple[int, bytes]]:
        if self._items is None:
            try:
                tree = self._fs._open_tree(c_btrfs.BTRFS_CSUM_TREE_OBJECTID)
                self._items = [(item.key.offset, bytes(data)) for item, data in tree.cursor().walk()]
            except (KeyError, ValueError):
                self._items = []
        return self._items

    def verify(self, fs: object, logical: int) -> bool:
        for offset, data in self._load():
            span = (len(data) // 4) * fs.sector_size
            if offset <= logical < offset + span:
                index = (logical - offset) // fs.sector_size
                stored = data[index * 4 : index * 4 + 4]
                return any(
                    stored == (crc32c.update(0, raw) & 0xFFFFFFFF).to_bytes(4, "little")
                    for raw in _read_mirrors(fs, logical)
                )
        return False


def _read_mirrors(fs: object, logical: int) -> Iterator[bytes]:
    """Yield the sector bytes from every supplied mirror of a logical address.

    Reading only one mirror would understate recoverability: a corrupt first
    copy does not make the range unproven if another supplied mirror matches the
    checksum.
    """
    for chunk in fs._logical_fh.chunks:
        if chunk.offset <= logical < chunk.offset + chunk.length:
            stripe_num, stripe_offset = divmod(logical - chunk.offset, chunk.stripe_length)
            for stripe in chunk.stripes:
                if stripe.fh is None:
                    continue
                stripe.fh.seek(stripe.offset + stripe_offset + stripe_num * chunk.stripe_length)
                yield stripe.fh.read(fs.sector_size)
            return
