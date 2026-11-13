"""Read-only reverse lookups from a Btrfs logical extent to file references.

The functions in this module answer a forensic question: given a logical disk range, which files
in a set of explicitly selected subvolume roots still reference (part of) that range?

Only physical extent identity is used. Two files with equal content but independently allocated
extents are deliberately *not* reported as sharing. Every result keeps its provenance: subvolume,
root item, inode, the extent data key and the leaf block that contains it.

References are limited to the selected roots. Subvolume roots that were not selected are reported
as outside the scan scope, and roots that cannot be traversed are reported as coverage gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from dissect.btrfs.c_btrfs import c_btrfs

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from dissect.btrfs.btrfs import Btrfs, Subvolume
    from dissect.btrfs.tree import BTree


_LAST_FREE_OBJECTID = 0xFFFFFFFFFFFFFF00

RefPrecision = Literal["byte", "extent"]
"""How precisely a file range could be mapped back to the queried disk range.

``byte`` means the file bytes that depend on the queried disk bytes are known exactly.
``extent`` means only the decompressed extent slice is known, because Btrfs compression framing
does not allow mapping individual compressed bytes back to individual uncompressed bytes.
"""


@dataclass(frozen=True)
class SubvolumeScope:
    """Identity of a subvolume root that was selected for, or excluded from, the scan.

    Args:
        objectid: The Btrfs subvolume object ID (tree root object ID).
        path: The path of the subvolume root, if it could be resolved.
        uuid: The subvolume UUID from the root item, if available.
        root_bytenr: The logical address of the subvolume root node.
        generation: The transaction ID of the root item.
    """

    objectid: int
    path: str | None
    uuid: UUID | None
    root_bytenr: int | None
    generation: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "objectid": self.objectid,
            "path": self.path,
            "uuid": str(self.uuid) if self.uuid else None,
            "root_bytenr": self.root_bytenr,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class CoverageGap:
    """An explicitly reported hole in the scan coverage.

    A gap means the result must not be presented as a complete list of references for the
    selected roots. ``kind`` is one of ``root_not_found``, ``root_unreadable``,
    ``partial_scan`` or ``enumeration_failed``.

    Args:
        kind: Machine-readable category of the gap.
        subvolume_objectid: The affected subvolume root, if the gap is root specific.
        reason: Human-readable explanation.
        items_scanned: How many B-tree leaf items were inspected before the gap occurred.
        extent_items_scanned: How many of those were file extent items.
    """

    kind: str
    subvolume_objectid: int | None
    reason: str
    items_scanned: int = 0
    extent_items_scanned: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "kind": self.kind,
            "subvolume_objectid": self.subvolume_objectid,
            "reason": self.reason,
            "items_scanned": self.items_scanned,
            "extent_items_scanned": self.extent_items_scanned,
        }


@dataclass(frozen=True)
class ScanStats:
    """Counters describing how much of the selected roots was actually inspected.

    Args:
        roots_requested: Number of distinct subvolume roots requested by the caller.
        roots_scanned: Number of requested roots whose file extent items were fully walked.
        extent_items_scanned: Number of ``BTRFS_EXTENT_DATA_KEY`` items inspected.
        inline_extents_skipped: Inline extents, which have no disk extent to match.
        sparse_extents_skipped: Extent records with ``disk_bytenr == 0``.
        prealloc_extents_matched: PREALLOC records that overlap the query. PREALLOC extents are
            reported, but forward reads may expose zeros instead of the on-disk bytes.
    """

    roots_requested: int
    roots_scanned: int
    extent_items_scanned: int
    inline_extents_skipped: int
    sparse_extents_skipped: int
    prealloc_extents_matched: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""
        return {
            "roots_requested": self.roots_requested,
            "roots_scanned": self.roots_scanned,
            "extent_items_scanned": self.extent_items_scanned,
            "inline_extents_skipped": self.inline_extents_skipped,
            "sparse_extents_skipped": self.sparse_extents_skipped,
            "prealloc_extents_matched": self.prealloc_extents_matched,
        }


@dataclass(frozen=True)
class ExtentAllocation:
    """Filesystem-level record of a queried physical extent in the extent tree.

    This is independent evidence from the file extent items: it says how many references the
    filesystem itself accounts for, not how many file records the scan could reach.

    Args:
        bytenr: First logical disk byte of the allocation.
        length: Allocation length from the extent item key offset.
        refs: Reference count recorded by the extent tree.
        generation: Transaction that created the extent.
        flags: ``BTRFS_EXTENT_FLAG_*`` value.
        item_leaf: Logical address of the extent tree leaf containing the item.
    """

    bytenr: int
    length: int
    refs: int
    generation: int
    flags: int
    item_leaf: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "bytenr": self.bytenr,
            "length": self.length,
            "refs": self.refs,
            "generation": self.generation,
            "flags": self.flags,
            "item_leaf": self.item_leaf,
        }


@dataclass(frozen=True)
class ExtentReference:
    """One file extent record that physically references the queried disk range.

    The physical extent is ``[disk_start, disk_end)`` (``disk_bytenr`` and ``disk_num_bytes`` from
    the file extent item). ``overlap_start``/``overlap_end`` are the intersection of that extent
    with the query.

    ``exposed_start``/``exposed_end`` are the file-visible bytes of this record, in file logical
    coordinates. For uncompressed extents, ``file_start``/``file_end`` narrow that down to the file
    bytes that actually depend on ``overlap``. For compressed extents, ``file_start``/``file_end``
    are the whole exposed slice and ``precision`` is ``extent``: Btrfs compresses a whole extent
    and an individual compressed byte range cannot be attributed to individual file bytes without
    decompressing, and even then the compressed framing is coarser than a byte mapping.

    ``file_start`` and ``file_end`` are ``None`` when the queried physical bytes lie in the extent
    allocation but outside the slice this particular record exposes.

    Args:
        subvolume_objectid: Subvolume root object ID that owns the inode.
        subvolume_path: Path of that subvolume root.
        subvolume_uuid: UUID of that subvolume root.
        root_bytenr: Logical address of the subvolume root node.
        root_generation: Transaction ID of the subvolume root item.
        inode: Inode number.
        inode_generation: Generation field of the inode item.
        inode_transid: Transaction ID that last touched the inode item.
        inode_size: File size recorded by the inode item.
        paths: All paths (hard links) of the inode, if resolvable.
        extent_key_offset: Offset of the ``BTRFS_EXTENT_DATA_KEY`` item, i.e. the file offset
            at which this extent record starts.
        extent_key_leaf: Logical address of the B-tree leaf containing the extent item.
        extent_generation: Generation field of the file extent item.
        disk_start: First logical disk byte of the referenced extent.
        disk_end: End (exclusive) of the referenced extent.
        overlap_start: First logical disk byte shared with the query.
        overlap_end: End (exclusive) of the disk overlap.
        exposed_start: First file byte exposed by this extent record.
        exposed_end: End (exclusive) of the file slice exposed by this record.
        file_start: First file byte backed by the disk overlap, or ``None``.
        file_end: End (exclusive) of the file range backed by the disk overlap, or ``None``.
        precision: ``byte`` or ``extent``, see :data:`RefPrecision`.
        compression: Btrfs compression type.
        encryption: Btrfs extent encryption type.
        extent_type: ``BTRFS_FILE_EXTENT_REG`` or ``BTRFS_FILE_EXTENT_PREALLOC``.
        disk_num_bytes: On-disk bytes of the extent (compressed size for compressed extents).
        extent_offset: Offset of this record's data inside the (uncompressed) extent.
        extent_num_bytes: Number of file bytes exposed by this record.
        ram_bytes: Uncompressed size of the whole extent.
        notes: Per-reference caveats, e.g. partial exposure of a compressed extent.
    """

    subvolume_objectid: int
    subvolume_path: str | None
    subvolume_uuid: UUID | None
    root_bytenr: int | None
    root_generation: int | None
    inode: int
    inode_generation: int | None
    inode_transid: int | None
    inode_size: int | None
    paths: tuple[str, ...]
    extent_key_offset: int
    extent_key_leaf: int | None
    extent_generation: int
    disk_start: int
    disk_end: int
    overlap_start: int
    overlap_end: int
    exposed_start: int
    exposed_end: int
    file_start: int | None
    file_end: int | None
    precision: RefPrecision
    compression: int
    encryption: int
    extent_type: int
    disk_num_bytes: int
    extent_offset: int
    extent_num_bytes: int
    ram_bytes: int
    notes: tuple[str, ...] = ()

    @property
    def compressed(self) -> bool:
        """Return whether the referenced extent is compressed."""
        return self.compression != c_btrfs.BTRFS_COMPRESS_NONE

    @property
    def exposes_query(self) -> bool:
        """Return whether this record's file slice includes bytes backed by the query."""
        return self.file_start is not None and self.file_end is not None

    @property
    def preallocated(self) -> bool:
        """Return whether this record is a PREALLOC extent.

        A preallocated extent physically reserves the disk range, but its content is not
        guaranteed to be exposed by a forward read.
        """
        return self.extent_type == c_btrfs.BTRFS_FILE_EXTENT_PREALLOC

    @property
    def partial_exposure(self) -> bool:
        """Return whether this record exposes only a part of the physical extent."""
        if self.compressed:
            return self.extent_offset != 0 or self.extent_num_bytes != self.ram_bytes
        return self.extent_offset != 0 or self.extent_num_bytes != self.disk_num_bytes

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "subvolume_objectid": self.subvolume_objectid,
            "subvolume_path": self.subvolume_path,
            "subvolume_uuid": str(self.subvolume_uuid) if self.subvolume_uuid else None,
            "root_bytenr": self.root_bytenr,
            "root_generation": self.root_generation,
            "inode": self.inode,
            "inode_generation": self.inode_generation,
            "inode_transid": self.inode_transid,
            "inode_size": self.inode_size,
            "paths": list(self.paths),
            "extent_key_offset": self.extent_key_offset,
            "extent_key_leaf": self.extent_key_leaf,
            "extent_generation": self.extent_generation,
            "disk_start": self.disk_start,
            "disk_end": self.disk_end,
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end,
            "exposed_start": self.exposed_start,
            "exposed_end": self.exposed_end,
            "file_start": self.file_start,
            "file_end": self.file_end,
            "precision": self.precision,
            "compression": self.compression,
            "encryption": self.encryption,
            "extent_type": self.extent_type,
            "disk_num_bytes": self.disk_num_bytes,
            "extent_offset": self.extent_offset,
            "extent_num_bytes": self.extent_num_bytes,
            "ram_bytes": self.ram_bytes,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExtentReferenceReport:
    """Result of a reverse extent lookup.

    ``complete`` only refers to the selected roots: it is false if any selected root could not be
    fully walked. ``exhaustive`` additionally requires that no other subvolume roots exist outside
    the selection. Neither flag ever means "all references in the image", because roots that were
    not selected are not inspected.

    Args:
        bytenr: Query start in logical disk coordinates.
        length: Query length.
        selected: Roots that were explicitly selected for the scan.
        references: File extent records that physically overlap the query.
        coverage_gaps: Roots or scan phases that could not be traversed.
        unselected: Subvolume roots that exist but were not selected.
        stats: Scan counters.
        allocations: Extent tree records for allocations that overlap the query.
        warnings: Non-fatal inconsistencies found while parsing extent items.
    """

    bytenr: int
    length: int
    selected: tuple[SubvolumeScope, ...]
    references: tuple[ExtentReference, ...]
    coverage_gaps: tuple[CoverageGap, ...]
    unselected: tuple[SubvolumeScope, ...]
    stats: ScanStats
    allocations: tuple[ExtentAllocation, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def end(self) -> int:
        """Return the exclusive end of the queried disk range."""
        return self.bytenr + self.length

    def __iter__(self) -> Iterator[ExtentReference]:
        """Iterate the references, so a report can be used like a result list."""
        return iter(self.references)

    def __len__(self) -> int:
        """Return the number of references."""
        return len(self.references)

    @property
    def complete(self) -> bool:
        """Return whether all selected roots were fully traversed."""
        return not self.coverage_gaps

    @property
    def covering(self) -> tuple[ExtentReference, ...]:
        """Return only references whose exposed file slice includes part of the query."""
        return tuple(reference for reference in self.references if reference.exposes_query)

    @property
    def exhaustive(self) -> bool:
        """Return whether the scan covered every discovered subvolume root without gaps."""
        return self.complete and not self.unselected

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "bytenr": self.bytenr,
            "length": self.length,
            "end": self.end,
            "selected": [scope.to_dict() for scope in self.selected],
            "unselected": [scope.to_dict() for scope in self.unselected],
            "references": [reference.to_dict() for reference in self.references],
            "coverage_gaps": [gap.to_dict() for gap in self.coverage_gaps],
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "stats": self.stats.to_dict(),
            "warnings": list(self.warnings),
            "complete": self.complete,
            "exhaustive": self.exhaustive,
        }


@dataclass
class _InodeFacts:
    generation: int | None
    transid: int | None
    size: int | None
    paths: tuple[str, ...]
    notes: tuple[str, ...]


def find_extent_references(
    btrfs: Btrfs,
    bytenr: int,
    length: int,
    subvolumes: Iterable[int | Subvolume],
) -> ExtentReferenceReport:
    """Find files in explicitly selected subvolume roots that physically reference a disk range.

    The lookup is read-only: it walks file extent items and inode metadata, and never writes to the
    image or modifies filesystem state. Physical sharing is decided solely by the extent's
    ``disk_bytenr``/``disk_num_bytes`` allocation range overlapping the query. Content hashes are
    not used, so equal content in independently allocated extents is not reported as shared.

    Args:
        btrfs: The filesystem to scan.
        bytenr: Query start in Btrfs logical disk coordinates.
        length: Query length in bytes, must be positive.
        subvolumes: Explicit subvolume roots to scan. Each item is a :class:`~dissect.btrfs.btrfs.Subvolume`
            or a subvolume object ID. An empty selection is rejected: there is no implicit
            "scan everything" mode.

    Returns:
        An :class:`ExtentReferenceReport`. Check ``complete`` and ``exhaustive`` before treating
        the reference list as a complete answer.

    Raises:
        ValueError: If the range or selection is invalid, or a subvolume belongs to another filesystem.
    """
    if not isinstance(bytenr, int) or bytenr < 0:
        raise ValueError("bytenr must be a non-negative integer")
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")

    if isinstance(subvolumes, int) or getattr(subvolumes, "objectid", None) is not None:
        subvolumes = [subvolumes]

    selected_ids, provided = _normalize_selection(btrfs, subvolumes)
    if not selected_ids:
        raise ValueError("At least one subvolume root must be explicitly selected")

    query_start = bytenr
    query_end = bytenr + length

    discovered, discovery_gaps = _discover_subvolumes(btrfs)
    warnings: list[str] = []
    gaps = list(discovery_gaps)
    allocations = _find_allocations(btrfs, query_start, query_end, warnings)

    selected_scopes = []
    references = []
    stats_extent_items = 0
    stats_inline = 0
    stats_sparse = 0
    stats_prealloc = 0
    roots_scanned = 0

    for objectid in selected_ids:
        subvolume = provided.get(objectid)
        try:
            if subvolume is None:
                subvolume = btrfs.open_subvolume(objectid)
            scope, tree = _open_scope(subvolume)
        except Exception as exc:
            scope = _scope_from_root_item(objectid, discovered.get(objectid), btrfs)
            selected_scopes.append(scope)
            gaps.append(
                CoverageGap(
                    "root_not_found" if isinstance(exc, KeyError) else "root_unreadable",
                    objectid,
                    f"cannot open subvolume root {objectid}: {type(exc).__name__}: {exc}",
                )
            )
            continue

        selected_scopes.append(scope)
        inode_cache: dict[int, _InodeFacts] = {}
        root_items = 0
        root_extent_items = 0
        scan_error: Exception | None = None

        try:
            cursor = tree.cursor()
            # Visit every leaf: a reverse scan cannot seek by inode, and key order only groups
            # extent items per inode. The scan stays read-only and linear in the metadata size.
            for item, data in cursor.walk():
                root_items += 1
                if item.key.type != c_btrfs.BTRFS_EXTENT_DATA_KEY:
                    continue

                root_extent_items += 1
                leaf = cursor.node_bytenr
                extent_type = _extent_type(data)
                if extent_type is None:
                    warnings.append(
                        f"subvolume {objectid} inode {item.key.objectid}: short extent item at file offset "
                        f"{item.key.offset:#x}"
                    )
                    continue
                if extent_type == c_btrfs.BTRFS_FILE_EXTENT_INLINE:
                    stats_inline += 1
                    continue
                if extent_type not in (c_btrfs.BTRFS_FILE_EXTENT_REG, c_btrfs.BTRFS_FILE_EXTENT_PREALLOC):
                    warnings.append(
                        f"subvolume {objectid} inode {item.key.objectid}: unknown file extent type {extent_type}"
                    )
                    continue

                if len(data) < len(c_btrfs.btrfs_file_extent_item_reg):
                    warnings.append(
                        f"subvolume {objectid} inode {item.key.objectid}: truncated extent item at file offset "
                        f"{item.key.offset:#x}"
                    )
                    continue

                extent = c_btrfs.btrfs_file_extent_item_reg(data)
                if extent.disk_bytenr == 0:
                    stats_sparse += 1
                    continue
                if extent.disk_num_bytes == 0:
                    warnings.append(
                        f"subvolume {objectid} inode {item.key.objectid}: extent at file offset "
                        f"{item.key.offset:#x} has no disk length"
                    )
                    continue

                reference = _build_reference(
                    scope,
                    subvolume,
                    tree,
                    item,
                    extent,
                    leaf,
                    query_start,
                    query_end,
                    inode_cache,
                    warnings,
                )
                if reference is not None:
                    if extent.type == c_btrfs.BTRFS_FILE_EXTENT_PREALLOC:
                        stats_prealloc += 1
                    references.append(reference)
        except Exception as exc:
            scan_error = exc
        finally:
            stats_extent_items += root_extent_items

        if scan_error is not None:
            gaps.append(
                CoverageGap(
                    "partial_scan",
                    objectid,
                    f"scanning subvolume root {objectid} stopped after {root_items} items "
                    f"({root_extent_items} file extents): {type(scan_error).__name__}: {scan_error}",
                    root_items,
                    root_extent_items,
                )
            )
        else:
            roots_scanned += 1

    unselected = tuple(
        _scope_from_root_item(objectid, root_item, btrfs)
        for objectid, root_item in discovered.items()
        if objectid not in selected_ids
    )

    references.sort(
        key=lambda ref: (
            ref.subvolume_objectid,
            ref.inode,
            ref.extent_key_offset,
            ref.overlap_start,
        )
    )

    return ExtentReferenceReport(
        bytenr=bytenr,
        length=length,
        selected=tuple(selected_scopes),
        references=tuple(references),
        coverage_gaps=tuple(gaps),
        unselected=unselected,
        allocations=allocations,
        stats=ScanStats(
            roots_requested=len(selected_ids),
            roots_scanned=roots_scanned,
            extent_items_scanned=stats_extent_items,
            inline_extents_skipped=stats_inline,
            sparse_extents_skipped=stats_sparse,
            prealloc_extents_matched=stats_prealloc,
        ),
        warnings=tuple(warnings),
    )


def _find_allocations(
    btrfs: Btrfs, query_start: int, query_end: int, warnings: list[str]
) -> tuple[ExtentAllocation, ...]:
    """Find extent tree items for allocations overlapping the query.

    Extent items are keyed on ``(bytenr, BTRFS_EXTENT_ITEM_KEY, length)`` in ascending order.
    The forward walk stops as soon as an allocation starts at or after the query end. The range can
    also start in the middle of an allocation, so preceding data extent items are probed separately;
    metadata and block-group items may sit between them. Data allocations do not overlap, so the
    nearest one that overlaps the query is sufficient.
    """
    allocations: list[ExtentAllocation] = []
    seen: set[tuple[int, int]] = set()

    try:
        cursor = btrfs.extent_tree.cursor()
        if cursor.search(query_start, c_btrfs.BTRFS_EXTENT_ITEM_KEY, 0):
            while True:
                item, data = cursor.get()
                if item.key.objectid >= query_end:
                    break
                if item.key.type == c_btrfs.BTRFS_EXTENT_ITEM_KEY:
                    allocation = _parse_allocation(item, data, cursor.node_bytenr, query_start, query_end, warnings)
                    if allocation is not None:
                        _add_allocation(allocations, seen, allocation)

                try:
                    cursor.next()
                except ValueError:
                    break

        preceding = _preceding_allocation(btrfs, query_start, warnings)
        if preceding is not None:
            _add_allocation(allocations, seen, preceding)
    except Exception as exc:
        warnings.append(f"cannot read extent tree allocations: {type(exc).__name__}: {exc}")

    allocations.sort(key=lambda allocation: allocation.bytenr)
    return tuple(allocations)


def _preceding_allocation(btrfs: Btrfs, query_start: int, warnings: list[str]) -> ExtentAllocation | None:
    cursor = btrfs.extent_tree.cursor()
    if not cursor.search(query_start, c_btrfs.BTRFS_EXTENT_ITEM_KEY, 0):
        return None

    for _ in range(64):
        item = cursor.item()
        if item.key.type == c_btrfs.BTRFS_EXTENT_ITEM_KEY and item.key.objectid < query_start:
            allocation = _parse_allocation(
                item, cursor.data(), cursor.node_bytenr, query_start, query_start + 1, warnings
            )
            return allocation
        try:
            cursor.prev()
        except ValueError:
            return None

    return None


def _add_allocation(
    allocations: list[ExtentAllocation], seen: set[tuple[int, int]], allocation: ExtentAllocation
) -> None:
    key = (allocation.bytenr, allocation.length)
    if key not in seen:
        seen.add(key)
        allocations.append(allocation)


def _parse_allocation(
    item: c_btrfs.btrfs_item,
    data: memoryview,
    leaf: int | None,
    query_start: int,
    query_end: int,
    warnings: list[str],
) -> ExtentAllocation | None:
    if len(data) < len(c_btrfs.btrfs_extent_item):
        warnings.append(f"short extent item at {item.key.objectid:#x}")
        return None

    extent_item = c_btrfs.btrfs_extent_item(data)
    if not extent_item.flags & c_btrfs.BTRFS_EXTENT_FLAG_DATA:
        return None

    bytenr = item.key.objectid
    length = item.key.offset
    if bytenr + length <= query_start or bytenr >= query_end:
        return None

    return ExtentAllocation(
        bytenr=bytenr,
        length=length,
        refs=extent_item.refs,
        generation=extent_item.generation,
        flags=extent_item.flags,
        item_leaf=leaf,
    )


def _normalize_selection(btrfs: Btrfs, subvolumes: Iterable[int | Subvolume]) -> tuple[list[int], dict[int, Subvolume]]:
    provided: dict[int, Subvolume] = {}
    selected: list[int] = []

    for item in subvolumes:
        if isinstance(item, int):
            objectid = item
        else:
            if getattr(item, "btrfs", None) is not btrfs:
                raise ValueError("Selected subvolume belongs to a different filesystem")
            objectid = item.objectid
            provided[objectid] = item

        if objectid != c_btrfs.BTRFS_FS_TREE_OBJECTID and not (
            c_btrfs.BTRFS_FIRST_FREE_OBJECTID <= objectid < _LAST_FREE_OBJECTID
        ):
            raise ValueError(f"Object ID {objectid} is not a subvolume root")
        if objectid not in selected:
            selected.append(objectid)

    return selected, provided


def _discover_subvolumes(btrfs: Btrfs) -> tuple[dict[int, c_btrfs.btrfs_root_item], list[CoverageGap]]:
    """Enumerate subvolume roots, including roots that have no ROOT_REF directory entry."""
    discovered: dict[int, c_btrfs.btrfs_root_item] = {}
    gaps: list[CoverageGap] = []

    try:
        cursor = btrfs._root_tree.cursor()
        for item, data in cursor.walk(type=c_btrfs.BTRFS_ROOT_ITEM_KEY):
            objectid = item.key.objectid
            if objectid == c_btrfs.BTRFS_FS_TREE_OBJECTID or (
                c_btrfs.BTRFS_FIRST_FREE_OBJECTID <= objectid < _LAST_FREE_OBJECTID
            ):
                discovered[objectid] = c_btrfs.btrfs_root_item(data)
    except Exception as exc:
        gaps.append(
            CoverageGap(
                "enumeration_failed",
                None,
                f"cannot enumerate subvolume roots from the root tree: {type(exc).__name__}: {exc}",
            )
        )

    return discovered, gaps


def _open_scope(subvolume: Subvolume) -> tuple[SubvolumeScope, BTree]:
    tree = subvolume.tree
    root_item = tree.root_item

    scope = SubvolumeScope(
        objectid=subvolume.objectid,
        path=_subvolume_path(subvolume),
        uuid=_root_item_uuid(root_item),
        root_bytenr=tree.root_offset,
        generation=root_item.generation,
    )
    return scope, tree


def _scope_from_root_item(objectid: int, root_item: c_btrfs.btrfs_root_item | None, btrfs: Btrfs) -> SubvolumeScope:
    path = None
    try:
        path = _subvolume_path(btrfs.open_subvolume(objectid))
    except Exception:
        path = None

    if root_item is None:
        return SubvolumeScope(objectid, path, None, None, None)

    return SubvolumeScope(
        objectid=objectid,
        path=path,
        uuid=_root_item_uuid(root_item),
        root_bytenr=root_item.bytenr,
        generation=root_item.generation,
    )


def _subvolume_path(subvolume: Subvolume) -> str | None:
    try:
        return subvolume.path
    except Exception:
        return None


def _root_item_uuid(root_item: c_btrfs.btrfs_root_item) -> UUID | None:
    if not any(root_item.uuid):
        return None
    return UUID(bytes=root_item.uuid)


def _extent_type(data: memoryview) -> int | None:
    if len(data) < len(c_btrfs.btrfs_file_extent_item_inline):
        return None
    return c_btrfs.btrfs_file_extent_item_inline(data).type


def _build_reference(
    scope: SubvolumeScope,
    subvolume: Subvolume,
    tree: BTree,
    item: c_btrfs.btrfs_item,
    extent: c_btrfs.btrfs_file_extent_item_reg,
    leaf: int | None,
    query_start: int,
    query_end: int,
    inode_cache: dict[int, _InodeFacts],
    warnings: list[str],
) -> ExtentReference | None:
    disk_start = extent.disk_bytenr
    disk_end = disk_start + extent.disk_num_bytes

    overlap_start = max(query_start, disk_start)
    overlap_end = min(query_end, disk_end)
    if overlap_end <= overlap_start:
        return None

    inode = item.key.objectid
    facts = inode_cache.get(inode)
    if facts is None:
        facts = _read_inode_facts(subvolume, inode, warnings)
        inode_cache[inode] = facts

    mapping = map_overlap(
        query_start=query_start,
        query_end=query_end,
        disk_start=disk_start,
        disk_num_bytes=extent.disk_num_bytes,
        extent_offset=extent.offset,
        num_bytes=extent.num_bytes,
        ram_bytes=extent.ram_bytes,
        key_offset=item.key.offset,
        compression=extent.compression,
    )

    notes = list(mapping.notes)
    file_start = mapping.file_start
    file_end = mapping.file_end


    if facts.size is not None and file_start is not None and file_end is not None:
        if file_start >= facts.size:
            notes.append("extent record lies beyond the inode size")
            file_start = None
            file_end = None
        elif file_end > facts.size:
            notes.append("file range clamped to the inode size")
            file_end = facts.size

    if not mapping.within_allocation:
        warnings.append(
            f"subvolume {scope.objectid} inode {inode}: extent at file offset "
            f"{item.key.offset:#x} exceeds its recorded allocation"
        )
    if extent.type == c_btrfs.BTRFS_FILE_EXTENT_PREALLOC:
        notes.append("preallocated extent; forward reads may not expose the on-disk bytes")
    return ExtentReference(
        subvolume_objectid=scope.objectid,
        subvolume_path=scope.path,
        subvolume_uuid=scope.uuid,
        root_bytenr=scope.root_bytenr,
        root_generation=scope.generation,
        inode=inode,
        inode_generation=facts.generation,
        inode_transid=facts.transid,
        inode_size=facts.size,
        paths=facts.paths,
        extent_key_offset=item.key.offset,
        extent_key_leaf=leaf,
        extent_generation=extent.generation,
        disk_start=disk_start,
        disk_end=disk_end,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        exposed_start=mapping.exposed_start,
        exposed_end=mapping.exposed_end,
        file_start=file_start,
        file_end=file_end,
        precision=mapping.precision,
        compression=extent.compression,
        encryption=extent.encryption,
        extent_type=extent.type,
        disk_num_bytes=extent.disk_num_bytes,
        extent_offset=extent.offset,
        extent_num_bytes=extent.num_bytes,
        ram_bytes=extent.ram_bytes,
        notes=tuple(notes) + facts.notes,
    )


@dataclass(frozen=True)
class OverlapMapping:
    """File-side view of a logical disk range that overlaps one extent record.

    Args:
        exposed_start: First file byte exposed by the extent record.
        exposed_end: End (exclusive) of the file slice exposed by the record.
        file_start: First file byte backed by the disk overlap, or ``None`` if the overlap falls
            outside the slice this record exposes.
        file_end: End (exclusive) of the file range backed by the disk overlap, or ``None``.
        precision: ``byte`` when the file range is exact, ``extent`` when only the decoded extent
            slice is known (compressed extents).
        within_allocation: Whether the record's file slice stays inside its disk allocation.
        notes: Caveats explaining the precision and exposure.
    """

    exposed_start: int
    exposed_end: int
    file_start: int | None
    file_end: int | None
    precision: RefPrecision
    within_allocation: bool
    notes: tuple[str, ...]


def map_overlap(
    query_start: int,
    query_end: int,
    *,
    disk_start: int,
    disk_num_bytes: int,
    extent_offset: int,
    num_bytes: int,
    ram_bytes: int,
    key_offset: int,
    compression: int,
) -> OverlapMapping:
    """Map a logical disk overlap to the file bytes exposed by one extent record.

    This is the pure arithmetic part of the reverse lookup. It is kept separate so the mapping
    semantics can be audited without an image. If the query does not overlap the allocation at all,
    ``file_start`` and ``file_end`` are ``None``.

    Args:
        query_start: Query start in logical disk coordinates.
        query_end: Query end (exclusive).
        disk_start: First logical disk byte of the extent allocation.
        disk_num_bytes: On-disk length of the extent allocation.
        extent_offset: Offset of this record's data inside the extent.
        num_bytes: File bytes exposed by this record.
        ram_bytes: Uncompressed size of the whole extent.
        key_offset: File offset of this extent record.
        compression: Btrfs compression type of the extent.

    Returns:
        An :class:`OverlapMapping` with the exposed file slice and the narrower byte-accurate range.
    """
    disk_end = disk_start + disk_num_bytes
    overlap_start = max(query_start, disk_start)
    overlap_end = min(query_end, disk_end)
    has_overlap = overlap_end > overlap_start

    notes: list[str] = []
    exposed_start = key_offset
    exposed_end = key_offset + num_bytes

    compressed = compression != c_btrfs.BTRFS_COMPRESS_NONE

    if compressed:
        # For compressed extents ``extent_offset`` and ``num_bytes`` are in uncompressed space,
        # while ``disk_num_bytes`` is the compressed allocation size. Compare with ``ram_bytes``.
        within_allocation = extent_offset + num_bytes <= ram_bytes
        if not within_allocation:
            notes.append("compressed extent slice exceeds ram_bytes")
        notes.append("compressed extent; byte-level attribution within the extent is unresolved")
        if extent_offset != 0 or num_bytes != ram_bytes:
            notes.append("record exposes only part of a compressed extent")
        return OverlapMapping(
            exposed_start,
            exposed_end,
            exposed_start if has_overlap else None,
            exposed_end if has_overlap else None,
            "extent",
            within_allocation,
            tuple(notes),
        )

    within_allocation = extent_offset + num_bytes <= disk_num_bytes
    if not within_allocation:
        notes.append("file extent slice exceeds its recorded disk allocation")

    physical_start = disk_start + extent_offset
    physical_end = physical_start + num_bytes

    if extent_offset != 0 or num_bytes != disk_num_bytes:
        notes.append("record exposes only part of the physical extent")

    file_start: int | None = None
    file_end: int | None = None

    visible_start = max(overlap_start, physical_start)
    visible_end = min(overlap_end, physical_end)
    if visible_end > visible_start:
        file_start = key_offset + (visible_start - physical_start)
        file_end = key_offset + (visible_end - physical_start)
    else:
        notes.append("disk overlap lies outside the slice exposed by this record")

    return OverlapMapping(
        exposed_start,
        exposed_end,
        file_start,
        file_end,
        "byte",
        within_allocation,
        tuple(notes),
    )


def _read_inode_facts(subvolume: Subvolume, inode: int, warnings: list[str]) -> _InodeFacts:
    generation = None
    transid = None
    size = None
    paths: tuple[str, ...] = ()
    notes: list[str] = []

    tree = subvolume.tree
    try:
        _, data = tree.find(inode, c_btrfs.BTRFS_INODE_ITEM_KEY)
        inode_item = c_btrfs.btrfs_inode_item(data)
        generation = inode_item.generation
        transid = inode_item.transid
        size = inode_item.size
    except KeyError:
        notes.append("inode item not found")
    except Exception as exc:
        notes.append(f"cannot read inode item: {type(exc).__name__}: {exc}")

    try:
        node = subvolume.inode(inode)
    except Exception:
        node = None

    if node is not None:
        try:
            paths = tuple(node.paths(full=True))
        except Exception as exc:
            notes.append(f"cannot resolve paths: {type(exc).__name__}: {exc}")
            try:
                paths = tuple(node.paths(full=False))
            except Exception:
                paths = ()

    warnings.extend(f"inode {inode}: {note}" for note in notes)

    return _InodeFacts(generation, transid, size, paths, tuple(notes))
