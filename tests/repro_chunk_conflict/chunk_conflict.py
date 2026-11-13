"""Conflict classification for Btrfs chunk mappings.

Two independent sources describe how logical byte ranges map onto physical
devices:

1. the ``sys_chunk_array`` embedded in the super block, and
2. the chunk tree (``BTRFS_CHUNK_ITEM_KEY`` items under
   ``BTRFS_FIRST_CHUNK_TREE_OBJECTID``).

A healthy filesystem stores byte-for-byte identical chunk items in both
places. This module never picks a "newer" mapping itself. It only decides
whether a second description is a harmless duplicate (safe to de-duplicate)
or an evidence-changing conflict that must stop automated file extraction.

The comparison works on the normalised :class:`~dissect.btrfs.stream.Chunk`
named tuples produced by the real reader, so it evaluates exactly the fields
that actually drive a read: logical span, type, stripe geometry and the
(devid, physical offset, device UUID) of every stripe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dissect.btrfs.c_btrfs import BTRFS_BLOCK_GROUP_PROFILE_MASK
from dissect.btrfs.stream import Chunk

# Logical ranges smaller than one full chunk are reserved/metadata regions in
# practice; callers may override this when classifying arbitrary ranges.
DEFAULT_GRANULARITY = 0x10000

DUPLICATE = "duplicate"
OVERLAP_CONFLICT = "overlap_conflict"
HARD_CONFLICT = "hard_conflict"

BLOCK = "block_extraction"
DEDUP = "deduplicate"
INFORM = "inform_only"


@dataclass(frozen=True)
class ChunkFingerprint:
    """The evidence-relevant identity of a chunk mapping."""

    offset: int
    length: int
    stripe_length: int
    profile: int
    chunk_type: int
    num_stripes: int
    sub_stripes: int
    data_stripes: int
    stripes: tuple[tuple[int, int, str], ...]

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkFingerprint":
        return cls(
            offset=chunk.offset,
            length=chunk.length,
            stripe_length=chunk.stripe_length,
            profile=int(chunk.type) & int(BTRFS_BLOCK_GROUP_PROFILE_MASK),
            chunk_type=int(chunk.type),
            num_stripes=chunk.num_stripes,
            sub_stripes=chunk.sub_stripes,
            data_stripes=chunk.data_stripes,
            stripes=tuple(
                (stripe.devid, stripe.offset, str(stripe.dev_uuid)) for stripe in chunk.stripes
            ),
        )


@dataclass(frozen=True)
class Verdict:
    """Result of comparing a candidate chunk against the already-loaded set."""

    kind: str
    action: str
    reason: str
    incumbent: Chunk | None = None
    candidate: Chunk | None = None


def _same_mapping(existing: Chunk, candidate: Chunk) -> bool:
    """Return whether two chunks are byte-equivalent descriptions."""
    return ChunkFingerprint.from_chunk(existing) == ChunkFingerprint.from_chunk(candidate)


def _logically_overlaps(existing: Chunk, candidate: Chunk) -> bool:
    start = max(existing.offset, candidate.offset)
    end = min(existing.offset + existing.length, candidate.offset + candidate.length)
    return start < end


def _covering_chunk(chunks: list[Chunk], offset: int) -> Chunk | None:
    """Mimic the predecessor lookup the real ChunkStream.add() performs."""
    covering = [
        chunk
        for chunk in chunks
        if chunk.offset <= offset < chunk.offset + chunk.length
    ]
    if not covering:
        return None
    # ChunkStream keeps the list sorted by offset; the highest-starting
    # covering chunk is the predecessor bisect_right() would select.
    return max(covering, key=lambda chunk: chunk.offset)


def classify_candidate(loaded: list[Chunk], candidate: Chunk) -> Verdict:
    """Classify ``candidate`` against the chunks already loaded.

    Load order matters and is preserved by the caller: the super block array is
    always added before the chunk tree, mirroring ``Btrfs._initialize_chunks``.
    """
    incumbent = _covering_chunk(loaded, candidate.offset)

    if incumbent is None:
        # Not a same-start duplicate. It may still partially overlap a chunk
        # that starts earlier, which is itself a hard conflict.
        for chunk in loaded:
            if _logically_overlaps(chunk, candidate):
                return Verdict(
                    OVERLAP_CONFLICT,
                    BLOCK,
                    (
                        f"candidate at {candidate.offset:#x} starts in a gap but overlaps chunk "
                        f"{chunk.offset:#x}+{chunk.length:#x}; no single mapping owns the shared range"
                    ),
                    chunk,
                    candidate,
                )
        return Verdict(
            "new_range",
            "accept",
            f"candidate at {candidate.offset:#x} covers a previously unmapped range",
            None,
            candidate,
        )

    if incumbent.offset == candidate.offset and _same_mapping(incumbent, candidate):
        return Verdict(
            DUPLICATE,
            DEDUP,
            (
                f"identical chunk item at logical offset {candidate.offset:#x} from a second source; "
                "same span, type and every stripe (devid, physical offset, device UUID)"
            ),
            incumbent,
            candidate,
        )

    if incumbent.offset == candidate.offset:
        return Verdict(
            HARD_CONFLICT,
            BLOCK,
            (
                f"same logical start {candidate.offset:#x} but the physical mapping differs "
                "(stripe targets, device UUID, geometry or type); the logical bytes would come "
                "from a different place depending on which source wins"
            ),
            incumbent,
            candidate,
        )

    return Verdict(
        OVERLAP_CONFLICT,
        BLOCK,
        (
            f"candidate start {candidate.offset:#x} falls inside existing chunk "
            f"{incumbent.offset:#x}+{incumbent.length:#x}; the shared sub-range has two owners"
        ),
        incumbent,
        candidate,
    )


def explain_reader_choice(loaded: list[Chunk], candidate: Chunk) -> dict[str, Any]:
    """Describe what the *current* unmodified reader does with ``candidate``.

    This is diagnostic only. It reproduces the first-writer-wins rule in
    ``ChunkStream.add`` (the covering-offset early return) without changing it.
    """
    incumbent = _covering_chunk(loaded, candidate.offset)
    if incumbent is not None:
        return {
            "reader_keeps": "incumbent",
            "reader_drops": "candidate",
            "winning_source_hint": "whichever source was added first (super block sys_chunk_array)",
            "incumbent_offset": incumbent.offset,
            "candidate_offset": candidate.offset,
            "note": "ChunkStream.add() returns early whenever the new offset is covered; contents are never compared",
        }
    return {
        "reader_keeps": "candidate",
        "reader_drops": None,
        "winning_source_hint": "no covering chunk existed, so the candidate is inserted",
        "incumbent_offset": None,
        "candidate_offset": candidate.offset,
        "note": "insertion is by sorted logical offset; overlap with a later-starting chunk is not checked",
    }
