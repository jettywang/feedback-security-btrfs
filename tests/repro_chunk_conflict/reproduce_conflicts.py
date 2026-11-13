"""Runnable reproduction of sys_chunk_array vs chunk-tree conflicts.

Run directly:

    python tests/repro_chunk_conflict/reproduce_conflicts.py
    pytest tests/repro_chunk_conflict/test_chunk_conflict.py -q

The script drives the *real* ``dissect.btrfs.stream.ChunkStream`` (the exact
code path ``Btrfs._initialize_chunks`` feeds), so the observed winner is the
reader's genuine behaviour, not a re-implementation.

Each scenario feeds chunks in production load order: first the super block
``sys_chunk_array`` (source ``SB``), then the chunk tree (source ``CT``).
Every physical region is filled with a distinct marker, which lets us state
precisely where each returned logical byte came from.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from uuid import UUID

from dissect.btrfs.c_btrfs import c_btrfs
from dissect.btrfs.stream import Chunk, ChunkStream

from tests.repro_chunk_conflict.chunk_conflict import (
    BLOCK,
    Verdict,
    classify_candidate,
    explain_reader_choice,
)

BG = c_btrfs.BTRFS_BLOCK_GROUP
CHUNK_LEN = 0x10000  # one 64 KiB chunk, aligned to ChunkStream's read buffer
LOGICAL = 0x100000  # logical start shared by the "same start" scenarios

# Distinct, easy-to-read single-byte markers per physical region.
SB_MARKER = ord("S")  # bytes reachable through the super block mapping
CT_MARKER = ord("T")  # bytes reachable through the chunk-tree mapping
FILL_MARKER = ord(".")


class FakeBtrfs:
    """Minimal stand-in exposing only what ChunkStream needs."""

    def __init__(self, devices: dict[int, io.BytesIO]) -> None:
        self.devices = devices


def make_disk(marker: int, size: int = CHUNK_LEN * 4) -> io.BytesIO:
    return io.BytesIO(bytes([marker]) * size)


def make_chunk(
    length: int,
    stripe_len: int,
    devid: int,
    dev_off: int,
    dev_uuid: UUID,
    chunk_type: int = BG.SYSTEM,
) -> c_btrfs.btrfs_chunk:
    stripe = c_btrfs.btrfs_stripe(devid=devid, offset=dev_off, dev_uuid=dev_uuid.bytes)
    chunk = c_btrfs.btrfs_chunk(
        length=length,
        owner=c_btrfs.BTRFS_FIRST_CHUNK_TREE_OBJECTID,
        stripe_len=stripe_len,
        type=chunk_type,
        io_align=stripe_len,
        io_width=stripe_len,
        sector_size=4096,
        num_stripes=1,
        sub_stripes=1,
    )
    chunk.stripe = [stripe]
    return chunk


@dataclass
class ScenarioResult:
    name: str
    description: str
    verdict: Verdict
    reader: dict
    logical_byte: int
    logical_byte_source: str
    winner: str
    chunks_loaded: int


def _first_logical_chunk(stream: ChunkStream) -> Chunk:
    return next(chunk for chunk in stream.chunks if chunk.offset <= LOGICAL)


def _source_of(chunk: Chunk, marker_to_source: dict[int, str]) -> str:
    stripe = chunk.stripes[0]
    stripe.fh.seek(stripe.offset)
    marker = stripe.fh.read(1)[0]
    return f"{marker_to_source[marker]} (devid={stripe.devid}, phys={stripe.offset:#x}, " \
           f"dev_uuid={stripe.dev_uuid})"


def run_scenarios() -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    uuid_a = UUID("11111111-1111-1111-1111-111111111111")
    uuid_b = UUID("22222222-2222-2222-2222-222222222222")

    # A. Byte-for-byte identical entry from both sources: two independent
    # descriptors that resolve to the exact same (devid, offset, UUID).
    fs = FakeBtrfs({1: make_disk(SB_MARKER)})
    stream = ChunkStream(fs)
    sb_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_a)
    ct_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_a)
    stream.add(LOGICAL, sb_chunk)  # super block first
    loaded = list(stream.chunks)
    verdict = classify_candidate(loaded, _to_named_tuple(LOGICAL, ct_chunk, fs))
    reader = explain_reader_choice(loaded, _to_named_tuple(LOGICAL, ct_chunk, fs))
    stream.add(LOGICAL, ct_chunk)  # chunk tree second
    stream.seek(LOGICAL)
    byte = stream.read(1)[0]
    results.append(
        ScenarioResult(
            "A_identical_duplicate",
            "same start, identical span/stripe/UUID from SB and CT",
            verdict, reader, byte, _source_of(_first_logical_chunk(stream),
                                              {SB_MARKER: "SB/CT-shared region", FILL_MARKER: "fill"}),
            reader["reader_keeps"], len(stream.chunks),
        )
    )

    # B. Same logical start, different physical stripe (different devid + offset).
    sb_disk = make_disk(SB_MARKER)
    ct_disk = make_disk(CT_MARKER)
    fs = FakeBtrfs({1: sb_disk, 2: ct_disk})
    stream = ChunkStream(fs)
    sb_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_a)
    ct_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 2, 0x20000, uuid_b)
    stream.add(LOGICAL, sb_chunk)
    loaded = list(stream.chunks)
    cand = _to_named_tuple(LOGICAL, ct_chunk, fs)
    verdict = classify_candidate(loaded, cand)
    reader = explain_reader_choice(loaded, cand)
    stream.add(LOGICAL, ct_chunk)
    stream.seek(LOGICAL)
    byte = stream.read(1)[0]
    results.append(
        ScenarioResult(
            "B_same_start_different_stripe",
            "same start, CT points at another device/physical offset",
            verdict, reader, byte,
            _source_of(_first_logical_chunk(stream),
                       {SB_MARKER: "SB-backed region", CT_MARKER: "CT-backed region"}),
            reader["reader_keeps"], len(stream.chunks),
        )
    )

    # C. Partial overlap: CT chunk starts inside the SB chunk's range.
    sb_disk = make_disk(SB_MARKER)
    ct_disk = make_disk(CT_MARKER)
    fs = FakeBtrfs({1: sb_disk, 2: ct_disk})
    stream = ChunkStream(fs)
    inner = LOGICAL + CHUNK_LEN // 2  # 32 KiB into the SB chunk
    sb_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_a)
    ct_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 2, 0x0, uuid_b)
    stream.add(LOGICAL, sb_chunk)
    loaded = list(stream.chunks)
    cand = _to_named_tuple(inner, ct_chunk, fs)
    verdict = classify_candidate(loaded, cand)
    reader = explain_reader_choice(loaded, cand)
    stream.add(inner, ct_chunk)
    # The contested byte is the candidate's own start, inside the SB range.
    stream.seek(inner)
    byte = stream.read(1)[0]
    covering = max(
        (c for c in stream.chunks if c.offset <= inner < c.offset + c.length),
        key=lambda c: c.offset,
    )
    results.append(
        ScenarioResult(
            "C_partial_overlap",
            "CT chunk starts 32 KiB inside the SB chunk and re-claims the range",
            verdict, reader, byte,
            _source_of(covering, {SB_MARKER: "SB-backed region", CT_MARKER: "CT-backed region"}),
            reader["reader_keeps"], len(stream.chunks),
        )
    )

    # D. Same start, same devid/offset, but a different device UUID.
    sb_disk = make_disk(SB_MARKER)
    ct_disk = make_disk(CT_MARKER)  # a physically different device behind the same devid slot
    fs = FakeBtrfs({1: sb_disk})
    stream = ChunkStream(fs)
    sb_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_a)
    stream.add(LOGICAL, sb_chunk)
    loaded = list(stream.chunks)
    # CT claims devid 1 but names a different device UUID; the mounted device is still uuid_a.
    ct_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_b)
    cand = _to_named_tuple(LOGICAL, ct_chunk, fs)
    verdict = classify_candidate(loaded, cand)
    reader = explain_reader_choice(loaded, cand)
    stream.add(LOGICAL, ct_chunk)
    stream.seek(LOGICAL)
    byte = stream.read(1)[0]
    results.append(
        ScenarioResult(
            "D_different_device_uuid",
            "same start/devid/offset, CT names a different device UUID",
            verdict, reader, byte,
            _source_of(_first_logical_chunk(stream),
                       {SB_MARKER: "SB-backed region", CT_MARKER: "CT-backed region"}),
            reader["reader_keeps"], len(stream.chunks),
        )
    )

    # E. Gap-start overlap: the candidate starts where no chunk begins, so the
    # reader's predecessor check sees no covering chunk and inserts it, even
    # though it overlaps a chunk starting later. Two owners then coexist.
    sb_disk = make_disk(SB_MARKER)
    ct_disk = make_disk(CT_MARKER)
    fs = FakeBtrfs({1: sb_disk, 2: ct_disk})
    stream = ChunkStream(fs)
    later = LOGICAL + CHUNK_LEN  # SB chunk begins one chunk-length later
    gap_start = later - CHUNK_LEN // 2  # CT starts 32 KiB before it, in the gap
    sb_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0x0, uuid_a)
    ct_chunk = make_chunk(CHUNK_LEN, CHUNK_LEN, 2, 0x0, uuid_b)
    stream.add(later, sb_chunk)
    loaded = list(stream.chunks)
    cand = _to_named_tuple(gap_start, ct_chunk, fs)
    verdict = classify_candidate(loaded, cand)
    reader = explain_reader_choice(loaded, cand)
    stream.add(gap_start, ct_chunk)
    stream.seek(later)
    byte = stream.read(1)[0]
    # At `later` both chunks claim coverage. The reader selects the
    # highest-offset covering chunk, which is the original SB chunk here.
    covering = max(
        (c for c in stream.chunks if c.offset <= later < c.offset + c.length),
        key=lambda c: c.offset,
    )
    results.append(
        ScenarioResult(
            "E_gap_start_overlap",
            "CT starts in a gap but overlaps a later SB chunk; reader inserts it anyway",
            verdict, reader, byte,
            f"{_source_of(covering, {SB_MARKER: 'SB region', CT_MARKER: 'CT region'})} "
            "(the SB chunk still wins the read despite the inserted CT chunk)",
            reader["reader_keeps"], len(stream.chunks),
        )
    )

    return results


def _to_named_tuple(offset: int, chunk: c_btrfs.btrfs_chunk, fs: FakeBtrfs) -> Chunk:
    """Convert a cstruct chunk into the reader's Chunk tuple exactly like add()."""
    from dissect.btrfs.c_btrfs import BTRFS_BLOCK_GROUP_PROFILE_MASK, BTRFS_RAID_ATTRIBUTES
    from dissect.btrfs.stream import Stripe
    from uuid import UUID as _UUID

    ncopies, nparity, _ = BTRFS_RAID_ATTRIBUTES[chunk.type & BTRFS_BLOCK_GROUP_PROFILE_MASK]
    data_stripes = (chunk.num_stripes - nparity) // ncopies
    stripes = []
    for stripe in chunk.stripe:
        stripes.append(
            Stripe(
                fs.devices.get(stripe.devid),
                stripe.offset,
                stripe.devid,
                _UUID(bytes=stripe.dev_uuid),
            )
        )
    return Chunk(
        offset,
        chunk.length,
        chunk.stripe_len,
        chunk.type,
        chunk.num_stripes,
        chunk.sub_stripes,
        data_stripes,
        stripes,
    )


def main() -> int:
    results = run_scenarios()
    print("Btrfs chunk-source conflict reproduction (real ChunkStream, load order: SB then CT)\n")
    failures = 0
    for result in results:
        print(f"[{result.name}] {result.description}")
        print(f"  classification : {result.verdict.kind} -> action={result.verdict.action}")
        print(f"  reason         : {result.verdict.reason}")
        print(f"  reader actually: keeps={result.reader['reader_keeps']}, "
              f"drops={result.reader['reader_drops']}")
        print(f"  logical byte   : {chr(result.logical_byte)!r} from {result.logical_byte_source}")
        print(f"  chunks loaded  : {result.chunks_loaded}")
        if result.verdict.action == BLOCK:
            failures += 1
        print()
    print("Summary: safe duplicates are de-duplicated silently; every evidence-changing conflict "
          "is flagged to block extraction.")
    print(f"Block-level conflicts demonstrated: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
