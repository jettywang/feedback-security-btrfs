from __future__ import annotations

from uuid import UUID

import pytest

from dissect.btrfs.exceptions import Error
from dissect.btrfs.stream import ChunkStream

from tests.repro_chunk_conflict.chunk_conflict import (
    BLOCK,
    DEDUP,
    DUPLICATE,
    HARD_CONFLICT,
    OVERLAP_CONFLICT,
    classify_candidate,
)
from tests.repro_chunk_conflict.reproduce_conflicts import (
    CHUNK_LEN,
    CT_MARKER,
    LOGICAL,
    SB_MARKER,
    FakeBtrfs,
    main,
    make_chunk,
    make_disk,
    run_scenarios,
)

UUID_A = UUID("11111111-1111-1111-1111-111111111111")
UUID_B = UUID("22222222-2222-2222-2222-222222222222")


def _candidate_tuple(stream, offset, chunk, fs):
    from tests.repro_chunk_conflict.reproduce_conflicts import _to_named_tuple

    return _to_named_tuple(offset, chunk, fs)


def test_all_scenarios_block_or_dedup_and_match_reader():
    results = run_scenarios()
    by_name = {r.name: r for r in results}

    assert by_name["A_identical_duplicate"].verdict.kind == DUPLICATE
    assert by_name["A_identical_duplicate"].verdict.action == DEDUP

    for blocking in ("B_same_start_different_stripe", "C_partial_overlap",
                     "D_different_device_uuid", "E_gap_start_overlap"):
        result = by_name[blocking]
        assert result.verdict.action == BLOCK
        assert result.verdict.kind in (HARD_CONFLICT, OVERLAP_CONFLICT)

    # The current reader is first-writer-wins: the super block mapping is the
    # incumbent in B/C/D and its marker is what the logical byte resolves to.
    for name in ("A_identical_duplicate", "B_same_start_different_stripe",
                 "C_partial_overlap", "D_different_device_uuid"):
        assert by_name[name].logical_byte == SB_MARKER

    # E inserts the candidate (no covering predecessor) but the SB chunk still
    # owns the read at its own start.
    assert by_name["E_gap_start_overlap"].reader["reader_keeps"] == "candidate"
    assert by_name["E_gap_start_overlap"].logical_byte == SB_MARKER


def test_identical_entry_is_safe_duplicate():
    fs = FakeBtrfs({1: make_disk(SB_MARKER)})
    stream = ChunkStream(fs)
    sb = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_A)
    ct = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_A)
    stream.add(LOGICAL, sb)

    verdict = classify_candidate(list(stream.chunks), _candidate_tuple(stream, LOGICAL, ct, fs))
    assert verdict.kind == DUPLICATE
    assert verdict.action == DEDUP

    before = len(stream.chunks)
    stream.add(LOGICAL, ct)
    assert len(stream.chunks) == before  # de-duplicated, not duplicated


def test_same_start_different_stripe_is_hard_conflict():
    fs = FakeBtrfs({1: make_disk(SB_MARKER), 2: make_disk(CT_MARKER)})
    stream = ChunkStream(fs)
    stream.add(LOGICAL, make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_A))
    ct = make_chunk(CHUNK_LEN, CHUNK_LEN, 2, 0x20000, UUID_B)

    verdict = classify_candidate(list(stream.chunks), _candidate_tuple(stream, LOGICAL, ct, fs))
    assert verdict.kind == HARD_CONFLICT
    assert verdict.action == BLOCK


def test_different_device_uuid_alone_is_hard_conflict():
    # Same devid and physical offset, but a different device UUID changes which
    # physical device is the provenance of the bytes -> evidence-changing.
    fs = FakeBtrfs({1: make_disk(SB_MARKER)})
    stream = ChunkStream(fs)
    stream.add(LOGICAL, make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_A))
    ct = make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_B)

    verdict = classify_candidate(list(stream.chunks), _candidate_tuple(stream, LOGICAL, ct, fs))
    assert verdict.kind == HARD_CONFLICT
    assert verdict.action == BLOCK

    # The reader itself never validates dev_uuid against the mounted device;
    # it silently keeps the incumbent solely on logical-offset coverage.
    stream.add(LOGICAL, ct)
    stream.seek(LOGICAL)
    assert stream.read(1) == bytes([SB_MARKER])


def test_partial_overlap_is_blocked_even_when_start_differs():
    fs = FakeBtrfs({1: make_disk(SB_MARKER), 2: make_disk(CT_MARKER)})
    stream = ChunkStream(fs)
    stream.add(LOGICAL, make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_A))
    inner = LOGICAL + CHUNK_LEN // 2
    ct = make_chunk(CHUNK_LEN, CHUNK_LEN, 2, 0, UUID_B)

    verdict = classify_candidate(list(stream.chunks), _candidate_tuple(stream, inner, ct, fs))
    assert verdict.kind == OVERLAP_CONFLICT
    assert verdict.action == BLOCK


def test_main_runs_and_reports_four_blocking_conflicts(capsys):
    assert main() == 0
    out = capsys.readouterr().out
    assert "Block-level conflicts demonstrated: 4" in out
    assert "[D_different_device_uuid]" in out


def test_reader_raises_on_missing_device_for_unrecoverable_chunk():
    # A candidate referencing a device that is neither mounted nor within RAID
    # tolerance cannot be read at all; this must surface rather than be guessed.
    fs = FakeBtrfs({1: make_disk(SB_MARKER)})
    stream = ChunkStream(fs)
    stream.add(LOGICAL, make_chunk(CHUNK_LEN, CHUNK_LEN, 1, 0, UUID_A))
    orphan = make_chunk(CHUNK_LEN, CHUNK_LEN, 99, 0, UUID_B)  # devid 99 not mounted

    with pytest.raises(Error, match="Missing stripe disk"):
        stream.add(LOGICAL + CHUNK_LEN, orphan)
