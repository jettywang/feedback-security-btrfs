"""Deterministic, self-contained seed corpus and structural mutators.

Seeds are small valid zlib, LZO and zstd payloads paired with the exact
``ram_bytes``/``num_bytes`` declaration a real Btrfs image would carry.  All
mutators take an explicit :class:`random.Random`, so a case is fully
reproducible from ``(seed, case_index)``.
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass, field
from enum import Enum

try:
    from backports import zstd
except ImportError:  # pragma: no cover - optional at import time
    zstd = None  # type: ignore[assignment]


class Codec(str, Enum):
    ZLIB = "zlib"
    LZO = "lzo"
    ZSTD = "zstd"

    @property
    def compression_id(self) -> int:
        return {Codec.ZLIB: 1, Codec.LZO: 2, Codec.ZSTD: 3}[self]


class Entrypoint(str, Enum):
    INLINE = "inline"
    REGULAR = "regular"


ZLIKE_PLAIN = b"zlib" * 256 + b"\n"
LZO_PLAIN = b"lzo" * 256 + b"\n"
ZSTD_PLAIN = b"zstd" * 256 + b"\n"

# Smallest known-good LZO payload: outer total-size header plus one 32-byte
# segment that decodes to 769 bytes.  Byte-for-byte the payload used by the
# upstream btrfs-compression fixture.
LZO_SEED = bytes.fromhex(
    "2800000020000000036c7a6f6c7a6f200000cd14000c7a6f6c7a6f6c7a6f6c7a6f6c7a6f0a110000"
)


@dataclass(frozen=True)
class Seed:
    """A minimal valid starting point for one codec/entrypoint pair."""

    seed_id: str
    codec: Codec
    entry: Entrypoint
    payload: bytes
    declared: int
    plain: bytes

    def describe(self) -> dict[str, object]:
        return {
            "seed_id": self.seed_id,
            "codec": self.codec.value,
            "entry": self.entry.value,
            "payload_len": len(self.payload),
            "declared": self.declared,
            "plain_len": len(self.plain),
            "sha256_payload": _sha256(self.payload),
        }


@dataclass
class Sample:
    """A mutable extent sample: codec payload plus its on-disk declaration."""

    codec: Codec
    entry: Entrypoint
    payload: bytes
    declared: int
    inode_size: int
    disk_num_bytes: int
    disk_offset: int = 0
    lzo_outer: int | None = None
    frame_tail: bytes = b""
    mutations: list[str] = field(default_factory=list)

    def clone(self) -> Sample:
        return Sample(
            codec=self.codec,
            entry=self.entry,
            payload=self.payload,
            declared=self.declared,
            inode_size=self.inode_size,
            disk_num_bytes=self.disk_num_bytes,
            disk_offset=self.disk_offset,
            lzo_outer=self.lzo_outer,
            frame_tail=self.frame_tail,
            mutations=list(self.mutations),
        )


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _make_seeds() -> list[Seed]:
    zlib_payload = zlib.compress(ZLIKE_PLAIN, 9)
    seeds = [
        Seed("zlib-inline", Codec.ZLIB, Entrypoint.INLINE, zlib_payload, len(ZLIKE_PLAIN), ZLIKE_PLAIN),
        Seed("zlib-regular", Codec.ZLIB, Entrypoint.REGULAR, zlib_payload, len(ZLIKE_PLAIN), ZLIKE_PLAIN),
        Seed("lzo-inline", Codec.LZO, Entrypoint.INLINE, LZO_SEED, len(LZO_PLAIN), LZO_PLAIN),
        Seed("lzo-regular", Codec.LZO, Entrypoint.REGULAR, LZO_SEED, len(LZO_PLAIN), LZO_PLAIN),
    ]
    if zstd is not None:
        payload = zstd.compress(ZSTD_PLAIN, 9)
        seeds += [
            Seed("zstd-inline", Codec.ZSTD, Entrypoint.INLINE, payload, len(ZSTD_PLAIN), ZSTD_PLAIN),
            Seed("zstd-regular", Codec.ZSTD, Entrypoint.REGULAR, payload, len(ZSTD_PLAIN), ZSTD_PLAIN),
        ]
    return seeds


def corpus() -> list[Seed]:
    """Return the fixed seed corpus (independent of process state)."""
    return _make_seeds()


def seed_by_id(seed_id: str) -> Seed:
    for seed in corpus():
        if seed.seed_id == seed_id:
            return seed
    raise KeyError(f"unknown seed id: {seed_id!r}")


def sample_from_seed(seed: Seed) -> Sample:
    return Sample(
        codec=seed.codec,
        entry=seed.entry,
        payload=seed.payload,
        declared=seed.declared,
        inode_size=seed.declared,
        disk_num_bytes=len(seed.payload),
        lzo_outer=int.from_bytes(seed.payload[:4], "little") if seed.codec is Codec.LZO else None,
    )


# ---------------------------------------------------------------------------
# Mutators.  Each mutator mutates in place and records a human-readable tag.
# ---------------------------------------------------------------------------


def mutate_segment_length(sample: Sample, rng: random.Random) -> None:
    """LZO: rewrite one segment-length word, the axis unique to this codec."""
    if sample.codec is not Codec.LZO:
        return
    payload = bytearray(sample.payload)
    if len(payload) < 12:
        sample.mutations.append("lzo-segment:too-short")
        return
    slot = 4 + 4 * rng.randrange(0, max(1, (len(payload) - 8) // 4))
    if slot + 4 > len(payload):
        slot = len(payload) - 4
    choice = rng.choice(["zero", "huge", "truncate", "overlap"])
    if choice == "zero":
        payload[slot : slot + 4] = (0).to_bytes(4, "little")
    elif choice == "huge":
        payload[slot : slot + 4] = (0xFFFFFFFF).to_bytes(4, "little")
    elif choice == "truncate":
        payload[slot : slot + 4] = (max(1, len(payload) - slot)).to_bytes(4, "little")
    else:
        payload[slot : slot + 4] = (rng.randrange(1, 0x1000)).to_bytes(4, "little")
    sample.payload = bytes(payload)
    sample.mutations.append(f"lzo-segment:{choice}@{slot}")


def mutate_lzo_outer_total(sample: Sample, rng: random.Random) -> None:
    if sample.codec is not Codec.LZO or len(sample.payload) < 4:
        return
    payload = bytearray(sample.payload)
    value = rng.choice([0, 0xFFFFFFFF, len(payload), 1])
    payload[0:4] = value.to_bytes(4, "little")
    sample.payload = bytes(payload)
    sample.lzo_outer = value
    sample.mutations.append(f"lzo-total:{value:#x}")


def mutate_frame_tail(sample: Sample, rng: random.Random) -> None:
    """zlib/zstd: truncate the tail, append junk bytes, or concatenate frames."""
    if sample.codec is Codec.LZO:
        return
    mode = rng.choice(["truncate", "append", "concat", "bite"])
    payload = sample.payload
    if mode == "truncate":
        cut = rng.randrange(1, max(2, len(payload)))
        sample.payload = payload[:cut]
        sample.mutations.append(f"tail:truncate-{cut}")
    elif mode == "append":
        junk = rng.randbytes(rng.randrange(1, 32))
        sample.payload = payload + junk
        sample.frame_tail = junk
        sample.mutations.append(f"tail:append-{len(junk)}")
    elif mode == "concat":
        sample.payload = payload + payload
        sample.mutations.append("tail:concat-self")
    else:
        payload = bytearray(payload)
        payload[-1] ^= rng.randrange(1, 256)
        sample.payload = bytes(payload)
        sample.mutations.append("tail:flip-checksum")


def mutate_zstd_frame_header(sample: Sample, rng: random.Random) -> None:
    """zstd: widen/patch the frame content-size descriptor."""
    if sample.codec is not Codec.ZSTD or len(sample.payload) < 6:
        return
    payload = bytearray(sample.payload)
    mode = rng.choice(["widen-fcs", "set-fcs", "clear-single-segment"])
    if mode.startswith("widen"):
        payload[4] |= 0x40
        sample.mutations.append("zstd-header:widen-fcs")
    elif mode == "set-fcs":
        if payload[4] & 0x20:
            payload[4] |= 0x40
            payload[4] &= ~0x20
        payload[5] = rng.choice([0xFF, 0x80, 0x01])
        sample.mutations.append("zstd-header:set-fcs")
    else:
        payload[4] &= ~0x20
        sample.mutations.append("zstd-header:clear-single-segment")
    sample.payload = bytes(payload)


# Inline extents live inside a B-tree leaf; stay well inside one node so the
# harness never confuses "image malformed" with "decoder over-produced".
INLINE_MAX_PAYLOAD = 8192


def _clamp_inline(sample: Sample) -> None:
    if sample.entry is Entrypoint.INLINE and len(sample.payload) > INLINE_MAX_PAYLOAD:
        sample.payload = sample.payload[:INLINE_MAX_PAYLOAD]
        sample.mutations.append("payload:clamped-inline")


def mutate_declared(sample: Sample, rng: random.Random) -> None:
    """Coupled extent-declaration mutations (the third cross-cutting axis)."""
    mode = rng.choice(
        [
            "shrink-declared",
            "zero-declared",
            "max-declared",
            "inode-lies",
            "disk-lies",
            "offset-lies",
        ]
    )
    if mode == "shrink-declared":
        sample.declared = max(1, sample.declared // rng.choice([2, 4, 16]))
    elif mode == "zero-declared":
        sample.declared = 0
    elif mode == "max-declared":
        sample.declared = rng.choice([1 << 32, 1 << 48, (1 << 64) - 1])
    elif mode == "inode-lies":
        sample.inode_size = rng.choice([0, 1, 1 << 32, (1 << 64) - 1])
    elif mode == "disk-lies":
        sample.disk_num_bytes = rng.choice([0, 1, 1 << 40, (1 << 64) - 1])
    else:
        sample.disk_offset = rng.choice([0, 1, 1 << 40, (1 << 64) - 1])
    sample.mutations.append(f"declared:{mode}")


def mutate_random_bytes(sample: Sample, rng: random.Random) -> None:
    if not sample.payload:
        return
    payload = bytearray(sample.payload)
    for _ in range(rng.randrange(1, 4)):
        idx = rng.randrange(len(payload))
        payload[idx] ^= rng.randrange(1, 256)
    sample.payload = bytes(payload)
    sample.mutations.append("bytes:flip")


MUTATORS = (
    mutate_segment_length,
    mutate_lzo_outer_total,
    mutate_frame_tail,
    mutate_zstd_frame_header,
    mutate_declared,
    mutate_random_bytes,
)


def mutate_sample(seed: Seed, case_index: int, root_seed: int) -> Sample:
    """Apply a deterministic 1-3 step mutation stack to ``seed``."""
    rng = random.Random(f"{root_seed}:{seed.seed_id}:{case_index}")
    sample = sample_from_seed(seed)
    depth = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
    for mutator in rng.sample(MUTATORS, depth):
        mutator(sample, rng)
    _clamp_inline(sample)
    return sample


__all__ = [
    "Codec",
    "Entrypoint",
    "LZO_SEED",
    "MUTATORS",
    "Sample",
    "Seed",
    "corpus",
    "mutate_lzo_outer_total",
    "mutate_sample",
    "mutate_segment_length",
    "mutate_zstd_frame_header",
    "sample_from_seed",
    "seed_by_id",
]
