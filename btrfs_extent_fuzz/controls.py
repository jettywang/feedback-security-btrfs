"""Controlled samples used to prove the detector actually fires.

The positive control is a *deliberately* inflated extent: a valid zstd frame
that decodes to far more bytes than its extent header declares.  The detector
must flag it.  The negative controls are legitimate high-compression-ratio
files whose declarations are truthful; the detector must leave them alone,
which is what keeps "high ratio" from masquerading as a finding.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from btrfs_extent_fuzz.corpus import Codec, Entrypoint, Sample

try:
    from backports import zstd
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

# Decoded size of the positive control.  Small enough to stay cheap, large
# enough that any prose-level "ratio" heuristic is not what trips it.
CONTROL_DECODED = 2 << 20


@dataclass(frozen=True)
class Control:
    name: str
    sample: Sample
    expect: str  # verdict value the detector must produce
    rationale: str


def _zstd_frame(decoded: int) -> bytes:
    if zstd is None:  # pragma: no cover
        raise RuntimeError("backports.zstd is required for the zstd controls")
    return zstd.compress(b"\0" * decoded, 19)


def positive_control() -> Control:
    """A declared 1 KiB extent whose payload decodes to 2 MiB."""
    payload = _zstd_frame(CONTROL_DECODED)
    return Control(
        name="positive-inflate-zstd",
        sample=Sample(Codec.ZSTD, Entrypoint.REGULAR, payload, 1024, 1024, len(payload)),
        expect="output-over-declared",
        rationale="valid frame, truthful outer parsers, decoder output far above declared ram_bytes",
    )


def positive_control_inline() -> Control:
    """Same inflation reached through the inline entry point."""
    payload = _zstd_frame(CONTROL_DECODED)
    return Control(
        name="positive-inflate-zstd-inline",
        sample=Sample(Codec.ZSTD, Entrypoint.INLINE, payload, 1024, 1024, len(payload)),
        expect="output-over-declared",
        rationale="inline extent path must reach the same containment check",
    )


def negative_high_ratio() -> list[Control]:
    """Truthfully-declared, legitimately highly compressible files."""
    controls = []
    payload = _zstd_frame(CONTROL_DECODED)
    controls.append(
        Control(
            name="negative-high-ratio-zstd",
            sample=Sample(Codec.ZSTD, Entrypoint.REGULAR, payload, CONTROL_DECODED, CONTROL_DECODED, len(payload)),
            expect="ok",
            rationale="ratio is enormous but ram_bytes/num_bytes declare it truthfully",
        )
    )
    plain = b"zlib" * (CONTROL_DECODED // 4)
    payload = zlib.compress(plain, 9)
    controls.append(
        Control(
            name="negative-high-ratio-zlib",
            sample=Sample(Codec.ZLIB, Entrypoint.REGULAR, payload, len(plain), len(plain), len(payload)),
            expect="ok",
            rationale="≈10x ratio zlib file whose declaration matches the decoded size",
        )
    )
    return controls


def negative_bomb_truthful() -> Control:
    """The hardest false-positive case: a bomb whose declaration is honest.

    This decodes 128 MiB from ~4 KiB.  It must not be reported as
    ``output-over-declared``, because the extent truthfully says it holds that
    much; it is only a budget question, which is a separate verdict.
    """
    decoded = 128 << 20
    payload = _zstd_frame(decoded)
    return Control(
        name="negative-bomb-truthful-declaration",
        sample=Sample(Codec.ZSTD, Entrypoint.REGULAR, payload, decoded, decoded, len(payload)),
        expect="ok",
        rationale="huge decoded size but ram_bytes/num_bytes declare it exactly; ratio alone is not evidence",
    )


def resource_control(budget_bytes: int) -> Control:
    """Truthful declaration whose decode nevertheless exceeds a tight RSS budget."""
    decoded = 160 << 20
    payload = _zstd_frame(decoded)
    return Control(
        name="positive-resource-overrun",
        sample=Sample(Codec.ZSTD, Entrypoint.REGULAR, payload, decoded, decoded, len(payload)),
        expect="resource-overrun",
        rationale=f"truthful declaration, decoded {decoded} bytes exceeds a {budget_bytes} byte RSS budget",
    )


def all_controls() -> list[Control]:
    return [positive_control(), positive_control_inline(), *negative_high_ratio()]


__all__ = [
    "CONTROL_DECODED",
    "Control",
    "all_controls",
    "negative_bomb_truthful",
    "negative_high_ratio",
    "positive_control",
    "positive_control_inline",
    "resource_control",
]
