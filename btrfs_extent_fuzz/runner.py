"""Isolated, budgeted execution of one mutated extent sample.

Every case runs in a freshly forked child.  The child enforces its own CPU
budget with :func:`resource.setrlimit`; the parent enforces a wall-clock
budget with :mod:`select` and measures peak RSS through :func:`os.wait4`, so a
decoder that allocates without bound cannot take the campaign down.

Containment is measured at the decoder, not at the public read: Btrfs streams
truncate their output to the inode/extent declaration, so a bomb would look
harmless through ``read()`` alone.  The child wraps
``dissect.btrfs.stream.decode_extent`` -- the exact function used by both
:meth:`~dissect.btrfs.btrfs.INode.open` (inline) and
:class:`~dissect.btrfs.stream.ExtentStream` (regular) -- and records how many
bytes it really produced, then also records how many bytes the public entry
point returned.

Verdict classes are disjoint, and a ratio is never used as evidence:

``OK``
    Decoded output stayed within the declaration and inside every budget.
``DECODE_REJECT``
    The decoder or parser refused the sample.  Expected fuzz noise; not a
    vulnerability by itself.
``OUTPUT_OVER_DECLARED``
    The decoder produced more bytes than the extent header declares.
``RESOURCE_OVERRUN``
    Output stayed within the declaration but CPU, wall time or peak RSS left
    its budget.
``CRASH``
    The child died on an unexpected signal.
"""

from __future__ import annotations

import os
import resource
import select
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btrfs_extent_fuzz.corpus import Codec, Entrypoint, Sample

# Absolute slack allowed between decoded length and the declaration.  Btrfs
# extent lengths are byte-granular, so anything above this is a containment
# failure rather than codec padding.
DECLARED_SLACK = 8


class Verdict(str, Enum):
    OK = "ok"
    DECODE_REJECT = "decode-reject"
    OUTPUT_OVER_DECLARED = "output-over-declared"
    RESOURCE_OVERRUN = "resource-overrun"
    CRASH = "crash"


@dataclass
class Budget:
    """Independent per-case limits, enforced around the decoding child."""

    # Peak resident set of the decoding child, in bytes.  A normal image peaks
    # near 18 MiB here; the default leaves room for a real highly-compressed
    # file, while a decoder that materialises a large extent is caught.
    rss_bytes: int = 96 << 20
    # Wall clock for the whole child (parse + decode + harness).
    wall_seconds: float = 5.0
    # CPU time as enforced by RLIMIT_CPU (soft == hard).
    cpu_seconds: int = 4
    # Absolute cap on bytes one extent decode may produce.  This is an
    # independent read budget: it does not depend on the declaration (an
    # attacker can declare 2**64) nor on RSS (a decoder may stream output).
    max_output_bytes: int = 64 << 20


@dataclass
class CaseResult:
    case_id: str
    seed_id: str
    codec: "Codec"
    entry: "Entrypoint"
    verdict: Verdict
    declared: int
    decoded: int
    produced: int
    peak_rss: int
    wall_seconds: float
    detail: str = ""
    mutations: list[str] = field(default_factory=list)
    sample_path: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "seed_id": self.seed_id,
            "codec": self.codec.value,
            "entry": self.entry.value,
            "verdict": self.verdict.value,
            "flags": dict(self.flags),
            "declared": self.declared,
            "decoded": self.decoded,
            "produced": self.produced,
            "over_declared": max(0, self.decoded - self.declared),
            "peak_rss": self.peak_rss,
            "wall_seconds": round(self.wall_seconds, 4),
            "mutations": self.mutations,
            "sample_path": self.sample_path,
            "detail": self.detail,
        }


def _maxrss_bytes(rusage) -> int:
    """Return peak RSS in bytes, normalising the platform-dependent unit."""
    value = int(rusage.ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _child_limits(budget: Budget) -> None:
    """Apply child-side budgets, tolerating platforms that refuse a limit."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (budget.cpu_seconds, budget.cpu_seconds))
    except (ValueError, OSError):
        pass


def _instrument_decoder() -> dict[str, int]:
    """Wrap the real ``decode_extent`` so true decoder output is observable."""
    import dissect.btrfs.btrfs as btrfs_mod
    import dissect.btrfs.stream as stream_mod

    tally = {"decoded": 0, "max_decoded": 0, "calls": 0}
    original = stream_mod.decode_extent

    def recorder(buf, compression, encryption, sector_size):
        out = original(buf, compression, encryption, sector_size)
        tally["decoded"] = max(tally["decoded"], len(out))
        tally["max_decoded"] = max(tally["max_decoded"], len(out))
        tally["calls"] += 1
        return out

    stream_mod.decode_extent = recorder
    btrfs_mod.decode_extent = recorder
    return tally


def _read_child(image: bytes, declared_hint: int) -> dict[str, object]:
    """Decoding work performed inside the child.  Never writes user data."""
    import io

    from dissect.btrfs import Btrfs

    baseline = _maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF))
    tally = _instrument_decoder()

    fs = Btrfs(io.BytesIO(image))
    inode = fs.get("extent.bin")
    fh = inode.open()
    data = fh.read()
    peak = _maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF))

    declared = declared_hint
    bound = declared_hint
    try:
        _, item_data = next(inode.subvolume.tree.cursor().iter(inode.inum, 0x6C, 0, ignore_offset=True))
        raw = bytes(item_data)
        declared = int.from_bytes(raw[8:16], "little")
        kind = raw[20]
        if kind == 1 and len(raw) >= 53:  # BTRFS_FILE_EXTENT_REG
            extent_offset = int.from_bytes(raw[37:45], "little")
            num_bytes = int.from_bytes(raw[45:53], "little")
            bound = max(declared, extent_offset + num_bytes)
        else:
            bound = declared
    except Exception:
        pass

    return {
        "decoded": tally["decoded"],
        "decode_calls": tally["calls"],
        "produced": len(data),
        "declared": declared,
        "bound": bound,
        "baseline_rss": baseline,
        "peak_rss": peak,
        "entry_stream": type(fh).__name__,
    }


def run_child(image: bytes, declared_hint: int, budget: Budget) -> dict[str, object]:
    """Run :func:`_read_child` in a forked, budgeted process and return its report."""
    start = time.monotonic()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child branch
        try:
            os.close(r)
            # Mutations can name huge inode/extent sizes; those are the
            # detector's input, so never let the harness allocate for them.
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))
            except (ValueError, OSError):
                pass
            # Native (Rust) codec panics print to stderr before Python can
            # convert them; keep campaign output readable, the report still
            # records the panic through the exception type.
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
            except OSError:
                pass
            _child_limits(budget)
            payload = repr(_read_child(image, declared_hint)).encode()
            os.write(w, payload)
        except BaseException as exc:
            try:
                name = type(exc).__name__
                kind = "panic" if "Panic" in name else "error"
                os.write(w, repr({"error": f"{name}: {exc}", "error_kind": kind}).encode()[:65536])
            except OSError:
                pass
        finally:
            os._exit(0)

    os.close(w)
    os.set_blocking(r, False)
    chunks: list[bytes] = []
    deadline = start + budget.wall_seconds
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([r], [], [], remaining)
        if not ready:
            timed_out = True
            break
        chunk = os.read(r, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r)

    if timed_out:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _, status, usage = os.wait4(pid, 0)
    wall = time.monotonic() - start

    report: dict[str, object] = {}
    raw = b"".join(chunks)
    if raw:
        try:
            import ast

            report = ast.literal_eval(raw.decode())
            if not isinstance(report, dict):
                report = {"error": "unexpected child report"}
        except Exception:
            report = {"error": "unreadable child report"}
    report.setdefault("error", None)
    report["wall"] = wall
    report["timed_out"] = timed_out
    report["status"] = status
    report["child_peak_rss"] = _maxrss_bytes(usage) if usage is not None else 0
    return report


def classify(
    case_id: str,
    seed_id: str,
    codec: "Codec",
    entry: "Entrypoint",
    declared: int,
    report: dict[str, object],
    budget: Budget,
) -> CaseResult:
    """Map a child report onto a verdict using containment, never ratio."""
    reported_declared = report.get("declared")
    declared = declared if reported_declared is None else int(reported_declared)
    reported_bound = report.get("bound")
    bound = declared if reported_bound is None else int(reported_bound)
    decoded = int(report.get("decoded") or 0)
    produced = int(report.get("produced") or 0)
    peak = max(int(report.get("peak_rss") or 0), int(report.get("child_peak_rss") or 0))
    wall = float(report.get("wall") or 0.0)
    status = report.get("status")
    error = report.get("error")
    timed_out = bool(report.get("timed_out"))

    error_text = str(error) if error else ""
    flags = {
        "decode_rejected": bool(error) and not timed_out,
        "native_panic": report.get("error_kind") == "panic",
        "memory_exhausted": "MemoryError" in error_text or "Cannot allocate" in error_text,
        "output_over_declared": decoded > bound + DECLARED_SLACK,
        "output_budget_exceeded": decoded > budget.max_output_bytes,
        "resource_exceeded": (
            peak > budget.rss_bytes
            or wall > budget.wall_seconds
            or timed_out
            or decoded > budget.max_output_bytes
        ),
    }

    verdict = Verdict.OK
    detail = ""
    if status is not None and os.WIFSIGNALED(int(status)):
        sig = os.WTERMSIG(int(status))
        if sig in (signal.SIGXCPU, signal.SIGKILL):
            # SIGXCPU is our CPU budget; an unrequested SIGKILL is almost
            # always the OS reclaiming memory for an unbounded decode.
            flags["resource_exceeded"] = True
        else:
            verdict = Verdict.CRASH
            detail = f"signal {sig}"

    if verdict is Verdict.OK:
        if flags["memory_exhausted"]:
            verdict = Verdict.RESOURCE_OVERRUN
            detail = f"decoder exhausted memory: {error_text[:120]}"
        elif flags["output_budget_exceeded"]:
            verdict = Verdict.RESOURCE_OVERRUN
            detail = f"decoder produced {decoded} bytes over the {budget.max_output_bytes} byte read budget"
        elif flags["resource_exceeded"]:
            verdict = Verdict.RESOURCE_OVERRUN
            detail = f"peak_rss={peak} wall={wall:.3f}s timed_out={timed_out}"
        elif flags["output_over_declared"]:
            verdict = Verdict.OUTPUT_OVER_DECLARED
            detail = f"decoder produced {decoded} bytes for declared bound {bound}"
        elif flags["decode_rejected"]:
            verdict = Verdict.DECODE_REJECT
            detail = str(error)
        elif not decoded and declared > 0:
            verdict = Verdict.DECODE_REJECT
            flags["decode_rejected"] = True
            detail = "decoder produced no output for a non-empty declaration"

    return CaseResult(
        case_id=case_id,
        seed_id=seed_id,
        codec=codec,
        entry=entry,
        verdict=verdict,
        declared=declared,
        decoded=decoded,
        produced=produced,
        peak_rss=peak,
        wall_seconds=wall,
        detail=detail,
        flags=flags,
    )


@dataclass
class Task:
    """One scheduled case: a mutated sample plus its provenance."""

    case_id: str
    seed: object
    sample: "Sample"
    budget: Budget = field(default_factory=Budget)

    @property
    def image(self) -> bytes:
        from btrfs_extent_fuzz.image import build_image

        return build_image(self.sample)

    def run(self) -> CaseResult:
        report = run_child(self.image, self.sample.declared, self.budget)
        result = classify(
            self.case_id,
            self.seed.seed_id,
            self.sample.codec,
            self.sample.entry,
            self.sample.declared,
            report,
            self.budget,
        )
        result.mutations = list(self.sample.mutations)
        return result


__all__ = [
    "Budget",
    "CaseResult",
    "DECLARED_SLACK",
    "Task",
    "Verdict",
    "classify",
    "run_child",
]
