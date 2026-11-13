"""Campaign orchestration, minimal-sample replay and report generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from btrfs_extent_fuzz.corpus import (
    Codec,
    Entrypoint,
    Sample,
    corpus,
    mutate_sample,
)
from btrfs_extent_fuzz.runner import Budget, CaseResult, Task, Verdict, run_child, classify
from btrfs_extent_fuzz.image import build_image

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class CampaignResult:
    """Aggregate outcome of a campaign, plus replayable artifacts."""

    root_seed: int
    budget: Budget
    results: list[CaseResult] = field(default_factory=list)
    written: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(r.verdict.value for r in self.results))

    def findings(self) -> list[CaseResult]:
        return [r for r in self.results if r.verdict not in (Verdict.OK, Verdict.DECODE_REJECT)]

    def as_dict(self) -> dict[str, object]:
        return {
            "root_seed": self.root_seed,
            "budget": asdict(self.budget),
            "total_cases": len(self.results),
            "counts": self.counts(),
            "findings": [r.as_dict() for r in self.findings()],
            "written": self.written,
        }


def _is_finding(sample: Sample, budget: Budget | None = None) -> bool:
    budget = budget or Budget(wall_seconds=3.0)
    report = run_child(build_image(sample), sample.declared, budget)
    verdict = classify("mini", "mini", sample.codec, sample.entry, sample.declared, report, budget)
    return verdict.verdict not in (Verdict.OK, Verdict.DECODE_REJECT)


def _minimise(sample: Sample, budget: Budget | None = None) -> Sample:
    """Shrink a sample to the smallest payload that still reproduces its verdict."""
    current = sample.clone()
    step = max(1, len(current.payload) // 2)
    while step > 0:
        candidate = current.clone()
        if len(candidate.payload) > step:
            candidate.payload = candidate.payload[:-step]
            candidate.disk_num_bytes = len(candidate.payload)
            candidate.mutations = [*current.mutations, f"minimise:-{step}"]
            if _is_finding(candidate, budget):
                current = candidate
                continue
        step //= 2
    return current


def _write_minimal(out_dir: Path, case_id: str, sample: Sample, image: bytes, result: CaseResult) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = {
        "case_id": case_id,
        "seed_id": result.seed_id,
        "codec": sample.codec.value,
        "entry": sample.entry.value,
        "declared": sample.declared,
        "inode_size": sample.inode_size,
        "disk_num_bytes": sample.disk_num_bytes,
        "disk_offset": sample.disk_offset,
        "payload_hex": sample.payload.hex(),
        "mutations": sample.mutations,
        "verdict": result.verdict.value,
        "produced": result.produced,
        "image_sha256": hashlib.sha256(image).hexdigest(),
    }
    path = out_dir / f"{case_id}.min.json"
    path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    return str(path)


def run_campaign(
    *,
    root_seed: int = 0xB7F5,
    cases_per_seed: int = 40,
    out_dir: str | Path | None = "samples",
    budget: Budget | None = None,
    seed_ids: Iterable[str] | None = None,
    write_minimal: bool = True,
) -> CampaignResult:
    """Run the full deterministic campaign against every codec/entrypoint pair."""
    budget = budget or Budget()
    campaign = CampaignResult(root_seed=root_seed, budget=budget)
    selected = corpus()
    if seed_ids is not None:
        wanted = set(seed_ids)
        selected = [s for s in selected if s.seed_id in wanted]

    out = Path(out_dir) if out_dir is not None else None
    for seed in selected:
        for index in range(cases_per_seed):
            sample = mutate_sample(seed, index, root_seed)
            case_id = f"{seed.seed_id}-{index:04d}"
            task = Task(case_id=case_id, seed=seed, sample=sample, budget=budget)
            result = task.run()
            image = task.image
            if result.verdict not in (Verdict.OK, Verdict.DECODE_REJECT) and write_minimal and out is not None:
                minimal = _minimise(sample, budget)
                minimal_image = build_image(minimal)
                path = _write_minimal(out, case_id, minimal, minimal_image, result)
                result.sample_path = path
                campaign.written.append(path)
            campaign.results.append(result)
    return campaign


def summarize(campaign: CampaignResult) -> str:
    counts = campaign.counts()
    lines = [
        f"root_seed=0x{campaign.root_seed:X} cases={len(campaign.results)}",
        "budget: "
        f"rss={campaign.budget.rss_bytes}B output={campaign.budget.max_output_bytes}B "
        f"wall={campaign.budget.wall_seconds}s cpu={campaign.budget.cpu_seconds}s",
        "verdicts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
    ]
    for finding in campaign.findings():
        lines.append(
            f"  {finding.case_id} [{finding.codec.value}/{finding.entry.value}] {finding.verdict.value}: "
            f"{finding.detail} produced={finding.produced} declared={finding.declared} "
            f"rss={finding.peak_rss}"
        )
    return "\n".join(lines)


def replay(sample_json: str | Path, budget: Budget | None = None) -> CaseResult:
    """Rebuild the minimal image from a written sample and re-run it."""
    blob = json.loads(Path(sample_json).read_text())
    sample = Sample(
        codec=Codec(blob["codec"]),
        entry=Entrypoint(blob["entry"]),
        payload=bytes.fromhex(blob["payload_hex"]),
        declared=blob["declared"],
        inode_size=blob["inode_size"],
        disk_num_bytes=blob["disk_num_bytes"],
        disk_offset=blob["disk_offset"],
        mutations=list(blob["mutations"]),
    )
    task = Task(
        case_id=blob["case_id"],
        seed=_FakeSeed(blob["seed_id"]),
        sample=sample,
        budget=budget or Budget(),
    )
    result = task.run()
    expected = blob.get("image_sha256")
    actual = hashlib.sha256(task.image).hexdigest()
    result.flags["reconstructed_image_matches"] = expected is None or expected == actual
    return result


@dataclass(frozen=True)
class _FakeSeed:
    seed_id: str
    sample: Sample | None = None


__all__ = ["CampaignResult", "replay", "run_campaign", "summarize"]
