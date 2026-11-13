from __future__ import annotations

import hashlib

import pytest

from btrfs_extent_fuzz.controls import (
    negative_bomb_truthful,
    negative_high_ratio,
    positive_control,
    positive_control_inline,
    resource_control,
)
from btrfs_extent_fuzz.corpus import (
    Codec,
    Entrypoint,
    corpus,
    mutate_sample,
    sample_from_seed,
    seed_by_id,
)
from btrfs_extent_fuzz.image import build_image, read_target
from btrfs_extent_fuzz.report import replay, run_campaign
from btrfs_extent_fuzz.runner import Budget, Task, Verdict

CAMPAIGN_SEED = 0xB7F5


def test_corpus_is_complete_and_deterministic() -> None:
    seeds = corpus()
    assert {seed.codec for seed in seeds} == set(Codec)
    assert {seed.entry for seed in seeds} == set(Entrypoint)
    assert len(seeds) == 6
    assert [seed.describe()["sha256_payload"] for seed in seeds] == [
        seed.describe()["sha256_payload"] for seed in corpus()
    ]


@pytest.mark.parametrize("seed_id", [seed.seed_id for seed in corpus()])
def test_untouched_seed_decodes_and_is_ok(seed_id: str) -> None:
    seed = seed_by_id(seed_id)
    sample = sample_from_seed(seed)
    stream_name, data = read_target(build_image(sample))
    assert data == seed.plain
    assert stream_name in ("BufferedStream", "ExtentStream")

    result = Task(case_id="seed", seed=seed, sample=sample).run()
    assert result.verdict is Verdict.OK
    assert result.decoded == len(seed.plain)


def test_positive_control_is_detected() -> None:
    control = positive_control()
    result = Task(case_id=control.name, seed=corpus()[0], sample=control.sample).run()
    assert result.verdict is Verdict.OUTPUT_OVER_DECLARED
    assert result.decoded > control.sample.declared + 1024
    assert result.flags["output_over_declared"]
    assert not result.flags["resource_exceeded"]


def test_positive_control_reaches_inline_entrypoint() -> None:
    control = positive_control_inline()
    result = Task(case_id=control.name, seed=corpus()[0], sample=control.sample).run()
    assert result.verdict is Verdict.OUTPUT_OVER_DECLARED
    stream_name, data = read_target(build_image(control.sample))
    assert stream_name == "BufferedStream"
    # The inline path wraps the decoded buffer in a stream sized to the inode,
    # so callers only ever observe the declared 1024 bytes; the excess is only
    # visible at the decoder, which is why the runner records it there.
    assert len(data) == control.sample.inode_size
    assert data == b"\x00" * control.sample.inode_size


@pytest.mark.parametrize("control", negative_high_ratio(), ids=lambda c: c.name)
def test_legitimate_high_ratio_is_not_a_finding(control) -> None:
    result = Task(case_id=control.name, seed=corpus()[0], sample=control.sample).run()
    assert result.verdict is Verdict.OK
    assert result.decoded <= control.sample.declared
    assert result.decoded / len(control.sample.payload) > 8


def test_truthful_bomb_is_not_an_output_escape() -> None:
    control = negative_bomb_truthful()
    budget = Budget(rss_bytes=384 << 20, wall_seconds=8.0, max_output_bytes=256 << 20)
    result = Task(case_id=control.name, seed=corpus()[0], sample=control.sample, budget=budget).run()
    assert result.verdict is Verdict.OK
    assert not result.flags["output_over_declared"]
    assert result.decoded == control.sample.declared
    assert result.decoded / len(control.sample.payload) > 1000


def test_independent_output_budget() -> None:
    """A declared-unbounded sample is still stopped by the absolute read budget."""
    from btrfs_extent_fuzz.controls import CONTROL_DECODED

    control = positive_control()
    tight = Budget(rss_bytes=768 << 20, max_output_bytes=1 << 20)
    result = Task(case_id="out-budget", seed=corpus()[0], sample=control.sample, budget=tight).run()
    assert CONTROL_DECODED > tight.max_output_bytes
    assert result.verdict is Verdict.RESOURCE_OVERRUN
    assert result.flags["output_budget_exceeded"]


def test_resource_overrun_is_separate_from_output_escape() -> None:
    control = resource_control(96 << 20)
    budget = Budget(rss_bytes=96 << 20, max_output_bytes=256 << 20)
    result = Task(case_id=control.name, seed=corpus()[0], sample=control.sample, budget=budget).run()
    assert result.verdict is Verdict.RESOURCE_OVERRUN
    assert not result.flags["output_over_declared"]
    assert result.peak_rss > budget.rss_bytes

    roomy = Task(
        case_id="roomy",
        seed=corpus()[0],
        sample=control.sample,
        budget=Budget(rss_bytes=768 << 20, wall_seconds=8.0, max_output_bytes=256 << 20),
    ).run()
    assert roomy.verdict is Verdict.OK


def test_decode_rejection_is_not_a_vulnerability() -> None:
    seed = seed_by_id("zstd-regular")
    sample = sample_from_seed(seed)
    sample.payload = sample.payload[:5]
    sample.disk_num_bytes = 5
    result = Task(case_id="truncated", seed=seed, sample=sample).run()
    assert result.verdict is Verdict.DECODE_REJECT
    assert not result.flags["output_over_declared"]
    assert not result.flags["resource_exceeded"]


def test_mutation_axis_segment_length_for_lzo() -> None:
    seed = seed_by_id("lzo-regular")
    seen_segment = False
    for index in range(24):
        sample = mutate_sample(seed, index, CAMPAIGN_SEED)
        if any(tag.startswith("lzo-segment") for tag in sample.mutations):
            seen_segment = True
        assert sample.codec is Codec.LZO
    assert seen_segment


def test_campaign_minimal_samples_replay(tmp_path) -> None:
    campaign = run_campaign(
        root_seed=CAMPAIGN_SEED,
        cases_per_seed=3,
        out_dir=tmp_path / "samples",
        budget=Budget(wall_seconds=4.0),
        seed_ids=["zlib-inline", "zstd-regular"],
    )
    assert campaign.results
    assert campaign.findings(), "expected the coupled declaration mutation to be found"
    assert campaign.written

    for path in campaign.written:
        replayed = replay(path)
        original = next(r for r in campaign.findings() if r.sample_path == path)
        assert replayed.verdict is original.verdict

        blob = (tmp_path / "samples" / path.split("/")[-1]).read_text()
        assert blob
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert digest


def test_runner_reports_peak_rss_and_wall() -> None:
    seed = corpus()[0]
    result = Task(case_id="metrics", seed=seed, sample=sample_from_seed(seed)).run()
    assert result.peak_rss > 0
    assert result.wall_seconds > 0


def test_budget_defaults_are_independent() -> None:
    budget = Budget()
    assert budget.rss_bytes > 0
    assert budget.wall_seconds > 0
    assert budget.cpu_seconds > 0
    other = Budget(rss_bytes=32 << 20)
    assert other.wall_seconds == budget.wall_seconds


def test_image_is_stable_for_a_given_sample() -> None:
    seed = seed_by_id("zstd-regular")
    sample = sample_from_seed(seed)
    first = build_image(sample)
    second = build_image(sample)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
