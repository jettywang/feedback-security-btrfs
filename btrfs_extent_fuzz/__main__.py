"""Command line entry point for the extent resource-abuse campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from btrfs_extent_fuzz.corpus import corpus
from btrfs_extent_fuzz.controls import all_controls, negative_bomb_truthful, resource_control
from btrfs_extent_fuzz.report import replay, run_campaign, summarize
from btrfs_extent_fuzz.runner import Budget, Task


def _budget(args: argparse.Namespace) -> Budget:
    return Budget(
        rss_bytes=int(args.rss_mb * 1024 * 1024),
        wall_seconds=args.wall,
        cpu_seconds=args.cpu,
        max_output_bytes=int(args.max_output_mb * 1024 * 1024),
    )


def command_run(args: argparse.Namespace) -> int:
    budget = _budget(args)
    campaign = run_campaign(
        root_seed=args.seed,
        cases_per_seed=args.cases,
        out_dir=args.samples,
        budget=budget,
        write_minimal=not args.no_minimal,
    )
    print(summarize(campaign))
    if args.json:
        Path(args.json).write_text(json.dumps(campaign.as_dict(), indent=2, sort_keys=True) + "\n")
        print(f"report written to {args.json}")
    return 0


def command_controls(args: argparse.Namespace) -> int:
    budget = _budget(args)
    bomb = negative_bomb_truthful()
    # The truthful bomb needs a budget that fits its honest declaration; that is
    # the whole point: with room, a bomb is a normal file.
    cases = all_controls() + [bomb, resource_control(budget.rss_bytes)]
    failed = 0
    for index, control in enumerate(cases):
        case_budget = budget
        if control is bomb:
            case_budget = Budget(
                rss_bytes=384 << 20,
                wall_seconds=budget.wall_seconds,
                cpu_seconds=budget.cpu_seconds,
                max_output_bytes=256 << 20,
            )
        result = Task(
            case_id=control.name,
            seed=corpus()[index % len(corpus())],
            sample=control.sample,
            budget=case_budget,
        ).run()
        ok = result.verdict.value == control.expect
        failed += 0 if ok else 1
        print(
            f"[{'pass' if ok else 'FAIL'}] {control.name}: expected {control.expect}, got {result.verdict.value} "
            f"(decoded={result.decoded}, declared={result.declared}, peak_rss={result.peak_rss})"
        )
    return 1 if failed else 0


def command_replay(args: argparse.Namespace) -> int:
    result = replay(args.sample, budget=_budget(args))
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def command_seeds(_: argparse.Namespace) -> int:
    for seed in corpus():
        print(json.dumps(seed.describe(), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btrfs_extent_fuzz", description=__doc__)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xB7F5, help="campaign root seed")
    parser.add_argument("--rss-mb", type=int, default=96, help="per-case peak RSS budget in MiB")
    parser.add_argument("--wall", type=float, default=5.0, help="per-case wall-clock budget in seconds")
    parser.add_argument("--cpu", type=int, default=4, help="per-case CPU budget in seconds")
    parser.add_argument("--max-output-mb", type=int, default=64, help="per-extent decoded-output budget in MiB")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the mutation campaign")
    run.add_argument("--cases", type=int, default=40, help="cases per seed")
    run.add_argument("--samples", type=str, default="samples", help="directory for minimal samples")
    run.add_argument("--no-minimal", action="store_true", help="do not write minimal samples")
    run.add_argument("--json", type=str, default=None, help="write the full report as JSON here")
    run.set_defaults(func=command_run)

    controls = sub.add_parser("controls", help="run detector positive/negative controls")
    controls.set_defaults(func=command_controls)

    replay_parser = sub.add_parser("replay", help="replay one minimal sample")
    replay_parser.add_argument("sample", type=str)
    replay_parser.set_defaults(func=command_replay)

    seeds = sub.add_parser("seeds", help="print the seed corpus")
    seeds.set_defaults(func=command_seeds)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
