"""Resource-abuse fuzzing for Btrfs compressed extents.

The package drives the real ``dissect.btrfs`` read paths
(:meth:`dissect.btrfs.btrfs.INode.open` for inline data and
:class:`dissect.btrfs.stream.ExtentStream` for regular extents) with
deterministically mutated, self-contained Btrfs images.

Two invariants are checked per sample:

* *decode containment* -- the decoded byte count for an extent must stay
  within the size the on-disk extent header declares, otherwise output is
  escaping the declaration.
* *resource containment* -- wall time and peak RSS of the decoding process
  must stay within independent budgets.

Neither invariant uses the compression ratio of a sample, so a legitimately
high-ratio file is never reported merely for being highly compressible.
"""

from __future__ import annotations

from btrfs_extent_fuzz.corpus import Codec, Entrypoint, Seed, corpus, seed_by_id
from btrfs_extent_fuzz.image import build_image, extent_item, inline_item, regular_item
from btrfs_extent_fuzz.runner import CaseResult, Task, Verdict, run_child
from btrfs_extent_fuzz.report import run_campaign, summarize

__all__ = [
    "CaseResult",
    "Codec",
    "Entrypoint",
    "Seed",
    "Task",
    "Verdict",
    "build_image",
    "corpus",
    "extent_item",
    "inline_item",
    "regular_item",
    "run_campaign",
    "run_child",
    "seed_by_id",
    "summarize",
]
