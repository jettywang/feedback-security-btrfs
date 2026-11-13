from __future__ import annotations

from dissect.btrfs.btrfs import Btrfs, INode, Subvolume
from dissect.btrfs.exceptions import (
    Error,
    FileNotFoundError,
    NotADirectoryError,
    NotASymlinkError,
)
from dissect.btrfs.reference import (
    CoverageGap,
    ExtentAllocation,
    ExtentReference,
    ExtentReferenceReport,
    OverlapMapping,
    ScanStats,
    SubvolumeScope,
    find_extent_references,
    map_overlap,
)

__all__ = [
    "Btrfs",
    "CoverageGap",
    "Error",
    "ExtentAllocation",
    "ExtentReference",
    "ExtentReferenceReport",
    "FileNotFoundError",
    "INode",
    "NotADirectoryError",
    "NotASymlinkError",
    "OverlapMapping",
    "ScanStats",
    "Subvolume",
    "SubvolumeScope",
    "find_extent_references",
    "map_overlap",
]
