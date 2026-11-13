"""Example: trace a suspicious Btrfs disk range back to files in selected subvolumes.

This is read-only. It opens an image, scans only the subvolume roots you name, and prints the
references together with their provenance and any coverage gaps.

Run it against a Btrfs image::

    python examples/reverse_extent.py image.bin --bytenr 0x4500000 --length 0x1000 --subvolume 256 --subvolume 257

For the repository's ``btrfs-subvolume-snapshot`` fixture, decompress it first::

    gunzip -c tests/_data/btrfs-subvolume-snapshot.bin.gz > snapshot.bin
"""

from __future__ import annotations

import argparse
import json
import sys

from dissect.btrfs import Btrfs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a Btrfs image")
    parser.add_argument("--bytenr", type=lambda value: int(value, 0), help="Logical start offset")
    parser.add_argument("--length", type=lambda value: int(value, 0), help="Logical range length")
    parser.add_argument(
        "--subvolume",
        action="append",
        type=int,
        default=[],
        help="Subvolume object ID to scan, repeat for multiple roots",
    )
    parser.add_argument("--list-subvolumes", action="store_true", help="Only list subvolume roots and exit")
    args = parser.parse_args()

    with open(args.image, "rb") as fh:
        fs = Btrfs(fh)

        if args.list_subvolumes:
            for subvolume in fs.subvolumes():
                print(
                    json.dumps(
                        {
                            "objectid": subvolume.objectid,
                            "path": subvolume.path,
                            "uuid": str(subvolume.uuid),
                            "root_bytenr": subvolume.tree.root_offset,
                            "generation": subvolume.tree.root_item.generation,
                        }
                    )
                )
            return 0

        if args.bytenr is None or args.length is None or not args.subvolume:
            parser.error("--bytenr, --length and at least one --subvolume are required")

        report = fs.find_extent_references(args.bytenr, args.length, args.subvolume)

    print(json.dumps(report.to_dict(), indent=2))

    if report.coverage_gaps:
        print("\nINCOMPLETE: some selected roots or scan phases could not be traversed", file=sys.stderr)
        for gap in report.coverage_gaps:
            print(f"  {gap.kind} (subvolume {gap.subvolume_objectid}): {gap.reason}", file=sys.stderr)
    if report.unselected:
        print(
            "\nNote: these roots exist but were not selected, so their references are unknown: "
            + ", ".join(f"{scope.objectid} ({scope.path})" for scope in report.unselected),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
