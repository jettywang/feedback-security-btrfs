# dissect.btrfs

A Dissect module implementing a parser for the Btrfs file system, a commonly used Linux filesystem. For more
information, please see [the documentation](https://docs.dissect.tools/en/latest/projects/dissect.btrfs/index.html).

## Requirements

This project is part of the Dissect framework and requires Python.

Information on the supported Python versions can be found in the Getting Started section of [the
documentation](https://docs.dissect.tools/en/latest/index.html#getting-started).

## Installation

`dissect.btrfs` is available on [PyPI](https://pypi.org/project/dissect.btrfs/).

```bash
pip install dissect.btrfs
```

This project decompresses LZO compressed file systems and can use a faster, native (Rust-based) LZO implementation,
instead of a slower pure Python implementation. Both implementations are provided by `dissect.util`.
Pre-build wheels are available for most common platforms and the native implementation will automatically be used if
available.
In the rare case that a pre-build wheel is not available, please refer to `dissect.util` for build instructions.

## Build and test instructions

This project uses `tox` to build source and wheel distributions. Run the following command from the root folder to build
these:

```bash
tox -e build
```

The build artifacts can be found in the `dist/` directory.

`tox` is also used to run linting and unit tests in a self-contained environment. To run both linting and unit tests
using the default installed Python version, run:

```bash
tox
```

For a more elaborate explanation on how to build and test the project, please see [the
documentation](https://docs.dissect.tools/en/latest/contributing/tooling.html).

## Multi-device forensic intake

Btrfs devices of one filesystem often carry superblocks from different
transactions when they are collected at different times, and a transaction can
span devices. Selecting "the newest superblock" and merging whatever devices are
present is therefore not enough to claim a consistent state: it can combine a
newer metadata root with older data or chunk mappings.

The library refuses the two dangerous merges by default:

- two supplied images that claim the same `devid`
- two chunk items that map the same logical range differently, or a stripe whose
  `dev_uuid` does not match the supplied device

Use `dissect.btrfs.examine` before opening a filesystem to record what a
collection can and cannot prove:

```python
from dissect.btrfs import Btrfs, examine, verify_file_data

report = examine(images)
print(report.describe())          # verdict: proven / unverified / rejected

if report.verdict.value != "rejected":
    fs = Btrfs(images)
    print(verify_file_data(fs, "path/to/file"))
```

A `proven` verdict means every referenced member is present and every compared
chunk mapping agrees. It does **not** mean the directory tree is a consistent
snapshot: only reading every reachable tree block against its checksum and the
generation recorded by its parent establishes that, which `Btrfs` does while
parsing. A filesystem can be perfectly parseable and still be `unverified` when a
mirror is missing.

Intake rules in short:

- refuse a set with a duplicate `devid`, regardless of generation
- refuse a set whose members disagree on a chunk mapping or a stripe's `dev_uuid`
- treat a missing mirror as `unverified`, never as `proven`
- record per-member generations; different generations mean the collection is
  not one point in time, even when every block is individually valid
- verify tree blocks and file data separately; neither substitutes for the other

`Btrfs(..., strict=True)` verifies every reachable tree block's checksum,
`bytenr`, fsid, owner, level and the generation recorded by its parent pointer.
Use `strict=False` only for damaged media, and treat every result from it as
unverified.

## Contributing

The Dissect project encourages any contribution to the codebase. To make your contribution fit into the project, please
refer to [the development guide](https://docs.dissect.tools/en/latest/contributing/developing.html).

## Copyright and license

Dissect is released as open source by Fox-IT (<https://www.fox-it.com>) part of NCC Group Plc
(<https://www.nccgroup.com>).

Developed by the Dissect Team (<dissect@fox-it.com>) and made available at <https://github.com/fox-it/dissect>.

License terms: AGPL3 (<https://www.gnu.org/licenses/agpl-3.0.html>). For more information, see the LICENSE file.
