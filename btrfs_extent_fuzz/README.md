# Btrfs compressed-extent resource-abuse fuzzer

A small, self-contained fuzzing harness for the compressed data paths of
`dissect.btrfs`. It targets the two places a Btrfs image reaches a decompressor:

| Entry point | Code path |
| --- | --- |
| Inline file data | `INode.open()` → `decode_extent()`, returned as `BufferedStream` |
| Regular file extent | `INode.open()` → `ExtentStream._read()` → `decode_extent()` |

Every case is a complete, minimal Btrfs image, so the real parser and the real
decompressor run — no test doubles and no in-memory shortcuts.

## What is mutated

Starting from six fixed seeds (zlib / LZO / zstd × inline / regular), three
coupled axes are mutated, 1–3 steps per case, with a per-case RNG derived from
`(root_seed, seed_id, case_index)`:

* **Segment length** (LZO) — `lzo-segment:{zero,huge,truncate,overlap}` rewrites
  a segment-size word, plus `lzo-total:*` for the outer total-size header.
* **Frame tail** (zlib, zstd) — truncate, append junk, concatenate a second
  frame, or flip the trailing checksum (`tail:*`).
* **Frame declaration** (zstd) — widen or rewrite the frame content-size
  descriptor (`zstd-header:*`).
* **Extent declaration** (all codecs) — `declared:*` mutates `ram_bytes`,
  `num_bytes`, `disk_num_bytes`, the inode size and the extent offset,
  including `1 << 64 - 1` sentinels, independently of the payload.

Fixed seeds, so a run is reproducible byte for byte:

```
python -m btrfs_extent_fuzz run --seed 0xB7F5 --cases 40
```

## Budgets and isolation

Each case runs in a forked child. Load-bearing values are recorded per case and
can be changed from the CLI:

* `--rss-mb` (default 96) — peak RSS of the decoding child.
* `--wall` (default 5 s) — hard wall-clock cap; the child is `SIGKILL`ed.
* `--cpu` (default 4 s) — `RLIMIT_CPU` inside the child.
* `--max-output-mb` (default 64) — absolute cap on bytes one decode may
  produce. Independent of the declaration (which an attacker can set to `2**64`)
  and of RSS (a decoder can stream output), so it still bounds memory-free
  amplification.

A normal image peaks around 18 MiB here, so the default RSS budget leaves room
for a genuinely highly-compressed file while still catching a decoder that
materialises a large extent. Peak RSS is read from `ru_maxrss` of the reaped
child (normalised between macOS bytes and Linux KiB). If the sandbox refuses
`RLIMIT_CPU`, the wall-clock and RSS budgets still apply.

## Verdicts

The classes are deliberately disjoint, and compression ratio is never evidence:

| Verdict | Meaning |
| --- | --- |
| `ok` | Decoded inside the declared bound and inside every budget. |
| `decode-reject` | Decoder or parser refused the sample. Expected fuzz noise, **not** a vulnerability. |
| `output-over-declared` | The decoder produced more bytes than the extent declares. |
| `resource-overrun` | Output stayed in range but the output, CPU, wall or RSS budget was exceeded. |
| `crash` | Child died on an unexpected signal. |

Because Btrfs streams truncate output to the inode/extent size, a bomb looks
harmless through `read()` alone. The child therefore instruments
`dissect.btrfs.stream.decode_extent` — the exact function both entry points use —
and records true decoder output, while still driving the public API. Containment
is checked against a bound derived from the on-disk `ram_bytes` and
`offset + num_bytes`, whichever is larger, plus 8 bytes of slack.

A truthfully-declared high-ratio file is therefore `ok`; only output escaping
the declaration is a finding. Native Rust LZO panics
(`pyo3_runtime.PanicException`) are recorded via the `native_panic` flag instead
of being mistaken for a container escape.

## Minimal samples and replay

Every finding is shrunk in place (payload truncation, keeping the verdict) and
written to `samples/<case>.min.json`:

```json
{
  "case_id": "zstd-regular-0002",
  "codec": "zstd", "entry": "regular",
  "declared": 512, "inode_size": 1025, "disk_num_bytes": 22, "disk_offset": 0,
  "payload_hex": "28b52ffd...", "mutations": ["declared:shrink-declared"],
  "verdict": "output-over-declared", "image_sha256": "..."
}
```

Replay is deterministic and takes an independent budget:

```
python -m btrfs_extent_fuzz replay samples/zstd-regular-0002.min.json
```

## Detector controls

`python -m btrfs_extent_fuzz controls` runs the proof that the detector fires
and does not misfire:

* `positive-inflate-zstd` — valid frame decoding to 2 MiB while declaring 1 KiB;
  must be `output-over-declared`.
* `positive-inflate-zstd-inline` — same through the inline entry point.
* `negative-high-ratio-zstd`, `negative-high-ratio-zlib` — ~200× and ~10×
  ratios with truthful declarations; must be `ok`.
* `negative-bomb-truthful-declaration` — 128 MiB decoded from ~4 KiB, declared
  honestly; must be `ok` when the budget allows it, proving ratio alone is not
  evidence.
* `positive-resource-overrun` — truthful 160 MiB decode against a 96 MiB RSS
  budget; must be `resource-overrun`, and `ok` once the budget is raised.

## Layout

| File | Role |
| --- | --- |
| `corpus.py` | Fixed seeds, mutation axes, deterministic sampling |
| `image.py` | Minimal Btrfs image builder (superblock, chunk/root/FS trees) |
| `runner.py` | Forked child, budgets, decoder instrumentation, verdicts |
| `report.py` | Campaign, minimisation, sample files, replay |
| `controls.py` | Positive/negative detector controls |
| `tests/test_extent_fuzz.py` | Regression tests for all of the above |

