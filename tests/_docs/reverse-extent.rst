Reverse extent references
=========================

``Btrfs.find_extent_references(bytenr, length, subvolumes)`` answers a narrow forensic question:
which files in explicitly selected subvolume roots still reference a given Btrfs **logical** disk
range? It is read-only and never mounts or repairs the filesystem.

Why a range can map to several files
------------------------------------

Btrfs copy-on-write can make several file extent items point at the same physical extent. A file
extent item is keyed by inode and file offset, and records:

* ``disk_bytenr`` / ``disk_num_bytes`` — the physical extent allocation,
* ``offset`` — where this record starts inside that extent,
* ``num_bytes`` — how many file bytes the record exposes,
* ``compression`` — whether the extent is compressed.

Two snapshots can therefore expose different slices of one allocation, and one allocation can be
referenced by many inodes. The reverse lookup intersects the queried disk range with each record's
allocation and translates the intersection back into file offsets.

What the result contains
------------------------

Each reference keeps its provenance, so a finding can be re-checked against the image:

* subvolume object ID, resolved path, UUID, root node address and root generation,
* inode number, inode generation/transid and inode size,
* all paths of the inode,
* the extent data key offset, its generation, and the Btrfs leaf block that contains it,
* the physical extent, the disk overlap, the exposed file slice and the byte-accurate file range.

``precision`` is ``byte`` when the exact file bytes depending on the queried disk bytes are known.
It is ``extent`` for compressed extents: Btrfs compresses a whole extent, and individual compressed
bytes cannot be attributed to individual uncompressed bytes. For those records ``file_start`` and
``file_end`` describe the whole decoded slice, not a byte-exact sub-range.

The report also includes ``allocations`` from the extent tree. Each entry records ``refs``, the
filesystem's own reference count for that allocation. This is independent corroboration: a count
of 2 matches two file records, while a count of 1 shows a private allocation even if the content is
byte-identical to another file.

Coverage and completeness
-------------------------

The operation only scans the roots you name. It never implies "all files in the image".

* ``coverage_gaps`` lists roots that could not be opened or fully walked, and root enumeration
  failures. Any gap makes ``complete`` false.
* ``unselected`` lists subvolume roots that exist but were not selected. Any entry makes
  ``exhaustive`` false.
* ``stats`` records how many extent items were inspected, and counts inline and sparse records
  that cannot be mapped, plus preallocated records that were reported but may read as zeros.

Only ``exhaustive == True`` supports a statement that no other *discovered* subvolume root
references the range. Even then the claim is about file extent items in those roots, not about
every byte of the image.

Complete example: suspicious range to two snapshot files
--------------------------------------------------------

The ``btrfs-subvolume-snapshot`` fixture has a large file shared between a subvolume and its
snapshot. The suspicious range starts at logical ``0x4500000``::

    $ python examples/reverse_extent.py snapshot.bin --bytenr 0x4500000 --length 0x1000 \
          --subvolume 256 --subvolume 257

The report contains two references:

* subvolume ``256`` (``subvol``), inode ``259``, path ``subvol/large.txt``,
  file range ``0..4096``, ``precision="byte"``.
* subvolume ``257`` (``subvol-snapshot``), inode ``259``, path ``subvol-snapshot/large.txt``,
  file range ``0..4096``, ``precision="byte"``.

Both records reference the same allocation ``0x4500000`` of length ``0x501000``. The extent tree
item for that allocation has ``refs == 2``, matching the two file records. Forward reads of
``file_start..file_end`` in both files return the same bytes, and those bytes equal a direct read
of the queried logical range.

Subvolume ``5`` appears under ``unselected``, so ``exhaustive`` is false: references from that root
were not inspected.

What this proves and what it does not
-------------------------------------

Proves:

* both inodes' extent items declare the same physical allocation for the queried bytes,
* the queried file offsets depend on those bytes, and a forward read reproduces them,
* the extent tree independently accounts for two references to that allocation,
* both findings are attributable to exact subvolume, inode and Btree leaf provenance.

Does not prove:

* that the files have identical content overall — only the mapped slices are covered,
* that the data originated in either file, or which snapshot was created first,
* that no other file references the range — roots outside the selection and any reported coverage
  gaps are unknown,
* that equal-looking content elsewhere is shared: identical bytes in a separately allocated extent
  have a different ``disk_bytenr`` and are not reported,
* anything about deleted or partially overwritten history that is no longer described by current
  metadata,
* byte-level attribution inside a compressed extent beyond the decoded slice,
* that a reported preallocated extent actually contains the bytes a forward read returns.
