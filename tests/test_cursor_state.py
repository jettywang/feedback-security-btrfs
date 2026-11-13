"""Model based state machine tests for :class:`dissect.btrfs.tree.Cursor`.

The tests build a small multi level tree from handcrafted on disk nodes and replay fixed and generated
operation sequences against an independent ordering model. Every step asserts both the returned records and
the resulting cursor position, so a stale path or a bad boundary transition cannot hide behind a fresh cursor.
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterable

import pytest

from dissect.btrfs.btrfs import Btrfs
from dissect.btrfs.c_btrfs import c_btrfs
from dissect.btrfs.tree import BTree, Cursor

HEADER_SIZE = len(c_btrfs.btrfs_header)
ITEM_SIZE = len(c_btrfs.btrfs_item)


def _pack_key(key: tuple[int, int, int]) -> bytes:
    return struct.pack("<QBQ", *key)


def _header(bytenr: int, level: int, nritems: int, owner: int = 5) -> bytes:
    return c_btrfs.btrfs_header(
        csum=b"\x00" * c_btrfs.BTRFS_CSUM_SIZE,
        fsid=b"\x00" * c_btrfs.BTRFS_FSID_SIZE,
        bytenr=bytenr,
        flags=0,
        chunk_tree_uuid=b"\x00" * c_btrfs.BTRFS_UUID_SIZE,
        generation=1,
        owner=owner,
        nritems=nritems,
        level=level,
    ).dumps()


def _leaf(bytenr: int, records: list[tuple[tuple[int, int, int], bytes]]) -> bytes:
    data_start = HEADER_SIZE + ITEM_SIZE * len(records)
    items = []
    payloads = []
    offset = data_start
    for key, payload in records:
        items.append(_pack_key(key) + struct.pack("<II", offset - HEADER_SIZE, len(payload)))
        payloads.append(payload)
        offset += len(payload)

    return _header(bytenr, 0, len(records)) + b"".join(items) + b"".join(payloads)


def _node(bytenr: int, level: int, entries: list[tuple[tuple[int, int, int], int]]) -> bytes:
    return _header(bytenr, level, len(entries)) + b"".join(
        _pack_key(key) + struct.pack("<QQ", blockptr, 1) for key, blockptr in entries
    )


def _make_tree() -> tuple[BTree, list[tuple[int, int, int]], dict[tuple[int, int, int], bytes]]:
    """Build a three level tree with partially filled leaves and keys sharing object IDs."""
    leaves = [
        [((5, 1, 0), b"a"), ((5, 1, 3), b"b"), ((5, 1, 9), b"c")],
        [((5, 2, 0), b"d"), ((5, 2, 4), b"e"), ((6, 1, 0), b"f")],
        [((6, 1, 2), b"g"), ((6, 1, 5), b"h"), ((6, 2, 0), b"i")],
        [((6, 2, 1), b"j"), ((7, 1, 0), b"k"), ((7, 1, 7), b"l")],
        [((8, 1, 0), b"m"), ((8, 1, 2), b"n"), ((8, 2, 0), b"o")],
    ]
    nodes = {0x4000 + i: _leaf(0x4000 + i, records) for i, records in enumerate(leaves)}
    nodes[0x2000] = _node(0x2000, 1, [(leaves[0][0][0], 0x4000), (leaves[1][0][0], 0x4001), (leaves[2][0][0], 0x4002)])
    nodes[0x3000] = _node(0x3000, 1, [(leaves[3][0][0], 0x4003), (leaves[4][0][0], 0x4004)])
    nodes[0x1000] = _node(0x1000, 2, [(leaves[0][0][0], 0x2000), (leaves[3][0][0], 0x3000)])

    class FakeBtrfs:
        def _read_node(self, address: int) -> bytes:
            return nodes[address]

    keys = [key for leaf in leaves for key, _ in leaf]
    data = {key: payload for leaf in leaves for key, payload in leaf}
    return BTree(FakeBtrfs(), root_offset=0x1000), keys, data


TREE, KEYS, DATA = _make_tree()


def _compare(key: tuple[int, int, int], objectid: int | None, type: int | None, offset: int | None) -> int:
    """Independent implementation of B-tree key ordering."""
    if objectid is not None:
        if key[0] > objectid:
            return 1
        if key[0] < objectid:
            return -1
    if type is not None:
        if key[1] > type:
            return 1
        if key[1] < type:
            return -1
    if offset is not None:
        if key[2] > offset:
            return 1
        if key[2] < offset:
            return -1
    return 0


def _lower_bound(target: tuple[int | None, int | None, int | None]) -> int | None:
    """Return the first index whose key is greater than or equal to ``target``."""
    for index, key in enumerate(KEYS):
        if _compare(key, *target) >= 0:
            return index
    return None


@dataclass
class Op:
    """A single cursor operation in a replayed sequence."""

    kind: str
    args: tuple = ()
    label: str = ""

    def __str__(self) -> str:
        """Render the operation as it appears in a failing operation chain."""
        if self.label:
            return self.label
        return self.kind if not self.args else f"{self.kind}{self.args}"


@dataclass
class Model:
    """Independent state model: a position in the sorted key list, or ``None`` when unpositioned."""

    index: int | None = None


class ReplayFailure(AssertionError):
    """Raised with the shortest prefix of operations that exposes a divergence."""

    def __init__(self, exc: AssertionError, history: list[Op]) -> None:
        """Wrap ``exc`` and append the operation chain that triggered it."""
        self.exc = exc
        self.history = history
        chain = " -> ".join(str(op) for op in history)
        super().__init__(f"{exc}\nShortest failing operation chain ({len(history)} ops): {chain}")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _position(cursor: Cursor) -> tuple[int, int, int] | None:
    try:
        key = cursor.item().key
    except ValueError:
        return None
    return (key.objectid, key.type, key.offset)


class Transitions:
    """Apply operations to both the real cursor and the independent model, asserting after each step."""

    def __init__(self) -> None:
        """Start from an unpositioned cursor with a matching model."""
        self.cursor = TREE.cursor()
        self.model = Model()
        self.history: list[Op] = []
        self._factories: dict[str, Callable[[Op], None]] = {
            "search": self._search,
            "next": self._next,
            "prev": self._prev,
            "first": self._first,
            "last": self._last,
            "reset": self._reset,
            "iter": self._iter,
            "walk": self._walk,
        }

    def run(self, operations: Iterable[Op]) -> None:
        """Replay ``operations``, raising :class:`ReplayFailure` on the first divergence."""
        for operation in operations:
            self.history.append(operation)
            try:
                self._factories[operation.kind](operation)
            except AssertionError as exc:
                raise ReplayFailure(exc, list(self.history)) from None
            except Exception as exc:  # model and cursor must agree on raising
                raise ReplayFailure(AssertionError(f"unexpected {exc!r}"), list(self.history)) from None

    def _assert_position(self) -> None:
        actual = _position(self.cursor)
        if self.model.index is None:
            _expect(actual is None, f"expected unpositioned cursor, got {actual}")
        else:
            _expect(actual == KEYS[self.model.index], f"expected position {KEYS[self.model.index]}, got {actual}")

    def _assert_payload(self) -> None:
        _expect(_position(self.cursor) is not None, "expected a positioned cursor")
        item, payload = self.cursor.get()
        key = (item.key.objectid, item.key.type, item.key.offset)
        _expect(bytes(payload) == DATA[key], f"payload mismatch for {key}")

    def _search(self, op: Op) -> None:
        found = self.cursor.search(*op.args)
        index = _lower_bound(op.args)
        if index is None:
            _expect(not found, "search should report a miss past the end")
            self._assert_position()
            return

        _expect(found, f"search{op.args} should hit {KEYS[index]}")
        self.model.index = index
        self._assert_position()
        self._assert_payload()

    def _next(self, _op: Op) -> None:
        state = self.model.index
        try:
            self.cursor.next()
        except ValueError:
            _expect(state is None or state == len(KEYS) - 1, "next should only fail at the end")
            self.model.index = state
            self._assert_position()
            # A failed next must not desynchronize the cursor from the model on the next query.
            return

        _expect(state is not None and state < len(KEYS) - 1, "next should fail at the end")
        self.model.index = state + 1
        self._assert_position()

    def _prev(self, _op: Op) -> None:
        state = self.model.index
        try:
            self.cursor.prev()
        except ValueError:
            _expect(state is None or state == 0, "prev should only fail at the start")
            self.model.index = state
            self._assert_position()
            return

        _expect(state is not None and state > 0, "prev should fail at the start")
        self.model.index = state - 1
        self._assert_position()

    def _first(self, _op: Op) -> None:
        self.cursor.first()
        self.model.index = 0
        self._assert_position()

    def _last(self, _op: Op) -> None:
        self.cursor.last()
        self.model.index = len(KEYS) - 1
        self._assert_position()

    def _reset(self, _op: Op) -> None:
        self.cursor.reset()
        self.model.index = None
        self._assert_position()

    def _iter(self, op: Op) -> None:
        objectid, type, offset, ignore_offset = op.args
        expected = []
        index = _lower_bound((objectid, type, offset))
        if index is not None:
            for key in KEYS[index:]:
                if _compare(key, objectid, type, None if ignore_offset else offset) != 0:
                    break
                expected.append(key)

        actual = [(item.key.objectid, item.key.type, item.key.offset) for item, _ in self.cursor.iter(*op.args)]
        _expect(actual == expected, f"iter{op.args} yielded {actual}, expected {expected}")
        _expect([bytes(data) for _, data in self.cursor.iter(*op.args)] == [DATA[key] for key in expected], "payloads")

        # ``iter`` leaves the cursor on the item that terminated the iteration: the lower bound itself when the
        # range is empty, or the item after the last match (the last item when the range runs to the end).
        if index is None:
            # A miss leaves the cursor untouched, on whatever item it was before.
            pass
        elif expected:
            self.model.index = min(index + len(expected), len(KEYS) - 1)
        else:
            self.model.index = index
        self._assert_position()

    def _walk(self, op: Op) -> None:
        objectid, type, offset = op.args
        expected = [key for key in KEYS if _compare(key, objectid, type, offset) == 0]
        actual = [(item.key.objectid, item.key.type, item.key.offset) for item, _ in self.cursor.walk(*op.args)]
        _expect(actual == expected, f"walk{op.args} yielded {actual}, expected {expected}")
        self.model.index = len(KEYS) - 1
        self._assert_position()


def _replay(operations: Iterable[Op]) -> None:
    Transitions().run(operations)


def _fresh(operation: Op) -> None:
    _replay([operation])


SELECTORS: list[tuple[int | None, int | None, int | None]] = [
    *KEYS,
    (5, 1, 1),
    (5, 1, 4),
    (5, 1, 10),
    (5, 2, 1),
    (5, 2, 5),
    (6, 1, 1),
    (6, 1, 3),
    (6, 1, 6),
    (6, 2, 2),
    (7, 1, 1),
    (7, 1, 8),
    (8, 1, 1),
    (8, 2, 1),
    (4, 1, 0),
    (9, 1, 0),
    (0, 1, 0),
    (5, 0, 0),
    (5, None, None),
    (6, None, None),
    (8, None, None),
    (9, None, None),
    (5, 2, None),
    (6, 2, None),
    (None, None, None),
]


@pytest.mark.parametrize("selector", SELECTORS)
def test_search_from_unpositioned(selector: tuple[int | None, int | None, int | None]) -> None:
    """Search results must be absolute: exact hits, lower bounds and end-of-tree misses."""
    Transitions().run([Op("reset"), Op("search", selector)])


def test_search_distinguishes_exact_hit_and_lower_bound() -> None:
    """An exact hit and a lower-bound hit must be distinguishable by comparing the resulting key."""
    cursor = TREE.cursor()

    assert cursor.search(6, 1, 2) is True
    item = cursor.item()
    assert (item.key.objectid, item.key.type, item.key.offset) == (6, 1, 2)

    assert cursor.search(6, 1, 3) is True
    item = cursor.item()
    assert (item.key.objectid, item.key.type, item.key.offset) == (6, 1, 5)

    # A key one past the highest key at a leaf boundary lands on the first key of the next leaf.
    assert cursor.search(5, 1, 10) is True
    item = cursor.item()
    assert (item.key.objectid, item.key.type, item.key.offset) == (5, 2, 0)

    # A miss past the end of the entire tree reports ``False`` and does not raise.
    assert cursor.search(9, 1, 0) is False


def test_iter_ignore_offset_only_affects_initial_position() -> None:
    """With ``ignore_offset`` the offset only positions the cursor, so iteration runs to the end of the object ID."""
    cursor = TREE.cursor()
    assert [item.key.offset for item, _ in cursor.iter(5, 1, 3, ignore_offset=True)] == [3, 9]

    cursor = TREE.cursor()
    assert [item.key.offset for item, _ in cursor.iter(5, 1, 3)] == [3]

    cursor = TREE.cursor()
    assert [item.key.offset for item, _ in cursor.iter(6, 2, 0, ignore_offset=True)] == [0, 1]

    cursor = TREE.cursor()
    assert [item.key.offset for item, _ in cursor.iter(6, 2, 0)] == [0]

    # The initial position is still the lower bound for the given offset, even though it is ignored later.
    cursor = TREE.cursor()
    assert [item.key.offset for item, _ in cursor.iter(5, 1, 5, ignore_offset=True)] == [9]


def test_search_is_not_relative_to_current_leaf() -> None:
    """Regression: a search below the current position used to stay in the current leaf and miss earlier keys."""
    cursor = TREE.cursor()
    assert cursor.search(8, 1, 2) is True
    assert cursor.search(5, 1, 0) is True
    item = cursor.item()
    assert (item.key.objectid, item.key.type, item.key.offset) == (5, 1, 0)


def test_partial_key_search_reaches_previous_leaf() -> None:
    """Regression: matching only on object ID used to skip equal keys in the previous leaf."""
    cursor = TREE.cursor()
    assert cursor.search(5, 2, None) is True
    item = cursor.item()
    assert (item.key.objectid, item.key.type, item.key.offset) == (5, 2, 0)


@pytest.mark.parametrize("selector", SELECTORS)
def test_search_is_independent_of_history(selector: tuple[int | None, int | None, int | None]) -> None:
    """A stale path from an earlier query must not change the result of a later search."""
    for warmup in [(4, 1, 0), (5, 1, 9), (8, 1, 0), (9, 1, 0)]:
        _replay([Op("search", warmup), Op("search", selector), Op("next")])


@pytest.mark.parametrize(
    ("selector", "ignore_offset"),
    [
        ((5, 1, 1), False),  # empty range inside a leaf
        ((5, 1, 1), True),
        ((5, 1, 10), False),  # empty range between leaves/objects
        ((6, 1, 0), True),
        ((8, 1, 0), True),  # range that crosses a leaf boundary
        ((5, 2, 0), True),
        ((9, 1, 0), False),  # range beyond the end of the tree
        ((9, 1, 0), True),
    ],
)
def test_range_iterator_semantics(selector: tuple[int | None, int | None, int | None], ignore_offset: bool) -> None:
    """``iter`` positions on a lower bound and then either honours or ignores the offset."""
    _replay([Op("iter", (*selector, ignore_offset)), Op("reset"), Op("search", selector), Op("next")])
    _fresh(Op("iter", (*selector, ignore_offset)))


@pytest.mark.parametrize("selector", SELECTORS)
def test_walk_semantics(selector: tuple[int | None, int | None, int | None]) -> None:
    """``walk`` always rewinds to the first leaf item, regardless of the previous cursor position."""
    _replay([Op("search", (8, 2, 0)), Op("walk", selector), Op("reset")])


@pytest.mark.parametrize(
    ("position", "operation"),
    [
        ("first", "prev"),
        ("first", "next"),
        ("last", "next"),
        ("last", "prev"),
        ("middle", "prev"),
        ("middle", "next"),
    ],
)
def test_boundary_movement_keeps_position(position: str, operation: str) -> None:
    """Failing navigation at a tree boundary leaves the cursor where it was."""
    start = {"first": Op("first"), "last": Op("last"), "middle": Op("search", (6, 1, 5))}[position]
    _replay([start, Op(operation), Op("search", (5, 1, 0)), Op("next")])


def test_multi_level_leaf_boundaries() -> None:
    """Leaf edges must be crossed correctly in both directions across all levels of the tree."""
    operations = [Op("first")]
    operations += [Op("next") for _ in range(len(KEYS) - 1)]
    operations += [Op("next"), Op("next")]
    operations += [Op("prev") for _ in range(len(KEYS) - 1)]
    operations += [Op("prev"), Op("prev")]
    operations += [Op("last")]
    operations += [Op("prev") for _ in range(len(KEYS) - 1)]
    _replay(operations)


def test_unpositioned_cursor_queries() -> None:
    """After ``reset`` only positioning operations may be used."""
    _replay([Op("search", (6, 1, 5)), Op("reset"), Op("search", (5, 1, 0)), Op("reset"), Op("first")])
    _replay([Op("search", (6, 1, 5)), Op("reset"), Op("iter", (5, 1, 0, True)), Op("reset"), Op("last")])

    cursor = TREE.cursor()
    cursor.reset()
    with pytest.raises(ValueError, match="Cursor not set"):
        cursor.get()
    with pytest.raises(ValueError, match="Cursor not set"):
        cursor.item()
    # ``iter`` is a positioning operation, so it works on an unpositioned cursor.
    assert [(item.key.objectid, item.key.type, item.key.offset) for item, _ in cursor.iter(5, 1, 0)] == [(5, 1, 0)]


def test_failed_search_does_not_poison_cursor() -> None:
    """A miss must not leave the cursor in a state where the next query explodes."""
    _replay([Op("search", (9, 1, 0)), Op("search", (5, 1, 0))])
    _replay([Op("search", (9, 1, 0)), Op("first"), Op("search", (9, 1, 0)), Op("iter", (8, 1, 0, True))])


def test_fixed_operation_sequences() -> None:
    """Replay a curated interleaving of every operation kind."""
    _replay(
        [
            Op("search", (5, 1, 3)),
            Op("next"),
            Op("iter", (5, 1, 0, True)),
            Op("prev"),
            Op("search", (6, 1, 5)),
            Op("last"),
            Op("search", (9, 1, 0)),
            Op("reset"),
            Op("first"),
            Op("prev"),
            Op("next"),
            Op("walk", (5, 1, None)),
            Op("search", (6, 2, None)),
            Op("iter", (6, 2, None, True)),
            Op("prev"),
            Op("reset"),
            Op("search", (0, 0, 0)),
            Op("next"),
        ]
    )


def test_random_state_machine() -> None:
    """Generate deterministic pseudo-random sequences and compare against the model."""
    random_ops = [Op("search", selector) for selector in SELECTORS]
    random_ops += [Op("iter", (*selector, ignore)) for selector in SELECTORS for ignore in (False, True)]
    random_ops += [Op("walk", selector) for selector in SELECTORS]
    random_ops += [Op(kind) for kind in ("next", "prev", "first", "last", "reset")]

    rng = random.Random(0xB7F5)
    for _ in range(40):
        length = rng.randrange(2, 20)
        _replay([rng.choice(random_ops) for _ in range(length)])


def test_shortest_failing_chain_is_reported() -> None:
    """A divergence must report the shortest prefix that reproduces it, not the full random sequence."""
    operations = [Op("first"), Op("next"), Op("search", (9, 1, 0)), Op("next"), Op("reset")]

    class BrokenCursor(Cursor):
        def search(self, *args: object) -> bool:
            if args == (9, 1, 0):
                raise AssertionError("injected failure")
            return super().search(*args)

    transitions = Transitions()
    transitions.cursor = BrokenCursor(TREE)

    with pytest.raises(ReplayFailure) as exc_info:
        transitions.run(operations)

    message = str(exc_info.value)
    assert "injected failure" in message
    assert "first -> next -> search(9, 1, 0)" in message
    assert "reset" not in message


def test_search_agrees_with_reference_comparator() -> None:
    """The cursor search result must match the independent comparator for every selector."""
    for selector in SELECTORS:
        cursor = TREE.cursor()
        found = cursor.search(*selector)
        index = _lower_bound(selector)
        assert found == (index is not None)
        if index is not None:
            key = cursor.item().key
            assert (key.objectid, key.type, key.offset) == KEYS[index]
            assert _compare((key.objectid, key.type, key.offset), *selector) >= 0


def test_all_keys_reachable_by_iteration() -> None:
    """Full traversal must visit every key in order, including across every leaf boundary."""
    cursor = TREE.cursor()
    cursor.first()
    seen = []
    while True:
        item = cursor.item()
        seen.append((item.key.objectid, item.key.type, item.key.offset))
        try:
            cursor.next()
        except ValueError:
            break

    assert seen == KEYS
    assert seen == sorted(seen)


def test_real_image_queries_are_order_independent(btrfs_default: BinaryIO) -> None:
    """On a real image, a cursor that already traversed part of the tree must still find earlier records."""
    fs = Btrfs(btrfs_default)
    tree = fs.fs_tree.tree

    entry = fs.get("path/to/a/file.txt")

    cursor = tree.cursor()
    cursor.last()
    for _ in cursor.iter(c_btrfs.BTRFS_DIR_INDEX_KEY, ignore_offset=True):
        pass

    assert cursor.search(entry.inum, c_btrfs.BTRFS_INODE_ITEM_KEY, 0) is True
    item, data = cursor.get()
    assert (item.key.objectid, item.key.type, item.key.offset) == (entry.inum, c_btrfs.BTRFS_INODE_ITEM_KEY, 0)

    cursor.last()
    assert [(item.key.objectid) for item, _ in cursor.iter(entry.inum, c_btrfs.BTRFS_INODE_ITEM_KEY, 0)] == [entry.inum]
