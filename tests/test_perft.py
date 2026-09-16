"""Perft correctness gate (architecture.md §13, §14).

This is the single most important correctness gate in the whole project:
`perft(board, depth)` must match the published reference node counts
*exactly* for every position/depth pair below. A wrong count means some
legality rule (pin, check evasion, castling-rights bookkeeping, en passant,
promotion) was generated incorrectly somewhere in `movegen.py`.

Depths 1-4 run on every default `pytest` invocation, matching §14's "Depths
1-4 are run on every `pytest` invocation" rule. Depths 5-6 are marked
`@pytest.mark.slow` and are skipped by default; run them explicitly with:

    pytest -m slow tests/test_perft.py
    pytest -m "" tests/test_perft.py     # (or) run the whole suite including slow

If a position/depth pair ever fails, `divide(board, depth)` bisects the
mismatch down to the offending root move: compare its per-move breakdown
against a reference engine's `divide` output at the same depth, find the one
root move whose subtree count disagrees, and recurse `divide` one ply deeper
inside that subtree.
"""

from __future__ import annotations

import pytest

from chessengine.fen import parse_fen
from chessengine.perft import perft

# --- §14 reference positions -------------------------------------------------
#
# Standard positions with published node counts (Chess Programming Wiki's
# "Perft Results" page). Each entry is (position_id, name, fen, {depth: nodes}).

STARTPOS = (
    1,
    "Startpos",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609, 6: 119060324},
)

KIWIPETE = (
    2,
    "Kiwipete",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    {1: 48, 2: 2039, 3: 97862, 4: 4085603, 5: 193690690},
)

POSITION_3 = (
    3,
    "Position 3",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624, 6: 11030083},
)

POSITION_4 = (
    4,
    "Position 4",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    {1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292},
)

POSITION_5 = (
    5,
    "Position 5",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    {1: 44, 2: 1486, 3: 62379, 4: 2103487},
)

POSITION_6 = (
    6,
    "Position 6",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    {1: 46, 2: 2079, 3: 89890, 4: 3894594},
)

ALL_POSITIONS = (STARTPOS, KIWIPETE, POSITION_3, POSITION_4, POSITION_5, POSITION_6)

# Depths 1-4 run by default; depths 5-6 are marked slow (§14: "Depths 1-4 are
# run on every `pytest` invocation; depth 5-6 rows are marked
# @pytest.mark.slow").
FAST_DEPTHS = (1, 2, 3, 4)
SLOW_DEPTHS = (5, 6)


def _cases(depths: tuple[int, ...]) -> list[tuple[int, str, str, int, int]]:
    """Flatten ALL_POSITIONS into one (pos_id, name, fen, depth, expected)
    tuple per (position, depth) pair whose depth is in `depths` and that the
    position's reference table actually lists (not every position is
    published to depth 5/6, per §14's table)."""
    cases = []
    for pos_id, name, fen, depth_nodes in ALL_POSITIONS:
        for depth in depths:
            if depth in depth_nodes:
                cases.append((pos_id, name, fen, depth, depth_nodes[depth]))
    return cases


def _case_id(case: tuple[int, str, str, int, int]) -> str:
    pos_id, name, _fen, depth, expected = case
    return f"pos{pos_id}-{name.replace(' ', '_')}-depth{depth}-nodes{expected}"


FAST_CASES = _cases(FAST_DEPTHS)
SLOW_CASES = _cases(SLOW_DEPTHS)


@pytest.mark.parametrize("case", FAST_CASES, ids=[_case_id(c) for c in FAST_CASES])
def test_perft_fast(case: tuple[int, str, str, int, int]) -> None:
    """Depths 1-4 for every §14 reference position: run on every invocation."""
    _pos_id, _name, fen, depth, expected = case
    board = parse_fen(fen)
    assert perft(board, depth) == expected


@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW_CASES, ids=[_case_id(c) for c in SLOW_CASES])
def test_perft_slow(case: tuple[int, str, str, int, int]) -> None:
    """Depths 5-6 for the §14 reference positions that publish them.

    Skipped by default; run explicitly with `pytest -m slow` (or include
    them alongside the rest of the suite with `pytest -m "" tests/`).
    """
    _pos_id, _name, fen, depth, expected = case
    board = parse_fen(fen)
    assert perft(board, depth) == expected


def test_perft_depth_zero_is_one_for_every_position() -> None:
    """perft(board, 0) counts the current position itself as the single
    leaf, per perft.py's documented depth<=0 base case -- true regardless
    of which position it's called on."""
    for _pos_id, _name, fen, _depth_nodes in ALL_POSITIONS:
        board = parse_fen(fen)
        assert perft(board, 0) == 1


def test_perft_leaves_board_unchanged() -> None:
    """perft must make/unmake every move it plays, leaving the board exactly
    as it found it -- otherwise a bug here could silently corrupt every
    other test that reuses a Board across calls."""
    board = parse_fen(KIWIPETE[2])
    fen_before = board.to_fen()
    perft(board, 3)
    assert board.to_fen() == fen_before
    assert board.history == []
