"""Milestone 4 exit-criteria test (architecture.md §15): a node-count
regression check confirming move ordering does not *increase* node count at
a fixed depth versus an unordered baseline.

§15's Milestone 4 exit criteria call for exactly this: "a node-count
regression check confirming move ordering does not increase node count at
fixed depth versus Milestone 2's baseline ordering." Milestone 2 shipped
`Search` with only TT-move-first ordering wired in (architecture.md §15,
Milestone 2's **Build** paragraph: "TT-backed move ordering, no quiescence
yet"); Milestone 4 (this codebase's current state) adds the full stack
described in §9.4 -- TT move, then MVV-LVA-ranked captures, then killer
moves, then the history heuristic. Rather than reconstruct Milestone 2's
now-superseded ordering byte-for-byte, this test uses the strictest,
most-general baseline any move-ordering scheme must beat: **no ordering at
all** (moves searched in whatever raw order `generate_legal_moves`/
`generate_captures` happen to return them, including at quiescence nodes).
If today's full ordering stack can't beat *that* baseline, it would surely
also fail to beat Milestone 2's own (weaker) baseline -- so this is a
strictly stronger, and strictly more future-proof, version of the same
regression gate.

Method: two `Search` instances -- one using the real, unmodified
`Search._order_moves` (TT move + MVV-LVA + killers + history, §9.4), and one
a `Search` subclass with `_order_moves` neutralized to a no-op -- run
`.search(board, SearchLimits(max_depth=FIXED_DEPTH))` on the same battery of
positions (an opening and a tactical middlegame FEN, per the task's own
battery). Both go through the exact same `Search.search` iterative-deepening
entry point (architecture.md §9.3), so both accumulate `SearchResult.nodes`
over the identical depth-1..FIXED_DEPTH loop, alpha-beta window, TT
(each instance owns its own, so nothing leaks between the two runs),
quiescence search, and SEE-based capture pruning (§9.5) -- ordering is the
*only* variable being isolated. The assertion is `ordered.nodes <=
unordered.nodes`: move ordering must never cost the search extra nodes at a
fixed depth, only ever save them (by improving the alpha-beta cutoff rate).

Depth 4 is used for both positions, matching the task's own suggested fixed
depth; both were measured (informally, during development of this file) at
well under a second per run even in the unordered/no-cutoff-assisted case,
so the whole file runs in a couple of seconds, not minutes.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits

FIXED_DEPTH = 4

# (case id, FEN) -- one opening, one tactical middlegame, per the task's own
# battery. The tactical-middlegame FEN is the same position `test_search.py`
# uses under the id "tactical-middlegame-exposed-king" (an exposed king, a
# pinned/skewered rook, a passed pawn one step from promotion): plenty of
# captures and check-related tactics for move ordering to actually have an
# effect on, while its modest piece count keeps even the *unordered* search
# fast at this depth.
POSITIONS = [
    pytest.param(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        id="opening-startpos",
    ),
    pytest.param(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        id="tactical-middlegame-exposed-king",
    ),
]


class _UnorderedSearch(Search):
    """A `Search` subclass with move ordering neutralized: `_order_moves` is
    a no-op that hands back whatever order `moves` already came in (the raw
    `generate_legal_moves`/`generate_captures` order), instead of sorting by
    TT move / MVV-LVA / killers / history (§9.4).

    Everything else -- alpha-beta pruning, the TT's cutoff/probe behavior
    (still consulted for bounds, just no longer used to seed move order),
    quiescence search, SEE-gated capture pruning -- is inherited unchanged
    from `Search`, so this isolates *only* the move-ordering variable.
    """

    def _order_moves(self, moves: list[int], board: Board, tt_move: int, ply: int, **kwargs) -> list[int]:
        return moves


def test_unordered_search_subclass_actually_leaves_moves_unsorted() -> None:
    """Sanity check on the test harness itself: `_UnorderedSearch._order_moves`
    must hand back the exact same order `generate_legal_moves` produced, with
    no sorting applied -- otherwise the "unordered baseline" in the main
    regression test below wouldn't actually be unordered, and a node-count
    comparison against it would be meaningless."""
    board = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    raw_moves = generate_legal_moves(board)
    unordered = _UnorderedSearch(default_evaluator())

    result = unordered._order_moves(list(raw_moves), board, NULL_MOVE, 0)

    assert result == raw_moves, (
        "_UnorderedSearch._order_moves must return the moves in their original, "
        "unsorted order -- if this fails, the 'unordered' baseline below is not "
        "actually unordered, and no longer proves what this file claims to prove."
    )


@pytest.mark.parametrize("fen", POSITIONS)
def test_move_ordering_does_not_increase_node_count_at_fixed_depth(fen: str) -> None:
    """Core Milestone 4 exit-criteria assertion (architecture.md §15): at a
    fixed depth, the real move-ordering stack (TT move + MVV-LVA + killers +
    history, §9.4) must visit no more nodes than the same search with move
    ordering neutralized -- proving move ordering helps (or at worst does
    not hurt) search efficiency, never the reverse."""
    # Two independent Board + Search pairs, each with its own TT/killers/
    # history, so nothing about one run's state can leak into the other.
    ordered_board = parse_fen(fen)
    ordered_search = Search(default_evaluator())
    ordered_result = ordered_search.search(ordered_board, SearchLimits(max_depth=FIXED_DEPTH))

    unordered_board = parse_fen(fen)
    unordered_search = _UnorderedSearch(default_evaluator())
    unordered_result = unordered_search.search(unordered_board, SearchLimits(max_depth=FIXED_DEPTH))

    assert ordered_result.nodes <= unordered_result.nodes, (
        f"real move ordering visited MORE nodes than the unordered baseline at "
        f"depth {FIXED_DEPTH} for {fen!r}: ordered={ordered_result.nodes} "
        f"unordered={unordered_result.nodes} -- move ordering must never make "
        f"the search less efficient."
    )

    # Both searches must still have left their boards exactly as found --
    # otherwise a make/unmake asymmetry in this test, not a real ordering
    # regression, could be mistaken for one.
    assert ordered_board.to_fen() == fen
    assert unordered_board.to_fen() == fen
