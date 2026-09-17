"""Milestone 4 exit-criteria tests for `transposition.py` (architecture.md
§9.2, §13, §15): "the TT must never change the search's result, only its
speed."

Three gates, per §13/§15:

1. **TT-on/TT-off agreement.** A search run with a real, working TT and the
   same search run with a stub TT (`probe` always misses, `store` is a
   no-op) must reach the same score -- and the same best move, up to
   equal-score ties -- for every root move, at a fixed small depth, across a
   battery of tactical and quiet positions.
2. **Mate-distance adjustment.** `score_to_tt`/`score_from_tt` must correctly
   re-base a mate score found at one ply to a different ply it's later
   probed at.
3. **Replacement policy.** In a tiny TT, a shallower entry for a *different*
   position must never survive being overwritten by a deeper one, and a
   deeper entry must never be evicted by a shallower one.

Gate 1 does **not** go through `Search.search()`'s public API, deliberately:
`Search._extract_pv` (architecture.md §9.3) reconstructs the reported
`best_move`/`pv` by reading them back out of the TT after the fact, so an
always-miss/no-store stub would make `Search.search()` report `NULL_MOVE`
regardless of whether the *search itself* (`Search._negamax`) computed the
right answer -- that would be testing PV extraction's TT dependency, not the
§9.2 invariant this file is actually about. Instead, gate 1 scores every
legal root move directly through `Search._negamax`, exactly as `_negamax`'s
own root ply does internally, with a real TT and a stub TT swapped in for
`Search.tt` in turn -- precedented by `test_search.py`'s own differential
test, which pokes at `_negamax` directly for the same reason.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.constants import INF, MATE_SCORE
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits, _SearchCtx
from chessengine.transposition import TTEntry, TTFlag, TranspositionTable, score_from_tt, score_to_tt

# --- 1. TT-on/TT-off agreement (architecture.md §9.2, §13) ------------------


class _AlwaysMissTT(TranspositionTable):
    """A `TranspositionTable` stand-in that behaves as if it were empty and
    permanently inert: every `probe` misses, every `store` is silently
    dropped.

    Swapping this in for `Search.tt` isolates "does the TT change the
    answer" from "does the TT work at all" -- per architecture.md §9.2's
    invariant, a search using this stub must reach exactly the same
    per-move scores as one using a real, working TT; any difference points
    at the TT (not this stub) as the cause.
    """

    def __init__(self) -> None:  # deliberately skip TranspositionTable.__init__:
        pass  # no backing array is needed when every probe/store is a no-op.

    def probe(self, key: int) -> TTEntry | None:
        return None

    def store(self, key: int, depth: int, score: int, flag: TTFlag, best_move: int) -> None:
        pass

    def clear(self) -> None:
        pass


def _score_every_root_move(search: Search, board: Board, depth: int) -> dict[int, int]:
    """Mirrors what `Search._negamax`'s root ply computes for each legal
    move: one full-width (`-INF, INF`) child search per move, via the exact
    same `_negamax` the engine's own root loop calls.

    A full `(-INF, INF)` window always returns a subtree's *exact* minimax
    value (alpha-beta only ever returns a bound when the window has already
    been narrowed) -- so this is a faithful, TT-implementation-independent
    way to ask "what is this move actually worth," regardless of whether
    `search.tt` happens to be a real table or the always-miss stub above.
    """
    moves = generate_legal_moves(board)
    assert moves, f"expected at least one legal move in {board.to_fen()!r}"
    ctx = _SearchCtx(SearchLimits(max_depth=depth), deadline=None, stop_event=None, extra_stop=None)
    scores: dict[int, int] = {}
    for move in moves:
        board.make_move(move)
        scores[move] = -search._negamax(board, depth - 1, -INF, INF, 1, ctx)
        board.unmake_move()
    return scores


# (case id, FEN, depth). A mix of a quiet opening, two tactical middlegames
# (an exposed king with a skewered rook, and a promotion-heavy melee), and a
# quiet king-and-pawn endgame -- the "battery of tactical and quiet
# positions" the task calls for. All four FENs are already-vetted fixtures
# reused from architecture.md §14's perft table / `test_search.py`'s own
# differential-test battery, not new, unverified positions. Depths are fixed
# and small (3-4) so the whole file runs in well under a second.
TT_INVARIANCE_CASES = [
    pytest.param(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        3,
        id="opening-startpos",
    ),
    pytest.param(
        # architecture.md §14 Position 3: an exposed king, a
        # pinned/skewered rook, a passed pawn one step from promotion.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        3,
        id="tactical-exposed-king",
    ),
    pytest.param(
        # architecture.md §14 Position 5: mid-exchange promotion tactics
        # and a pinned knight, with a wider branching factor than the
        # other tactical case.
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        3,
        id="tactical-promotion-rich",
    ),
    pytest.param(
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        4,
        id="quiet-kp-endgame",
    ),
]


@pytest.mark.parametrize("fen, depth", TT_INVARIANCE_CASES)
def test_tt_never_changes_the_search_result(fen: str, depth: int) -> None:
    real_board = parse_fen(fen)
    real_search = Search(default_evaluator(), tt_size_mb=1)
    real_scores = _score_every_root_move(real_search, real_board, depth)
    assert real_board.to_fen() == fen, "the real-TT harness must leave the board exactly as it found it"

    stub_board = parse_fen(fen)
    stub_search = Search(default_evaluator(), tt_size_mb=1)
    stub_search.tt = _AlwaysMissTT()
    stub_scores = _score_every_root_move(stub_search, stub_board, depth)
    assert stub_board.to_fen() == fen, "the stub-TT harness must leave the board exactly as it found it"

    assert set(real_scores) == set(stub_scores), (
        f"the real-TT and stub-TT runs disagree about which moves are legal in {fen!r} "
        "-- that alone points at a bug outside the TT."
    )
    assert real_scores == stub_scores, (
        f"the TT changed the search result for {fen!r} at depth {depth}: "
        f"real-TT per-move scores {real_scores} != stub-TT per-move scores {stub_scores}. "
        "Per architecture.md §9.2, the TT must never change the score, only the speed."
    )

    real_best = max(real_scores.values())
    real_best_moves = {m for m, s in real_scores.items() if s == real_best}
    stub_best = max(stub_scores.values())
    stub_best_moves = {m for m, s in stub_scores.items() if s == stub_best}
    assert real_best == stub_best
    # "Same best move up to equal-score ties": every move either config
    # would pick achieves the one shared best score.
    assert real_best_moves == stub_best_moves


# --- 2. Mate-distance adjustment (architecture.md §9.2, §13) ----------------


def test_mate_distance_adjusts_correctly_across_a_different_ply() -> None:
    """A mate score is only meaningful relative to the node it was computed
    at. `score_to_tt` strips a node's own root-relative ply out of a mate
    score before it's stored, so the same physical "position, N plies from
    delivering mate" maps to one canonical TT value regardless of which
    path/ply first reached it; `score_from_tt` re-applies a (possibly
    different) node's own ply when the value is probed back.

    This directly constructs a `TranspositionTable` and drives
    `score_to_tt`/`score_from_tt` by hand -- no `Search`/`Board` involved --
    for the *winning* side: a mate found 3 plies deep from a node at
    ply=5 (one search), stored, then probed back as if reached at ply=2
    (a different, closer path to the very same position), must come back
    as "mate in 3 from here" re-based to the new ply, not "mate in 5"
    (the original absolute ply) and not the raw stored value unadjusted.
    """
    tt = TranspositionTable(size_mb=1)
    key = 0x1234_5678_9ABC_DEF0

    mate_distance = 3  # "mate in 3" from the node itself, true in *both* searches
    ply_when_found = 5  # first found 5 plies into one search
    ply_when_reached = 2  # later reached via a different path, only 2 plies into another

    # The root-relative score `_negamax` would compute at a node, at
    # ply=`ply_when_found`, whose side to move forces mate `mate_distance`
    # plies later (architecture.md §9.1: `MATE_SCORE - absolute_ply` for
    # the winning side, mirroring the losing side's raw leaf formula
    # `-MATE_SCORE + absolute_ply`).
    root_relative_score = MATE_SCORE - (ply_when_found + mate_distance)

    stored = score_to_tt(root_relative_score, ply_when_found)
    # The whole point of `score_to_tt`: the stored, ply-independent value
    # depends only on `mate_distance`, never on `ply_when_found`.
    assert stored == MATE_SCORE - mate_distance

    tt.store(key, depth=6, score=stored, flag=TTFlag.EXACT, best_move=NULL_MOVE)

    entry = tt.probe(key)
    assert entry is not None
    assert entry.score == stored  # the table itself never adjusts anything on its own

    corrected = score_from_tt(entry.score, ply_when_reached)
    # Re-based to the *new* search's root: mate at absolute ply
    # `ply_when_reached + mate_distance` -- still "mate in 3" from this
    # different, closer node.
    assert corrected == MATE_SCORE - (ply_when_reached + mate_distance)
    recovered_mate_distance = MATE_SCORE - corrected - ply_when_reached
    assert recovered_mate_distance == mate_distance
    assert corrected != root_relative_score  # the ply adjustment actually did something
    assert corrected != stored  # ...and it's distinct from the raw stored value too


def test_mate_distance_adjusts_correctly_for_the_losing_side_too() -> None:
    """Mirror of the test above for a *losing* mate score (`score <=
    -MATE_SCORE + 128`, architecture.md §9.2) -- `score_to_tt`/
    `score_from_tt`'s other branch, exercised the same way."""
    tt = TranspositionTable(size_mb=1)
    key = 0x0FED_CBA9_8765_4321

    mate_distance = 4
    ply_when_found = 6
    ply_when_reached = 1

    root_relative_score = -MATE_SCORE + (ply_when_found + mate_distance)
    stored = score_to_tt(root_relative_score, ply_when_found)
    assert stored == -MATE_SCORE + mate_distance

    tt.store(key, depth=8, score=stored, flag=TTFlag.EXACT, best_move=NULL_MOVE)
    entry = tt.probe(key)
    assert entry is not None
    assert entry.score == stored

    corrected = score_from_tt(entry.score, ply_when_reached)
    assert corrected == -MATE_SCORE + (ply_when_reached + mate_distance)
    recovered_mate_distance = corrected + MATE_SCORE - ply_when_reached
    assert recovered_mate_distance == mate_distance
    assert corrected != root_relative_score
    assert corrected != stored


# --- 3. Replacement policy (architecture.md §9.2, §13) ----------------------


def test_replacement_policy_never_lets_a_shallower_entry_evict_a_deeper_one() -> None:
    """architecture.md §9.2's depth-preferred replacement scheme: within one
    slot (`key & mask`), a **different** key may only replace what's
    already there if it was searched at least as deep; a shallower, cheaper
    re-search of some other position must never evict a deeper, more
    expensive-to-recompute one. (Re-storing the *same* key is a refresh,
    not an eviction, and is deliberately exempt from this -- not what this
    test is about.)

    Directly overrides `.size`/`.mask`/`.table` on a normally-constructed
    `TranspositionTable` to force a tiny, deterministic slot count (4),
    independent of the constructor's bytes-per-entry sizing constant --
    this test is about the *replacement policy*, not that arithmetic.
    """
    tt = TranspositionTable(size_mb=1)
    tt.size = 4
    tt.mask = tt.size - 1
    tt.table = [None] * tt.size

    # Three distinct keys, all landing in the same slot (built off
    # `tt.size` so they collide regardless of the exact slot count).
    key_shallow = 1
    key_deep = 1 + tt.size
    key_deeper_tie = 1 + 2 * tt.size
    assert key_shallow & tt.mask == key_deep & tt.mask == key_deeper_tie & tt.mask

    # An empty slot always accepts a store.
    tt.store(key_shallow, depth=2, score=10, flag=TTFlag.EXACT, best_move=NULL_MOVE)
    entry = tt.probe(key_shallow)
    assert entry is not None and entry.depth == 2

    # A different key, searched *deeper*, evicts the shallower entry.
    tt.store(key_deep, depth=5, score=20, flag=TTFlag.EXACT, best_move=NULL_MOVE)
    assert tt.probe(key_shallow) is None, "the deeper store must have evicted the shallower entry"
    entry = tt.probe(key_deep)
    assert entry is not None and entry.depth == 5 and entry.score == 20

    # A different key again, searched *shallower* than what's currently in
    # the slot, must NOT evict the deeper entry.
    tt.store(key_deeper_tie, depth=1, score=30, flag=TTFlag.EXACT, best_move=NULL_MOVE)
    assert tt.probe(key_deeper_tie) is None, "a shallower store must never evict a deeper entry"
    entry = tt.probe(key_deep)
    assert entry is not None and entry.depth == 5 and entry.score == 20, (
        "the deeper entry must survive a shallower different-key store attempt untouched"
    )

    # An equal-depth store of a different key IS allowed to replace (ties
    # go to the newcomer, per the `depth >= existing.depth` check) --
    # locking in that boundary explicitly rather than leaving it untested.
    tt.store(key_deeper_tie, depth=5, score=40, flag=TTFlag.EXACT, best_move=NULL_MOVE)
    assert tt.probe(key_deep) is None, "an equal-depth store of a different key must replace"
    entry = tt.probe(key_deeper_tie)
    assert entry is not None and entry.depth == 5 and entry.score == 40

    # And once more: attempting to re-introduce the original, shallow key
    # must still fail against what's now there.
    tt.store(key_shallow, depth=0, score=50, flag=TTFlag.EXACT, best_move=NULL_MOVE)
    assert tt.probe(key_shallow) is None
    entry = tt.probe(key_deeper_tie)
    assert entry is not None and entry.depth == 5 and entry.score == 40
