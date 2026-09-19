"""Milestone 2/4 exit-criteria tests for `search.py` (architecture.md §9, §15).

§15's Milestone 2 exit criteria call for exactly two things beyond
`test_evaluate.py` (which is out of scope for this file, since it exercises
`evaluate.py` in isolation with no search involved):

1. **A differential test** asserting `Search._negamax`'s result matches an
   unpruned, full-width minimax on small/shallow positions. This is the one
   test that can actually catch an alpha-beta *window* bug (a wrong sign on
   negation, `alpha`/`beta` not swapped -- or swapped the wrong way -- on
   the recursive call, a stale bound leaking across sibling subtrees):
   alpha-beta pruning is only ever a speed optimization over full-width
   minimax, so the two must agree on every node's *score* even though they
   visit wildly different numbers of nodes to get there. A bug that breaks
   this invariant is otherwise invisible -- the search still runs, still
   returns *a* move, just occasionally the wrong one, with nothing else in
   the suite positioned to catch it.

2. **A self-play smoke test** that plays a complete game from the start
   position using nothing but `Search.search` + `Board.make_move`, at a
   shallow fixed depth for test speed, asserting every move played was
   actually legal at the time and the game terminates cleanly (no
   exception, no infinite loop) within a generous ply cap.

Depths and positions below are deliberately kept small (fixed depth 3 for
the differential oracle, depth 2 for self-play) so the whole file runs in a
few seconds, not minutes.

**Milestone 4 update.** Since `Search._negamax`'s `depth == 0` leaf now hands
off to `Search._quiescence` rather than evaluating statically (architecture.md
§9.5), the oracle's own depth-0 case mirrors that with `_naive_quiescence`: an
unpruned (no alpha-beta window) capture search that nonetheless applies the
*same* `see_ge` capture filtering `Search._quiescence` does. That filtering is
deliberately kept identical between engine and oracle rather than dropped,
because SEE-based pruning is a *heuristic* move filter, not a window
optimization -- per §9.5 it can change which captures are ever searched
(unlike alpha-beta, which never changes the resulting score, only how many
nodes it takes to compute it). Comparing against an oracle that searched
*every* capture, with no SEE filtering at all, would then be expected to
diverge from the engine even with zero alpha-beta bugs, which would defeat
this test's actual purpose. Keeping the SEE filtering identical on both
sides isolates exactly what this test is for: alpha-beta window bugs, not
SEE's own known, accepted imprecision.
"""

from __future__ import annotations

import threading

import pytest

from chessengine.board import Board
from chessengine.constants import DRAW_SCORE, INF, MATE_SCORE
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_captures, generate_legal_moves
from chessengine.search import (
    CHECK_EXTENSION_MAX_PLIES,
    CHECK_EXTENSION_PLIES,
    Search,
    SearchInfo,
    SearchLimits,
    _SearchCtx,
    see_ge,
)

# --- 1. Differential test: Search._negamax vs. a naive, unpruned oracle -----


def _naive_quiescence(board: Board, ply: int) -> int:
    """A deliberately naive mirror of `Search._quiescence`: the same
    stand-pat bound, the same `see_ge`-gated capture set (so a SEE-pruned
    capture is skipped identically on both sides, per this module's
    docstring), but **no** alpha-beta window -- every surviving capture is
    searched and the true maximum taken, rather than cutting off as soon as
    one is known to be at least as good as `beta`.
    """
    stand_pat = _ORACLE_EVALUATOR.evaluate(board)
    best = stand_pat
    for move in generate_captures(board):
        if not see_ge(board, move, 0):
            continue
        board.make_move(move)
        score = -_naive_quiescence(board, ply + 1)
        board.unmake_move()
        if score > best:
            best = score
    return best


def _naive_full_width_negamax(
    board: Board, depth: int, ply: int = 0, ext_remaining: int = CHECK_EXTENSION_MAX_PLIES
) -> int:
    """A deliberately naive reference oracle: full-width negamax with **no**
    alpha-beta window (every legal move at every node is visited to the
    full requested depth, nothing is ever pruned), **no** transposition
    table, and **no** move ordering -- moves are searched in whatever order
    `generate_legal_moves` happens to return them.

    This shares only *what a position's minimax value means* with
    `Search._negamax` (draw scoring, mate scoring relative to `ply`, the
    same evaluator, the same `see_ge`-gated quiescence at the leaves
    (§9.5), and the same check-extension depth bonus (Milestone 5
    extension: a move that gives check is searched `CHECK_EXTENSION_PLIES`
    deeper, budgeted by `ext_remaining` exactly like `Search._negamax`'s
    own `ext_remaining` parameter)) -- never *how* it is computed. Check
    extensions are mirrored here for the same reason SEE filtering is:
    it's a search-structure change, not a pruning technique, so it can
    change which depth a subtree is searched to even with zero alpha-beta
    involved -- an oracle without it would diverge from the engine even
    with zero real bugs, defeating this test's purpose. Any remaining
    divergence between this and `Search._negamax`'s score at the same
    (position, depth) is therefore a real alpha-beta bug, not a difference
    in search strategy.
    """
    if board.is_fifty_move_draw() or board.is_repetition_draw():
        return DRAW_SCORE
    if depth == 0:
        return _naive_quiescence(board, ply)

    moves = generate_legal_moves(board)
    if not moves:
        return -MATE_SCORE + ply if board.in_check() else DRAW_SCORE

    best = -INF
    for move in moves:
        board.make_move(move)
        gives_check = board.in_check()
        extend = CHECK_EXTENSION_PLIES if (gives_check and ext_remaining > 0) else 0
        score = -_naive_full_width_negamax(board, depth - 1 + extend, ply + 1, ext_remaining - extend)
        board.unmake_move()
        if score > best:
            best = score
    return best


_ORACLE_EVALUATOR = default_evaluator()

# (case id, FEN, depth). One opening, one tactical middlegame, one simple
# endgame, per §15's "small/shallow positions" and the task's explicit
# battery. Depth 3 is small enough that even the unpruned oracle -- which
# visits every node in the full-width tree, not just the leaves -- finishes
# in well under a second on each of these.
DIFFERENTIAL_CASES = [
    pytest.param(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        3,
        id="opening-startpos",
    ),
    pytest.param(
        # A tactical position with an exposed king, a pinned/skewered rook,
        # and a passed pawn one step from promotion -- plenty of captures
        # and checks for alpha-beta to have something to (mis)prune, while
        # its modest piece count keeps the branching factor, and so the
        # oracle's runtime, small.
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        3,
        id="tactical-middlegame-exposed-king",
    ),
    pytest.param(
        # A minimal king-and-pawn endgame: one side pushing a pawn toward
        # promotion against a lone king.
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        3,
        id="simple-kp-endgame",
    ),
]


@pytest.mark.parametrize("fen, depth", DIFFERENTIAL_CASES)
def test_negamax_matches_naive_full_width_oracle(
    fen: str, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chessengine.search as search_mod

    monkeypatch.setattr(search_mod, "NULL_MOVE_MIN_DEPTH", 10_000)
    monkeypatch.setattr(search_mod, "LMR_MIN_MOVE_INDEX", 10_000)
    monkeypatch.setattr(search_mod, "PVS_ENABLED", False)
    monkeypatch.setattr(search_mod, "FUTILITY_DEPTH", 0)
    monkeypatch.setattr(search_mod, "RFP_DEPTH", 0)
    monkeypatch.setattr(search_mod, "LMP_DEPTH", 0)
    monkeypatch.setattr(search_mod, "DELTA_MARGIN", 99_999)

    oracle_board = parse_fen(fen)
    oracle_score = _naive_full_width_negamax(oracle_board, depth)
    assert oracle_board.to_fen() == fen

    engine_board = parse_fen(fen)
    search = Search(default_evaluator())
    ctx = _SearchCtx(SearchLimits(max_depth=depth), deadline=None, stop_event=None, extra_stop=None)
    engine_score = search._negamax(engine_board, depth, -INF, INF, 0, ctx)

    assert engine_score == oracle_score, (
        f"Search._negamax disagreed with the unpruned full-width oracle at "
        f"depth {depth} for {fen!r}: engine={engine_score} oracle={oracle_score}"
    )
    assert engine_board.to_fen() == fen


# --- 2. Self-play smoke test -------------------------------------------------

_SELF_PLAY_MAX_PLIES = 150


def test_self_play_never_returns_an_illegal_move_and_terminates_cleanly() -> None:
    """Play a complete game from the start position using nothing but
    `Search.search` + `Board.make_move`, at a shallow fixed depth (2 ply)
    for test speed.

    Two things are asserted at every single ply:

    - the move `search.search(...)` hands back was actually present in
      `board`'s own legal move list *at the time it was generated* (the
      search must never fabricate or mis-decode a move it never actually
      considered legal);
    - nothing raises -- no exception anywhere in search, make_move, or
      unmake_move, for as long as the game runs.

    The loop stops the instant the position is itself terminal (checkmate,
    stalemate, the fifty-move rule, or the engine's own -- stricter than
    threefold -- repetition-draw rule, architecture.md §9.6), and otherwise
    after `_SELF_PLAY_MAX_PLIES` plies as a safety cap: a depth-2,
    material+PST-only engine has no obligation to ever reach an actual
    mate, so the cap (not "game over") is the realistic, guaranteed way
    this test terminates.
    """
    board = Board.starting_position()
    search = Search(default_evaluator())
    limits = SearchLimits(max_depth=2)

    plies_played = 0
    game_over_reason: str | None = None

    for _ in range(_SELF_PLAY_MAX_PLIES):
        legal_moves = generate_legal_moves(board)
        if not legal_moves:
            game_over_reason = "checkmate" if board.in_check() else "stalemate"
            break
        if board.is_fifty_move_draw():
            game_over_reason = "fifty-move draw"
            break
        if board.is_repetition_draw():
            game_over_reason = "repetition draw"
            break

        result = search.search(board, limits)

        assert result.best_move != NULL_MOVE, (
            f"search.search returned the NULL_MOVE sentinel at ply {plies_played} "
            f"in a position with {len(legal_moves)} legal move(s) -- it must always "
            f"pick one of them."
        )
        assert result.best_move in legal_moves, (
            f"search.search returned an illegal move "
            f"({move_to_uci(result.best_move)!r}) at ply {plies_played}; legal moves "
            f"were {sorted(move_to_uci(m) for m in legal_moves)!r} in position "
            f"{board.to_fen()!r}."
        )

        board.make_move(result.best_move)
        plies_played += 1
    else:
        # Hit the safety cap without the position itself ever going
        # terminal -- an accepted, common outcome at this depth/evaluator,
        # not a failure: the point of the cap is precisely to bound this
        # case instead of hanging.
        game_over_reason = f"ply cap ({_SELF_PLAY_MAX_PLIES})"

    assert plies_played <= _SELF_PLAY_MAX_PLIES
    assert game_over_reason is not None


@pytest.mark.parametrize(
    "fen",
    [
        None,  # startpos
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete
    ],
)
def test_search_never_returns_null_move_on_immediate_stop(fen: str | None) -> None:
    """Regression test for a real bug caught end-to-end via the UCI layer:
    `Search.search` must never hand back `NULL_MOVE` just because a stop
    signal (a real `stop_event`, or a node/time budget) fires before the
    very first iterative-deepening iteration (depth 1) finishes even one
    root move.

    Before the fix, `search()` seeded `best` with `NULL_MOVE` and only
    discarded an aborted iteration's result when `depth > 1` -- so an
    immediate stop during depth 1 left `best.best_move` at its `NULL_MOVE`
    default (no TT entry had been stored yet for `_extract_pv` to find).
    Reported over UCI, this became an illegal `bestmove a1a1` -- a genuine
    forfeit/crash risk for any GUI whose `stop`/short `movetime` can land
    before a slow position's first root move completes (quiescence search
    makes root nodes expensive enough that this is a real scenario, not
    just a pathological one).

    The fix seeds an *ordered* legal fallback move before the loop starts,
    so an immediate stop still returns a legal (if unoptimized) move.
    """
    board = Board.starting_position() if fen is None else parse_fen(fen)
    legal_moves = generate_legal_moves(board)
    assert legal_moves, "test position must not itself be terminal"

    search = Search(default_evaluator())
    stop_event = threading.Event()
    stop_event.set()  # simulate a stop that fires before search even starts

    result = search.search(board, SearchLimits(max_depth=10), stop_event=stop_event)

    assert result.best_move != NULL_MOVE, (
        "search.search returned NULL_MOVE under an immediate stop -- this would "
        "be reported as an illegal 'bestmove a1a1' over UCI"
    )
    assert result.best_move in legal_moves, (
        f"fallback move {move_to_uci(result.best_move)!r} is not even legal "
        f"in position {board.to_fen()!r}"
    )


# --- 3. Triangular PV stability test (Milestone 5) -------------------------

# Positions chosen to exercise different kinds of PV lines: an opening with
# many quiet moves, a tactical middlegame with captures/checks, and a
# pawn-endgame with a clear plan.
_TRIANGULAR_PV_CASES = [
    pytest.param(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        id="startpos",
    ),
    pytest.param(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        id="tactical-middlegame",
    ),
    pytest.param(
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        id="kp-endgame",
    ),
    pytest.param(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        id="kiwipete",
    ),
]


@pytest.mark.parametrize("fen", _TRIANGULAR_PV_CASES)
@pytest.mark.parametrize("depth", [5, 6])
def test_triangular_pv_at_least_as_long_as_tt_extract_pv(
    fen: str, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The triangular PV array (built inside _negamax as it runs) should be
    at least as long as the TT-walk PV (_extract_pv) for the same search,
    confirming that the triangular approach is more stable -- TT entries can
    be overwritten during a search, truncating the TT-walk PV, while the
    triangular array is immune to that."""
    import chessengine.search as search_mod
    monkeypatch.setattr(search_mod, "LMP_DEPTH", 0)
    monkeypatch.setattr(search_mod, "FUTILITY_DEPTH", 0)

    board = parse_fen(fen)
    search = Search(default_evaluator())
    # We need to capture the triangular PV from the last completed depth's
    # info callback.
    triangular_pvs: list[list[int]] = []

    def capture_info(info: SearchInfo) -> None:
        triangular_pvs.append(list(info.pv))

    result = search.search(board, SearchLimits(max_depth=depth), on_info=capture_info)

    # The result PV should itself come from the triangular array.
    assert result.pv == triangular_pvs[-1], (
        "SearchResult.pv should match the last on_info PV (both from the triangular array)"
    )

    # Now extract the TT-based PV for comparison.
    tt_pv = search._extract_pv(board, depth)

    # The triangular PV must be at least as long as the TT-walk PV.
    assert len(result.pv) >= len(tt_pv), (
        f"Triangular PV (len={len(result.pv)}) should be at least as long as "
        f"the TT-walk PV (len={len(tt_pv)}) for {fen!r} at depth {depth}. "
        f"Triangular: {[move_to_uci(m) for m in result.pv]}, "
        f"TT-walk: {[move_to_uci(m) for m in tt_pv]}"
    )

    # Both must start with the same best move (when non-empty).
    if result.pv and tt_pv:
        assert result.pv[0] == tt_pv[0], (
            f"Triangular and TT-walk PVs disagree on best move: "
            f"{move_to_uci(result.pv[0])} vs {move_to_uci(tt_pv[0])}"
        )
