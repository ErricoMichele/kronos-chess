"""Milestone 2 exit-criteria tests for `search.py` (architecture.md §9, §15).

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
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.constants import DRAW_SCORE, INF, MATE_SCORE
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits, _SearchCtx

# --- 1. Differential test: Search._negamax vs. a naive, unpruned oracle -----


def _naive_full_width_negamax(board: Board, depth: int, ply: int = 0) -> int:
    """A deliberately naive reference oracle: full-width negamax with **no**
    alpha-beta window (every legal move at every node is visited to the
    full requested depth, nothing is ever pruned), **no** transposition
    table, and **no** move ordering -- moves are searched in whatever order
    `generate_legal_moves` happens to return them.

    This shares only *what a position's minimax value means* with
    `Search._negamax` (draw scoring, mate scoring relative to `ply`, the
    same evaluator at the leaves) -- never *how* it is computed. Any
    divergence between this and `Search._negamax`'s score at the same
    (position, depth) is therefore a real alpha-beta bug, not a difference
    in search strategy.
    """
    if board.is_fifty_move_draw() or board.is_repetition_draw():
        return DRAW_SCORE
    if depth == 0:
        return _ORACLE_EVALUATOR.evaluate(board)

    moves = generate_legal_moves(board)
    if not moves:
        return -MATE_SCORE + ply if board.in_check() else DRAW_SCORE

    best = -INF
    for move in moves:
        board.make_move(move)
        score = -_naive_full_width_negamax(board, depth - 1, ply + 1)
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
def test_negamax_matches_naive_full_width_oracle(fen: str, depth: int) -> None:
    # Two independent Board instances (and Search instances), so nothing
    # about one call's make/unmake bookkeeping or TT/killer/history state
    # can leak into the other.
    oracle_board = parse_fen(fen)
    oracle_score = _naive_full_width_negamax(oracle_board, depth)
    # The oracle must leave the board exactly as it found it -- otherwise a
    # make/unmake bug in *this file*, not in `search.py`, could be mistaken
    # for a real alpha-beta divergence.
    assert oracle_board.to_fen() == fen

    engine_board = parse_fen(fen)
    search = Search(default_evaluator())
    ctx = _SearchCtx(SearchLimits(max_depth=depth), deadline=None, stop_event=None, extra_stop=None)
    engine_score = search._negamax(engine_board, depth, -INF, INF, 0, ctx)

    assert engine_score == oracle_score, (
        f"Search._negamax disagreed with the unpruned full-width oracle at "
        f"depth {depth} for {fen!r}: engine={engine_score} oracle={oracle_score} "
        f"-- a real alpha-beta window bug, since pruning must never change the score."
    )
    # The engine's own make/unmake bookkeeping must also be symmetric.
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
