"""Check extension tests (architecture.md §9, Milestone 5 extension) --
mirrors `test_null_move.py`/`test_lmr.py`'s structure for the last of the
four Milestone 5 "optional search extensions" (null-move pruning, late move
reductions, aspiration windows, check extensions -- architecture.md §15).

Three things, matching this extension's own risk profile:

1. **Basic sanity.** A normal search at a moderate depth on a battery of
   varied positions still returns a legal move and completes without
   raising, with check extensions active (the default, unmodified `Search`).

2. **The extension actually matters, not just "it's harmless".** A genuine
   mate-in-2 position -- independently verified by this project's own
   brute-force mate oracle (`_move_forces_checkmate`/`_mate_search`,
   duplicated here from `test_search_tactics.py`; see that file's own
   docstring for why an independent, `board.py`/`movegen.py`-only oracle is
   used rather than a hardcoded "known solution") -- whose *first* move
   gives check, searched at a nominal depth exactly one ply short of what
   the full 3-ply mate line needs. With check extensions active, the extra
   ply the checking move earns is enough to see the forced mate through to
   the end and report a mate-range score; with `CHECK_EXTENSION_MAX_PLIES`
   disabled (monkeypatched to 0), the position after Black's forced reply
   falls exactly on `depth == 0` and is handed to `_quiescence` -- which,
   unlike the "no legal moves" branch at the top of `_negamax`, never checks
   for checkmate at all -- so the search reports a plain material
   evaluation instead, oblivious to the mate one move away. A second,
   narrower test isolates the exact same mechanism to a single node via a
   direct `_negamax` call with an explicit `ext_remaining`, independent of
   the rest of iterative deepening.

3. **Mutual exclusion with LMR is already covered elsewhere, not duplicated
   here.** `test_lmr.py`'s own
   `test_lmr_never_reduces_a_capture_promotion_or_checking_move` already
   proves no move that gives check is ever handed to `_lmr_reduction` --
   exactly the guarantee (`reduce_this_move` requires `not gives_check`)
   that makes a single move never both reduced and extended.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.constants import INF, MATE_SCORE
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits, _SearchCtx
from chessengine import search as search_mod

# --- Shared FEN fixtures ------------------------------------------------------

OPENING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
TACTICAL_MIDDLEGAME_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"


# --- 1. Basic sanity: search still works with check extensions active -------

SANITY_CASES = [
    pytest.param(OPENING_FEN, 6, id="opening-startpos"),
    pytest.param(KIWIPETE_FEN, 6, id="kiwipete-middlegame"),
    pytest.param(TACTICAL_MIDDLEGAME_FEN, 7, id="tactical-middlegame-exposed-king"),
]


@pytest.mark.parametrize("fen, depth", SANITY_CASES)
def test_search_completes_and_returns_legal_move_with_check_extensions_active(
    fen: str, depth: int
) -> None:
    """A normal, moderate-depth search on an opening/middlegame/tactical
    position still returns a legal move and completes without raising, with
    check extensions active (the default, unmodified `Search`)."""
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=depth))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )


# --- Independent oracle: "does this move force checkmate within N of the
# engine's own moves?" (duplicated from test_search_tactics.py -- see that
# file's docstring for the full rationale for keeping this self-contained
# per test file rather than cross-importing between test modules).


def _mate_search(board: Board, plies_left: int, attacker_to_move: bool) -> bool:
    if plies_left <= 0:
        return False
    legal = generate_legal_moves(board)
    if attacker_to_move:
        for move in legal:
            board.make_move(move)
            opponent_legal = generate_legal_moves(board)
            if not opponent_legal:
                forced = board.in_check()
            else:
                forced = _mate_search(board, plies_left - 1, attacker_to_move=False)
            board.unmake_move()
            if forced:
                return True
        return False
    else:
        if not legal:
            return board.in_check()
        for move in legal:
            board.make_move(move)
            forced = _mate_search(board, plies_left - 1, attacker_to_move=True)
            board.unmake_move()
            if not forced:
                return False
        return True


def _move_forces_checkmate(board: Board, move: int, max_plies_after: int) -> bool:
    board.make_move(move)
    try:
        opponent_legal = generate_legal_moves(board)
        if not opponent_legal:
            return board.in_check()
        if max_plies_after == 0:
            return False
        for opponent_move in opponent_legal:
            board.make_move(opponent_move)
            forced = _mate_search(board, max_plies_after - 1, attacker_to_move=True)
            board.unmake_move()
            if not forced:
                return False
        return True
    finally:
        board.unmake_move()


# --- 2. The extension actually matters ---------------------------------------
#
# White Ka5, Qa1; Black Ka8 (bare kings + one queen -- nothing else on the
# board to complicate the line). 1.Kb6+ is a DISCOVERED check: the White
# king merely steps off the a-file, unmasking Qa1's own check along it.
# Black's only legal reply is Ka8-b8 (a7 is still covered by the queen along
# the a1-a8 file/a1-h8-ish reach once the king vacates a5, and every other
# adjacent square is off the board or attacked). 2.Qa1-h8# (among other
# mating replies) finishes it. Independently verified below (see
# `test_discovered_check_mate_in_2_fixture_is_valid`) via the oracle above,
# not asserted on faith.
_DISCOVERED_CHECK_MATE_IN_2_FEN = "k7/8/8/K7/8/8/8/Q7 w - - 0 1"
_DISCOVERED_CHECK_UCI = "a5b6"
# The exact position reached after 1.Kb6+ Kb8, White to move with a forced
# mate one ply away -- used directly by test 2b below, isolated from the
# rest of iterative deepening.
_FEN_AFTER_FORCED_REPLY = "1k6/8/1K6/8/8/8/8/Q7 w - - 2 2"


def test_discovered_check_mate_in_2_fixture_is_valid() -> None:
    """Sanity-checks the fixture itself against the independent oracle
    before either search-based test below relies on it."""
    board = parse_fen(_DISCOVERED_CHECK_MATE_IN_2_FEN)
    move = next(m for m in generate_legal_moves(board) if move_to_uci(m) == _DISCOVERED_CHECK_UCI)

    board.make_move(move)
    gives_check = board.in_check()
    board.unmake_move()
    assert gives_check, (
        f"{_DISCOVERED_CHECK_UCI!r} must actually give check for this fixture to "
        f"exercise check extensions at all"
    )

    assert _move_forces_checkmate(board, move, max_plies_after=2), (
        f"{_DISCOVERED_CHECK_UCI!r} does not force mate in 2 from "
        f"{_DISCOVERED_CHECK_MATE_IN_2_FEN!r} per the independent oracle"
    )
    assert board.to_fen() == _DISCOVERED_CHECK_MATE_IN_2_FEN


def test_check_extension_finds_a_mate_a_fixed_depth_search_would_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At a nominal depth exactly one ply short of the full 3-ply mate line,
    check extensions are the only thing standing between "sees the forced
    mate" and "evaluates the pre-mate position statically and never notices
    it".
    """
    board_with = parse_fen(_DISCOVERED_CHECK_MATE_IN_2_FEN)
    with_extensions = Search(default_evaluator()).search(board_with, SearchLimits(max_depth=2))
    assert board_with.to_fen() == _DISCOVERED_CHECK_MATE_IN_2_FEN

    assert with_extensions.score_cp >= MATE_SCORE - 128, (
        f"expected a mate-range score with check extensions active (the default), "
        f"got {with_extensions.score_cp}"
    )
    assert move_to_uci(with_extensions.best_move) == _DISCOVERED_CHECK_UCI, (
        f"expected the discovered-check move {_DISCOVERED_CHECK_UCI!r}, got "
        f"{move_to_uci(with_extensions.best_move)!r}"
    )

    monkeypatch.setattr(search_mod, "CHECK_EXTENSION_MAX_PLIES", 0)
    board_without = parse_fen(_DISCOVERED_CHECK_MATE_IN_2_FEN)
    without_extensions = Search(default_evaluator()).search(board_without, SearchLimits(max_depth=2))
    assert board_without.to_fen() == _DISCOVERED_CHECK_MATE_IN_2_FEN

    assert without_extensions.score_cp < MATE_SCORE - 128, (
        f"with check extensions disabled, a nominal depth-2 search should fall "
        f"exactly one ply short of the forced mate and report an ordinary "
        f"material evaluation instead of a mate score -- got "
        f"{without_extensions.score_cp}, suggesting either the disable didn't "
        f"take effect or this fixture no longer needs the extension to be solved"
    )


def test_ext_remaining_budget_directly_gates_the_extension() -> None:
    """The same mechanism as the test above, isolated to a single node via a
    direct `_negamax` call on the exact post-forced-reply position (White to
    move, a forced mate one ply away): with `ext_remaining=0`, the mating
    move's own check is denied its extension, so the child call lands on
    `depth == 0` and is handed to `_quiescence` -- which never checks for
    checkmate -- reporting a plain material evaluation instead. With a real
    budget, the same mating move is extended, the child call still generates
    moves normally, and the forced mate is found and scored.
    """
    board_no_budget = parse_fen(_FEN_AFTER_FORCED_REPLY)
    ctx_no_budget = _SearchCtx(SearchLimits(max_depth=1), deadline=None, stop_event=None, extra_stop=None)
    score_no_budget = Search(default_evaluator())._negamax(
        board_no_budget, 1, -INF, INF, 0, ctx_no_budget, ext_remaining=0
    )
    assert board_no_budget.to_fen() == _FEN_AFTER_FORCED_REPLY
    assert score_no_budget < MATE_SCORE - 128, (
        f"ext_remaining=0 should suppress the extension entirely, landing on "
        f"depth == 0 -> quiescence (no mate detection at all) -- got {score_no_budget}"
    )

    board_with_budget = parse_fen(_FEN_AFTER_FORCED_REPLY)
    ctx_with_budget = _SearchCtx(
        SearchLimits(max_depth=1), deadline=None, stop_event=None, extra_stop=None
    )
    score_with_budget = Search(default_evaluator())._negamax(
        board_with_budget,
        1,
        -INF,
        INF,
        0,
        ctx_with_budget,
        ext_remaining=search_mod.CHECK_EXTENSION_MAX_PLIES,
    )
    assert board_with_budget.to_fen() == _FEN_AFTER_FORCED_REPLY
    assert score_with_budget >= MATE_SCORE - 128, (
        f"a real ext_remaining budget should let the mating move's own check "
        f"extend the child call past depth == 0, finding the forced mate -- got "
        f"{score_with_budget}"
    )
