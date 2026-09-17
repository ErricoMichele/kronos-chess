"""Milestone 4 exit-criteria tests for `search.py` (architecture.md §13, §15):
a short, fast battery of mate-in-1, mate-in-2, and "don't hang a piece"
positions that `Search.search` must solve within a small, fixed depth
budget -- the whole file running in well under a second.

Per §15's Milestone 4 exit criteria: "`test_search_tactics.py` (mate-in-N
suite) solved within budget." Per §13: "a short list of mate-in-1/2/3 and
'don't hang a piece' FENs that `Search.search` must solve (correct
`best_move`) within a small fixed depth/node budget."

**Independent verification, not a hardcoded 'book' move.** Rather than
asserting the engine's `best_move` equals one specific memorized algebraic
move (fragile: several different moves can equally deliver/force the same
mate, and a hand-transcribed "known solution" is itself a plausible source
of an incorrect test), every assertion below is checked against a small,
self-contained oracle built only on `board.py`/`movegen.py` -- the two
modules Milestone 1's perft suite already proves correct (architecture.md
§13, §14) -- plus, for the "don't hang a piece" cases, `search.py`'s own
`see_ge` static-exchange helper (§9.5), which is a pure position-analysis
function, not the negamax/iterative-deepening machinery under test here
(the same reasoning `test_search.py`'s own differential oracle already
relies on for its capture filtering).

- For mate positions, `_move_forces_checkmate` brute-force verifies that
  the engine's move actually forces checkmate within the stated number of
  the engine's own moves, trying *every* legal reply for the opponent at
  each of their turns (adversarial: the opponent is assumed to defend as
  well as it possibly can) -- exactly what "mate in N" means.
- For "don't hang a piece" positions, `_hangs_material` checks that no
  legal reply available to the opponent, immediately after the engine's
  move, has a static-exchange value at or above a given threshold -- i.e.
  nothing the opponent could do wins material for free.

Every FEN below was independently confirmed, while writing this file, to
actually support its claimed tactic (via these same two oracle functions
run over every root move, not just the engine's), so a failure here means
`Search.search` regressed, not that the fixture's premise is wrong.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_captures, generate_legal_moves
from chessengine.search import Search, SearchLimits, see_ge

# A small, fixed depth budget (architecture.md §15's "small fixed
# depth/node budget"), well within §13's "a few seconds" for the whole file.
# Iterative deepening's own "forced mate found -> stop deepening" early exit
# (`Search.search`) means a deeper ceiling costs nothing extra once a mate
# is actually found, so the same budget is reused for every case below
# rather than tuned per position.
_TACTICS_LIMITS = SearchLimits(max_depth=6)


# --- Independent oracle: "does this move force checkmate within N of the
# engine's own moves?" (architecture.md §13's mate-in-N gate) ---------------
#
# Built only on `board.py`/`movegen.py` (Milestone 1's perft-validated rules
# engine, §14) -- deliberately never on `search.py`'s own negamax, so this
# can't be trivially "validated by the thing it's validating."


def _mate_search(board: Board, plies_left: int, attacker_to_move: bool) -> bool:
    """True if, from `board`'s current position, the *attacking* side can
    force checkmate within `plies_left` half-moves, assuming the *defending*
    side always plays whichever legal reply best avoids/delays it.

    `attacker_to_move` says whose turn it currently is: the attacker
    (`True`, an existential search -- at least one of its moves must work)
    or the defender (`False`, a universal search -- *every* legal reply
    must still lead to a forced mate, since the defender will pick
    whichever one doesn't, if any does).
    """
    if plies_left <= 0:
        return False
    legal = generate_legal_moves(board)
    if attacker_to_move:
        for move in legal:
            board.make_move(move)
            opponent_legal = generate_legal_moves(board)
            if not opponent_legal:
                forced = board.in_check()  # checkmate delivered; stalemate would be a failure
            else:
                forced = _mate_search(board, plies_left - 1, attacker_to_move=False)
            board.unmake_move()
            if forced:
                return True
        return False
    else:
        if not legal:
            return board.in_check()  # already mated (good for the attacker) vs. stalemated (bad)
        for move in legal:
            board.make_move(move)
            forced = _mate_search(board, plies_left - 1, attacker_to_move=True)
            board.unmake_move()
            if not forced:
                return False  # the defender found an escape: not forced
        return True


def _move_forces_checkmate(board: Board, move: int, max_plies_after: int) -> bool:
    """True if playing `move` from `board` (mutates and fully restores
    `board`) forces checkmate against best defense within `max_plies_after`
    more half-moves. This is exactly "`move` is a correct mate-in-N move"
    for `max_plies_after == 2 * (N - 1)`: `N == 1` (an immediate mate) means
    `max_plies_after == 0`; `N == 2` means one opponent reply plus one more
    mating move (`max_plies_after == 2`); and so on.
    """
    board.make_move(move)
    try:
        opponent_legal = generate_legal_moves(board)
        if not opponent_legal:
            return board.in_check()  # mate delivered immediately by `move` itself
        if max_plies_after == 0:
            return False  # an immediate mate was required; this position isn't one
        for opponent_move in opponent_legal:
            board.make_move(opponent_move)
            forced = _mate_search(board, max_plies_after - 1, attacker_to_move=True)
            board.unmake_move()
            if not forced:
                return False
        return True
    finally:
        board.unmake_move()


# --- Independent oracle: "does the opponent have a free capture after this
# move?" (architecture.md §13's "don't hang a piece" gate) ------------------


def _hangs_material(board: Board, move: int, threshold_cp: int = 1) -> bool:
    """True if, after playing `move` (mutates and fully restores `board`),
    the opponent has some legal capture whose static-exchange value is
    `>= threshold_cp` -- i.e. `move` left something capturable for a net
    gain, rather than leaving every capture at best equal for the opponent.
    """
    board.make_move(move)
    try:
        return any(see_ge(board, reply, threshold_cp) for reply in generate_captures(board))
    finally:
        board.unmake_move()


# --- Fixtures: mate-in-1 / mate-in-2 -----------------------------------------

MATE_IN_N_CASES = [
    pytest.param(
        # Back-rank mate: Black's own f7/g7/h7 pawns block every escape
        # square, and nothing can interpose on or capture the rook's
        # 8th-rank check. 1.Ra8#.
        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
        1,
        id="mate-in-1-back-rank",
    ),
    pytest.param(
        # Scholar's mate final position (1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? and
        # now White to move): the queen takes the f7 pawn with check,
        # defended by the bishop on c4 so the king can't recapture, and
        # Black's own queen/bishop still on d8/f8 block every other escape.
        # 4.Qxf7#.
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        1,
        id="mate-in-1-scholars-mate",
    ),
    pytest.param(
        # King+queen vs. lone king, the standard mating technique's final
        # phase: 1.Kc7 (the only move that doesn't allow an escape via a7)
        # Ka7 (Black's only legal reply, since b7/b8 are controlled by the
        # White king) 2.Qa1# (or Qa3#).
        "k7/8/3K4/8/8/8/8/2Q5 w - - 0 1",
        2,
        id="mate-in-2-kq-vs-k-a",
    ),
    pytest.param(
        # Same technique from a different starting square: 1.Kc6 Kb8
        # (Black's only legal reply) 2.Qb7#.
        "k7/7Q/3K4/8/8/8/8/8 w - - 0 1",
        2,
        id="mate-in-2-kq-vs-k-b",
    ),
]


@pytest.mark.parametrize("fen, mate_in", MATE_IN_N_CASES)
def test_search_finds_forced_mate(fen: str, mate_in: int) -> None:
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, _TACTICS_LIMITS)

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )

    max_plies_after = 2 * (mate_in - 1)
    assert _move_forces_checkmate(board, result.best_move, max_plies_after), (
        f"{move_to_uci(result.best_move)!r} does not force mate in {mate_in} from {fen!r} "
        f"(engine score was {result.score_cp} at depth {result.depth})"
    )
    assert board.to_fen() == fen, "the independent mate oracle must also restore the board"


# --- Fixtures: "don't hang a piece" ------------------------------------------

HANG_PIECE_CASES = [
    pytest.param(
        # White's queen is attacked for free by the c6 pawn and is
        # otherwise undefended; it must move (here, simply capturing the
        # undefended attacker) rather than be left en prise.
        "4k3/8/2p5/3Q4/8/8/8/4K3 w - - 0 1",
        id="dont-hang-the-queen",
    ),
    pytest.param(
        # White's rook is attacked for free by the bishop on d8 along the
        # d8-h4 diagonal and is undefended; it must move off that diagonal
        # rather than be left en prise.
        "3bk3/8/8/8/7R/8/8/6K1 w - - 0 1",
        id="dont-hang-the-rook",
    ),
]


@pytest.mark.parametrize("fen", HANG_PIECE_CASES)
def test_search_does_not_hang_material(fen: str) -> None:
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, _TACTICS_LIMITS)

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )

    assert not _hangs_material(board, result.best_move), (
        f"{move_to_uci(result.best_move)!r} leaves a free capture on the board for the "
        f"opponent in {fen!r} (engine score was {result.score_cp} at depth {result.depth})"
    )
    assert board.to_fen() == fen, "the independent hang-material oracle must also restore the board"
