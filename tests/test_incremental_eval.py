"""Incremental material+PST score correctness tests (architecture.md §10.1,
§13).

`board.material_pst_score` is a White-relative running total ("White's
PIECE_VALUE+PST total minus Black's", per its own docstring in board.py)
maintained incrementally inside `Board._add_piece`/`_remove_piece` — the same
single-writer discipline that keeps `pieces`/`occupied_co`/`occupied`/
`mailbox`/`zobrist_hash` in sync (architecture.md §3.2, §6, §13). This module
mirrors tests/test_zobrist.py's pattern (an incrementally-maintained field
must always agree with an independent from-scratch oracle, and `unmake_move`
must restore its *exact* prior value, not merely one the oracle would also
accept) and tests/test_make_unmake_roundtrip.py's battery-of-positions
round-trip pattern, applied to this one additional field:

1. After every move of many random legal games (seeded from the standard
   start position plus the architecture.md §14 perft reference positions,
   exactly as tests/test_zobrist.py does for `board.zobrist_hash`),
   `board.material_pst_score` must equal
   `evaluate._side_score(board, WHITE) - evaluate._side_score(board, BLACK)`
   -- `_side_score` is evaluate.py's own from-scratch, non-incremental oracle
   for this quantity (see its docstring: "Kept as the differential-test
   oracle for the incremental running total").

2. For every pseudo-legal move in a battery of positions (the same battery
   tests/test_make_unmake_roundtrip.py uses: the §14 perft references plus
   hand-built en passant/promotion/castling/rights-spoiling edge cases),
   `make_move` followed by `unmake_move` must restore `material_pst_score`
   to its *exact* prior value.

3. A fresh `Board.starting_position()` and a parse_fen(...) battery over the
   §14 reference FENs must all have `material_pst_score` exactly equal to
   the from-scratch oracle, before any move is ever played -- ruling out
   `fen.py`/`Board._add_piece` (rather than `make_move`/`unmake_move`) as the
   source of any later mismatch.
"""

from __future__ import annotations

import random

import pytest

from chessengine import evaluate
from chessengine.board import Board
from chessengine.constants import BLACK, WHITE
from chessengine.fen import parse_fen
from chessengine.move import move_to_uci
from chessengine.movegen import generate_legal_moves, generate_pseudo_legal_moves

# --- Starting positions (architecture.md §14 perft reference set) ----------
#
# Reused verbatim from tests/test_zobrist.py's STARTING_FENS: between them
# these exercise promotion/underpromotion, en passant, castling through/
# into/out of check, and rights lost on rook capture -- the same position
# features that would expose a broken incremental material_pst_score if the
# _add_piece/_remove_piece bookkeeping ever drifted from a from-scratch
# recompute.
STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POSITION_3_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POSITION_4_FEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POSITION_5_FEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POSITION_6_FEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

STARTING_FENS = [
    STARTPOS_FEN,
    KIWIPETE_FEN,
    POSITION_3_FEN,
    POSITION_4_FEN,
    POSITION_5_FEN,
    POSITION_6_FEN,
]

# How many plies each random game is played for (games that reach
# checkmate/stalemate first simply stop early).
MAX_PLIES = 60

# Several independent RNG seeds per starting position, so the random walk
# through the game tree differs across runs of the parametrized test without
# being nondeterministic (a fixed seed makes a failure reproducible).
SEEDS = range(8)


def _oracle_material_pst_score(board: Board) -> int:
    """From-scratch, White-relative material+PST total, independent of
    `board.material_pst_score`'s incremental bookkeeping -- built directly
    from `evaluate._side_score`, evaluate.py's own from-scratch oracle for
    `material_pst_term` (architecture.md §10.1's closing note), the same
    contract `board.material_pst_score`'s docstring documents: White's total
    minus Black's."""
    return evaluate._side_score(board, WHITE) - evaluate._side_score(board, BLACK)


def _random_legal_move(board: Board, rng: random.Random) -> int | None:
    """Return a uniformly random legal move for `board`'s side to move, or
    None if the game has ended (checkmate or stalemate)."""
    moves = generate_legal_moves(board)
    if not moves:
        return None
    return rng.choice(moves)


# --- 1. Incremental score matches the from-scratch oracle after every move -


@pytest.mark.parametrize("start_fen", STARTING_FENS)
@pytest.mark.parametrize("seed", SEEDS)
def test_material_pst_score_matches_oracle_after_every_move(start_fen: str, seed: int) -> None:
    """`board.material_pst_score` must equal
    `evaluate._side_score(board, WHITE) - evaluate._side_score(board, BLACK)`
    after every move of a random legal game, regardless of which random
    branch of the game tree is taken."""
    board = parse_fen(start_fen)
    rng = random.Random(seed)

    # Sanity check on the starting position itself before any moves are
    # played: `fen.parse_fen` seeds `material_pst_score` via `_add_piece`
    # (board.py), so this should hold trivially, but a passing check here
    # rules out the starting fixture itself as a source of later mismatches.
    assert board.material_pst_score == _oracle_material_pst_score(board)

    for ply in range(1, MAX_PLIES + 1):
        move = _random_legal_move(board, rng)
        if move is None:
            break
        board.make_move(move)
        assert board.material_pst_score == _oracle_material_pst_score(board), (
            f"material_pst_score diverged from the from-scratch oracle at ply {ply} "
            f"starting from {start_fen!r} (seed={seed}): "
            f"board.material_pst_score={board.material_pst_score}, "
            f"oracle={_oracle_material_pst_score(board)}"
        )


# --- 2. unmake_move restores the exact prior score, one move at a time -----
#
# Battery of positions reused verbatim from tests/test_make_unmake_roundtrip.py:
# the six §14 perft reference positions plus hand-built en passant/promotion/
# castling/rights-spoiling edge cases, so the incremental score is exercised
# across every kind of move `make_move`/`unmake_move` treats specially
# (capture, en passant capture, promotion incl. underpromotion, castling, and
# a capture that spoils castling rights without moving the king or rook).

PERFT3_FEN = POSITION_3_FEN
PERFT4_FEN = POSITION_4_FEN
PERFT5_FEN = POSITION_5_FEN
PERFT6_FEN = POSITION_6_FEN

# White pawn on e5 may capture en passant onto d6.
EP_WHITE_TO_MOVE_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 4"
# Black pawn on d4 may capture en passant onto e3.
EP_BLACK_TO_MOVE_FEN = "rnbqkbnr/ppp1pppp/8/8/3pP3/5N2/PPPP1PPP/RNBQKB1R b KQkq e3 0 3"
# White pawn one step from promoting on every file-adjacent capture/quiet case.
WHITE_PROMOTION_FEN = "n1n2n2/1P1P1P2/8/4k3/8/8/8/4K3 w - - 0 1"
# Black pawn one step from promoting (quiet + capturing), mirrored.
BLACK_PROMOTION_FEN = "4k3/8/8/8/8/8/1p1p1p2/N1N2N2 b - - 0 1"
# Only queenside rights remain for both sides.
QUEENSIDE_ONLY_FEN = "r3k3/pppppppp/8/8/8/8/PPPPPPPP/R3K3 w Qq - 4 5"
# Both kings/rooks untouched, all four rights present, black to move.
BLACK_CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
# White knight can capture black's untouched rook on h8, spoiling black's
# kingside right via the capture rather than a king/rook move.
ROOK_CAPTURE_SPOILS_RIGHTS_FEN = "4k2r/8/6N1/8/8/8/8/4K3 w k - 0 1"

BATTERY_FENS = [
    STARTPOS_FEN,
    KIWIPETE_FEN,
    PERFT3_FEN,
    PERFT4_FEN,
    PERFT5_FEN,
    PERFT6_FEN,
    EP_WHITE_TO_MOVE_FEN,
    EP_BLACK_TO_MOVE_FEN,
    WHITE_PROMOTION_FEN,
    BLACK_PROMOTION_FEN,
    QUEENSIDE_ONLY_FEN,
    BLACK_CASTLE_FEN,
    ROOK_CAPTURE_SPOILS_RIGHTS_FEN,
]


def _assert_material_pst_score_roundtrips(board: Board, move: int) -> None:
    """`make_move` then `unmake_move` for a single pseudo-legal move must
    restore `board.material_pst_score` to its *exact* prior value -- not
    merely one that happens to still agree with the from-scratch oracle."""
    uci = move_to_uci(move)
    fen_before = board.to_fen()
    score_before = board.material_pst_score

    board.make_move(move)
    board.unmake_move()

    assert board.material_pst_score == score_before, (
        f"unmake_move failed to restore the exact prior material_pst_score "
        f"(move={uci!r} from fen={fen_before!r}): "
        f"before={score_before}, after={board.material_pst_score}"
    )


@pytest.mark.parametrize("position_fen", BATTERY_FENS)
def test_material_pst_score_roundtrips_every_pseudo_legal_move(position_fen: str) -> None:
    board = parse_fen(position_fen)
    pseudo_legal_moves = generate_pseudo_legal_moves(board)
    assert pseudo_legal_moves, f"no pseudo-legal moves generated for fen={position_fen!r}"
    for move in pseudo_legal_moves:
        _assert_material_pst_score_roundtrips(board, move)


# --- 2b. Same check along randomized walks several plies deep --------------
#
# Beyond the fixed battery above, walk a short randomized sequence of *legal*
# moves from each seed position and round-trip *every pseudo-legal move*
# available at each visited node along the way -- exercising many more
# positions than the hand-picked battery alone (castling rights disappearing
# one at a time, en-passant squares appearing then expiring, captures
# thinning out material, promoted pieces appearing on the board, etc.),
# mirroring tests/test_make_unmake_roundtrip.py's
# `test_roundtrip_along_randomized_legal_walks`.

_RANDOM_WALK_PLIES = 6
_RANDOM_WALK_TRIALS_PER_SEED = 3


@pytest.mark.parametrize("seed_fen", STARTING_FENS)
def test_material_pst_score_roundtrips_along_randomized_legal_walks(seed_fen: str) -> None:
    rng = random.Random((hash(seed_fen) ^ 0xC0FFEE) & 0xFFFF_FFFF)
    for _trial in range(_RANDOM_WALK_TRIALS_PER_SEED):
        board = parse_fen(seed_fen)
        for _ply in range(_RANDOM_WALK_PLIES):
            for move in generate_pseudo_legal_moves(board):
                _assert_material_pst_score_roundtrips(board, move)

            legal_moves = generate_legal_moves(board)
            if not legal_moves:
                break  # checkmate/stalemate reached: nothing left to walk into
            move = rng.choice(legal_moves)
            board.make_move(move)
        # Also exercise the final position reached by the walk.
        for move in generate_pseudo_legal_moves(board):
            _assert_material_pst_score_roundtrips(board, move)


# --- 3. Freshly built positions match the oracle before any move is played -


def test_starting_position_material_pst_score_matches_oracle() -> None:
    """`Board.starting_position()` (which lazily delegates to
    `fen.parse_fen`) must have `material_pst_score` exactly equal to the
    from-scratch oracle, before any move is ever played."""
    board = Board.starting_position()
    assert board.material_pst_score == _oracle_material_pst_score(board)
    # Sanity: the standard starting position is materially symmetric, so the
    # White-relative total should be exactly 0.
    assert board.material_pst_score == 0


@pytest.mark.parametrize("start_fen", STARTING_FENS)
def test_parse_fen_battery_material_pst_score_matches_oracle(start_fen: str) -> None:
    """Every §14 reference FEN, freshly parsed with no moves played, must
    have `material_pst_score` exactly equal to the from-scratch oracle -- a
    failure here would point at `fen.py`/`Board._add_piece`, not at
    `make_move`/`unmake_move`."""
    board = parse_fen(start_fen)
    assert board.material_pst_score == _oracle_material_pst_score(board), (
        f"material_pst_score mismatch on freshly parsed fen={start_fen!r}: "
        f"board.material_pst_score={board.material_pst_score}, "
        f"oracle={_oracle_material_pst_score(board)}"
    )
