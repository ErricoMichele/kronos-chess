"""Zobrist hashing correctness tests (architecture.md §7, §13).

Two invariants are checked after every move of many random legal games,
seeded from a battery of starting positions (the standard start position
plus positions from architecture.md §14 that exercise castling rights on
both sides, en passant, and promotion — exactly the position features the
incremental side/castling/en-passant/piece keys need to track correctly):

1. `board.zobrist_hash`, maintained incrementally inside `make_move`/
   `unmake_move` (§6), must always agree with `zobrist.compute_hash(board)`,
   an independent, from-scratch, non-incremental oracle (§7).
2. `unmake_move` must restore the *exact* prior hash value -- not merely one
   that happens to satisfy (1) -- both for a single make/unmake pair and
   across a whole chain of them unwound in reverse order.
"""

from __future__ import annotations

import random

import pytest

from chessengine import zobrist
from chessengine.board import Board
from chessengine.fen import board_to_fen, parse_fen
from chessengine.movegen import generate_legal_moves, move_from_uci

# --- Starting positions -------------------------------------------------
#
# Reused from architecture.md §14's perft reference positions: between
# them they exercise promotion/underpromotion, en passant, castling
# through/into/out of check, and rights lost on rook capture -- the exact
# position features that would expose a broken piece/side/castling/ep
# Zobrist key if the incremental bookkeeping in `make_move`/`unmake_move`
# ever drifted from a from-scratch recompute.
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
# checkmate/stalemate first simply stop early -- see `_random_legal_move`
# call sites below, all of which break out of their loop when
# `generate_legal_moves` returns an empty list).
MAX_PLIES = 60

# Several independent RNG seeds per starting position, so the random walk
# through the game tree differs across runs of the parametrized test
# without being nondeterministic (a fixed seed makes a failure
# reproducible).
SEEDS = range(8)


def _random_legal_move(board: Board, rng: random.Random) -> int | None:
    """Return a uniformly random legal move for `board`'s side to move, or
    None if the game has ended (checkmate or stalemate)."""
    moves = generate_legal_moves(board)
    if not moves:
        return None
    return rng.choice(moves)


# --- 1. Incremental hash matches the from-scratch oracle after every move ---


@pytest.mark.parametrize("start_fen", STARTING_FENS)
@pytest.mark.parametrize("seed", SEEDS)
def test_hash_matches_oracle_after_every_move(start_fen: str, seed: int) -> None:
    """`board.zobrist_hash` must equal `zobrist.compute_hash(board)` after
    every move of a random legal game, regardless of which random branch of
    the game tree is taken."""
    board = parse_fen(start_fen)
    rng = random.Random(seed)

    # Sanity check on the starting position itself before any moves are
    # played: `fen.parse_fen` seeds `zobrist_hash` via the same oracle
    # (fen.py), so this should hold trivially, but a passing test here
    # rules out the starting fixture itself as a source of later mismatches.
    assert board.zobrist_hash == zobrist.compute_hash(board)

    for ply in range(1, MAX_PLIES + 1):
        move = _random_legal_move(board, rng)
        if move is None:
            break
        board.make_move(move)
        assert board.zobrist_hash == zobrist.compute_hash(board), (
            f"zobrist_hash diverged from compute_hash oracle at ply {ply} "
            f"starting from {start_fen!r} (seed={seed})"
        )


# --- 2a. unmake_move restores the exact prior hash, one move at a time ------


@pytest.mark.parametrize("start_fen", STARTING_FENS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unmake_restores_exact_hash_single_ply(start_fen: str, seed: int) -> None:
    """For every move played along a random legal game, making the move and
    immediately unmaking it must restore *exactly* the pre-move hash value
    (bit-for-bit, not merely a value the oracle would also accept) -- the
    move is then re-applied so the game continues down the same random
    trajectory for the next iteration."""
    board = parse_fen(start_fen)
    rng = random.Random(seed)

    for ply in range(1, MAX_PLIES + 1):
        move = _random_legal_move(board, rng)
        if move is None:
            break

        hash_before = board.zobrist_hash
        board.make_move(move)
        board.unmake_move()
        assert board.zobrist_hash == hash_before, (
            f"unmake_move failed to restore the exact prior hash at ply {ply} "
            f"starting from {start_fen!r} (seed={seed})"
        )

        # Re-apply the same move so the random walk continues as if the
        # make/unmake probe above had never happened.
        board.make_move(move)


# --- 2b. unmake_move restores exact hashes across a whole unwound chain -----


@pytest.mark.parametrize("start_fen", STARTING_FENS)
@pytest.mark.parametrize("seed", SEEDS)
def test_unmake_chain_restores_hashes_in_order(start_fen: str, seed: int) -> None:
    """Play a full random game recording the hash immediately before each
    move, then unmake every move in reverse order and assert the hash lands
    back on exactly the recorded value at each step. This catches drift
    that might only appear across a longer make/unmake chain (e.g. a bug in
    `UndoInfo`/`unmake_move` that happens to cancel out for a single pair
    but not for a whole game's worth of undo history)."""
    board = parse_fen(start_fen)
    rng = random.Random(seed)
    initial_hash = board.zobrist_hash

    hashes_before_move: list[int] = []
    for _ in range(MAX_PLIES):
        move = _random_legal_move(board, rng)
        if move is None:
            break
        hashes_before_move.append(board.zobrist_hash)
        board.make_move(move)

    # Unwind every move played, in reverse order.
    for expected_hash in reversed(hashes_before_move):
        board.unmake_move()
        assert board.zobrist_hash == expected_hash

    # Fully unwound: back to the exact starting hash, an empty undo stack,
    # and (redundantly, via the independent oracle) the same value again.
    assert board.zobrist_hash == initial_hash
    assert board.history == []
    assert board.zobrist_hash == zobrist.compute_hash(board)


# --- 3. Targeted regression: the en-passant key must not outlive its ply ----


def test_ep_key_cleared_once_ep_square_expires() -> None:
    """Direct regression test for the rule documented in architecture.md §7:
    the en-passant Zobrist key represents 'en passant was available in this
    position', so it must be XORed back out on the very next move even when
    that move is not the en-passant capture itself -- otherwise a position
    with a long-expired ep square would hash differently from the legally
    identical position reached without ever passing through one."""
    board = Board.starting_position()

    double_push = move_from_uci(board, "e2e4")
    board.make_move(double_push)
    assert board.ep_square is not None
    assert board.zobrist_hash == zobrist.compute_hash(board)

    # Black's reply does not capture en passant, so the ep square (and its
    # Zobrist key) must disappear after this move, not linger.
    quiet_reply = move_from_uci(board, "b8c6")
    board.make_move(quiet_reply)
    assert board.ep_square is None
    assert board.zobrist_hash == zobrist.compute_hash(board)

    # Build the identical resulting position entirely independently (via a
    # FEN round trip, which re-seeds the hash from scratch in fen.py) and
    # confirm it hashes identically -- this is what would fail if a stale
    # en-passant key had survived inside `board`'s incremental hash.
    fresh = parse_fen(board_to_fen(board))
    assert fresh.zobrist_hash == board.zobrist_hash

    # Unmaking both moves must retrace the hash exactly, ep key included.
    hash_after_double_push = zobrist.compute_hash(parse_fen(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    ))
    board.unmake_move()
    assert board.zobrist_hash == hash_after_double_push
    board.unmake_move()
    assert board.zobrist_hash == zobrist.compute_hash(Board.starting_position())
