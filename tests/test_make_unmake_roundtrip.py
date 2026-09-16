"""Make/unmake round-trip correctness (architecture.md §13, Milestone 1).

For every pseudo-legal move in a battery of positions, `Board.make_move`
followed by `Board.unmake_move` must restore *every* field exactly:

    - all 12 piece bitboards (`board.pieces[color][piece_type]`)
    - both occupancy unions (`board.occupied_co[WHITE]`, `board.occupied_co[BLACK]`)
    - the combined occupancy (`board.occupied`)
    - the mailbox cache (`board.mailbox`)
    - castling rights (`board.castling_rights`)
    - the en-passant target square (`board.ep_square`)
    - the halfmove clock (`board.halfmove_clock`)
    - side to move (`board.side_to_move`)
    - the Zobrist hash (`board.zobrist_hash`)

This is deliberately independent of `test_zobrist.py` (which checks the hash
against the from-scratch oracle) and of `test_board_invariants.py` (which
checks internal cross-field consistency): this file only checks that
`unmake_move` is a perfect inverse of `make_move`, move by move, field by
field, on a wide battery of positions (including the perft reference
positions from §14 and hand-built edge cases: en passant, promotion,
underpromotion, castling in both directions, and castling-rights loss via
rook capture).

We also test round-tripping across a randomized walk of *legal* moves several
plies deep from each seed position, so the pseudo-legal batteries are
exercised not just at the handful of hand-picked FENs but at many positions
reachable from them (different castling-rights combinations, en-passant
squares appearing and expiring, promoted material, etc.).
"""

from __future__ import annotations

import random

import pytest

from chessengine.constants import BLACK, WHITE
from chessengine.fen import parse_fen
from chessengine.move import move_to_uci
from chessengine.movegen import generate_legal_moves, generate_pseudo_legal_moves

# --- Battery of positions -----------------------------------------------------
#
# The six §14 perft reference positions (they between them exercise every
# legality-adjacent rule the generator has a dedicated path for: promotion
# incl. underpromotion, en passant incl. the discovered-check edge case,
# castling through/into/out of check, and castling rights lost by rook
# capture) plus a handful of hand-built positions targeting en passant and
# promotion/underpromotion specifically, so those flags are exercised even
# if a particular perft position's move list happens not to hit every flag
# value.

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
PERFT3_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
PERFT4_FEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
PERFT5_FEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
PERFT6_FEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

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
# Both kings/rooks untouched, all four rights present, black to move: used to
# exercise black's castling moves specifically (perft position 4 already
# covers white's).
BLACK_CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
# White knight can capture black's rook sitting on its untouched home square
# h8: this must strip black's kingside right via the capture, not a move of
# the black king or rook itself.
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


# --- Snapshot / comparison machinery ------------------------------------------


def _snapshot(board):
    """Capture every field `make_move`/`unmake_move` touches, deeply enough
    that later mutation of `board` cannot retroactively change the snapshot
    (bitboards are plain ints, so only the outer/inner lists need copying)."""
    return {
        "pieces": [row[:] for row in board.pieces],
        "occupied_co": board.occupied_co[:],
        "occupied": board.occupied,
        "mailbox": board.mailbox[:],
        "side_to_move": board.side_to_move,
        "castling_rights": board.castling_rights,
        "ep_square": board.ep_square,
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
        "zobrist_hash": board.zobrist_hash,
        "history_len": len(board.history),
        "position_history": board.position_history[:],
    }


_PIECE_NAMES = ("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING")
_COLOR_NAMES = ("WHITE", "BLACK")


def _assert_roundtrip_restored(before: dict, after: dict, *, fen: str, uci: str) -> None:
    """Field-by-field comparison with a message that pinpoints exactly which
    part of the position failed to round-trip, for which move, from which
    starting FEN."""
    ctx = f"move={uci!r} from fen={fen!r}"

    for color in (WHITE, BLACK):
        for ptype in range(6):
            b_before = before["pieces"][color][ptype]
            b_after = after["pieces"][color][ptype]
            assert b_after == b_before, (
                f"{_COLOR_NAMES[color]} {_PIECE_NAMES[ptype]} bitboard not restored "
                f"({ctx}): before={b_before:#018x} after={b_after:#018x}"
            )

    for color in (WHITE, BLACK):
        assert after["occupied_co"][color] == before["occupied_co"][color], (
            f"occupied_co[{_COLOR_NAMES[color]}] not restored ({ctx})"
        )

    assert after["occupied"] == before["occupied"], f"occupied not restored ({ctx})"

    if after["mailbox"] != before["mailbox"]:
        diffs = [
            sq
            for sq in range(64)
            if after["mailbox"][sq] != before["mailbox"][sq]
        ]
        raise AssertionError(f"mailbox not restored at squares {diffs} ({ctx})")

    assert after["side_to_move"] == before["side_to_move"], f"side_to_move not restored ({ctx})"
    assert after["castling_rights"] == before["castling_rights"], (
        f"castling_rights not restored ({ctx}): "
        f"before={before['castling_rights']:#06b} after={after['castling_rights']:#06b}"
    )
    assert after["ep_square"] == before["ep_square"], f"ep_square not restored ({ctx})"
    assert after["halfmove_clock"] == before["halfmove_clock"], f"halfmove_clock not restored ({ctx})"
    assert after["fullmove_number"] == before["fullmove_number"], f"fullmove_number not restored ({ctx})"
    assert after["zobrist_hash"] == before["zobrist_hash"], (
        f"zobrist_hash not restored ({ctx}): "
        f"before={before['zobrist_hash']:#018x} after={after['zobrist_hash']:#018x}"
    )
    assert after["history_len"] == before["history_len"], (
        f"board.history was not popped back to its prior length ({ctx})"
    )
    assert after["position_history"] == before["position_history"], (
        f"position_history was not popped back to its prior contents ({ctx})"
    )


def _assert_full_roundtrip(board, move) -> None:
    """`make_move` then `unmake_move` for a single pseudo-legal move must
    restore `board` to bit-for-bit, field-for-field the state it was in
    beforehand."""
    fen_before = board.to_fen()
    uci = move_to_uci(move)
    before = _snapshot(board)

    board.make_move(move)
    board.unmake_move()

    after = _snapshot(board)
    _assert_roundtrip_restored(before, after, fen=fen_before, uci=uci)


def _assert_all_pseudo_legal_roundtrip(board) -> None:
    """Round-trip every pseudo-legal move available in `board`'s current
    position, one at a time (each move is made and unmade before the next
    is tried, so every trial starts from the identical, confirmed-restored
    position)."""
    pseudo_legal_moves = generate_pseudo_legal_moves(board)
    # A sanity floor: every position in the battery below has at least one
    # legal (hence pseudo-legal) move available, so an empty list here would
    # itself indicate a broken fixture rather than a genuine dead position.
    assert pseudo_legal_moves, f"no pseudo-legal moves generated for fen={board.to_fen()!r}"
    for move in pseudo_legal_moves:
        _assert_full_roundtrip(board, move)


# --- Tests: the fixed battery -------------------------------------------------


@pytest.mark.parametrize("position_fen", BATTERY_FENS)
def test_roundtrip_every_pseudo_legal_move(position_fen: str) -> None:
    board = parse_fen(position_fen)
    _assert_all_pseudo_legal_roundtrip(board)


# --- Tests: dedicated edge cases -----------------------------------------------
#
# These duplicate positions already present in BATTERY_FENS in spirit, but
# are named individually so a failure immediately identifies *which* rule
# broke, without needing to cross-reference the FEN back to a comment.


def test_roundtrip_en_passant_capture_white_to_move() -> None:
    board = parse_fen(EP_WHITE_TO_MOVE_FEN)
    assert board.ep_square is not None
    moves = generate_pseudo_legal_moves(board)
    ep_moves = [m for m in moves if move_to_uci(m).startswith("e5d6")]
    assert ep_moves, "expected an en passant capture e5d6 to be generated"
    for move in ep_moves:
        _assert_full_roundtrip(board, move)


def test_roundtrip_en_passant_capture_black_to_move() -> None:
    board = parse_fen(EP_BLACK_TO_MOVE_FEN)
    assert board.ep_square is not None
    moves = generate_pseudo_legal_moves(board)
    ep_moves = [m for m in moves if move_to_uci(m).startswith("d4e3")]
    assert ep_moves, "expected an en passant capture d4e3 to be generated"
    for move in ep_moves:
        _assert_full_roundtrip(board, move)


def test_roundtrip_quiet_and_capturing_promotion_white() -> None:
    board = parse_fen(WHITE_PROMOTION_FEN)
    moves = generate_pseudo_legal_moves(board)
    promo_moves = [m for m in moves if move_to_uci(m)[-1] in "nbrq"]
    # b7 has both a quiet promotion (b8) and two capturing promotions
    # (a8, c8) available, times four promotion pieces each: 12 promotion
    # moves from that pawn alone.
    assert len(promo_moves) >= 12, f"expected several promotion moves, got {len(promo_moves)}"
    for move in promo_moves:
        _assert_full_roundtrip(board, move)


def test_roundtrip_quiet_and_capturing_promotion_black() -> None:
    board = parse_fen(BLACK_PROMOTION_FEN)
    moves = generate_pseudo_legal_moves(board)
    promo_moves = [m for m in moves if move_to_uci(m)[-1] in "nbrq"]
    assert len(promo_moves) >= 12, f"expected several promotion moves, got {len(promo_moves)}"
    for move in promo_moves:
        _assert_full_roundtrip(board, move)


def test_roundtrip_kingside_castle_white() -> None:
    board = parse_fen(KIWIPETE_FEN)
    moves = generate_pseudo_legal_moves(board)
    castles = [m for m in moves if move_to_uci(m) == "e1g1"]
    assert castles, "expected white kingside castle e1g1 to be generated"
    _assert_full_roundtrip(board, castles[0])


def test_roundtrip_queenside_castle_white() -> None:
    board = parse_fen(KIWIPETE_FEN)
    moves = generate_pseudo_legal_moves(board)
    castles = [m for m in moves if move_to_uci(m) == "e1c1"]
    assert castles, "expected white queenside castle e1c1 to be generated"
    _assert_full_roundtrip(board, castles[0])


def test_roundtrip_queenside_castle_black() -> None:
    board = parse_fen(BLACK_CASTLE_FEN)
    moves = generate_pseudo_legal_moves(board)
    castles = [m for m in moves if move_to_uci(m) == "e8c8"]
    assert castles, "expected black queenside castle e8c8 to be generated"
    _assert_full_roundtrip(board, castles[0])


def test_roundtrip_kingside_castle_black() -> None:
    board = parse_fen(BLACK_CASTLE_FEN)
    moves = generate_pseudo_legal_moves(board)
    castles = [m for m in moves if move_to_uci(m) == "e8g8"]
    assert castles, "expected black kingside castle e8g8 to be generated"
    _assert_full_roundtrip(board, castles[0])


def test_roundtrip_rook_capture_spoils_castling_rights() -> None:
    # White's knight captures black's untouched rook on h8, which must strip
    # black's kingside castling right even though no black king or rook
    # *move* is involved — only `board.castling_rights`'s spoiler table
    # keyed by both `frm` and `to` (§5.7, §6) catches this.
    board = parse_fen(ROOK_CAPTURE_SPOILS_RIGHTS_FEN)
    assert board.castling_rights != 0
    moves = generate_pseudo_legal_moves(board)
    capturing_moves = [m for m in moves if move_to_uci(m) == "g6h8"]
    assert capturing_moves, "expected the knight capture g6xh8 to be generated"
    _assert_full_roundtrip(board, capturing_moves[0])


# --- Tests: randomized deeper walks --------------------------------------------
#
# Beyond the fixed battery above, walk a short randomized sequence of *legal*
# moves from each seed position and round-trip *every pseudo-legal move*
# available at each visited node along the way. This exercises many more
# positions than the hand-picked battery alone: castling rights disappearing
# one at a time, en-passant squares appearing then expiring, captures
# thinning out material, promoted pieces appearing on the board, etc.

_RANDOM_WALK_SEED_FENS = [
    STARTPOS_FEN,
    KIWIPETE_FEN,
    PERFT3_FEN,
    PERFT4_FEN,
    PERFT5_FEN,
    PERFT6_FEN,
]
_RANDOM_WALK_PLIES = 6
_RANDOM_WALK_TRIALS_PER_SEED = 3


@pytest.mark.parametrize("seed_fen", _RANDOM_WALK_SEED_FENS)
def test_roundtrip_along_randomized_legal_walks(seed_fen: str) -> None:
    rng = random.Random((hash(seed_fen) ^ 0xC0FFEE) & 0xFFFF_FFFF)
    for trial in range(_RANDOM_WALK_TRIALS_PER_SEED):
        board = parse_fen(seed_fen)
        for _ply in range(_RANDOM_WALK_PLIES):
            # Round-trip every pseudo-legal move at this node before
            # advancing the walk, so every visited position (not just the
            # seed) is fully exercised.
            _assert_all_pseudo_legal_roundtrip(board)

            legal_moves = generate_legal_moves(board)
            if not legal_moves:
                break  # checkmate/stalemate reached: nothing left to walk into
            move = rng.choice(legal_moves)
            board.make_move(move)
        # Also exercise the final position reached by the walk.
        _assert_all_pseudo_legal_roundtrip(board)


# --- Test: nested make/unmake (search-shaped usage) ----------------------------
#
# `Search._negamax` makes a move, recurses, and unmakes — many levels deep,
# not just one, with `UndoInfo`s pushed and popped in a genuinely nested (LIFO
# stack) fashion rather than the flat make/unmake/make/unmake sequence the
# tests above exercise. A bug that only manifests once the history stack is
# more than one entry deep would slip past those but not this one.
#
# Every pseudo-legal move at every *visited* node is round-trip-checked (not
# just the moves actually descended into); the fan-out actually recursed into
# is capped at each level so the total node count stays small regardless of
# how many legal moves a given position has.

# Number of moves actually recursed *into* at each level (levels beyond the
# end of this list are leaves: every move there is round-trip-checked, but
# none are descended into further). Keeps the total node count bounded
# regardless of how many legal moves a given position has (~30-50 here),
# while still reaching a real stack depth of len(_NESTED_FANOUT) + 1.
_NESTED_FANOUT = [4, 2, 1]


def _nested_roundtrip_check(board, level: int) -> None:
    moves = generate_pseudo_legal_moves(board)
    can_recurse = level < len(_NESTED_FANOUT)
    fanout = _NESTED_FANOUT[level] if can_recurse else 0
    for i, move in enumerate(moves):
        uci = move_to_uci(move)
        fen_here = board.to_fen()
        before = _snapshot(board)

        board.make_move(move)
        if can_recurse and i < fanout:
            _nested_roundtrip_check(board, level + 1)
        board.unmake_move()

        after = _snapshot(board)
        _assert_roundtrip_restored(before, after, fen=fen_here, uci=uci)


@pytest.mark.parametrize("position_fen", [STARTPOS_FEN, KIWIPETE_FEN, PERFT4_FEN])
def test_roundtrip_nested_make_unmake_restores_every_level(position_fen: str) -> None:
    board = parse_fen(position_fen)
    _nested_roundtrip_check(board, level=0)
