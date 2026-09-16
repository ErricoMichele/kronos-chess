"""Dedicated regression tests for the Milestone 1 exit criteria named
explicitly in architecture.md §15 ("Milestone 1 -- Board representation +
move generation + perft correctness"):

    "... dedicated regression tests for en passant discovered check,
    castling through/into/out of check, castling rights lost on rook
    capture, and under-promotion."

These are hand-built positions targeting one rule each, as a companion to
(not a replacement for) the exact perft node counts in architecture.md §14
and §13's `test_perft.py`: a wrong perft count tells you *that* one of these
rules is broken somewhere in the tree, these tests tell you *which* rule,
directly, and pin its exact expected behavior down as a regression.

Every "must not be legal" assertion here is paired with a positive-control
counterpart proving the same assertion would fail if the relevant rule were
simply never implemented (a generator that always returns no moves, or
never generates castling/promotions at all, would otherwise make the
negative assertions trivially true).
"""

from __future__ import annotations

import pytest

from chessengine import attacks, fen
from chessengine.constants import (
    BISHOP,
    BLACK,
    CASTLE_WK,
    CASTLE_WQ,
    KNIGHT,
    NO_PIECE,
    PAWN,
    QUEEN,
    ROOK,
    SQUARE_NAMES,
    WHITE,
    color_of,
    piece_type_of,
)
from chessengine.move import (
    EN_PASSANT,
    KING_CASTLE,
    PROMO_BISHOP,
    PROMO_BISHOP_CAP,
    PROMO_KNIGHT,
    PROMO_KNIGHT_CAP,
    PROMO_QUEEN,
    PROMO_QUEEN_CAP,
    PROMO_ROOK,
    PROMO_ROOK_CAP,
    QUEEN_CASTLE,
    move_flag,
    move_from,
    move_to,
    move_to_uci,
)
from chessengine.movegen import generate_legal_moves, generate_pseudo_legal_moves
from chessengine.zobrist import compute_hash

# --- Small test-local helpers -------------------------------------------------


def _legal_ucis(board) -> set[str]:
    return {move_to_uci(m) for m in generate_legal_moves(board)}


def _find(board, uci: str) -> int:
    """The legal move whose UCI string is exactly `uci`. UCI (from + to +
    optional promotion letter) uniquely identifies one legal move in a given
    position (architecture.md §4, §14's `divide` docstring), so a plain
    linear scan is unambiguous."""
    for m in generate_legal_moves(board):
        if move_to_uci(m) == uci:
            return m
    raise AssertionError(
        f"move {uci!r} not found among legal moves: {sorted(_legal_ucis(board))}"
    )


def _sq(name: str) -> int:
    return SQUARE_NAMES.index(name)


# ==============================================================================
# 1. En passant discovered check -- the pin-through-the-fifth-rank case
#    (architecture.md §5.8, §14 position 3)
# ==============================================================================
#
# White king a5, white pawn b5, black rook h5: all on rank 5. Black is about
# to double-push c7-c5, landing directly next to the white pawn. Capturing
# en passant (b5xc6) removes *two* pieces from rank 5 in a single move -- the
# capturing pawn (which leaves b5 for c6) and the captured pawn (which sits
# on c5, not on the destination square c6) -- laying the rank bare between
# the king and the rook. Neither the pin table (only tracks one blocker per
# ray) nor the evasion mask (the king isn't in check *before* the capture)
# can see this; architecture.md §5.8 mandates a make/unmake fallback for
# every en passant candidate specifically because of this case.

_EP_PINNED_FEN = "7k/2p5/8/KP5r/8/8/8/8 b - - 0 1"
_EP_UNPINNED_FEN = "7k/2p5/8/KP6/8/8/8/8 b - - 0 1"  # same idea, rook removed


def test_en_passant_forbidden_when_capture_exposes_discovered_check():
    board = fen.parse_fen(_EP_PINNED_FEN)
    board.make_move(_find(board, "c7c5"))
    assert board.ep_square == _sq("c6")

    # The capture must still be proposed by the *pseudo*-legal generator --
    # it obeys the piece-movement rule for en passant -- so its absence from
    # the legal list below is the legality filter's doing, not a generation
    # bug that would make this test pass for the wrong reason.
    pseudo = generate_pseudo_legal_moves(board)
    ep_candidate = next(m for m in pseudo if move_to_uci(m) == "b5c6")
    assert move_flag(ep_candidate) == EN_PASSANT

    # Prove *why* it's illegal: making it for real does expose check.
    board.make_move(ep_candidate)
    assert attacks.is_attacked(board, board.king_square(WHITE), BLACK), (
        "expected the en passant capture to expose a rank-5 discovered check"
    )
    board.unmake_move()
    assert board.ep_square == _sq("c6"), "unmake must restore the post-double-push state"

    legal = _legal_ucis(board)
    assert "b5c6" not in legal, f"illegal discovered-check en passant capture was generated: {sorted(legal)}"


def test_en_passant_allowed_when_not_pinned():
    """Positive control for the test above: remove the rook and the exact
    same capture must be legal, proving the previous test's absence
    assertion discriminates a real pin rather than en passant being broken
    in general."""
    board = fen.parse_fen(_EP_UNPINNED_FEN)
    board.make_move(_find(board, "c7c5"))
    assert board.ep_square == _sq("c6")

    legal = _legal_ucis(board)
    assert "b5c6" in legal

    fen_before_capture = board.to_fen()
    ep_move = _find(board, "b5c6")
    assert move_flag(ep_move) == EN_PASSANT

    board.make_move(ep_move)
    assert not board.in_check(WHITE)
    assert board.mailbox[_sq("c5")] == NO_PIECE  # the captured pawn is gone, not the destination square
    assert board.mailbox[_sq("c6")] != NO_PIECE  # the capturing pawn landed on the ep square
    assert board.zobrist_hash == compute_hash(board)

    board.unmake_move()
    assert board.to_fen() == fen_before_capture


# ==============================================================================
# 2. Castling through / into / out of check (architecture.md §5.7)
# ==============================================================================
#
# Four independent FENs, each isolating exactly one of the three ways a
# castle can be illegal despite the right being held and the in-between
# squares being empty, plus a positive control proving castling still works
# when none of the three applies.

_OUT_OF_CHECK_FEN = "4r2k/8/8/8/8/8/8/4K2R w K - 0 1"  # e1 itself is attacked
_THROUGH_CHECK_FEN = "5r1k/8/8/8/8/8/8/4K2R w K - 0 1"  # f1 (pass-through) is attacked
_INTO_CHECK_FEN = "6rk/8/8/8/8/8/8/4K2R w K - 0 1"  # g1 (landing square) is attacked
_QUEENSIDE_INTO_CHECK_FEN = "2r2k2/8/8/8/8/8/8/R3K3 w Q - 0 1"  # c1 (landing square) is attacked
_BOTH_SIDES_CLEAR_FEN = "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"


def test_castle_kingside_illegal_out_of_check():
    board = fen.parse_fen(_OUT_OF_CHECK_FEN)
    assert board.in_check()
    legal = generate_legal_moves(board)
    assert "e1g1" not in _legal_ucis(board)
    assert not any(move_flag(m) in (KING_CASTLE, QUEEN_CASTLE) for m in legal), (
        "a king in check must not be able to castle on either side"
    )


def test_castle_kingside_illegal_through_attacked_square():
    board = fen.parse_fen(_THROUGH_CHECK_FEN)
    assert not board.in_check()  # the king's own square is safe...
    assert attacks.is_attacked(board, _sq("f1"), BLACK)  # ...but f1, which it must pass through, is not
    assert "e1g1" not in _legal_ucis(board)


def test_castle_kingside_illegal_into_attacked_square():
    board = fen.parse_fen(_INTO_CHECK_FEN)
    assert not board.in_check()
    assert attacks.is_attacked(board, _sq("g1"), BLACK)  # the landing square itself is attacked
    assert "e1g1" not in _legal_ucis(board)


def test_castle_queenside_illegal_into_attacked_square():
    board = fen.parse_fen(_QUEENSIDE_INTO_CHECK_FEN)
    assert not board.in_check()
    assert attacks.is_attacked(board, _sq("c1"), BLACK)
    assert "e1c1" not in _legal_ucis(board)


def test_castle_both_sides_legal_when_unobstructed_and_unattacked():
    """Positive control for the four tests above: with nothing attacking the
    king's start/pass-through/landing squares on either side, both castles
    must be legal and correctly flagged -- otherwise the absent-move
    assertions above could pass merely because castling never works at
    all."""
    board = fen.parse_fen(_BOTH_SIDES_CLEAR_FEN)
    legal = _legal_ucis(board)
    assert "e1g1" in legal
    assert "e1c1" in legal
    assert move_flag(_find(board, "e1g1")) == KING_CASTLE
    assert move_flag(_find(board, "e1c1")) == QUEEN_CASTLE


# ==============================================================================
# 3. Castling rights lost when a rook is captured on its home square
#    (architecture.md §5.7/§6's CASTLE_SPOILER table, applied to both `frm`
#    and `to`)
# ==============================================================================

_KINGSIDE_ROOK_CAPTURED_FEN = "4k3/8/8/8/8/6n1/8/R3K2R b KQ - 0 1"  # ...Ng3xh1
_QUEENSIDE_ROOK_CAPTURED_FEN = "4k3/8/8/8/8/1n6/8/R3K2R b KQ - 0 1"  # ...Nb3xa1


def test_castling_rights_lost_when_kingside_rook_captured_on_home_square():
    board = fen.parse_fen(_KINGSIDE_ROOK_CAPTURED_FEN)
    assert board.castling_rights == (CASTLE_WK | CASTLE_WQ)

    capture = _find(board, "g3h1")
    board.make_move(capture)

    assert not (board.castling_rights & CASTLE_WK), "capturing the h1 rook must strip kingside rights"
    assert board.castling_rights & CASTLE_WQ, "the untouched a1 rook's queenside right must survive"
    assert board.zobrist_hash == compute_hash(board)

    board.unmake_move()
    assert board.castling_rights == (CASTLE_WK | CASTLE_WQ), "unmake must restore the pre-capture rights"


def test_castling_rights_lost_when_queenside_rook_captured_on_home_square():
    board = fen.parse_fen(_QUEENSIDE_ROOK_CAPTURED_FEN)
    assert board.castling_rights == (CASTLE_WK | CASTLE_WQ)

    capture = _find(board, "b3a1")
    board.make_move(capture)

    assert not (board.castling_rights & CASTLE_WQ), "capturing the a1 rook must strip queenside rights"
    assert board.castling_rights & CASTLE_WK, "the untouched h1 rook's kingside right must survive"
    assert board.zobrist_hash == compute_hash(board)

    board.unmake_move()
    assert board.castling_rights == (CASTLE_WK | CASTLE_WQ)


# ==============================================================================
# 4. Under-promotion: knight, bishop and rook, not just queen
#    (architecture.md §5.9)
# ==============================================================================
#
# White pawn on b7 can either push quietly to b8 or capture a black knight
# diagonally on c8; both destinations are the back rank, so both must expand
# into all four promotion choices (architecture.md §5.9's four-move
# expansion), not just queen.

_PROMOTION_FEN = "2n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1"


def test_promotion_generates_all_four_piece_choices_quiet_and_capturing():
    board = fen.parse_fen(_PROMOTION_FEN)
    legal = _legal_ucis(board)
    assert {"b7b8q", "b7b8r", "b7b8b", "b7b8n"} <= legal, sorted(legal)
    assert {"b7c8q", "b7c8r", "b7c8b", "b7c8n"} <= legal, sorted(legal)


@pytest.mark.parametrize(
    "uci,expected_piece,expected_flag",
    [
        ("b7b8q", QUEEN, PROMO_QUEEN),
        ("b7b8r", ROOK, PROMO_ROOK),
        ("b7b8b", BISHOP, PROMO_BISHOP),
        ("b7b8n", KNIGHT, PROMO_KNIGHT),
    ],
)
def test_quiet_underpromotion_places_exactly_the_chosen_piece(uci, expected_piece, expected_flag):
    board = fen.parse_fen(_PROMOTION_FEN)
    move = _find(board, uci)
    assert move_flag(move) == expected_flag

    fen_before = board.to_fen()
    board.make_move(move)
    to_sq = move_to(move)

    assert piece_type_of(board.mailbox[to_sq]) == expected_piece
    assert color_of(board.mailbox[to_sq]) == WHITE
    assert board.pieces[WHITE][expected_piece] & (1 << to_sq)
    assert not (board.pieces[WHITE][PAWN] & (1 << to_sq)), "the pawn must be gone, not merely relabeled"
    assert board.mailbox[move_from(move)] == NO_PIECE
    assert board.zobrist_hash == compute_hash(board)

    board.unmake_move()
    assert board.to_fen() == fen_before, "unmake must restore the pawn, not leave the promoted piece behind"


@pytest.mark.parametrize(
    "uci,expected_piece,expected_flag",
    [
        ("b7c8q", QUEEN, PROMO_QUEEN_CAP),
        ("b7c8r", ROOK, PROMO_ROOK_CAP),
        ("b7c8b", BISHOP, PROMO_BISHOP_CAP),
        ("b7c8n", KNIGHT, PROMO_KNIGHT_CAP),
    ],
)
def test_capturing_underpromotion_places_the_chosen_piece_and_removes_the_victim(
    uci, expected_piece, expected_flag
):
    board = fen.parse_fen(_PROMOTION_FEN)
    move = _find(board, uci)
    assert move_flag(move) == expected_flag

    fen_before = board.to_fen()
    board.make_move(move)
    to_sq = move_to(move)

    assert piece_type_of(board.mailbox[to_sq]) == expected_piece
    assert color_of(board.mailbox[to_sq]) == WHITE
    assert board.pieces[BLACK][KNIGHT] == 0, "the captured knight must be removed from its bitboard"
    assert board.zobrist_hash == compute_hash(board)

    board.unmake_move()
    assert board.to_fen() == fen_before
    assert board.pieces[BLACK][KNIGHT] & (1 << to_sq), "unmake must restore the captured knight"


_UNDERPROMOTION_CHECK_FEN = "8/4P3/3k4/8/8/8/8/K7 w - - 0 1"


def test_underpromotion_to_knight_gives_check_that_queen_promotion_does_not():
    """A concrete demonstration of why under-promotion is a correctness
    requirement, not a style nicety: a knight lands on squares (by an
    L-shaped attack pattern) a queen cannot reach along any single line, so
    a move generator that only ever emitted PROMO_QUEEN would silently miss
    a legal, check-giving move. White pawn e7 promotes with the black king
    on d6: a knight on e8 gives check (e8-d6 is a knight's move); a queen on
    e8 does not (d6 is neither on e8's rank/file nor its diagonal)."""
    board = fen.parse_fen(_UNDERPROMOTION_CHECK_FEN)
    fen_before = board.to_fen()

    knight_promo = _find(board, "e7e8n")
    board.make_move(knight_promo)
    assert board.in_check(BLACK), "knight promotion should check the king on d6"
    board.unmake_move()
    assert board.to_fen() == fen_before

    queen_promo = _find(board, "e7e8q")
    board.make_move(queen_promo)
    assert not board.in_check(BLACK), "queen promotion has no line to d6 and should not check"
    board.unmake_move()
    assert board.to_fen() == fen_before
