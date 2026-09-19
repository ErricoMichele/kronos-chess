"""Tests for the knight outpost evaluation term."""
from chessengine.board import Board
from chessengine.fen import parse_fen
from chessengine.evaluate import knight_outpost_term, KNIGHT_OUTPOST_BONUS_CP


def test_outpost_knight_gets_bonus() -> None:
    """White knight on e5 protected by d4 pawn, no black pawns on d/f files ahead."""
    # Knight e5, pawn d4, black has NO pawns on d or f files
    board = parse_fen("rnbqkb1r/ppp1p1pp/8/4N3/3P4/8/PPP1PPPP/R1BQKBNR w KQkq - 0 1")
    score = knight_outpost_term(board)
    assert score >= KNIGHT_OUTPOST_BONUS_CP


def test_no_outpost_without_pawn_support() -> None:
    """Knight on e5 without pawn protection is not an outpost."""
    board = parse_fen("rnbqkb1r/ppp1p1pp/8/4N3/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
    score = knight_outpost_term(board)
    assert score == 0


def test_no_outpost_if_enemy_pawn_can_attack() -> None:
    """Knight on e5 with d4 pawn, but black has f7 pawn that can attack e5."""
    board = parse_fen("rnbqkb1r/ppp1pppp/8/4N3/3P4/8/PPP1PPPP/R1BQKBNR w KQkq - 0 1")
    score = knight_outpost_term(board)
    assert score == 0


def test_symmetric_position_returns_zero() -> None:
    """Starting position: symmetric, no outpost knights."""
    board = Board.starting_position()
    assert knight_outpost_term(board) == 0


def test_knight_on_low_rank_not_outpost() -> None:
    """Knight on e2 (rank 1 for white) is not advanced enough."""
    board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPNPPP/RNBQKB1R w KQkq - 0 1")
    assert knight_outpost_term(board) == 0
