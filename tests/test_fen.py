"""Round-trip tests for FEN parsing/serialization (architecture.md §8, §13).

`fen.py` is the *only* place FEN grammar is parsed or serialized (architecture.md
§8), so this module is responsible for the one load-bearing invariant §13
calls out for it: `board_to_fen(parse_fen(fen)) == fen`, exactly (byte for
byte), for a battery of FENs covering every castling-rights combination, a
position with an en-passant target square set, and non-standard
halfmove/fullmove counters.

Note that `parse_fen` does not cross-validate the castling/en-passant fields
against piece placement (e.g. it does not check that a rook actually sits on
its home square before honoring a castling letter) -- it simply parses each
field independently. That is exactly what makes it possible to exercise all
16 castling-rights combinations against one fixed, valid piece placement
below, without needing 16 different "realistic" positions.
"""

from __future__ import annotations

import itertools

import pytest

from chessengine.fen import board_to_fen, parse_fen

STARTPOS_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Canonical castling-letter order. This matches fen.py's own
# `_CASTLE_BIT_TO_CHAR` emission order (K, Q, k, q), so any castling field
# spelled out in this order round-trips byte-for-byte: `board_to_fen` always
# emits the letters present in exactly this order, regardless of the order
# they appeared in on input.
CASTLE_LETTERS = "KQkq"


def _all_castling_fields() -> list[str]:
    """Every one of the 16 subsets of {K, Q, k, q}, each spelled out in
    canonical KQkq order (so it matches what `board_to_fen` will emit back),
    including the empty-rights "-" case."""
    fields = []
    for r in range(len(CASTLE_LETTERS) + 1):
        for combo in itertools.combinations(CASTLE_LETTERS, r):
            fields.append("".join(combo) if combo else "-")
    return fields


def test_all_castling_fields_covers_every_combination() -> None:
    # Sanity check on the parametrization itself, so a bug in
    # `_all_castling_fields` can't silently shrink the battery below "every
    # combination": 2**4 subsets of {K, Q, k, q}, from "-" to "KQkq".
    fields = _all_castling_fields()
    assert len(fields) == 16
    assert len(set(fields)) == 16
    assert fields[0] == "-"
    assert fields[-1] == "KQkq"


@pytest.mark.parametrize("castling", _all_castling_fields())
def test_round_trip_every_castling_rights_combination(castling: str) -> None:
    fen = f"{STARTPOS_PLACEMENT} w {castling} - 0 1"
    board = parse_fen(fen)
    assert board_to_fen(board) == fen


EN_PASSANT_FENS = [
    # After 1. e4: White's double push leaves e3 as the ep target square,
    # Black to move.
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # After 1. e4 e5: Black's double push leaves e6 as the ep target square,
    # White to move.
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    # En-passant target on the b-file, no castling rights left, Black to move.
    "4k3/8/8/8/pP6/8/8/4K3 b - b3 0 1",
    # En-passant target on the h-file (board edge), non-standard counters.
    "4k3/8/8/7p/6P1/8/8/4K3 w - h6 12 30",
]


@pytest.mark.parametrize("fen", EN_PASSANT_FENS)
def test_round_trip_en_passant_square(fen: str) -> None:
    board = parse_fen(fen)
    assert board.ep_square is not None
    assert board_to_fen(board) == fen


NON_STANDARD_COUNTER_FENS = [
    STARTPOS_PLACEMENT + " w KQkq - 0 1",  # baseline: standard counters
    STARTPOS_PLACEMENT + " w KQkq - 15 1",  # non-zero halfmove clock
    STARTPOS_PLACEMENT + " b KQkq - 0 42",  # non-standard fullmove number
    STARTPOS_PLACEMENT + " w KQkq - 99 250",  # both non-standard, large fullmove
    STARTPOS_PLACEMENT + " b - - 100 500",  # halfmove clock at the 50-move boundary
    "8/8/8/8/8/8/8/4K2k w - - 0 999",  # bare kings, very large fullmove number
]


@pytest.mark.parametrize("fen", NON_STANDARD_COUNTER_FENS)
def test_round_trip_non_standard_move_counters(fen: str) -> None:
    board = parse_fen(fen)
    assert board_to_fen(board) == fen


# architecture.md §14's perft reference positions: real, well-known
# positions (not just synthetic ones) exercised through the same round trip.
REFERENCE_POSITION_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # startpos
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # position 3
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",  # position 4
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",  # position 5
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",  # position 6
]


@pytest.mark.parametrize("fen", REFERENCE_POSITION_FENS)
def test_round_trip_perft_reference_positions(fen: str) -> None:
    board = parse_fen(fen)
    assert board_to_fen(board) == fen


def test_round_trip_combined_castling_ep_and_counters() -> None:
    """One FEN exercising all three battery dimensions at once: a
    non-trivial (non-full, non-empty) castling-rights subset, an en-passant
    square, and non-standard halfmove/fullmove counters."""
    fen = "r3k2r/8/8/8/pP6/8/8/R3K2R b Kq b3 7 88"
    board = parse_fen(fen)
    assert board.castling_rights != 0
    assert board.ep_square is not None
    assert board.halfmove_clock == 7
    assert board.fullmove_number == 88
    assert board_to_fen(board) == fen


def test_parse_fen_relaxed_missing_halfmove_and_fullmove_defaults() -> None:
    """`parse_fen` accepts a relaxed, 4-field FEN (halfmove/fullmove omitted,
    per architecture.md §8) and defaults them to 0 and 1 respectively;
    `board_to_fen` always emits all six fields, so re-serializing such a
    FEN does not reproduce the original (shorter) string but does reproduce
    the fully-specified, defaulted one."""
    board = parse_fen(STARTPOS_PLACEMENT + " w KQkq -")
    assert board.halfmove_clock == 0
    assert board.fullmove_number == 1
    assert board_to_fen(board) == STARTPOS_PLACEMENT + " w KQkq - 0 1"


def test_round_trip_black_to_move() -> None:
    fen = STARTPOS_PLACEMENT + " b KQkq - 0 1"
    board = parse_fen(fen)
    assert board_to_fen(board) == fen
