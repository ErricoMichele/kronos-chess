"""Board representation invariants across random legal games
(architecture.md §13, "test_board_invariants.py").

`Board.pieces`/`occupied_co`/`occupied`/`mailbox` are four redundant
encodings of the same position, kept in sync by a single-writer discipline
inside `Board._add_piece`/`_remove_piece`/`_move_piece` (architecture.md
§3.2, §6). This module is the dedicated test for that discipline: it plays
many randomly generated legal games (using the real `generate_legal_moves`
to pick only legal moves, then `Board.make_move`/`unmake_move` to play and
undo them) and, after *every single* `make_move` and `unmake_move` call,
asserts the three invariants architecture.md §13 calls out by name:

1. `board.occupied == board.occupied_co[WHITE] | board.occupied_co[BLACK]`.
2. `mailbox` agrees with the bitboards, square by square, in both
   directions: a non-empty mailbox entry's (color, piece_type) bitboard has
   that square's bit set (and no other (color, piece_type) bitboard does),
   and an empty mailbox entry has no bitboard bit set at that square at
   all.
3. Each side has exactly one king bit set in `pieces[color][KING]`.

Several starting positions from §14's perft reference set are used (not
just the startpos) specifically to raise the odds that castling, en
passant, and promotion moves -- the fiddliest paths through
`make_move`/`unmake_move` -- actually get exercised by the random walk, not
just quiet pawn/piece shuffles.
"""

from __future__ import annotations

import random

import pytest

from chessengine.board import Board
from chessengine.constants import BLACK, KING, NO_PIECE, WHITE, color_of, piece_type_of
from chessengine.fen import STARTPOS_FEN, parse_fen
from chessengine.movegen import generate_legal_moves

# --- Starting positions exercised ---------------------------------------
#
# Reused from architecture.md §14's perft fixture set: between them these
# hit castling (Kiwipete, Position 4), en passant on a near-empty board
# (Position 3), rook-capture castling-rights spoilage and promotion
# (Position 4), and pawns one step from promotion/under-promotion territory
# (Position 5) -- not just the quiet startpos.
STARTING_POSITIONS = (
    ("startpos", STARTPOS_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
)

PLIES_PER_GAME = 80
GAMES_PER_POSITION = 5


def _assert_board_invariants(board: Board) -> None:
    """architecture.md §13's three load-bearing `Board` invariants."""

    # 1. occupied is exactly the union of both colors' occupancy.
    assert board.occupied == (board.occupied_co[WHITE] | board.occupied_co[BLACK]), (
        "board.occupied is not the union of occupied_co[WHITE] and occupied_co[BLACK]"
    )

    # 3. Each side has exactly one king bit set. (Checked ahead of the
    # per-square mailbox loop below since a missing/duplicated king is the
    # most catastrophic possible invariant violation.)
    for color in (WHITE, BLACK):
        king_bits = board.pieces[color][KING]
        assert king_bits.bit_count() == 1, (
            f"color {color} has {king_bits.bit_count()} king bits set "
            f"(expected exactly 1): {king_bits:#018x}"
        )

    # 2. mailbox agrees with the bitboards, square by square, in both
    # directions.
    for sq in range(64):
        mailbox_code = board.mailbox[sq]

        if mailbox_code == NO_PIECE:
            assert not (board.occupied >> sq) & 1, (
                f"square {sq}: mailbox says empty (NO_PIECE) but occupied has its bit set"
            )
            for color in (WHITE, BLACK):
                for ptype in range(6):
                    assert not (board.pieces[color][ptype] >> sq) & 1, (
                        f"square {sq}: mailbox says empty (NO_PIECE) but "
                        f"pieces[{color}][{ptype}] has its bit set"
                    )
            continue

        owner_color, owner_ptype = color_of(mailbox_code), piece_type_of(mailbox_code)
        assert (board.pieces[owner_color][owner_ptype] >> sq) & 1, (
            f"square {sq}: mailbox code {mailbox_code} (color={owner_color}, "
            f"piece_type={owner_ptype}) but that bitboard's bit is not set"
        )
        assert (board.occupied_co[owner_color] >> sq) & 1, (
            f"square {sq}: mailbox code {mailbox_code} but "
            f"occupied_co[{owner_color}] does not have its bit set"
        )
        assert (board.occupied >> sq) & 1, (
            f"square {sq}: mailbox has a piece but occupied does not have its bit set"
        )
        # Exactly one piece per square: no *other* (color, piece_type)
        # bitboard may also claim this square.
        for other_color in (WHITE, BLACK):
            for other_ptype in range(6):
                if (other_color, other_ptype) == (owner_color, owner_ptype):
                    continue
                assert not (board.pieces[other_color][other_ptype] >> sq) & 1, (
                    f"square {sq}: claimed by both pieces[{owner_color}][{owner_ptype}] "
                    f"(matching mailbox code {mailbox_code}) and "
                    f"pieces[{other_color}][{other_ptype}]"
                )


def _play_random_legal_game(board: Board, rng: random.Random, max_plies: int) -> list[int]:
    """Play up to `max_plies` randomly chosen *legal* moves from `board`'s
    current position, asserting the §13 invariants after every single
    `make_move`. Stops early if the game ends (checkmate/stalemate leaves no
    legal moves). Returns the list of moves actually played, in order, so
    the caller can unmake them one by one and check the same invariants
    after every `unmake_move` too."""
    played: list[int] = []
    _assert_board_invariants(board)  # sanity baseline before any move at all
    for _ in range(max_plies):
        legal_moves = generate_legal_moves(board)
        if not legal_moves:
            break  # checkmate or stalemate: no legal move to play
        move = rng.choice(legal_moves)
        board.make_move(move)
        _assert_board_invariants(board)
        played.append(move)
    return played


def _unmake_all(board: Board, played: list[int]) -> None:
    """Unmake every move in `played`, most recent first, asserting the §13
    invariants after every single `unmake_move`."""
    for _ in reversed(played):
        board.unmake_move()
        _assert_board_invariants(board)


@pytest.mark.parametrize(
    "position_name,start_fen",
    STARTING_POSITIONS,
    ids=[name for name, _fen in STARTING_POSITIONS],
)
def test_invariants_hold_through_random_legal_games(position_name: str, start_fen: str) -> None:
    """After every make_move/unmake_move of a randomly generated legal game
    from each starting position, the board's redundant representations stay
    in sync (architecture.md §13)."""
    for game_index in range(GAMES_PER_POSITION):
        # Deterministic, distinct seed per (position, game) pair so a
        # failure is exactly reproducible without depending on interpreter
        # hash randomization (random.Random accepts a str seed directly).
        rng = random.Random(f"{position_name}-{game_index}")
        board = parse_fen(start_fen)
        fen_before = board.to_fen()

        played = _play_random_legal_game(board, rng, PLIES_PER_GAME)
        _unmake_all(board, played)

        # Unmaking every played move must restore the *exact* starting
        # position, not merely one that happens to satisfy the invariants
        # above -- a strong end-to-end sanity check on top of the per-step
        # assertions.
        assert board.to_fen() == fen_before
        assert board.history == []
        assert board.position_history == []


def test_invariants_hold_for_a_long_single_game_from_startpos() -> None:
    """A single longer random legal game (more plies than
    `test_invariants_hold_through_random_legal_games` uses per game),
    specifically to raise the odds of reaching promotions, multiple
    castles, and repeated en-passant opportunities within one game's
    history stack."""
    rng = random.Random("long-random-legal-game-from-startpos")
    board = parse_fen(STARTPOS_FEN)
    fen_before = board.to_fen()

    played = _play_random_legal_game(board, rng, max_plies=200)

    assert len(played) > 0  # a fresh startpos always has legal moves available
    assert len(board.history) == len(played)

    _unmake_all(board, played)

    assert board.to_fen() == fen_before
    assert board.history == []
    assert board.position_history == []


def test_invariants_hold_on_a_freshly_parsed_board_with_no_moves_played() -> None:
    """Baseline: the invariants must already hold on a board that has had no
    moves played on it at all, for every §14 starting position used above --
    a failure here would point at `fen.parse_fen`/`Board._add_piece`, not at
    `make_move`/`unmake_move`."""
    for _name, start_fen in STARTING_POSITIONS:
        board = parse_fen(start_fen)
        _assert_board_invariants(board)
