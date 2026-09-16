"""FEN (Forsyth-Edwards Notation) parsing and serialization
(architecture.md §8).

`fen.py` is the *only* place FEN grammar is parsed or serialized — nothing
else in the package touches FEN strings directly. Per the module-boundary
DAG (architecture.md §11), `fen.py` depends on `board.py` (it constructs and
reads a `Board`) plus `constants.py` and `zobrist.py` (both leaf-ish modules
that sit *before* `board.py` in the DAG, so importing them here creates no
cycle: `zobrist.py` never imports `board.py` at module scope, only under
`TYPE_CHECKING`). `board.py` never imports `fen.py` at module scope — only
lazily, inside `Board.starting_position()`/`Board.to_fen()` — which is what
keeps the DAG acyclic despite `Board` wanting FEN-based convenience methods.
"""

from __future__ import annotations

from .board import Board
from .constants import (
    BISHOP,
    BLACK,
    CASTLE_BK,
    CASTLE_BQ,
    CASTLE_WK,
    CASTLE_WQ,
    KING,
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
from .zobrist import compute_hash

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Piece-type <-> FEN letter, indexed/keyed exactly like constants.py's
# PAWN..KING enum (0..5). Uppercase is White, lowercase is Black — the FEN
# convention `board.py`'s own debug `__str__` also follows (§3.1).
_PIECE_TYPE_TO_CHAR = "PNBRQK"
_PIECE_CHAR_TO_TYPE = {
    "P": PAWN,
    "N": KNIGHT,
    "B": BISHOP,
    "R": ROOK,
    "Q": QUEEN,
    "K": KING,
}

# FEN castling-availability letters <-> the 4-bit rights mask (§3.1).
# "KQkq" is the field's own canonical letter order.
_CASTLE_CHAR_TO_BIT = {
    "K": CASTLE_WK,
    "Q": CASTLE_WQ,
    "k": CASTLE_BK,
    "q": CASTLE_BQ,
}
_CASTLE_BIT_TO_CHAR = [
    (CASTLE_WK, "K"),
    (CASTLE_WQ, "Q"),
    (CASTLE_BK, "k"),
    (CASTLE_BQ, "q"),
]


def parse_fen(fen: str) -> Board:
    """Parse a FEN string into a fresh `Board`. The only place FEN grammar
    is parsed (§8).

    A FEN has six space-separated fields: piece placement, active color,
    castling availability, en-passant target square, halfmove clock, and
    fullmove number. The last two are optional in a relaxed FEN (default to
    0 and 1 respectively) so that e.g. UCI `position fen ...` strings that
    omit them still parse.
    """
    fields = fen.strip().split()
    if len(fields) < 4:
        raise ValueError(f"invalid FEN (expected at least 4 fields): {fen!r}")

    placement, active_color, castling, ep_field = fields[0], fields[1], fields[2], fields[3]
    halfmove_clock = int(fields[4]) if len(fields) > 4 else 0
    fullmove_number = int(fields[5]) if len(fields) > 5 else 1

    board = Board()

    # --- Piece placement --------------------------------------------------
    #
    # FEN ranks run top (rank 8) to bottom (rank 1), '/'-separated; each
    # rank is a run of piece letters and digit run-lengths for empty
    # squares, left (file a) to right (file h).
    rank_strs = placement.split("/")
    if len(rank_strs) != 8:
        raise ValueError(f"invalid FEN piece placement (expected 8 ranks): {placement!r}")

    for rank_idx, rank_str in enumerate(rank_strs):
        rank = 7 - rank_idx  # rank_idx 0 is rank 8 (board index 7), ... rank_idx 7 is rank 1 (index 0)
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                run = int(ch)
                if run < 1 or run > 8:
                    raise ValueError(f"invalid FEN empty-square run in rank {rank_str!r}")
                file += run
            else:
                piece_type = _PIECE_CHAR_TO_TYPE.get(ch.upper())
                if piece_type is None:
                    raise ValueError(f"invalid FEN piece letter {ch!r} in rank {rank_str!r}")
                if file > 7:
                    raise ValueError(f"invalid FEN rank (overflows 8 files): {rank_str!r}")
                color = WHITE if ch.isupper() else BLACK
                sq = rank * 8 + file
                board._add_piece(color * 6 + piece_type, sq)
                file += 1
        if file != 8:
            raise ValueError(f"invalid FEN rank (must total exactly 8 files): {rank_str!r}")

    # --- Active color -------------------------------------------------------
    if active_color not in ("w", "b"):
        raise ValueError(f"invalid FEN active color: {active_color!r}")
    board.side_to_move = WHITE if active_color == "w" else BLACK

    # --- Castling availability -----------------------------------------------
    board.castling_rights = 0
    if castling != "-":
        for ch in castling:
            bit = _CASTLE_CHAR_TO_BIT.get(ch)
            if bit is None:
                raise ValueError(f"invalid FEN castling availability: {castling!r}")
            board.castling_rights |= bit

    # --- En-passant target square ---------------------------------------------
    if ep_field == "-":
        board.ep_square = None
    else:
        if ep_field not in SQUARE_NAMES:
            raise ValueError(f"invalid FEN en-passant square: {ep_field!r}")
        board.ep_square = SQUARE_NAMES.index(ep_field)

    board.halfmove_clock = halfmove_clock
    board.fullmove_number = fullmove_number

    # `_add_piece` already folded the piece-placement component into
    # `zobrist_hash` incrementally; rather than also hand-XOR the
    # side/castling/ep components in here, seed the hash from scratch via
    # the from-scratch oracle (zobrist.py §7) — this is exactly the "seed a
    # Board built directly from FEN" use case that function's docstring
    # calls out, and it can never drift from `compute_hash`'s own
    # definition of a position's hash by construction.
    board.zobrist_hash = compute_hash(board)

    return board


def board_to_fen(board: Board) -> str:
    """Serialize `board`'s current position to a FEN string. The only place
    FEN grammar is serialized (§8)."""
    rank_strs = []
    for rank in range(7, -1, -1):
        rank_str = ""
        empty_run = 0
        for file in range(8):
            sq = rank * 8 + file
            code = board.mailbox[sq]
            if code == NO_PIECE:
                empty_run += 1
                continue
            if empty_run:
                rank_str += str(empty_run)
                empty_run = 0
            ch = _PIECE_TYPE_TO_CHAR[piece_type_of(code)]
            rank_str += ch if color_of(code) == WHITE else ch.lower()
        if empty_run:
            rank_str += str(empty_run)
        rank_strs.append(rank_str)
    placement = "/".join(rank_strs)

    active_color = "w" if board.side_to_move == WHITE else "b"

    castling = "".join(ch for bit, ch in _CASTLE_BIT_TO_CHAR if board.castling_rights & bit)
    if not castling:
        castling = "-"

    ep_field = "-" if board.ep_square is None else SQUARE_NAMES[board.ep_square]

    return (
        f"{placement} {active_color} {castling} {ep_field} "
        f"{board.halfmove_clock} {board.fullmove_number}"
    )
