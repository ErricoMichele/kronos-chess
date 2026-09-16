"""Shared vocabulary for the whole engine: enums, masks, and small pure helpers.

Per the architecture doc (§3.1, §11), `constants.py` is a dependency-free leaf
module — every other module may import from it, but it imports nothing from
the rest of `chessengine`. It carries no behavior beyond tiny, side-effect-free
helper functions: no `Board`, no I/O, no mutable global state.
"""

# --- Color enum -------------------------------------------------------------

WHITE, BLACK = 0, 1

# --- Piece type enum ---------------------------------------------------------

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)

NO_PIECE = 12  # mailbox sentinel: not a valid (color * 6 + piece_type) code


def piece_type_of(code: int) -> int:
    return code % 6


def color_of(code: int) -> int:
    return code // 6


# --- Square indexing (§3.1) --------------------------------------------------
#
# Little-endian rank-file (LERF): square = rank * 8 + file, rank/file in
# [0, 7]. a1 = 0, b1 = 1, ..., h1 = 7, a8 = 56, ..., h8 = 63.


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H = range(8)

FILE_MASK = [0x0101010101010101 << f for f in range(8)]
RANK_MASK = [0xFF << (8 * r) for r in range(8)]

# Named squares (LERF layout above), spelled out via bounds-checked
# construction (`range(64)` in a1..h8 reading order) rather than raw
# arithmetic, so a typo can't silently alias two squares. These are the
# common vocabulary that castling tables built elsewhere (`move.py`,
# `movegen.py` — see §5.7) index by name, e.g. `CASTLE_SPOILER[E1]`.
(
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
) = range(64)

SQUARE_NAMES = [f + r for r in "12345678" for f in "abcdefgh"]  # index -> "e4" etc.

# --- Castling rights (4-bit mask, §3.1) --------------------------------------

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

# --- Search-adjacent shared constants ----------------------------------------

MAX_PLY = 128

# Search score constants (shared by transposition.py and search.py; homed
# here, not in either, so neither module depends on the other for a constant)
INF = 1_000_000
MATE_SCORE = 100_000
DRAW_SCORE = 0
