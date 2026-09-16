"""Move encoding: a move is a packed 16-bit ``int`` — 6 bits ``from``, 6 bits
``to``, 4 bits ``flag`` (architecture.md §4).

A move never records the moving piece or the captured piece itself:
``Board.make_move`` reads the moving piece from ``board.mailbox[frm]`` and
the captured piece from ``board.mailbox[to]`` (or the en-passant square) at
apply time. This keeps moves immutable, comparable by plain integer
equality (used for TT best-move matching and killer-move tables), and
trivially storable in flat ``list[int]``s.

Per the architecture doc (§4, §11), `move.py` is a leaf module that depends
only on `constants` — it never imports `board.py` or `movegen.py`.
`move_from_uci` deliberately lives in `movegen.py`, not here, because
resolving a UCI string requires matching against the legal move list, and
`movegen.py` already depends on `move.py`; putting it here would create an
import cycle.
"""

from .constants import BISHOP, KNIGHT, QUEEN, ROOK, SQUARE_NAMES

# --- Bit layout (§4) ----------------------------------------------------
#
# Bits  0-5  (6 bits): `from` square (0-63)
# Bits  6-11 (6 bits): `to` square (0-63)
# Bits 12-15 (4 bits): `flag`

_FROM_MASK = 0x3F
_TO_MASK = 0x3F
_FLAG_MASK = 0xF
_TO_SHIFT = 6
_FLAG_SHIFT = 12

# --- Flags (§4) -----------------------------------------------------------
#
# The encoding is deliberately bit-patterned so two single-bit tests answer
# the two questions callers ask most: `flag & 0x4` is set on `CAPTURE`,
# `EN_PASSANT`, and all four `*_CAP` promotions ("is this a capture");
# `flag & 0x8` is set on all eight promotion flags ("is this a promotion").

QUIET = 0x0  # normal, non-capturing move
DOUBLE_PAWN_PUSH = 0x1  # sets ep_square
KING_CASTLE = 0x2  # O-O
QUEEN_CASTLE = 0x3  # O-O-O
CAPTURE = 0x4  # ordinary capture
EN_PASSANT = 0x5  # captured square != `to`
# 0x6, 0x7 — reserved, unused

PROMO_KNIGHT = 0x8  # quiet promotion
PROMO_BISHOP = 0x9
PROMO_ROOK = 0xA
PROMO_QUEEN = 0xB
PROMO_KNIGHT_CAP = 0xC  # capturing promotion
PROMO_BISHOP_CAP = 0xD
PROMO_ROOK_CAP = 0xE
PROMO_QUEEN_CAP = 0xF

PROMO_PIECE_OF = {
    PROMO_KNIGHT: KNIGHT,
    PROMO_KNIGHT_CAP: KNIGHT,
    PROMO_BISHOP: BISHOP,
    PROMO_BISHOP_CAP: BISHOP,
    PROMO_ROOK: ROOK,
    PROMO_ROOK_CAP: ROOK,
    PROMO_QUEEN: QUEEN,
    PROMO_QUEEN_CAP: QUEEN,
}

NULL_MOVE = 0  # a1a1 quiet; never a legal move, used as a "no move" sentinel


def encode_move(frm: int, to: int, flag: int = QUIET) -> int:
    """Pack a move into its 16-bit representation."""
    return frm | (to << _TO_SHIFT) | (flag << _FLAG_SHIFT)


def move_from(m: int) -> int:
    """The `from` square (0-63)."""
    return m & _FROM_MASK


def move_to(m: int) -> int:
    """The `to` square (0-63)."""
    return (m >> _TO_SHIFT) & _TO_MASK


def move_flag(m: int) -> int:
    """The 4-bit flag field."""
    return (m >> _FLAG_SHIFT) & _FLAG_MASK


def is_capture(m: int) -> bool:
    """True for CAPTURE, EN_PASSANT, and any of the four `*_CAP` promotions."""
    return bool(move_flag(m) & 0x4)


def is_promotion(m: int) -> bool:
    """True for any of the eight promotion flags (quiet or capturing)."""
    return bool(move_flag(m) & 0x8)


def move_to_uci(m: int) -> str:
    """Pure string conversion — no Board needed, since a move's own fields
    are enough to print it."""
    s = SQUARE_NAMES[move_from(m)] + SQUARE_NAMES[move_to(m)]
    flag = move_flag(m)
    if flag in PROMO_PIECE_OF:
        s += "nbrq"[[KNIGHT, BISHOP, ROOK, QUEEN].index(PROMO_PIECE_OF[flag])]
    return s
