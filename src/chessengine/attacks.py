"""Attack generation: precomputed leaper tables, classical ray-scanning
sliding attacks, and attacked-square/check-detection queries
(architecture.md §5.2-5.4).

Per the module-boundary DAG (architecture.md §11), this module depends only
on `constants` and `bitboard` — it never imports `board.py`, which is what
lets `board.py` import *this* module for `in_check`/`checkers` with no
import cycle. The `Board` type only ever appears here as a quoted forward
reference (never evaluated at runtime, and only resolved by a type checker
under `TYPE_CHECKING`); the functions that take one only ever read its
public attributes (`occupied`, `occupied_co`, `pieces`, `mailbox`,
`king_square`) and never construct or import the class itself.

No magic bitboards (architecture.md §5.1): sliding attacks are computed by
unioning a precomputed full-length ray in each direction and chopping it off
at the first blocker. `bishop_attacks`/`rook_attacks`/`queen_attacks` are the
one, narrow seam a future magic-bitboard implementation would replace
(§5.11) — every other module only ever calls through these three
signatures.
"""

from typing import TYPE_CHECKING

from .bitboard import lsb_index, msb_index
from .constants import (
    BISHOP,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    file_of,
    rank_of,
)

if TYPE_CHECKING:
    from .board import Board

# --- Directions (§5.3) -------------------------------------------------------

NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST = range(8)

DIR_FILE_RANK_DELTA = {
    NORTH: (0, 1),
    NORTH_EAST: (1, 1),
    EAST: (1, 0),
    SOUTH_EAST: (1, -1),
    SOUTH: (0, -1),
    SOUTH_WEST: (-1, -1),
    WEST: (-1, 0),
    NORTH_WEST: (-1, 1),
}

# Along a "positive" direction, squares increase with distance from the
# source, so the nearest blocker is the lowest set bit; along a "negative"
# direction the nearest blocker is the highest set bit.
POSITIVE_DIRS = (NORTH, NORTH_EAST, EAST, NORTH_WEST)
NEGATIVE_DIRS = (SOUTH, SOUTH_EAST, SOUTH_WEST, WEST)

ORTHO_DIRS = (NORTH, EAST, SOUTH, WEST)
DIAG_DIRS = (NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST)

# --- Leaper attack tables (§5.2) ---------------------------------------------

KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _leaper_table(deltas: list[tuple[int, int]]) -> list[int]:
    """Build a `square -> attack bitboard` table for a fixed set of
    (file, rank) deltas. Bounds-checked by construction: a delta that would
    land off the board (nf/nr outside [0, 7]) is simply skipped, never
    wrapped around via raw index arithmetic."""
    table = [0] * 64
    for sq in range(64):
        f, r = file_of(sq), rank_of(sq)
        bb = 0
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= 1 << (nr * 8 + nf)
        table[sq] = bb
    return table


KNIGHT_ATTACKS = _leaper_table(KNIGHT_DELTAS)
KING_ATTACKS = _leaper_table(KING_DELTAS)

# PAWN_ATTACKS[color][sq] = squares a color's pawn on sq attacks (captures
# towards). White captures NE/NW (towards higher ranks); Black captures
# SE/SW (towards lower ranks).
PAWN_ATTACKS = [
    _leaper_table([(1, 1), (-1, 1)]),  # WHITE
    _leaper_table([(1, -1), (-1, -1)]),  # BLACK
]

# --- Ray tables and sliding attacks (§5.3) -----------------------------------


def _build_ray_attacks() -> list[list[int]]:
    """RAY_ATTACKS[d][sq] = every square from sq to the board edge along
    direction d, ignoring occupancy. Bounds-checked via file/rank deltas,
    never via raw ``<< n`` past the edge."""
    rays = [[0] * 64 for _ in range(8)]
    for sq in range(64):
        f0, r0 = file_of(sq), rank_of(sq)
        for d in range(8):
            df, dr = DIR_FILE_RANK_DELTA[d]
            bb, f, r = 0, f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                bb |= 1 << (r * 8 + f)
                f, r = f + df, r + dr
            rays[d][sq] = bb
    return rays


RAY_ATTACKS = _build_ray_attacks()


def sliding_attacks(sq: int, occupied: int, dirs: tuple[int, ...]) -> int:
    """Union the full ray in each of ``dirs``, then chop off everything
    beyond the first blocker in that direction (the blocker square itself
    stays attacked, since sliders can capture into it)."""
    attacks = 0
    for d in dirs:
        ray = RAY_ATTACKS[d][sq]
        attacks |= ray
        blockers = ray & occupied
        if blockers:
            blocker_sq = lsb_index(blockers) if d in POSITIVE_DIRS else msb_index(blockers)
            attacks &= ~RAY_ATTACKS[d][blocker_sq]
    return attacks


def bishop_attacks(sq: int, occupied: int) -> int:
    return sliding_attacks(sq, occupied, DIAG_DIRS)


def rook_attacks(sq: int, occupied: int) -> int:
    return sliding_attacks(sq, occupied, ORTHO_DIRS)


def queen_attacks(sq: int, occupied: int) -> int:
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)


def _build_squares_between() -> list[list[int]]:
    """SQUARES_BETWEEN[a][b] is 0 unless a and b share a rank, file, or
    diagonal, in which case it is exactly the squares strictly between them
    (used for pin masks and check-block masks, §5.5/5.6). Built from
    RAY_ATTACKS: for each square a and direction d, every square b on that
    ray gets the portion of the ray strictly between a and b (i.e. the ray
    from a truncated at b, with b itself removed)."""
    between = [[0] * 64 for _ in range(64)]
    for a in range(64):
        for d in range(8):
            ray = RAY_ATTACKS[d][a]
            segment = 0
            bb = ray
            while bb:
                b = lsb_index(bb) if d in POSITIVE_DIRS else msb_index(bb)
                between[a][b] = segment
                bb &= ~(1 << b)
                segment |= 1 << b
    return between


SQUARES_BETWEEN = _build_squares_between()

# --- Attacked-square queries and check detection (§5.4) ----------------------


def attackers_to(
    board: "Board", sq: int, by_color: int, occupied: int | None = None
) -> int:
    """Bitboard of every ``by_color`` piece attacking ``sq``, given an
    explicit occupancy (so callers can probe 'what if this square were
    empty', needed for king-move legality, §5.10)."""
    occ = board.occupied if occupied is None else occupied
    p = board.pieces[by_color]
    # "attacked by enemy pawn" == "pawn of the opposite color would attack
    # from sq" — hence indexing PAWN_ATTACKS by the *other* color.
    attackers = PAWN_ATTACKS[1 - by_color][sq] & p[PAWN]
    attackers |= KNIGHT_ATTACKS[sq] & p[KNIGHT]
    attackers |= KING_ATTACKS[sq] & p[KING]
    attackers |= bishop_attacks(sq, occ) & (p[BISHOP] | p[QUEEN])
    attackers |= rook_attacks(sq, occ) & (p[ROOK] | p[QUEEN])
    return attackers


def is_attacked(board: "Board", sq: int, by_color: int) -> bool:
    return attackers_to(board, sq, by_color) != 0


def checkers(board: "Board", color: int) -> int:
    return attackers_to(board, board.king_square(color), 1 - color)
