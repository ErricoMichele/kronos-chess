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

Magic bitboards (architecture.md §5.11) are now active for
`bishop_attacks`/`rook_attacks`/`queen_attacks`: this is the profiling-gated
optimization described in §5.1/§5.11, activated because cProfile on
`Search.search(kiwipete_position, SearchLimits(max_depth=5))` showed
`attacks.sliding_attacks` as the single highest-tottime function in the
entire search (1.494s own time / 2.419s cumulative out of 12.515s total) —
not assumed up front, measured. The classical ray-scanning `sliding_attacks`
below is unchanged and still public (`movegen.py` calls it directly for
pseudo-legal sliding-move generation, and it is exercised directly by
`tests/test_attacks.py`); it additionally now serves as the ground-truth
oracle every magic number is exhaustively verified against at import time,
via the private `_classical_bishop_attacks`/`_classical_rook_attacks`
wrappers. `bishop_attacks`/`rook_attacks`/`queen_attacks` keep the exact
public signatures `(square: int, occupied: int) -> int` described in §5.11,
so `movegen.py`, `evaluate.py`, and `search.py` (which all call through
these three names, not through `sliding_attacks` itself) need no changes.
"""

import random
from typing import TYPE_CHECKING

from .bitboard import BB_ALL, lsb_index, msb_index, popcount
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


def _classical_bishop_attacks(sq: int, occupied: int) -> int:
    """The classical ray-scanning bishop attack (§5.3), unchanged. Retained
    as (a) the ground-truth oracle every candidate magic number is
    exhaustively verified against below, and (b) an importable oracle for a
    differential test."""
    return sliding_attacks(sq, occupied, DIAG_DIRS)


def _classical_rook_attacks(sq: int, occupied: int) -> int:
    """The classical ray-scanning rook attack (§5.3), unchanged. See
    `_classical_bishop_attacks`."""
    return sliding_attacks(sq, occupied, ORTHO_DIRS)


# --- Magic bitboards (§5.11) --------------------------------------------------
#
# For each square and each of {bishop, rook}: compute the "relevant
# occupancy mask" (every square along that piece's rays from the square,
# excluding the outermost square of each ray). Whether that outermost
# square is occupied never changes the attack set — the ray already
# terminates there (board edge) either way — so it is never worth treating
# as a blocker, and excluding it keeps the mask (and so the table) small.
#
# With a fixed seed (so generation is fully deterministic and reproducible
# on every run/machine), random candidate 64-bit "magic" multipliers are
# tried, biased toward sparse bit patterns via the standard
# `getrandbits(64) & getrandbits(64) & getrandbits(64)` trick. For each
# candidate, *every* subset of the relevant mask (Carry-Rippler enumeration)
# is hashed to `(subset * magic) >> (64 - popcount(mask))` and checked
# against the classical oracle; a magic is accepted only once every subset
# has been checked with zero incorrect collisions (two different subsets
# that produce two different attack sets are never allowed to share an
# index — the same attack set landing on the same index from two different
# subsets is fine and expected). This exhaustive verification-before-
# acceptance is what makes the result unconditionally correct regardless of
# how the magic numbers were found.

_MAGIC_SEED = 0xC0FFEE  # fixed: magic-table generation must be reproducible

# Pre-searched magic multipliers (found once, offline, by exactly the random
# search `_find_magic` below implements, seeded with `_MAGIC_SEED`) — hard-
# coded here purely as a STARTUP-TIME OPTIMIZATION. Running that search live
# at every import was measured at ~18s (rook squares with a 10-12 bit
# relevant mask can need tens to hundreds of thousands of rejected
# candidates before one exhaustively verifies clean), which is unacceptable
# for a process that a UCI GUI expects to start near-instantly. Embedding the
# winning numbers turns "search for a working magic" into "verify this one
# already-known-good magic," which is a single O(table size) pass per square
# with no rejected candidates — see `_build_magic_tables` below, which still
# exhaustively re-verifies every one of these against the classical oracle
# before trusting it, and falls back to a live `_find_magic` search for any
# single square where that verification would ever fail (e.g. if
# `RAY_ATTACKS`/direction ordering ever changes upstream) — so correctness
# never depends on these constants being trustworthy, only startup speed does.
_PRECOMPUTED_BISHOP_MAGICS = (
    0x10420184108204, 0xc810424188001, 0x88084500200008, 0x11040080010810,
    0xb084050460000015, 0x189241040054000, 0x8104140108082007, 0x22018e06900410,
    0x202002118112, 0x832a04102008102, 0x405224800428000, 0x4400044040800028,
    0x8200a61210010190, 0x2211006100001, 0x80020504200400, 0x420108684132012,
    0x240000808080081, 0x9848000408080062, 0x108021000444008, 0x808c029201220200,
    0x4003080a08008, 0x8512008048020800, 0x200810c04010810, 0xa0010012016e0200,
    0x200405c110200800, 0x990120849780100, 0xb110281204014401, 0x10480084012020,
    0x290101001004000, 0x808001006000, 0x821810a421800, 0x101025001040080,
    0x804a642400802, 0x402880804143000, 0x3040824040840405, 0x8880800020a00,
    0x2241120400020102, 0x7c1010a00010801, 0x8810400010080, 0xe2c0040010440,
    0x4100804000918, 0xc04020802803508, 0xca01402081008, 0x8808404206808801,
    0x804022009001a08, 0xa0640180a00200, 0x20880283000090, 0x9080200449091,
    0x6000840318400055, 0x100480208004c, 0x1210045200900900, 0x200140084042100,
    0x862011c108620408, 0x610206120010, 0x8090860840160, 0x24010801250105,
    0x2028208944202000, 0x8010002118080c01, 0x1040020100880404, 0x4040000208820,
    0x11000010420202, 0x12044820080220, 0x230201441880900, 0x104010010100a080,
)
_PRECOMPUTED_ROOK_MAGICS = (
    0x480084000812010, 0x40200150004000, 0x200104a00824020, 0x8880100028002580,
    0x2080140008000280, 0x100010006080400, 0x1004a0000810004, 0x8002a841000080,
    0x21800140002081, 0x14044010012008c1, 0x852004020108200, 0x560020400a0011,
    0x8008800401800802, 0x1209000a04010028, 0x8c00800100801200, 0x30010010c08a0b00,
    0x1080004002a00040, 0x50004000482000, 0x88802000d000, 0x101808010000806,
    0x8a2020020440890, 0x1010002040008, 0x1500840002080190, 0x10020004008041,
    0xe80034040002000, 0x8400842100400100, 0x20200080801000, 0x208080080100080,
    0x8040440080800800, 0x280400801a0080, 0x600080400121019, 0x8000041a00014081,
    0x80002000400840, 0x80900020004004c1, 0x201202004080, 0x4010010061001009,
    0x480804800800401, 0x412a00140a005810, 0x3300410804000290, 0x14c02000881,
    0x4c400088608004, 0x602000804000802a, 0xa00100c10010, 0x401001000210048,
    0x2080140008008080, 0x100160c010048, 0x280a9008140001, 0x80230408c020021,
    0x1188400080002880, 0x44020008a400480, 0x880104822008200, 0x4050001810210300,
    0x1001008000d00, 0xa02000280040080, 0x8000210802101400, 0x108004804d141200,
    0x4848102082004302, 0x408810420400091, 0x20008c0802012, 0x100200082440a092,
    0x2430008000c1003, 0x800100081a040041, 0x8160023890090804, 0x40400104e0c40082,
)


def _relevant_occupancy_mask(sq: int, dirs: tuple[int, ...]) -> int:
    """Every square along `sq`'s rays in `dirs`, excluding the outermost
    (farthest-from-`sq`) square of each ray — the standard magic-bitboard
    relevant-occupancy convention (see the module comment above)."""
    mask = 0
    for d in dirs:
        ray = RAY_ATTACKS[d][sq]
        if not ray:
            continue
        far_sq = msb_index(ray) if d in POSITIVE_DIRS else lsb_index(ray)
        mask |= ray & ~(1 << far_sq)
    return mask


def _iter_mask_subsets(mask: int):
    """Yield every subset of `mask` exactly once — 2**popcount(mask)
    subsets, including 0 and `mask` itself — via the standard
    Carry-Rippler trick (`subset = (subset - 1) & mask`, starting from
    `mask` and descending to 0)."""
    subset = mask
    while True:
        yield subset
        if subset == 0:
            return
        subset = (subset - 1) & mask


def _find_magic(
    sq: int, dirs: tuple[int, ...], rng: random.Random
) -> tuple[int, int, int, list[int]]:
    """Random-search a collision-free magic multiplier for `sq` along
    `dirs`. Returns `(relevant_mask, magic, shift, table)` where
    `table[(occ_subset * magic & MASK64) >> shift]` is the correct attack
    bitboard (per the classical oracle, `sliding_attacks(sq, occ_subset,
    dirs)`) for every occupancy subset of `relevant_mask`.

    The per-candidate collision check below uses a "generation stamp"
    trick (`stamp`/`value`, keyed by a per-attempt counter) instead of
    reallocating and rescanning a fresh `[None] * table_size` table for
    every candidate — an important constant-factor speedup in CPython
    given how many candidates a 10-12 relevant-bit square (rook corners
    and edges) can need. It is exactly equivalent to the straightforward
    "fresh table per candidate" check: a slot is only ever trusted for the
    *current* candidate's `generation`, so a stale value from a previous,
    rejected candidate can never be mistaken for a real collision."""
    mask = _relevant_occupancy_mask(sq, dirs)
    bits = popcount(mask)
    shift = 64 - bits
    subsets_and_attacks = [
        (subset, sliding_attacks(sq, subset, dirs)) for subset in _iter_mask_subsets(mask)
    ]
    table_size = 1 << bits
    stamp = [0] * table_size
    value = [0] * table_size
    generation = 0
    while True:
        generation += 1
        magic = rng.getrandbits(64) & rng.getrandbits(64) & rng.getrandbits(64)
        ok = True
        for subset, attack_set in subsets_and_attacks:
            index = ((subset * magic) & BB_ALL) >> shift
            if stamp[index] != generation:
                stamp[index] = generation
                value[index] = attack_set
            elif value[index] != attack_set:
                ok = False
                break  # real collision: two different attack sets, same index
        if ok:
            # Every subset checked with zero incorrect collisions: build the
            # real output table from this winning magic (a plain, one-off
            # O(table_size) pass — no further candidates are considered).
            table = [0] * table_size
            for subset, attack_set in subsets_and_attacks:
                table[((subset * magic) & BB_ALL) >> shift] = attack_set
            return mask, magic, shift, table


def _try_magic(sq: int, dirs: tuple[int, ...], magic: int) -> tuple[int, int, list[int]] | None:
    """Exhaustively verify a single CANDIDATE `magic` (typically one of the
    `_PRECOMPUTED_*_MAGICS` constants) against the classical oracle, the
    exact same collision rule `_find_magic` enforces before ever accepting a
    magic: every subset of the relevant occupancy mask must hash to an
    index that is never shared with a different, incompatible attack set.
    Returns `(mask, shift, table)` if `magic` verifies cleanly, or `None` if
    it doesn't (a real collision), so the caller can fall back to a fresh
    search rather than trusting an unverified constant."""
    mask = _relevant_occupancy_mask(sq, dirs)
    bits = popcount(mask)
    shift = 64 - bits
    table_size = 1 << bits
    table: list[int | None] = [None] * table_size
    for subset in _iter_mask_subsets(mask):
        attack_set = sliding_attacks(sq, subset, dirs)
        index = ((subset * magic) & BB_ALL) >> shift
        if table[index] is None:
            table[index] = attack_set
        elif table[index] != attack_set:
            return None  # real collision: this magic doesn't work for this square
    return mask, shift, table  # type: ignore[return-value]


def _build_magic_tables(
    dirs: tuple[int, ...], precomputed_magics: tuple[int, ...]
) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    """Build the per-square `(masks, magics, shifts, tables)` for one piece
    type's ray set.

    Fast path (the expected, common case): verify each square's pre-searched
    `precomputed_magics[sq]` in one O(table size) pass via `_try_magic` — no
    rejected candidates, unlike a live search. Defensive fallback: if a
    precomputed magic ever fails verification for some square (e.g.
    `RAY_ATTACKS`/direction ordering changed upstream since the constants
    were captured), fall back to a fresh `_find_magic` random search for
    that square only, so correctness never silently depends on the
    precomputed constants being trustworthy — only startup speed does."""
    rng: random.Random | None = None  # lazily built only if ever needed
    masks: list[int] = [0] * 64
    magics: list[int] = [0] * 64
    shifts: list[int] = [0] * 64
    tables: list[list[int]] = [[] for _ in range(64)]
    for sq in range(64):
        verified = _try_magic(sq, dirs, precomputed_magics[sq])
        if verified is not None:
            mask, shift, table = verified
            magic = precomputed_magics[sq]
        else:
            if rng is None:
                rng = random.Random(_MAGIC_SEED)
            mask, magic, shift, table = _find_magic(sq, dirs, rng)
        masks[sq] = mask
        magics[sq] = magic
        shifts[sq] = shift
        tables[sq] = table
    return masks, magics, shifts, tables


# Built once at import time. Each square's `_PRECOMPUTED_*_MAGICS` entry is
# exhaustively verified (every subset of its relevant mask, exactly as
# `_find_magic` itself would check) in a single O(table size) pass — no
# rejected candidates — which is what keeps this import-time cost small
# (well under a second) despite rook squares needing up to a 4096-entry
# table. An earlier version of this module searched for magics live at
# import time instead of using precomputed constants: with the same fixed
# `_MAGIC_SEED`, several rook squares needed tens to hundreds of thousands of
# rejected candidates before an exhaustively-verified one turned up, costing
# ~18s total — unacceptable for a process a UCI GUI expects to start
# near-instantly. Correctness is unaffected either way (every candidate,
# precomputed or freshly searched, is fully verified against the classical
# oracle before acceptance) — only startup speed differs.
BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS, BISHOP_TABLES = _build_magic_tables(
    DIAG_DIRS, _PRECOMPUTED_BISHOP_MAGICS
)
ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS, ROOK_TABLES = _build_magic_tables(
    ORTHO_DIRS, _PRECOMPUTED_ROOK_MAGICS
)


def bishop_attacks(sq: int, occupied: int) -> int:
    """O(1) magic-bitboard table lookup. Same signature and semantics as
    the classical `_classical_bishop_attacks` this replaces (verified
    exhaustively against it at import time, above)."""
    index = (((occupied & BISHOP_MASKS[sq]) * BISHOP_MAGICS[sq]) & BB_ALL) >> BISHOP_SHIFTS[sq]
    return BISHOP_TABLES[sq][index]


def rook_attacks(sq: int, occupied: int) -> int:
    """O(1) magic-bitboard table lookup. Same signature and semantics as
    the classical `_classical_rook_attacks` this replaces (verified
    exhaustively against it at import time, above)."""
    index = (((occupied & ROOK_MASKS[sq]) * ROOK_MAGICS[sq]) & BB_ALL) >> ROOK_SHIFTS[sq]
    return ROOK_TABLES[sq][index]


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
