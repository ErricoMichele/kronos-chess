"""Correctness gate for classical ray-scanning sliding attacks
(architecture.md Sec 5.3, Sec 13).

The design (architecture.md Sec 5.1) explicitly calls for checking
`sliding_attacks`/`bishop_attacks`/`rook_attacks` against "a single property
test against an even-more-naive one-square-at-a-time tracer" -- this file is
that test.

The reference tracer below is written independently of `attacks.py`: it does
not import or reuse `RAY_ATTACKS`, `POSITIVE_DIRS`/`NEGATIVE_DIRS`, or the
lsb/msb-based "chop the ray at the first blocker" trick that
`sliding_attacks` uses internally. Instead it walks one square at a time in
each direction, appending squares to the attack set until it either falls off
the board or lands on an occupied square (in which case that occupied square
is the last one included -- a slider can capture into it, but nothing
further). Two independently-written implementations of the same rule
agreeing across a large, varied sample of occupancies is much stronger
evidence of correctness than either one read in isolation.
"""

import random

import pytest

from chessengine.attacks import (
    DIAG_DIRS,
    ORTHO_DIRS,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
    sliding_attacks,
)

# --- Independent naive reference tracer -------------------------------------

FULL_BOARD = (1 << 64) - 1  # all 64 bits set; spelled out, not imported

# (delta_file, delta_rank) for the four orthogonal / four diagonal rays,
# written from scratch here rather than reusing attacks.DIR_FILE_RANK_DELTA.
_NAIVE_ORTHO_DELTAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_NAIVE_DIAG_DELTAS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def naive_ray_trace(sq: int, occupied: int, deltas: list[tuple[int, int]]) -> int:
    """One-square-at-a-time walk in each of `deltas` from `sq`, stopping at
    the board edge or at (and including) the first occupied square. No ray
    tables, no bit-scan blocker resolution -- just a step counter."""
    f0, r0 = sq % 8, sq // 8
    attacks = 0
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            attacks |= 1 << target
            if (occupied >> target) & 1:
                break  # blocker: included above, but the ray stops here
            f += df
            r += dr
    return attacks


def naive_bishop_attacks(sq: int, occupied: int) -> int:
    return naive_ray_trace(sq, occupied, _NAIVE_DIAG_DELTAS)


def naive_rook_attacks(sq: int, occupied: int) -> int:
    return naive_ray_trace(sq, occupied, _NAIVE_ORTHO_DELTAS)


def naive_queen_attacks(sq: int, occupied: int) -> int:
    return naive_bishop_attacks(sq, occupied) | naive_rook_attacks(sq, occupied)


# --- Sanity-check the oracle itself against a few hand-computed positions ---
#
# The fuzz tests below are only as good as the oracle they compare against,
# so a handful of positions are checked by hand first, independently of both
# `naive_ray_trace` and the real implementation.

def test_naive_rook_from_a1_on_empty_board_is_hand_verified():
    # Rook on a1 (square 0), nothing else on the board: attacks the whole
    # first rank (b1..h1) and the whole a-file (a2..a8), nothing else.
    expected = 0
    for sq in range(1, 8):  # b1..h1
        expected |= 1 << sq
    for rank in range(1, 8):  # a2..a8
        expected |= 1 << (rank * 8)
    assert naive_rook_attacks(0, 0) == expected


def test_naive_bishop_from_d4_on_empty_board_is_hand_verified():
    # Bishop on d4 (file 3, rank 3 -> square 27): the four full diagonals to
    # the edges: a1-h8 diagonal, a7-g1 diagonal (through d4).
    d4 = 3 + 3 * 8
    expected_squares = [
        # NE: e5, f6, g7, h8
        4 + 4 * 8, 5 + 5 * 8, 6 + 6 * 8, 7 + 7 * 8,
        # SW: c3, b2, a1
        2 + 2 * 8, 1 + 1 * 8, 0 + 0 * 8,
        # NW: c5, b6, a7
        2 + 4 * 8, 1 + 5 * 8, 0 + 6 * 8,
        # SE: e3, f2, g1
        4 + 2 * 8, 5 + 1 * 8, 6 + 0 * 8,
    ]
    expected = 0
    for sq in expected_squares:
        expected |= 1 << sq
    assert naive_bishop_attacks(d4, 0) == expected


def test_naive_bishop_stops_at_and_includes_first_blocker():
    # Bishop on d4, a blocker placed on f6 (two squares up the NE diagonal).
    d4 = 3 + 3 * 8
    e5 = 4 + 4 * 8
    f6 = 5 + 5 * 8
    g7 = 6 + 6 * 8
    h8 = 7 + 7 * 8
    occupied = 1 << f6
    attacks = naive_bishop_attacks(d4, occupied)
    assert attacks & (1 << e5)  # square before the blocker: attacked
    assert attacks & (1 << f6)  # the blocker itself: attacked (capturable)
    assert not attacks & (1 << g7)  # beyond the blocker: not attacked
    assert not attacks & (1 << h8)


def test_naive_rook_double_blocker_only_first_one_reachable():
    # Rook on a1, blockers on a4 and a6: only a4 (the nearer one) should be
    # reachable/attacked; a5 and a6 must not be.
    a1 = 0
    a4 = 0 + 3 * 8
    a5 = 0 + 4 * 8
    a6 = 0 + 5 * 8
    occupied = (1 << a4) | (1 << a6)
    attacks = naive_rook_attacks(a1, occupied)
    assert attacks & (1 << a4)
    assert not attacks & (1 << a5)
    assert not attacks & (1 << a6)


# --- Fixed-seed random occupancy generation ---------------------------------


def _random_occupancy(rng: random.Random) -> int:
    """One random occupancy with a randomly chosen bit count, so the fuzz
    corpus covers everything from near-empty to near-full boards (and
    therefore both 'no blocker on this ray' and 'blocker one/several
    squares away' cases in roughly equal measure)."""
    num_bits = rng.randint(0, 64)
    bb = 0
    for sq in rng.sample(range(64), num_bits):
        bb |= 1 << sq
    return bb


def _build_random_occupancies(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    # Deterministic edge cases are always present, in addition to the random
    # sample, so a regression in either extreme is never left to chance.
    occupancies = [0, FULL_BOARD]
    occupancies.extend(_random_occupancy(rng) for _ in range(count))
    return occupancies


# Fixed seed: failures are reproducible across runs/machines.
RANDOM_OCCUPANCIES = _build_random_occupancies(count=200, seed=20260916)


def _occ_id(occupied: int) -> str:
    return f"occ{occupied:#018x}"


# --- Property test: real implementation vs. the independent naive tracer ---


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_bishop_attacks_matches_naive_tracer_for_every_square(occupied):
    for sq in range(64):
        expected = naive_bishop_attacks(sq, occupied)
        actual = bishop_attacks(sq, occupied)
        assert actual == expected, (
            f"bishop_attacks mismatch at sq={sq}: "
            f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
        )


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_rook_attacks_matches_naive_tracer_for_every_square(occupied):
    for sq in range(64):
        expected = naive_rook_attacks(sq, occupied)
        actual = rook_attacks(sq, occupied)
        assert actual == expected, (
            f"rook_attacks mismatch at sq={sq}: "
            f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
        )


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_queen_attacks_matches_naive_tracer_for_every_square(occupied):
    for sq in range(64):
        expected = naive_queen_attacks(sq, occupied)
        actual = queen_attacks(sq, occupied)
        assert actual == expected, (
            f"queen_attacks mismatch at sq={sq}: "
            f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
        )


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_sliding_attacks_matches_naive_tracer_directly_for_every_square(occupied):
    """`sliding_attacks` is the primitive `bishop_attacks`/`rook_attacks`
    build on (architecture.md Sec 5.3); exercise it directly with each
    direction tuple too, not only through its two wrappers."""
    for sq in range(64):
        expected_diag = naive_bishop_attacks(sq, occupied)
        expected_ortho = naive_rook_attacks(sq, occupied)
        assert sliding_attacks(sq, occupied, DIAG_DIRS) == expected_diag
        assert sliding_attacks(sq, occupied, ORTHO_DIRS) == expected_ortho


# --- Structural / consistency invariants ------------------------------------


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_bishop_and_rook_wrappers_agree_with_sliding_attacks(occupied):
    """bishop_attacks/rook_attacks must be exactly sliding_attacks called
    with DIAG_DIRS/ORTHO_DIRS -- no hidden divergence between the wrapper
    and the primitive it wraps."""
    for sq in range(64):
        assert bishop_attacks(sq, occupied) == sliding_attacks(sq, occupied, DIAG_DIRS)
        assert rook_attacks(sq, occupied) == sliding_attacks(sq, occupied, ORTHO_DIRS)


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_queen_attacks_is_exactly_bishop_union_rook(occupied):
    for sq in range(64):
        assert queen_attacks(sq, occupied) == bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_attacks_never_include_the_slider_own_square(occupied):
    """A slider is never counted as attacking the square it stands on --
    every ray starts at the *adjacent* square, never `sq` itself."""
    for sq in range(64):
        own_bit = 1 << sq
        assert bishop_attacks(sq, occupied) & own_bit == 0
        assert rook_attacks(sq, occupied) & own_bit == 0
        assert queen_attacks(sq, occupied) & own_bit == 0


@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_attacks_stay_within_64_bit_range(occupied):
    """Bounds-by-construction check (architecture.md Sec 3.1): a sliding
    attack bitboard must never carry a bit at index >= 64 or a negative
    value, for any occupancy."""
    for sq in range(64):
        for value in (
            bishop_attacks(sq, occupied),
            rook_attacks(sq, occupied),
            queen_attacks(sq, occupied),
        ):
            assert 0 <= value <= FULL_BOARD


# --- Targeted, human-checkable corner/edge cases -----------------------------
# (Corners and edge files/ranks are where off-by-one bounds bugs in a ray
# walk are most likely to surface, so they get named, explicit cases in
# addition to the blanket sq in range(64) coverage above.)

CORNER_AND_EDGE_SQUARES = {
    "a1": 0,
    "h1": 7,
    "a8": 56,
    "h8": 63,
    "a4": 24,
    "h5": 39,
    "d1": 3,
    "e8": 60,
}


@pytest.mark.parametrize("sq", sorted(CORNER_AND_EDGE_SQUARES.values()))
@pytest.mark.parametrize("occupied", RANDOM_OCCUPANCIES, ids=_occ_id)
def test_corner_and_edge_squares_match_naive_tracer(occupied, sq):
    assert bishop_attacks(sq, occupied) == naive_bishop_attacks(sq, occupied)
    assert rook_attacks(sq, occupied) == naive_rook_attacks(sq, occupied)
