"""Second, independent correctness gate for the magic-bitboard sliding
attacks (architecture.md Sec 5.11), on top of `test_attacks.py`'s
differential test against the naive one-square-at-a-time tracer.

`attacks.py` already exhaustively verifies every magic number against the
classical ray-scanning oracle *at import time* (every subset of every
square's relevant-occupancy mask, zero tolerance for a real collision)
before `BISHOP_TABLES`/`ROOK_TABLES` are ever built. That makes a
magic-number/table-generation bug in the *current* code effectively
impossible to smuggle past import. But the whole point of a magic
bitboard is that its correctness is a global property of the generation
algorithm (masks, magic multipliers, table construction), not something
visible from reading `bishop_attacks`/`rook_attacks` in isolation -- so if
that generation code is ever touched later (a "small" refactor of
`_find_magic`, `_build_magic_tables`, `_relevant_occupancy_mask`, the
Carry-Rippler subset enumeration, the generation-stamp collision check,
...) and the import-time verification is accidentally weakened, narrowed,
or removed in the same change, nothing else would catch it -- unless the
test suite *also* independently cross-checks the public
`bishop_attacks`/`rook_attacks`/`queen_attacks` API against the retained
classical oracle. That is what this file does.

This is deliberately independent of `test_attacks.py`:
  - It compares against `_classical_bishop_attacks`/`_classical_rook_attacks`
    (the private ray-scanning oracle the magic tables were verified against
    at import time), not against `test_attacks.py`'s from-scratch
    one-square-at-a-time `naive_ray_trace` tracer.
  - It uses its own fixed seed and its own occupancy-generation strategy
    (uniform `getrandbits(64)`, plus targeted random subsets of each
    square's own relevant-occupancy mask -- the exact domain a magic index
    is computed over, and so the most sensitive space for a collision).

`test_attacks.py` needs no changes: it validates the *behavior* of
`bishop_attacks`/`rook_attacks` (whatever implements them), and this file
separately validates that the magic-bitboard *implementation* of that
behavior agrees with the classical implementation of the same behavior.
"""

import random

import pytest

from chessengine.attacks import (
    BISHOP_MASKS,
    ROOK_MASKS,
    _classical_bishop_attacks,
    _classical_rook_attacks,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)

FULL_BOARD = (1 << 64) - 1


def _classical_queen_attacks(sq: int, occupied: int) -> int:
    """Classical-oracle queen attacks, built the same way `queen_attacks`
    is built from its magic-based halves (bishop union rook) -- but from
    the two *classical* halves, never from `bishop_attacks`/`rook_attacks`
    themselves. Keeps the queen check from silently depending on the code
    under test."""
    return _classical_bishop_attacks(sq, occupied) | _classical_rook_attacks(sq, occupied)


# --- Occupancy generation, independent of test_attacks.py -------------------
#
# A different fixed seed and a different sampling method (uniform
# `getrandbits(64)` here, vs. test_attacks.py's `rng.sample` over a random
# bit count) than test_attacks.py's RANDOM_OCCUPANCIES, so the two fuzz
# corpora don't just happen to be the same numbers under a different name.

_SEED = 0x5EC0D_0F1210


def _build_full_random_occupancies(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    occupancies = [0, FULL_BOARD]
    occupancies.extend(rng.getrandbits(64) for _ in range(count))
    return occupancies


FULL_RANDOM_OCCUPANCIES = _build_full_random_occupancies(count=300, seed=_SEED)


def _occ_id(occupied: int) -> str:
    return f"occ{occupied:#018x}"


# --- Cross-check vs. the classical oracle, full-width random occupancies ----


@pytest.mark.parametrize("occupied", FULL_RANDOM_OCCUPANCIES, ids=_occ_id)
def test_bishop_attacks_matches_classical_oracle_for_every_square(occupied):
    for sq in range(64):
        expected = _classical_bishop_attacks(sq, occupied)
        actual = bishop_attacks(sq, occupied)
        assert actual == expected, (
            f"bishop_attacks (magic) mismatch vs classical oracle at sq={sq}: "
            f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
        )


@pytest.mark.parametrize("occupied", FULL_RANDOM_OCCUPANCIES, ids=_occ_id)
def test_rook_attacks_matches_classical_oracle_for_every_square(occupied):
    for sq in range(64):
        expected = _classical_rook_attacks(sq, occupied)
        actual = rook_attacks(sq, occupied)
        assert actual == expected, (
            f"rook_attacks (magic) mismatch vs classical oracle at sq={sq}: "
            f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
        )


@pytest.mark.parametrize("occupied", FULL_RANDOM_OCCUPANCIES, ids=_occ_id)
def test_queen_attacks_matches_classical_oracle_for_every_square(occupied):
    for sq in range(64):
        expected = _classical_queen_attacks(sq, occupied)
        actual = queen_attacks(sq, occupied)
        assert actual == expected, (
            f"queen_attacks (magic) mismatch vs classical oracle at sq={sq}: "
            f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
        )


# --- Cross-check restricted to each square's own relevant-occupancy mask ----
#
# `bishop_attacks(sq, occ)`/`rook_attacks(sq, occ)` only ever look at
# `occ & {BISHOP,ROOK}_MASKS[sq]` before hashing through the magic
# multiplier -- that masked value *is* the entire input domain a magic
# index is computed over, and so the space where a collision between two
# different attack sets would actually bite. Fuzzing random subsets of
# each square's own mask -- rather than bits scattered uniformly across all
# 64 squares, most of which fall outside any one square's small mask --
# concentrates the search directly on that domain, per square, for all 64
# squares.

_MASK_RNG_SEED = 0xA1A5_FF1C


def _random_mask_subsets(mask: int, count: int, rng: random.Random) -> list[int]:
    """`count` random subsets of `mask`, each bit of `mask` included
    independently with probability 1/2, plus the two deterministic
    extremes (empty and the full mask) so every square's fuzz corpus always
    covers "no blocker in the relevant window" and "every relevant square
    occupied"."""
    subsets = [0, mask]
    bit_positions = [i for i in range(64) if (mask >> i) & 1]
    for _ in range(count):
        subset = 0
        for i in bit_positions:
            if rng.random() < 0.5:
                subset |= 1 << i
        subsets.append(subset)
    return subsets


def _build_per_square_mask_cases(count_per_square: int, seed: int):
    """`[(sq, occupied), ...]` covering every square, with `occupied`
    confined to (random subsets of) that square's own relevant-occupancy
    mask for the relevant piece."""
    rng = random.Random(seed)
    bishop_cases = []
    rook_cases = []
    for sq in range(64):
        for occ in _random_mask_subsets(BISHOP_MASKS[sq], count_per_square, rng):
            bishop_cases.append((sq, occ))
        for occ in _random_mask_subsets(ROOK_MASKS[sq], count_per_square, rng):
            rook_cases.append((sq, occ))
    return bishop_cases, rook_cases


BISHOP_MASK_CASES, ROOK_MASK_CASES = _build_per_square_mask_cases(
    count_per_square=64, seed=_MASK_RNG_SEED
)


def _case_id(case: tuple[int, int]) -> str:
    sq, occupied = case
    return f"sq{sq}_occ{occupied:#018x}"


@pytest.mark.parametrize("case", BISHOP_MASK_CASES, ids=_case_id)
def test_bishop_attacks_matches_classical_oracle_within_relevant_mask(case):
    sq, occupied = case
    expected = _classical_bishop_attacks(sq, occupied)
    actual = bishop_attacks(sq, occupied)
    assert actual == expected, (
        f"bishop_attacks (magic) mismatch vs classical oracle at sq={sq} "
        f"(occupancy confined to this square's relevant mask): "
        f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
    )


@pytest.mark.parametrize("case", ROOK_MASK_CASES, ids=_case_id)
def test_rook_attacks_matches_classical_oracle_within_relevant_mask(case):
    sq, occupied = case
    expected = _classical_rook_attacks(sq, occupied)
    actual = rook_attacks(sq, occupied)
    assert actual == expected, (
        f"rook_attacks (magic) mismatch vs classical oracle at sq={sq} "
        f"(occupancy confined to this square's relevant mask): "
        f"occupied={occupied:#018x} expected={expected:#018x} actual={actual:#018x}"
    )


# --- Bits outside a square's relevant mask must never move its index -------
#
# Directly exercises the masking step (`occupied & {BISHOP,ROOK}_MASKS[sq]`)
# inside `bishop_attacks`/`rook_attacks`: two occupancies that agree on the
# relevant mask but disagree everywhere else must still produce identical
# attack sets, since the classical oracle only ever "sees" a blocker on the
# outermost ray square as the board edge itself, never as an extra blocker.

_IRRELEVANT_BITS_SEED = 0x0FF_B175


@pytest.mark.parametrize("sq", range(64))
def test_bits_outside_relevant_mask_do_not_affect_bishop_or_rook_attacks(sq):
    rng = random.Random(_IRRELEVANT_BITS_SEED + sq)
    bishop_mask = BISHOP_MASKS[sq]
    rook_mask = ROOK_MASKS[sq]
    for _ in range(32):
        base_occupied = rng.getrandbits(64)
        noisy_occupied = base_occupied ^ (rng.getrandbits(64) & ~bishop_mask & ~rook_mask)
        assert bishop_attacks(sq, base_occupied) == bishop_attacks(sq, noisy_occupied)
        assert bishop_attacks(sq, noisy_occupied) == _classical_bishop_attacks(sq, noisy_occupied)
        assert rook_attacks(sq, base_occupied) == rook_attacks(sq, noisy_occupied)
        assert rook_attacks(sq, noisy_occupied) == _classical_rook_attacks(sq, noisy_occupied)
