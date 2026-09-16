"""Bitboard primitives: the 64-bit-int convention, bounds masking, and the
bit-scan helpers used throughout attack generation, move generation, and
Zobrist hashing (architecture.md §3.1, §5.3, §11).

This module is a leaf of the dependency DAG (architecture.md §11): it
depends on nothing else in the package, so every other module is free to
import it without risking a cycle.

Square indexing is little-endian rank-file (LERF): ``square = rank * 8 +
file``, ``a1 == 0`` through ``h8 == 63``. Bit ``i`` of a bitboard corresponds
to square ``i``. A bitboard is a plain Python ``int``, always kept in
``[0, 2**64 - 1]``.
"""

BB_ALL = 0xFFFF_FFFF_FFFF_FFFF


def mask64(x: int) -> int:
    """Clamp an arbitrary Python int to the 64-bit range a bitboard must
    stay within. Python ints are arbitrary-precision and never silently
    truncate, so this is only needed where a construction (e.g. an
    intermediate shift) could otherwise carry bits past bit 63."""
    return x & BB_ALL


def lsb_index(bb: int) -> int:
    """Index of the least-significant set bit. ``bb`` must be nonzero;
    callers (attacks.py, movegen.py) only ever call this after checking a
    mask is nonzero."""
    return (bb & -bb).bit_length() - 1


def msb_index(bb: int) -> int:
    """Index of the most-significant set bit. ``bb`` must be nonzero, same
    calling convention as ``lsb_index``."""
    return bb.bit_length() - 1


def pop_lsb(bb: int) -> tuple[int, int]:
    """Return ``(index_of_lsb, bb_with_lsb_cleared)``."""
    idx = lsb_index(bb)
    return idx, bb & (bb - 1)


def popcount(bb: int) -> int:
    """Number of set bits. Native ``int.bit_count()`` (Python 3.10+)."""
    return bb.bit_count()


def iter_bits(bb: int):
    """Yield the index of every set bit, from least- to most-significant."""
    while bb:
        idx, bb = pop_lsb(bb)
        yield idx


def print_bitboard(bb: int) -> None:
    """Debug helper: print an 8x8 ASCII grid of ``bb``, rank 8 at the top
    and file a at the left (the conventional way a chess board is drawn),
    '1' for a set bit and '.' for an empty square. Not used on any hot
    path — for interactive debugging and test failure output only."""
    ranks = []
    for rank in range(7, -1, -1):
        squares = (rank * 8 + file for file in range(8))
        ranks.append(" ".join("1" if (bb >> sq) & 1 else "." for sq in squares))
    print("\n".join(ranks))
