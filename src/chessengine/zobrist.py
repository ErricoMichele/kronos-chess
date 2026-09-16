"""Zobrist hashing: the incremental-hash key tables and a from-scratch
hash function used as their correctness oracle (architecture.md §7).

Per the module-boundary DAG (architecture.md §11), `zobrist.py` depends
only on `constants`/`bitboard` — it never imports `board.py`. This is what
lets `board.py` import `zobrist.py` (to XOR these tables into
`zobrist_hash` incrementally inside `make_move`/`unmake_move`, §6) with no
import cycle. `compute_hash`'s `board` parameter is therefore typed as a
forward-reference string and only imported under `TYPE_CHECKING`, purely
for static type checkers — never at runtime.

The tables are built once at import time from a fixed-seed RNG
(`random.Random(0xC0FFEE)`), not `os.urandom` or the default seed, so
hashes are reproducible run-to-run and test-to-test: `test_zobrist.py`
(architecture.md §13) checks `board.zobrist_hash == compute_hash(board)`
after every move of a random legal game, which only works if the keys
never change between the run that produced `board` and the run that
computes the oracle value.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .bitboard import iter_bits
from .constants import BLACK, WHITE, file_of

if TYPE_CHECKING:
    from .board import Board

_rng = random.Random(0xC0FFEE)  # fixed seed: reproducible hashes across runs/tests

# [color][piece_type][square] -> random 64-bit key, XORed in/out whenever a
# piece of that color/type is added to or removed from that square.
ZOBRIST_PIECE: list[list[list[int]]] = [
    [[_rng.getrandbits(64) for _ in range(64)] for _ in range(6)] for _ in range(2)
]

# XORed in whenever it's Black to move (and back out when it's White's turn
# again), so make_move/unmake_move just XOR this once per ply (§6).
ZOBRIST_SIDE: int = _rng.getrandbits(64)

# Indexed by the 4-bit castling-rights mask directly (0..15), one key per
# distinct combination of rights rather than one key per individual right —
# this matches how `board.castling_rights` is XORed in `make_move` (§6):
# XOR out the key for the old mask, then XOR in the key for the new one.
ZOBRIST_CASTLING: list[int] = [_rng.getrandbits(64) for _ in range(16)]

# Keyed by file only (0..7) — the en-passant-capturable rank is always
# implied by whichever side is to move, so the file alone fully identifies
# "en passant is available on this file right now." Represents "en passant
# was *available* in this position" (an evasion-relevant, legal-move-
# affecting property of the position), so it is XORed in the instant a
# double pawn push creates `ep_square` and XORed back out on the very next
# move regardless of whether the capture happened — otherwise two positions
# differing only in a long-expired ep square would hash differently despite
# being legally identical.
ZOBRIST_EP_FILE: list[int] = [_rng.getrandbits(64) for _ in range(8)]


def compute_hash(board: "Board") -> int:
    """From-scratch Zobrist hash of `board`'s current position, independent
    of `make_move`/`unmake_move`'s incremental XOR bookkeeping (§6). Used
    only as a test oracle (§13) and to seed a `Board` built directly from
    FEN — never called from a hot path, since it rescans every bitboard.
    """
    h = 0
    for color in (WHITE, BLACK):
        for piece_type in range(6):
            for sq in iter_bits(board.pieces[color][piece_type]):
                h ^= ZOBRIST_PIECE[color][piece_type][sq]
    h ^= ZOBRIST_CASTLING[board.castling_rights]
    if board.ep_square is not None:
        h ^= ZOBRIST_EP_FILE[file_of(board.ep_square)]
    if board.side_to_move == BLACK:
        h ^= ZOBRIST_SIDE
    return h
