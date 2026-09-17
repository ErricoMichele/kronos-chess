"""Zobrist-keyed transposition table (architecture.md §9.2).

A depth-preferred, always-store-on-tie-or-deeper replacement scheme over a
flat, power-of-two-sized array indexed by the low bits of the full 64-bit
Zobrist key. The full key is stored in every entry precisely so that an
index collision (two different positions hashing to the same table slot) is
detected on `probe` and rejected, rather than silently returning a wrong
hit — see `Board.zobrist_hash` / `zobrist.py` for how that key is
maintained.

Per the architecture doc (§9.2, §11), `transposition.py` depends only on
`constants` (for `MATE_SCORE`) — it never imports `board.py`, `movegen.py`,
or `search.py`. `search.py` (Milestone 3) is the only consumer.

**Invariant the rest of the search relies on (§9.2): the TT must never
change the search's result, only its speed.** That invariant lives in the
depth-preferred replacement policy below and in the mate-distance
adjustment functions (`score_to_tt`/`score_from_tt`), not in this module's
tests alone — `search.py` is expected to verify it end-to-end.
"""

from dataclasses import dataclass
from enum import IntEnum

from .constants import MATE_SCORE


class TTFlag(IntEnum):
    """What kind of bound `TTEntry.score` represents, from a prior search of
    the same position to at least `TTEntry.depth`."""

    EXACT = 0  # true score: the search's window wasn't broken out of
    LOWERBOUND = 1  # true score >= this (a beta cutoff occurred)
    UPPERBOUND = 2  # true score <= this (no move raised alpha)


@dataclass(slots=True)
class TTEntry:
    key: int  # full 64-bit zobrist key, stored so index collisions are
    # detected and rejected instead of returning a wrong hit
    depth: int
    score: int
    flag: TTFlag
    best_move: int


class TranspositionTable:
    """Flat array of `TTEntry | None`, indexed by `key & self.mask`.

    Sized in mebibytes rather than entry count so `search.py`'s
    `Search.__init__(tt_size_mb=...)` (and eventually a UCI `Hash` option)
    can size it the way every other engine's UCI interface does; the entry
    count and the power-of-two table size it implies are derived here, once,
    at construction time.
    """

    def __init__(self, size_mb: int = 64) -> None:
        entry_count = (size_mb * 1024 * 1024) // 40  # ~40 bytes/entry incl. list overhead
        self.size = 1 << (entry_count.bit_length() - 1)  # round down to a power of two
        self.mask = self.size - 1
        self.table: list[TTEntry | None] = [None] * self.size

    def probe(self, key: int) -> TTEntry | None:
        """Return the stored entry for `key`, or `None` on a miss or an
        index collision with a different position's key."""
        entry = self.table[key & self.mask]
        return entry if entry is not None and entry.key == key else None

    def store(self, key: int, depth: int, score: int, flag: TTFlag, best_move: int) -> None:
        idx = key & self.mask
        existing = self.table[idx]
        # Depth-preferred replacement: only overwrite a same-key or
        # shallower-or-equal entry, so a shallow re-search never evicts a
        # deeper, more expensive-to-recompute one.
        if existing is None or existing.key == key or depth >= existing.depth:
            self.table[idx] = TTEntry(key, depth, score, flag, best_move)

    def clear(self) -> None:
        """Called on UCI `ucinewgame` (§9.1): drop every entry so a stale
        position from a previous, unrelated game can never be probed into
        this one."""
        self.table = [None] * self.size


# --- Mate-distance adjustment (§9.2) ----------------------------------------
#
# A mate score found N plies below the *current* node is only meaningful
# relative to that node. Storing/retrieving it unadjusted would misreport a
# mate reached via a transposition as closer to (or further from) the search
# root than it really is, since the same stored position can be reached at a
# different ply on a different path. `score_to_tt` converts a from-this-node
# score to a from-this-key (ply-independent) score before storing; conversely,
# `score_from_tt` converts a stored ply-independent score back to a
# from-this-node score right after a probe.


def score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 128:
        return score + ply
    if score <= -MATE_SCORE + 128:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 128:
        return score - ply
    if score <= -MATE_SCORE + 128:
        return score + ply
    return score
