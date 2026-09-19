#!/usr/bin/env python3
"""A/B gate: LMP + delta pruning + countermove heuristic enabled vs disabled.

  - 12 positions (DEFAULT_POSITIONS + 6 extras)
  - 24 games (each position played twice, swapping colors)
  - SearchLimits(movetime_ms=200)
  - ply_cap=120
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessengine.evaluate import default_evaluator
from chessengine.search import Search, SearchLimits
from chessengine import search as search_mod
from match_harness import DEFAULT_POSITIONS, play_match_searches

EXTRA_POSITIONS = [
    "r1b1kb1r/1p3ppp/p1n1pn2/q1ppP3/3P4/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 0 7",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 6",
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
]

POSITIONS = DEFAULT_POSITIONS + EXTRA_POSITIONS

ORIG_LMP_DEPTH = search_mod.LMP_DEPTH
ORIG_DELTA_MARGIN = search_mod.DELTA_MARGIN


def main():
    limits = SearchLimits(movetime_ms=200)
    ply_cap = 120

    print("A/B gate: LMP + delta pruning + countermove heuristic")
    print(f"  Positions: {len(POSITIONS)}")
    print(f"  Games: {len(POSITIONS) * 2}")
    print(f"  Limits: movetime_ms={limits.movetime_ms}")
    print(f"  Ply cap: {ply_cap}")
    print()

    class EnabledSearch(Search):
        def search(self, board, limits, **kwargs):
            search_mod.LMP_DEPTH = ORIG_LMP_DEPTH
            search_mod.DELTA_MARGIN = ORIG_DELTA_MARGIN
            return super().search(board, limits, **kwargs)

    class DisabledSearch(Search):
        def search(self, board, limits, **kwargs):
            search_mod.LMP_DEPTH = 0
            search_mod.DELTA_MARGIN = 99_999
            return super().search(board, limits, **kwargs)

    result = play_match_searches(
        lambda: EnabledSearch(default_evaluator()),
        lambda: DisabledSearch(default_evaluator()),
        POSITIONS,
        limits,
        ply_cap=ply_cap,
    )

    search_mod.LMP_DEPTH = ORIG_LMP_DEPTH
    search_mod.DELTA_MARGIN = ORIG_DELTA_MARGIN

    print(f"Result: Enabled {result.score_a} vs Disabled {result.score_b}")
    print()
    print("Game details:")
    for game in result.games:
        print(f"  {game}")
    print()
    total = result.score_a + result.score_b
    if total > 0:
        pct = result.score_a / total * 100
        print(f"Win rate: {pct:.1f}%")
    if result.score_a > result.score_b:
        print("PASS: optimizations are at least as strong")
    else:
        print("FAIL: optimizations may have weakened the engine")


if __name__ == "__main__":
    main()
