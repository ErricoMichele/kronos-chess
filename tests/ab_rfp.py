#!/usr/bin/env python3
"""A/B gate for reverse futility pruning: RFP-enabled vs RFP-disabled.

Uses match_harness.play_match_searches with the same battery and parameters
as the NMP/LMR/aspiration/check-extensions/futility gates:
  - 12 positions (DEFAULT_POSITIONS + 6 extras)
  - 24 games (each position played twice, swapping colors)
  - SearchLimits(movetime_ms=200)
  - ply_cap=120
"""

import sys
import os

# Ensure the worktree's own src is on PYTHONPATH so changes are picked up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chessengine.evaluate import default_evaluator
from chessengine.search import Search, SearchLimits
from chessengine import search as search_mod
from match_harness import DEFAULT_POSITIONS, play_match_searches

# 6 extra positions (varied openings/middlegames), same as the
# NMP/LMR/aspiration/check-extensions/futility gates used.
EXTRA_POSITIONS = [
    # Sicilian-ish middlegame
    "r1b1kb1r/1p3ppp/p1n1pn2/q1ppP3/3P4/2N2N2/PPP1BPPP/R1BQK2R w KQkq - 0 7",
    # Open Ruy Lopez-ish middlegame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Queen's Gambit Declined-ish middlegame
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # King's Indian-ish middlegame
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 6",
    # Caro-Kann-ish middlegame
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
    # Italian-ish middlegame
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
]

POSITIONS = DEFAULT_POSITIONS + EXTRA_POSITIONS


def main():
    limits = SearchLimits(movetime_ms=200)
    ply_cap = 120

    print("A/B gate: reverse futility pruning")
    print(f"  Positions: {len(POSITIONS)}")
    print(f"  Games: {len(POSITIONS) * 2}")
    print(f"  Limits: movetime_ms={limits.movetime_ms}")
    print(f"  Ply cap: {ply_cap}")
    print()

    original_rfp_depth = search_mod.RFP_DEPTH

    class RFPEnabledSearch(Search):
        def search(self, board, limits, **kwargs):
            search_mod.RFP_DEPTH = original_rfp_depth
            return super().search(board, limits, **kwargs)

    class RFPDisabledSearch(Search):
        def search(self, board, limits, **kwargs):
            search_mod.RFP_DEPTH = 0
            return super().search(board, limits, **kwargs)

    result = play_match_searches(
        lambda: RFPEnabledSearch(default_evaluator()),
        lambda: RFPDisabledSearch(default_evaluator()),
        POSITIONS,
        limits,
        ply_cap=ply_cap,
    )

    # Restore
    search_mod.RFP_DEPTH = original_rfp_depth

    print(f"Result: RFP-enabled {result.score_a} vs RFP-disabled {result.score_b}")
    print()
    print("Game details:")
    for g in result.games:
        print(f"  {g['fen'][:40]}...  white={g['white']}  result={g['result']}")

    return result


if __name__ == "__main__":
    result = main()
