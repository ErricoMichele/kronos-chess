#!/usr/bin/env python3
"""A/B gate for futility pruning: futility-enabled vs futility-disabled.

Uses match_harness.play_match_searches with the same battery and parameters
as the NMP/LMR/aspiration/check-extensions gates:
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
# NMP/LMR/aspiration/check-extensions gates used.
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

    print("A/B gate: futility pruning")
    print(f"  Positions: {len(POSITIONS)}")
    print(f"  Games: {len(POSITIONS) * 2}")
    print(f"  Limits: movetime_ms={limits.movetime_ms}")
    print(f"  Ply cap: {ply_cap}")
    print()

    # Engine A: futility enabled (default)
    def factory_a():
        return Search(default_evaluator())

    # Engine B: futility disabled (FUTILITY_DEPTH = 0)
    original_futility_depth = search_mod.FUTILITY_DEPTH

    class _FutilityDisabledSearch(Search):
        """Search with futility pruning disabled."""
        pass

    def factory_b():
        search_mod.FUTILITY_DEPTH = 0
        s = Search(default_evaluator())
        return s

    # We need to toggle FUTILITY_DEPTH around each game. The simplest
    # approach: wrap the factories so A restores the original and B sets 0.
    class _ToggleFactory:
        def __init__(self, enabled: bool):
            self.enabled = enabled

        def __call__(self):
            if self.enabled:
                search_mod.FUTILITY_DEPTH = original_futility_depth
            else:
                search_mod.FUTILITY_DEPTH = 0
            return Search(default_evaluator())

    # Actually, play_match_searches creates fresh Search instances per game
    # via the factory, but both engines share the same module-level constant.
    # We need a different approach: subclass Search to override _negamax
    # behavior. But that's complex. Instead, let's just do two separate
    # sets of games, controlling the module constant.

    # Simpler: run the match with factories that patch around themselves.
    # Since play_match_searches alternates A/B per game and creates fresh
    # Search instances, the module-level FUTILITY_DEPTH is read at search
    # time, not at construction time. So we need per-search patching.

    # Best approach: use a wrapper that sets the constant before each search.
    from chessengine.board import Board

    class FutilityEnabledSearch(Search):
        def search(self, board, limits, **kwargs):
            search_mod.FUTILITY_DEPTH = original_futility_depth
            return super().search(board, limits, **kwargs)

    class FutilityDisabledSearch(Search):
        def search(self, board, limits, **kwargs):
            search_mod.FUTILITY_DEPTH = 0
            return super().search(board, limits, **kwargs)

    result = play_match_searches(
        lambda: FutilityEnabledSearch(default_evaluator()),
        lambda: FutilityDisabledSearch(default_evaluator()),
        POSITIONS,
        limits,
        ply_cap=ply_cap,
    )

    # Restore
    search_mod.FUTILITY_DEPTH = original_futility_depth

    print(f"Result: futility-enabled {result.score_a} vs futility-disabled {result.score_b}")
    print()
    print("Game details:")
    for g in result.games:
        print(f"  {g['fen'][:40]}...  white={g['white']}  result={g['result']}")

    return result


if __name__ == "__main__":
    result = main()
