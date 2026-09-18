#!/usr/bin/env python3
"""A/B gate for Principal Variation Search (PVS).

Compares PVS-enabled (the default Search) vs PVS-disabled (a subclass
that monkeypatches the PVS_ENABLED flag to False before each search)
over a 12-position, 24-game battery at SearchLimits(movetime_ms=200),
ply_cap=120. Uses the same match harness and position set as the
NMP/LMR/aspiration/check-extension gates.

Run:
    python -m tests.ab_pvs
or:
    python tests/ab_pvs.py
"""

from __future__ import annotations

import sys
import os

# Ensure the project root and src/ are importable when run as a script,
# overriding any editable-install path that might point elsewhere.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _project_root)

from chessengine.evaluate import default_evaluator
from chessengine.search import Search, SearchLimits
from chessengine import search as search_mod
from tests.match_harness import DEFAULT_POSITIONS, play_match_searches

# The same 6 additional positions used by the NMP/LMR/aspiration/check-
# extension gates, giving a 12-position battery that exercises openings,
# middlegames, tactical positions, and endgames.
EXTRA_POSITIONS: list[str] = [
    # Sicilian Najdorf-ish middlegame
    "r1bq1rk1/1p2bppp/p1np1n2/4p3/4P3/2N1BN2/PPPQ1PPP/R3KB1R w KQ - 0 9",
    # Queen's Gambit Declined structure
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    # Open Ruy Lopez middlegame
    "r1bqkb1r/1pp2ppp/p1np1n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 6",
    # Endgame: rook + pawns
    "8/5pk1/6p1/4R3/5P2/6P1/5K2/3r4 w - - 0 40",
    # Endgame: opposite-color bishops
    "8/5p2/4pkp1/8/2B5/4K1P1/5P2/2b5 w - - 0 35",
    # Middlegame with imbalanced material (queen vs two rooks)
    "r4rk1/pp3ppp/2n1bn2/q3p3/4P3/2N2N2/PPP1QPPP/R1B2RK1 w - - 0 12",
]

ALL_POSITIONS = DEFAULT_POSITIONS + EXTRA_POSITIONS


class _NoPVSSearch(Search):
    """A Search subclass that disables PVS by temporarily patching the
    module-level PVS_ENABLED flag around each search call."""

    def search(self, board, limits, **kwargs):
        saved = search_mod.PVS_ENABLED
        search_mod.PVS_ENABLED = False
        try:
            return super().search(board, limits, **kwargs)
        finally:
            search_mod.PVS_ENABLED = saved


def main() -> None:
    limits = SearchLimits(movetime_ms=200)
    ply_cap = 120

    print(f"PVS A/B gate: {len(ALL_POSITIONS)} positions, "
          f"{len(ALL_POSITIONS) * 2} games, "
          f"movetime={limits.movetime_ms}ms, ply_cap={ply_cap}")
    print("A = PVS-enabled (default Search)")
    print("B = PVS-disabled (_NoPVSSearch)")
    print()

    result = play_match_searches(
        search_factory_a=lambda: Search(default_evaluator()),
        search_factory_b=lambda: _NoPVSSearch(default_evaluator()),
        positions=ALL_POSITIONS,
        limits=limits,
        ply_cap=ply_cap,
    )

    print(f"Result: PVS-enabled {result.score_a} vs PVS-disabled {result.score_b}")
    print()
    for g in result.games:
        print(f"  {g['result']:7s}  white={g['white']}  fen={g['fen'][:40]}...")
    print()

    diff = result.score_a - result.score_b
    if diff >= 0:
        print(f"PVS-enabled wins by +{diff} -- passes the 'must not lose strength' bar.")
    else:
        print(f"PVS-enabled loses by {diff} -- FAILS the 'must not lose strength' bar.")


if __name__ == "__main__":
    main()
