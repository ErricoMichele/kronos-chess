"""Reverse futility pruning tests (architecture.md S9, Milestone 5 extension)
-- mirrors `test_futility.py`'s structure for the Milestone 5 search
extensions.

Two things:

1. **Basic sanity.** Search still returns legal moves at various depths
   with reverse futility pruning active (the default, unmodified `Search`).

2. **RFP actually prunes.** On the same position at the same depth,
   the node count with RFP enabled is strictly less than the node
   count with RFP disabled (RFP_DEPTH monkeypatched to 0),
   proving the pruning code path is exercised and saves work.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits, _SearchCtx
from chessengine.constants import INF
from chessengine import search as search_mod


# --- Shared FEN fixtures ---------------------------------------------------

OPENING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
KP_ENDGAME_FEN = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
# A position where one side is up material -- static eval is well above
# beta for many nodes, which is exactly where reverse futility should fire.
UNBALANCED_FEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"


# --- 1. Basic sanity: search returns legal moves with RFP active ----------

SANITY_CASES = [
    pytest.param(OPENING_FEN, 4, id="opening-depth-4"),
    pytest.param(OPENING_FEN, 6, id="opening-depth-6"),
    pytest.param(KIWIPETE_FEN, 4, id="kiwipete-depth-4"),
    pytest.param(KIWIPETE_FEN, 6, id="kiwipete-depth-6"),
    pytest.param(KP_ENDGAME_FEN, 5, id="kp-endgame-depth-5"),
    pytest.param(UNBALANCED_FEN, 5, id="unbalanced-depth-5"),
]


@pytest.mark.parametrize("fen, depth", SANITY_CASES)
def test_search_returns_legal_move_with_rfp_active(fen: str, depth: int) -> None:
    """A normal search at various depths still returns a legal move and
    completes without raising, with reverse futility pruning active (the
    default, unmodified `Search`)."""
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=depth))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.depth == depth, f"expected search to reach depth {depth} for {fen!r}"
    legal_moves = generate_legal_moves(board)
    assert result.best_move in legal_moves, (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )
    assert result.nodes > 0


# --- 2. RFP actually prunes (node count comparison) -----------------------

# Positions and depths chosen to maximize the chance RFP fires:
# busy middlegame positions at depth 4+ have internal nodes at depth 1-3
# where the static eval minus margin is well above beta.
PRUNE_CASES = [
    pytest.param(KIWIPETE_FEN, 5, id="kiwipete-depth-5"),
    pytest.param(OPENING_FEN, 5, id="opening-depth-5"),
    pytest.param(UNBALANCED_FEN, 5, id="unbalanced-depth-5"),
]


@pytest.mark.parametrize("fen, depth", PRUNE_CASES)
def test_rfp_reduces_node_count(
    fen: str, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the same position at the same depth, the node count with RFP
    enabled is strictly less than with RFP disabled (RFP_DEPTH
    monkeypatched to 0), proving the pruning code path is exercised and
    saves work.

    Both searches use _negamax directly with a full (-INF, INF) window to
    eliminate aspiration window variability from the comparison."""
    board_enabled = parse_fen(fen)
    search_enabled = Search(default_evaluator())
    ctx_enabled = _SearchCtx(
        SearchLimits(max_depth=depth), deadline=None, stop_event=None, extra_stop=None
    )
    search_enabled._negamax(board_enabled, depth, -INF, INF, 0, ctx_enabled)
    nodes_enabled = ctx_enabled.nodes
    assert board_enabled.to_fen() == fen

    # Disable reverse futility pruning by setting RFP_DEPTH to 0 (no depth
    # qualifies, so the `depth <= RFP_DEPTH` guard never holds).
    monkeypatch.setattr(search_mod, "RFP_DEPTH", 0)
    board_disabled = parse_fen(fen)
    search_disabled = Search(default_evaluator())
    ctx_disabled = _SearchCtx(
        SearchLimits(max_depth=depth), deadline=None, stop_event=None, extra_stop=None
    )
    search_disabled._negamax(board_disabled, depth, -INF, INF, 0, ctx_disabled)
    nodes_disabled = ctx_disabled.nodes
    assert board_disabled.to_fen() == fen

    assert nodes_enabled < nodes_disabled, (
        f"reverse futility pruning did not reduce the node count for {fen!r} at depth {depth}: "
        f"enabled={nodes_enabled} disabled={nodes_disabled} -- the pruning code path "
        f"may not be exercised on this position/depth"
    )
