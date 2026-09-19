"""PVS (Principal Variation Search) tests (Milestone 5 extension) --
mirrors test_futility.py / test_null_move.py / test_lmr.py structure.

Two things:

1. **Basic sanity.** Search still returns legal moves at various depths
   with PVS active (the default, unmodified `Search`).

2. **Correctness.** PVS is a pure optimisation — it must return the same
   best move as plain alpha-beta.  (Node count can go up or down depending
   on move-ordering quality; the A/B match validates net strength.)
"""

from __future__ import annotations

import pytest

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
TACTICAL_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


# --- 1. Basic sanity: search returns legal moves with PVS active ----------

SANITY_CASES = [
    pytest.param(OPENING_FEN, 4, id="opening-depth-4"),
    pytest.param(OPENING_FEN, 6, id="opening-depth-6"),
    pytest.param(KIWIPETE_FEN, 4, id="kiwipete-depth-4"),
    pytest.param(KIWIPETE_FEN, 6, id="kiwipete-depth-6"),
    pytest.param(KP_ENDGAME_FEN, 5, id="kp-endgame-depth-5"),
    pytest.param(TACTICAL_FEN, 5, id="tactical-depth-5"),
]


@pytest.mark.parametrize("fen, depth", SANITY_CASES)
def test_search_returns_legal_move_with_pvs_active(fen: str, depth: int) -> None:
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=depth))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.depth == depth
    legal_moves = generate_legal_moves(board)
    assert result.best_move in legal_moves, (
        f"search returned an illegal move ({move_to_uci(result.best_move)!r}) for {fen!r}"
    )
    assert result.nodes > 0


# --- 2. PVS preserves the same best move ----------------------------------

CORRECTNESS_CASES = [
    pytest.param(KIWIPETE_FEN, 5, id="kiwipete-depth-5"),
    pytest.param(OPENING_FEN, 5, id="opening-depth-5"),
    pytest.param(TACTICAL_FEN, 5, id="tactical-depth-5"),
]


@pytest.mark.parametrize("fen, depth", CORRECTNESS_CASES)
def test_pvs_same_best_move(
    fen: str, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PVS is a pure optimisation — it must return the same best move.

    Futility and RFP are disabled so their alpha-dependent thresholds don't
    interact with PVS's scout windows to produce different prune decisions."""
    monkeypatch.setattr(search_mod, "FUTILITY_DEPTH", 0)
    monkeypatch.setattr(search_mod, "RFP_DEPTH", 0)

    board_pvs = parse_fen(fen)
    result_pvs = Search(default_evaluator()).search(board_pvs, SearchLimits(max_depth=depth))

    monkeypatch.setattr(search_mod, "PVS_ENABLED", False)
    board_no = parse_fen(fen)
    result_no = Search(default_evaluator()).search(board_no, SearchLimits(max_depth=depth))

    assert result_pvs.best_move == result_no.best_move, (
        f"PVS changed best move for {fen!r}: "
        f"pvs={move_to_uci(result_pvs.best_move)} no_pvs={move_to_uci(result_no.best_move)}"
    )
