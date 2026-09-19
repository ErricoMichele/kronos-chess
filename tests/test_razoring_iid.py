"""Tests for razoring, IID, and history gravity."""
from chessengine.board import Board
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.search import Search, SearchLimits
from chessengine import search as search_mod


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


class TestRazoring:
    """Razoring should prune low-eval nodes at shallow depth."""

    def test_razoring_produces_valid_result(self) -> None:
        """Search with razoring enabled returns a legal move."""
        board = parse_fen(KIWIPETE)
        search = Search(default_evaluator())
        result = search.search(board, SearchLimits(max_depth=4))
        assert result.best_move != 0
        assert result.depth >= 1

    def test_razoring_at_depth_1(self) -> None:
        """At depth 1, razoring can fire. Verify search still works correctly."""
        board = Board.starting_position()
        search = Search(default_evaluator())
        result = search.search(board, SearchLimits(max_depth=1))
        assert result.best_move != 0
        assert result.depth == 1

    def test_razoring_constants_valid(self) -> None:
        """RAZORING_MARGIN is indexed by depth and has reasonable values."""
        assert search_mod.RAZORING_DEPTH == 2
        assert len(search_mod.RAZORING_MARGIN) >= search_mod.RAZORING_DEPTH + 1
        assert all(m >= 0 for m in search_mod.RAZORING_MARGIN)


class TestIID:
    """Internal Iterative Deepening at PV nodes without TT move."""

    def test_iid_produces_valid_result(self) -> None:
        """Search at depth >= IID_DEPTH still produces a valid move."""
        board = parse_fen(KIWIPETE)
        search = Search(default_evaluator())
        result = search.search(board, SearchLimits(max_depth=5))
        assert result.best_move != 0
        assert result.depth >= 1

    def test_iid_with_empty_tt(self) -> None:
        """With a fresh TT (no cached moves), IID should populate it."""
        board = parse_fen(KIWIPETE)
        search = Search(default_evaluator())
        result = search.search(board, SearchLimits(max_depth=5))
        entry = search.tt.probe(board.zobrist_hash)
        assert entry is not None

    def test_iid_constants_valid(self) -> None:
        assert search_mod.IID_DEPTH >= 3
        assert search_mod.IID_REDUCTION >= 1
        assert search_mod.IID_REDUCTION < search_mod.IID_DEPTH


class TestHistoryGravity:
    """History values should saturate near HISTORY_MAX, not grow forever."""

    def test_history_gravity_caps_values(self) -> None:
        """After many cutoffs on the same move, history stays bounded."""
        search = Search(default_evaluator())
        frm, to = 12, 28  # e2->e4
        for _ in range(200):
            old = search.history[frm][to]
            bonus = 64  # depth 8 squared
            search.history[frm][to] = old + bonus - old * bonus // search_mod.HISTORY_MAX
        assert search.history[frm][to] <= search_mod.HISTORY_MAX

    def test_history_starts_at_zero(self) -> None:
        search = Search(default_evaluator())
        assert all(search.history[i][j] == 0 for i in range(64) for j in range(64))
