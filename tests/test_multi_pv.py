"""Tests for Multi-PV search support (Milestone 5).

Multi-PV allows the engine to report the N best lines, not just the best
one.  Chess GUIs use this for analysis mode.  Tests here exercise both the
search-level API (``Search.search`` with ``SearchLimits.multi_pv``) and the
UCI-level integration (``setoption name MultiPV value N`` followed by ``go``).
"""

from __future__ import annotations

import io
import re
import time

from chessengine.board import Board
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits, SearchResult
from chessengine.uci import UCIEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLL_INTERVAL_S = 0.02
_POLL_TIMEOUT_S = 5.0


def _drive(command_text: str, *, timeout: float = _POLL_TIMEOUT_S) -> tuple[UCIEngine, str]:
    """Run *command_text* through a fresh ``UCIEngine`` over in-memory IO,
    then poll the captured output for a ``bestmove`` line before returning.
    """
    inp = io.StringIO(command_text)
    out = io.StringIO()
    engine = UCIEngine()
    engine.run(inp, out)
    deadline = time.monotonic() + timeout
    while "bestmove" not in out.getvalue() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
    if engine.search_thread is not None:
        remaining = max(0.0, deadline - time.monotonic())
        engine.search_thread.join(timeout=remaining)
    return engine, out.getvalue()


def _lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line]


# A position outside the opening book (1...Nf6) so a real search is forced.
_TEST_FEN = "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"


# ---------------------------------------------------------------------------
# 1.  multi_pv=1 returns the same result as the default (no multi-PV)
# ---------------------------------------------------------------------------


def test_multi_pv_one_same_as_default() -> None:
    """With ``multi_pv=1`` (the default), Multi-PV must not alter the search
    result compared to omitting the field entirely."""
    board = parse_fen(_TEST_FEN)

    search1 = Search(default_evaluator())
    result1 = search1.search(board, SearchLimits(max_depth=3))

    search2 = Search(default_evaluator())
    result2 = search2.search(board, SearchLimits(max_depth=3, multi_pv=1))

    assert result1.best_move == result2.best_move
    assert result1.score_cp == result2.score_cp
    assert result1.multi_pv_lines is None
    assert result2.multi_pv_lines is None


# ---------------------------------------------------------------------------
# 2.  multi_pv=3 returns 3 different best moves
# ---------------------------------------------------------------------------


def test_multi_pv_three_returns_three_different_moves() -> None:
    """Requesting 3 PV lines on a position with >= 3 legal moves must
    produce exactly 3 lines, each with a distinct best move."""
    board = parse_fen(_TEST_FEN)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=3, multi_pv=3))

    assert result.multi_pv_lines is not None
    assert len(result.multi_pv_lines) == 3

    moves = [line.best_move for line in result.multi_pv_lines]
    assert len(set(moves)) == 3, (
        f"Expected 3 distinct best moves, got: "
        f"{[move_to_uci(m) for m in moves]}"
    )


# ---------------------------------------------------------------------------
# 3.  multi_pv lines are sorted by score (best first)
# ---------------------------------------------------------------------------


def test_multi_pv_sorted_by_score() -> None:
    board = parse_fen(_TEST_FEN)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=3, multi_pv=3))

    assert result.multi_pv_lines is not None
    scores = [line.score_cp for line in result.multi_pv_lines]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"PV lines not sorted by score (best first): {scores}"
        )


# ---------------------------------------------------------------------------
# 4.  multi_pv > number of legal moves returns all legal moves
# ---------------------------------------------------------------------------


def test_multi_pv_exceeds_legal_moves() -> None:
    """When multi_pv is larger than the number of legal root moves, the
    engine must return exactly as many lines as there are legal moves,
    not more."""
    # King on a8 vs king on h1: White has exactly 3 legal moves (a7, b8, b7).
    board = parse_fen("K7/8/8/8/8/8/8/7k w - - 0 1")
    num_legal = len(generate_legal_moves(board))
    assert num_legal == 3, f"Expected 3 legal moves, got {num_legal}"

    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=3, multi_pv=10))

    assert result.multi_pv_lines is not None
    assert len(result.multi_pv_lines) == num_legal


# ---------------------------------------------------------------------------
# 5.  Each PV line has a non-empty PV
# ---------------------------------------------------------------------------


def test_multi_pv_lines_have_nonempty_pvs() -> None:
    board = parse_fen(_TEST_FEN)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=3, multi_pv=3))

    assert result.multi_pv_lines is not None
    for k, line in enumerate(result.multi_pv_lines):
        assert line.pv, f"PV line {k + 1} has an empty PV"
        assert line.best_move == line.pv[0], (
            f"PV line {k + 1}: best_move does not match pv[0]"
        )


# ---------------------------------------------------------------------------
# 6.  UCI output format includes ``multipv`` tag
# ---------------------------------------------------------------------------


def test_uci_multi_pv_output_format() -> None:
    """When MultiPV is set via ``setoption``, every ``info`` line at each
    completed depth must contain a ``multipv K`` tag for K in 1..N."""
    commands = (
        "uci\n"
        "setoption name MultiPV value 3\n"
        "isready\n"
        "position startpos moves e2e4 g8f6\n"
        "go depth 3\n"
    )
    _engine, output = _drive(commands)
    lines = _lines(output)

    info_lines = [l for l in lines if l.startswith("info ")]
    assert info_lines, f"No info lines in output:\n{output}"

    # Every info line must contain 'multipv <number>'
    for line in info_lines:
        assert re.search(r"\bmultipv \d+\b", line), (
            f"info line missing 'multipv' tag: {line}"
        )

    # At least one line for each of multipv 1, 2, 3
    for k in (1, 2, 3):
        matching = [l for l in info_lines if f"multipv {k} " in l]
        assert matching, (
            f"No info line found for multipv {k} in output:\n{output}"
        )


def test_uci_multi_pv_option_advertised() -> None:
    """The ``uci`` command must advertise the ``MultiPV`` option."""
    _engine, output = _drive("uci\n", timeout=0.5)
    assert "option name MultiPV type spin default 1 min 1 max 500" in output


def test_uci_setoption_multipv_is_stored() -> None:
    """``setoption name MultiPV value N`` must update the engine's internal
    ``multi_pv`` state."""
    engine = UCIEngine()
    out = io.StringIO()
    engine.handle_command("setoption name MultiPV value 5", out)
    assert engine.multi_pv == 5


# ---------------------------------------------------------------------------
# 7.  bestmove matches the first PV line
# ---------------------------------------------------------------------------


def test_multi_pv_bestmove_matches_first_line() -> None:
    """The overall ``bestmove`` must agree with the best-scored PV line."""
    board = parse_fen(_TEST_FEN)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=3, multi_pv=3))

    assert result.multi_pv_lines is not None
    assert result.best_move == result.multi_pv_lines[0].best_move
    assert result.score_cp == result.multi_pv_lines[0].score_cp
