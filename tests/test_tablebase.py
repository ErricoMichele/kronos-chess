"""Tests for Syzygy tablebase probing (``tablebase.py``).

Tests are organized into three tiers:

1. **Always run**: the module loads cleanly, ``AVAILABLE`` reflects whether
   python-chess is installed, and the ``SyzygyProber`` degrades gracefully
   when initialized without real tablebase files.

2. **Requires python-chess**: the Board-to-FEN-to-chess.Board conversion
   round-trips correctly. Skipped if python-chess is not installed.

3. **Requires tablebase files**: actual WDL/DTZ probing against real
   Syzygy tables. Skipped if no tablebase directory is available (the
   vast majority of CI/dev environments).
"""

from __future__ import annotations

import os

import pytest

from chessengine import fen
from chessengine.tablebase import AVAILABLE, SyzygyProber


# ---------------------------------------------------------------------------
# Tier 1: always runs, no python-chess required
# ---------------------------------------------------------------------------


def test_module_loads_and_available_is_bool() -> None:
    """``AVAILABLE`` is a bool reflecting whether python-chess could be
    imported. The module itself must always import cleanly regardless."""
    assert isinstance(AVAILABLE, bool)


def test_prober_with_invalid_path_degrades_gracefully() -> None:
    """A ``SyzygyProber`` initialized with a nonexistent path must not
    crash; probes must return ``None``."""
    prober = SyzygyProber("/nonexistent/path/to/syzygy/tables")
    board = fen.parse_fen("8/8/8/8/8/5k2/8/4K3 w - - 0 1")  # KvK
    assert prober.probe_wdl(board) is None
    assert prober.probe_dtz(board) is None


def test_prober_close_is_safe_on_unavailable() -> None:
    """Calling ``close()`` on a prober that never opened anything must
    not raise."""
    prober = SyzygyProber("/nonexistent")
    prober.close()  # must not raise


# ---------------------------------------------------------------------------
# Tier 2: requires python-chess (but not tablebase files)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AVAILABLE, reason="python-chess not installed")
def test_board_to_fen_roundtrip_with_python_chess() -> None:
    """Our Board's FEN output must be parseable by python-chess and
    produce an identical FEN (proving the conversion bridge works)."""
    import chess

    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/8/8/5k2/8/4K3 w - - 0 1",  # KvK
        "8/8/8/4k3/8/8/4KP2/8 w - - 0 1",  # KPvK
        "8/5k2/8/8/8/8/3RK3/8 w - - 0 1",  # KRvK
    ]
    for fen_str in test_fens:
        our_board = fen.parse_fen(fen_str)
        our_fen = our_board.to_fen()
        pc_board = chess.Board(our_fen)
        # python-chess normalizes FEN the same way we do; the round-trip
        # must match field by field.
        assert pc_board.fen() == our_fen, (
            f"FEN round-trip mismatch for {fen_str!r}: "
            f"our={our_fen!r}, python-chess={pc_board.fen()!r}"
        )


@pytest.mark.skipif(not AVAILABLE, reason="python-chess not installed")
def test_prober_with_empty_dir_returns_none_for_nontrivial() -> None:
    """A prober pointed at an empty (but existing) directory should open
    without error. A non-trivial endgame position (KR vs K) that requires
    actual .rtbw files should return None when no files are present.

    Note: KvK (bare kings) is trivially resolved by python-chess without
    any table files, so we use a 3-piece position instead."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        prober = SyzygyProber(tmpdir)
        # KR vs K -- requires KRvK.rtbw to probe; without it, returns None.
        board = fen.parse_fen("8/5k2/8/8/8/8/3RK3/8 w - - 0 1")
        assert prober.probe_wdl(board) is None
        assert prober.probe_dtz(board) is None
        prober.close()


# ---------------------------------------------------------------------------
# Tier 3: requires actual Syzygy tablebase files
# ---------------------------------------------------------------------------

_TB_PATH = os.environ.get("SYZYGY_PATH", "")
_HAS_TB_FILES = bool(_TB_PATH) and os.path.isdir(_TB_PATH)


@pytest.mark.skipif(
    not AVAILABLE or not _HAS_TB_FILES,
    reason="python-chess not installed or SYZYGY_PATH not set / not a directory",
)
def test_probe_wdl_krk_white_wins() -> None:
    """KR vs K is a trivial win for the side with the rook. With White to
    move and a rook, WDL must be +2 (win)."""
    prober = SyzygyProber(_TB_PATH)
    board = fen.parse_fen("8/5k2/8/8/8/8/3RK3/8 w - - 0 1")
    wdl = prober.probe_wdl(board)
    assert wdl == 2, f"Expected WDL=+2 (win) for KRvK, got {wdl}"
    prober.close()


@pytest.mark.skipif(
    not AVAILABLE or not _HAS_TB_FILES,
    reason="python-chess not installed or SYZYGY_PATH not set / not a directory",
)
def test_probe_wdl_kvk_is_draw() -> None:
    """K vs K is always a draw."""
    prober = SyzygyProber(_TB_PATH)
    board = fen.parse_fen("8/8/8/8/8/5k2/8/4K3 w - - 0 1")
    wdl = prober.probe_wdl(board)
    assert wdl == 0, f"Expected WDL=0 (draw) for KvK, got {wdl}"
    prober.close()


@pytest.mark.skipif(
    not AVAILABLE or not _HAS_TB_FILES,
    reason="python-chess not installed or SYZYGY_PATH not set / not a directory",
)
def test_probe_dtz_returns_integer_or_none() -> None:
    """DTZ probe must return an int (positive, negative, or zero) or None."""
    prober = SyzygyProber(_TB_PATH)
    board = fen.parse_fen("8/5k2/8/8/8/8/3RK3/8 w - - 0 1")
    dtz = prober.probe_dtz(board)
    assert dtz is None or isinstance(dtz, int), f"Expected int or None, got {type(dtz)}"
    prober.close()
