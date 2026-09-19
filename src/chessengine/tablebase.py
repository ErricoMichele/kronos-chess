"""Optional Syzygy endgame tablebase probing via python-chess.

This module wraps python-chess's ``chess.syzygy`` reader to provide WDL
(Win/Draw/Loss) and DTZ (Distance To Zeroing move) probing for endgame
positions with few pieces on the board. The engine works perfectly without
python-chess installed: ``AVAILABLE`` is ``False`` in that case, and every
probe returns ``None``.

Per the module-boundary DAG (architecture.md Section 11), ``tablebase.py``
depends on ``board`` and ``fen`` (it converts a ``Board`` to a FEN string,
then parses it with python-chess). It is consumed by ``search.py`` (for
in-search probing) and ``uci.py`` (for the ``SyzygyPath`` option).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .board import Board

# --- Optional python-chess import -------------------------------------------
#
# python-chess is an optional dependency (pyproject.toml
# [project.optional-dependencies] tablebase). If it is not installed, the
# module still loads cleanly with AVAILABLE=False and all probes return None.

try:
    import chess
    import chess.syzygy

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _board_to_chess_board(board: "Board") -> "chess.Board":
    """Convert our engine's Board to a python-chess Board via FEN.

    This is the only bridge between the two board representations. It goes
    through FEN (a universal interchange format both sides already
    support) rather than trying to map internal bitboard layouts directly,
    which would be fragile and couple the two implementations tightly.
    """
    return chess.Board(board.to_fen())


# --- Tablebase score constants ----------------------------------------------
#
# Scores returned by the search when a tablebase probe is conclusive.
# These must be large enough to dominate material evaluation but small
# enough to stay below MATE_SCORE so the search does not confuse a
# tablebase win with an actual checkmate.

TB_WIN_SCORE = 20_000   # returned for WDL = +2 (win)
TB_LOSS_SCORE = -20_000  # returned for WDL = -2 (loss)
TB_CURSED_WIN = 50       # WDL = +1 (cursed win: win but 50-move rule draws)
TB_BLESSED_LOSS = -50    # WDL = -1 (blessed loss: loss but 50-move rule saves)


class SyzygyProber:
    """Wraps a python-chess Syzygy tablebase reader.

    If python-chess is not installed, ``__init__`` still succeeds but
    ``self.available`` is ``False`` and every probe returns ``None``.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.available = False
        self._reader: object | None = None
        if not AVAILABLE:
            return
        try:
            reader = chess.syzygy.open_tablebase(path)
            self._reader = reader
            self.available = True
        except Exception:
            # Path might be invalid or contain no tablebase files -- degrade
            # gracefully rather than crashing the engine.
            self.available = False

    def probe_wdl(self, board: "Board") -> int | None:
        """Probe the WDL tables for the given position.

        Returns:
            An integer in {-2, -1, 0, 1, 2} from the side-to-move's
            perspective, or ``None`` if the position is not in the tables
            or the library is unavailable.

            +2 = win, +1 = cursed win (50-move rule draws it),
             0 = draw,
            -1 = blessed loss (50-move rule saves it), -2 = loss.
        """
        if not self.available or self._reader is None:
            return None
        try:
            cb = _board_to_chess_board(board)
            return self._reader.probe_wdl(cb)
        except Exception:
            return None

    def probe_dtz(self, board: "Board") -> int | None:
        """Probe the DTZ tables for the given position.

        Returns:
            The distance-to-zeroing-move (a move that resets the 50-move
            counter -- a capture or pawn push), or ``None`` if the position
            is not in the tables or the library is unavailable. Positive
            values mean the side to move wins; negative means it loses.
        """
        if not self.available or self._reader is None:
            return None
        try:
            cb = _board_to_chess_board(board)
            return self._reader.probe_dtz(cb)
        except Exception:
            return None

    def close(self) -> None:
        """Release the tablebase reader's resources."""
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
            self.available = False
