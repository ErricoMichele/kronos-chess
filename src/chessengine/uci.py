"""UCI (Universal Chess Interface) protocol handler (architecture.md §12).

Per the module-boundary DAG (architecture.md §11), `uci.py` depends on
`board`, `movegen`, `move`, `search`, `book`, `fen`, and `constants` — it is
the outermost layer besides `cli.py`, and nothing below it in the DAG ever
imports it. It is also the only module that touches `sys.stdin`/`stdout`
and `threading`.

`cmd_go` consults `book.probe_book` before ever calling `Search.search`
(architecture.md §15): a book hit for the current position short-circuits
straight to `bestmove` and never spawns a search thread at all, keeping the
book and the search engine fully decoupled — `search.py` has no knowledge
of `book.py` and its own contract is unchanged by this.

`UCIEngine` owns the engine's UCI-visible state: the current `Board`, a
persistent `Search` instance (so its transposition table/killers/history
survive across moves within one game and are reset by `ucinewgame`), and
the `threading.Event`/`Thread` pair that let `go` run a search without ever
blocking the main thread's ability to react to `stop`/`quit` while stdin is
read line-by-line (§9.3, §12).
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from . import book, fen, movegen
from .board import Board
from .constants import WHITE
from .evaluate import default_evaluator
from .move import move_to_uci
from .search import Search, SearchInfo, SearchLimits
from .tablebase import SyzygyProber

# --- `go` time/limit token parsing (architecture.md §12) --------------------
#
# Adaptive time management, isolated entirely here so it can be refined
# without touching `search.py`'s `SearchLimits` contract.
#
# The allocation formula estimates how many moves remain in the game and
# divides the remaining clock accordingly, with adjustments for:
#   - move number (fewer moves remaining -> larger share per move)
#   - increment vs. sudden death (no increment -> more conservative)
#   - safety cap (never spend more than 50% of remaining time on one move)
#   - minimum allocation (always at least 50ms, with a buffer when near-flagging)

_INT_TOKENS = {"depth", "nodes", "movetime", "wtime", "btime", "winc", "binc"}

_MIN_MOVETIME_MS = 50  # safety floor so a near-flagged clock still gets to move
_NEAR_FLAG_THRESHOLD_MS = 100  # below this we keep a buffer for the clock
_SAFETY_BUFFER_MS = 50  # buffer kept when near-flagging
_MAX_TIME_FRACTION = 0.5  # never use more than 50% of remaining time on one move


def parse_go_limits(
    args: list[str], side_to_move: int, move_number: int = 1
) -> SearchLimits:
    """Turn `go` command tokens into a `SearchLimits`.

    Recognizes `depth`, `nodes`, `movetime`, `wtime`/`btime`/`winc`/`binc`,
    and `infinite`. Unrecognized tokens (`ponder`, `mate`, `searchmoves`, ...)
    are silently skipped, matching `handle_command`'s "unknown input is
    ignored, not fatal" policy for the rest of the protocol.

    Precedence, highest first: `infinite` (no depth/nodes limit this
    function would otherwise add — an explicit `depth`/`nodes` alongside it
    still bounds the search, since both fields are set independently),
    `movetime` (an exact per-move budget straight from the GUI), then
    `wtime`/`btime` (+`winc`/`binc`) time-control math. With none of these,
    `SearchLimits`' own default (`max_depth=64`, unlimited time/nodes) means
    the search runs until an explicit `stop`.

    Time-control math (when wtime/btime are present):

    *With increment*: ``time_left / estimated_moves_remaining + increment``,
    where ``estimated_moves_remaining = max(20, 40 - move_number)``.  This
    assumes a typical game lasts ~40 moves with a floor of 20 to avoid
    over-spending in long endgames.

    *Sudden death (no increment)*: more conservative —
    ``time_left / max(30, 50 - move_number)`` — keeping a deeper reserve
    because there is no per-move replenishment.

    Both paths are then clamped by:
    - a *safety cap* of 50% of remaining time (avoid flagging on a single move),
    - a *minimum* of 50ms (or ``time_left - 50ms`` when less than 100ms
      remains, preserving a small buffer for the clock).

    Parameters
    ----------
    args : list[str]
        The tokens after ``go`` on the UCI command line.
    side_to_move : int
        ``WHITE`` or ``BLACK`` — selects wtime/winc vs. btime/binc.
    move_number : int
        The current full-move number (from ``board.fullmove_number``),
        used to estimate how many moves remain in the game.
    """
    limits = SearchLimits()
    wtime = btime = winc = binc = movetime = None
    infinite = False

    tokens = iter(args)
    for token in tokens:
        if token in _INT_TOKENS:
            try:
                value = int(next(tokens))
            except (StopIteration, ValueError):
                break
            if token == "depth":
                limits.max_depth = value
            elif token == "nodes":
                limits.nodes = value
            elif token == "movetime":
                movetime = value
            elif token == "wtime":
                wtime = value
            elif token == "btime":
                btime = value
            elif token == "winc":
                winc = value
            elif token == "binc":
                binc = value
        elif token == "infinite":
            infinite = True
        # else: unrecognized token, ignored.

    if infinite:
        return limits

    if movetime is not None:
        limits.movetime_ms = max(movetime, 0)
        return limits

    my_time = wtime if side_to_move == WHITE else btime
    my_inc = (winc if side_to_move == WHITE else binc) or 0
    if my_time is not None:
        if my_inc > 0:
            # With increment: divide remaining time by estimated moves left,
            # then add the full increment (we'll get it back next move).
            est_moves = max(20, 40 - move_number)
            allocated = my_time / est_moves + my_inc
        else:
            # Sudden death: no increment to replenish, so be more
            # conservative with a higher divisor and deeper floor.
            est_moves = max(30, 50 - move_number)
            allocated = my_time / est_moves

        # Safety cap: never spend more than 50% of remaining time.
        safety_cap = my_time * _MAX_TIME_FRACTION
        allocated = min(allocated, safety_cap)

        # Minimum allocation: at least 50ms, but if near-flagging
        # (< 100ms left), keep a small buffer for the clock.
        if my_time < _NEAR_FLAG_THRESHOLD_MS:
            min_alloc = max(my_time - _SAFETY_BUFFER_MS, 1)
        else:
            min_alloc = _MIN_MOVETIME_MS

        limits.movetime_ms = max(int(allocated), min_alloc)

    return limits


# --- The engine itself (architecture.md §12) --------------------------------


class UCIEngine:
    """Drives one UCI session: reads commands line-by-line from `inp`, writes
    responses to `out`, and dispatches each command to a `cmd_<name>` method
    (unknown commands are silently ignored, per the UCI spec)."""

    def __init__(self) -> None:
        self.board: Board = Board.starting_position()
        self.search: Search = Search(default_evaluator(), tt_size_mb=64)
        self.stop_event: threading.Event = threading.Event()
        self.search_thread: threading.Thread | None = None
        self.quit: bool = False
        self.ponder_enabled: bool = True
        self.pondering: bool = False
        self._ponder_limits: SearchLimits | None = None
        self.multi_pv: int = 1

    # --- Main loop -----------------------------------------------------------

    def run(self, inp: TextIO = sys.stdin, out: TextIO = sys.stdout) -> None:
        """Read commands from `inp` one line at a time until `quit` (or
        EOF). Kept to a plain blocking-readline loop on the main thread —
        `go` never runs here, only on the background thread it spawns
        (§9.3, §12), so this loop is always free to notice `stop`/`quit` on
        the very next line."""
        for line in inp:
            self.handle_command(line.strip(), out)
            if self.quit:
                break

    def handle_command(self, line: str, out: TextIO) -> None:
        if not line:
            return
        cmd, *args = line.split()
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            return  # unrecognized command: silently ignored, per the UCI spec
        try:
            handler(args, out)
        except Exception:
            # A malformed command (bad FEN, unresolvable move, non-numeric
            # option value, ...) must not take down the whole engine
            # process mid-game — the same "ignore what you don't understand"
            # spirit the UCI spec applies to unknown commands, extended to
            # unparsable arguments of a known one.
            pass

    # --- Handshake / readiness -------------------------------------------------

    def cmd_uci(self, args: list[str], out: TextIO) -> None:
        out.write("id name death-Token 0.1\n")
        out.write("id author Francesco Errico\n")
        out.write("option name Hash type spin default 64 min 1 max 1024\n")
        out.write("option name Ponder type check default true\n")
        out.write("option name MultiPV type spin default 1 min 1 max 500\n")
        out.write("option name SyzygyPath type string default \n")
        out.write("uciok\n")
        out.flush()

    def cmd_isready(self, args: list[str], out: TextIO) -> None:
        out.write("readyok\n")
        out.flush()

    def cmd_setoption(self, args: list[str], out: TextIO) -> None:
        """Recognized options: Hash, Ponder, MultiPV."""
        if "name" not in args or "value" not in args:
            return
        name_idx = args.index("name")
        value_idx = args.index("value")
        if value_idx + 1 >= len(args):
            return
        name = " ".join(args[name_idx + 1 : value_idx])
        value_str = args[value_idx + 1]

        if name == "Hash":
            self.search = Search(
                self.search.evaluator,
                tt_size_mb=int(value_str),
                tablebase=self.search.tablebase,
            )
        elif name == "Ponder":
            self.ponder_enabled = value_str.lower() == "true"
        elif name == "MultiPV":
            self.multi_pv = max(1, int(value_str))
        elif name == "SyzygyPath":
            path = " ".join(args[value_idx + 1:])
            tb = SyzygyProber(path) if path else None
            self.search = Search(
                self.search.evaluator,
                tt_size_mb=self.search.tt_size_mb,
                tablebase=tb,
            )

    def cmd_ucinewgame(self, args: list[str], out: TextIO) -> None:
        self._stop_and_join_search()
        self.board = Board.starting_position()
        self.search.new_game()

    # --- Position setup --------------------------------------------------------

    def cmd_position(self, args: list[str], out: TextIO) -> None:
        self._stop_and_join_search()
        if not args:
            return
        if args[0] == "startpos":
            self.board = Board.starting_position()
            rest = args[1:]
        elif args[0] == "fen":
            moves_idx = args.index("moves") if "moves" in args else len(args)
            self.board = fen.parse_fen(" ".join(args[1:moves_idx]))
            rest = args[moves_idx:]
        else:
            return
        if rest and rest[0] == "moves":
            for uci_move in rest[1:]:
                self.board.make_move(movegen.move_from_uci(self.board, uci_move))

    # --- Search control ----------------------------------------------------------

    def _stop_and_join_search(self) -> None:
        """Ensure no search is still in flight before the main thread
        mutates shared state (`self.board`, `self.search`). A no-op if no
        search is running. §12 notes that `self.board`/`self.search` are
        safe to read from the search thread without a lock only because
        "a compliant GUI always sends `stop` or waits for `bestmove`
        before a new `position`/`go`" — but that assumption doesn't hold
        for every real caller (e.g. a scripted UCI session with no
        pacing), and violating it races the main thread's board mutation
        against the still-running search thread reading that same
        `Board`, corrupting move generation and interleaving `info`/
        `bestmove` output out of order. Called defensively from
        `cmd_ucinewgame`, `cmd_position`, and `cmd_go` so those races
        can't happen regardless of caller compliance."""
        self.pondering = False
        if self.search_thread is not None and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join()

    def cmd_go(self, args: list[str], out: TextIO) -> None:
        """Book lookup, then search. Supports `go ponder` for pondering."""
        self._stop_and_join_search()

        is_ponder = "ponder" in args

        if not is_ponder:
            book_move = book.probe_book(self.board)
            if book_move is not None:
                self.search_thread = None
                out.write(f"bestmove {move_to_uci(book_move)}\n")
                out.flush()
                return

        limits = parse_go_limits(
            args, self.board.side_to_move, self.board.fullmove_number
        )
        limits.multi_pv = self.multi_pv

        if is_ponder:
            self._ponder_limits = limits
            limits = SearchLimits(max_depth=limits.max_depth, multi_pv=self.multi_pv)
            self.pondering = True

        self.stop_event = threading.Event()
        self.search_thread = threading.Thread(
            target=self._search_and_report, args=(limits, out), daemon=True
        )
        self.search_thread.start()

    def cmd_stop(self, args: list[str], out: TextIO) -> None:
        self.pondering = False
        self.stop_event.set()
        if self.search_thread is not None:
            self.search_thread.join()

    def cmd_ponderhit(self, args: list[str], out: TextIO) -> None:
        """Switch from pondering to normal timed search."""
        if not self.pondering:
            return
        self.pondering = False
        if self._ponder_limits is not None and self._ponder_limits.movetime_ms is not None:
            deadline = time.monotonic() + self._ponder_limits.movetime_ms / 1000
            self.search.set_deadline(deadline)
        self._ponder_limits = None

    def cmd_quit(self, args: list[str], out: TextIO) -> None:
        # Must join the search thread before returning, exactly like
        # cmd_stop: `run()`'s loop exits the moment `self.quit` is set, and
        # if the process (and its stdout stream) starts tearing down while
        # the daemon search thread is still mid-write, CPython can abort
        # with a fatal "could not acquire lock for <stdout> at interpreter
        # shutdown" error instead of exiting cleanly.
        self.stop_event.set()
        if self.search_thread is not None:
            self.search_thread.join()
        self.quit = True

    def _search_and_report(self, limits: SearchLimits, out: TextIO) -> None:
        """Runs on the background thread spawned by `cmd_go`."""

        def on_info(info: SearchInfo) -> None:
            if info.multi_pv_lines:
                for k, line in enumerate(info.multi_pv_lines, 1):
                    pv_str = " ".join(move_to_uci(m) for m in line.pv)
                    out.write(
                        f"info depth {info.depth} score cp {line.score_cp} "
                        f"nodes {info.nodes} multipv {k} pv {pv_str}\n"
                    )
            else:
                pv_str = " ".join(move_to_uci(m) for m in info.pv)
                out.write(
                    f"info depth {info.depth} score cp {info.score_cp} "
                    f"nodes {info.nodes} pv {pv_str}\n"
                )
            out.flush()

        result = self.search.search(
            self.board, limits, stop_event=self.stop_event, on_info=on_info
        )
        bm = move_to_uci(result.best_move)
        if self.ponder_enabled and len(result.pv) >= 2:
            pm = move_to_uci(result.pv[1])
            out.write(f"bestmove {bm} ponder {pm}\n")
        else:
            out.write(f"bestmove {bm}\n")
        out.flush()
