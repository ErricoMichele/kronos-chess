"""UCI (Universal Chess Interface) protocol handler (architecture.md §12).

Per the module-boundary DAG (architecture.md §11), `uci.py` depends on
`board`, `movegen`, `move`, `search`, `fen`, and `constants` — it is the
outermost layer besides `cli.py`, and nothing below it in the DAG ever
imports it. It is also the only module that touches `sys.stdin`/`stdout`
and `threading`.

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
from typing import TextIO

from . import fen, movegen
from .board import Board
from .constants import WHITE
from .evaluate import default_evaluator
from .move import move_to_uci
from .search import Search, SearchInfo, SearchLimits

# --- `go` time/limit token parsing (architecture.md §12) --------------------
#
# Time-control math (a fixed fraction of remaining time plus increment,
# clamped to a safety minimum) is intentionally simple through Milestone 4
# and isolated entirely here, so it can be refined later without touching
# `search.py`'s `SearchLimits` contract.

_INT_TOKENS = {"depth", "nodes", "movetime", "wtime", "btime", "winc", "binc"}

_TIME_FRACTION_DIVISOR = 20  # allocate ~1/20th of the remaining clock per move
_INC_FRACTION_DIVISOR = 2  # plus about half of the increment
_MOVE_OVERHEAD_MS = 50  # never allocate the *entire* remaining clock
_MIN_MOVETIME_MS = 50  # safety floor so a near-flagged clock still gets to move


def parse_go_limits(args: list[str], side_to_move: int) -> SearchLimits:
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
        allocated = my_time // _TIME_FRACTION_DIVISOR + my_inc // _INC_FRACTION_DIVISOR
        safety_cap = max(my_time - _MOVE_OVERHEAD_MS, _MIN_MOVETIME_MS)
        limits.movetime_ms = max(min(allocated, safety_cap), _MIN_MOVETIME_MS)

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
        out.write("uciok\n")
        out.flush()

    def cmd_isready(self, args: list[str], out: TextIO) -> None:
        out.write("readyok\n")
        out.flush()

    def cmd_setoption(self, args: list[str], out: TextIO) -> None:
        """Only `Hash` (transposition table size, in MB) is a recognized
        option; everything else is silently ignored. Changing `Hash`
        rebuilds `Search` (a fresh, empty TT at the new size) but keeps the
        same evaluator — this drops killers/history too, which is fine
        since `setoption` is a rare, out-of-game-flow event, not something
        a GUI does mid-search."""
        if "Hash" not in args or "value" not in args:
            return
        size_mb = int(args[args.index("value") + 1])
        self.search = Search(self.search.evaluator, tt_size_mb=size_mb)

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
        if self.search_thread is not None and self.search_thread.is_alive():
            self.stop_event.set()
            self.search_thread.join()

    def cmd_go(self, args: list[str], out: TextIO) -> None:
        """Parse `go`'s limits and hand the search off to a background
        thread so this method (and thus the main stdin-reading loop) returns
        immediately — `stop`/`quit` must never wait on a long search call
        stack to unwind on its own (§9.3, §12)."""
        self._stop_and_join_search()
        limits = parse_go_limits(args, self.board.side_to_move)
        self.stop_event = threading.Event()
        self.search_thread = threading.Thread(
            target=self._search_and_report, args=(limits, out), daemon=True
        )
        self.search_thread.start()

    def cmd_stop(self, args: list[str], out: TextIO) -> None:
        self.stop_event.set()
        if self.search_thread is not None:
            self.search_thread.join()

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
        """Runs on the background thread spawned by `cmd_go`. The only
        cross-thread state touched is `self.stop_event` (safe to `.set()`
        from another thread) and `self.search`/`self.board` (read-only here;
        a compliant GUI always sends `stop` or waits for `bestmove` before
        the next `position`/`go`, which is what makes this safe without an
        explicit lock, §12)."""

        def on_info(info: SearchInfo) -> None:
            pv_str = " ".join(move_to_uci(m) for m in info.pv)
            out.write(
                f"info depth {info.depth} score cp {info.score_cp} "
                f"nodes {info.nodes} pv {pv_str}\n"
            )
            out.flush()

        result = self.search.search(
            self.board, limits, stop_event=self.stop_event, on_info=on_info
        )
        out.write(f"bestmove {move_to_uci(result.best_move)}\n")
        out.flush()
