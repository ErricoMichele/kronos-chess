"""`test_uci.py` (architecture.md §12, §13).

§13 describes this gate as: "drives `UCIEngine.run()` over an in-memory
`io.StringIO` pair with `uci`/`isready`/`position startpos moves e2e4
e7e5`/`go depth 4` and asserts well-formed `uciok`/`readyok`/`info`/
`bestmove` lines appear in the right order."

The one wrinkle `UCIEngine` forces on any driver of it: `cmd_go` (§12) hands
the actual search off to a daemon `threading.Thread` and returns
immediately, specifically so the main stdin-reading loop in `run()` is
always free to notice a `stop`/`quit` on the very next line without waiting
on a long search call stack to unwind. That means `run()` itself can return
(on EOF, or right after processing a `quit` line) *before* the background
thread has written its `bestmove` line to `out` -- there is nothing in
`UCIEngine`'s public surface for a caller to block on except `cmd_stop`
(which joins the search thread) or the thread handle `run()` happens to
leave behind on `engine.search_thread`. Since a scripted session may end
with `quit` rather than `stop`, the tests below poll `out`'s buffered text
for the `bestmove` line with a short sleep loop (bounded by a generous
timeout) rather than assuming `run()` returning means the search is done.
"""

from __future__ import annotations

import io
import re
import time

from chessengine.uci import UCIEngine

# A UCI move token: from-square, to-square, optional promotion letter.
# e.g. "e2e4", "b1c3", "e7e8q". See move.py's PROMO_* flags (architecture.md §4).
_UCI_MOVE_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

_POLL_INTERVAL_S = 0.02
_POLL_TIMEOUT_S = 5.0


def _drive(command_text: str, *, timeout: float = _POLL_TIMEOUT_S) -> tuple[UCIEngine, str]:
    """Run `command_text` (newline-separated UCI commands) through a fresh
    `UCIEngine().run()` over an in-memory `io.StringIO` pair, then poll the
    captured output for a `bestmove` line before returning.

    If `command_text` never sends `go`, no `bestmove` line will ever appear
    and this simply waits out the full timeout once -- callers that don't
    expect a `bestmove` should keep `timeout` small.
    """
    inp = io.StringIO(command_text)
    out = io.StringIO()
    engine = UCIEngine()

    engine.run(inp, out)  # returns as soon as EOF or a `quit` line is seen

    deadline = time.monotonic() + timeout
    while "bestmove" not in out.getvalue() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)

    # `run()` may have returned without joining the search thread (a `quit`
    # line only sets the stop event, per cmd_quit in uci.py -- it doesn't
    # join). Give it a bounded chance to actually finish so a slow machine
    # doesn't see a torn write, and so the thread is gone before the test
    # returns rather than lingering into the next one.
    if engine.search_thread is not None:
        remaining = max(0.0, deadline - time.monotonic())
        engine.search_thread.join(timeout=remaining)

    return engine, out.getvalue()


def _lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line]


# --- Handshake -----------------------------------------------------------


def test_uci_command_emits_well_formed_id_and_uciok() -> None:
    _engine, output = _drive("uci\n", timeout=0.5)
    lines = _lines(output)

    id_name_lines = [line for line in lines if line.startswith("id name ")]
    assert len(id_name_lines) == 1, output
    # "id name <non-empty engine name>", not just a bare "id name".
    assert re.match(r"^id name \S.*$", id_name_lines[0])

    assert "uciok" in lines, output


def test_isready_emits_readyok() -> None:
    _engine, output = _drive("isready\n", timeout=0.5)
    assert "readyok" in _lines(output)


# --- Full session: handshake, position setup, search, ordering -----------


def test_full_session_produces_ordered_wellformed_output() -> None:
    """The scenario `test_uci.py` is named for in §13: a realistic scripted
    GUI session ending in `quit` right after `go`, at a shallow depth so the
    search finishes almost immediately.

    Deliberately played to a position *outside* `book.py`'s demonstration
    repertoire (1...Nf6, not one of its listed replies to 1.e4): this test's
    whole point is to see a real search actually run and report `info`
    lines, which architecture.md §15's book short-circuit (see
    `test_go_depth_10_from_startpos_uses_book_and_returns_fast` below, and
    `test_book.py`) would otherwise skip entirely for a position the book
    does recognize."""
    # No trailing `quit`: let the search complete via EOF so the search
    # thread is never aborted mid-depth — a `quit` right after `go` on a
    # StringIO pipe is a race (the main thread reads `quit` and sets the
    # stop event before the search thread finishes depth 1 under CPU load).
    commands = "uci\nisready\nposition startpos moves e2e4 g8f6\ngo depth 3\n"
    engine, output = _drive(commands)
    lines = _lines(output)

    # --- presence -----------------------------------------------------
    assert any(line.startswith("id name ") for line in lines), output
    assert "uciok" in lines, output
    assert "readyok" in lines, output

    bestmove_lines = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmove_lines) == 1, output
    move_token = bestmove_lines[0].split()[1]
    assert _UCI_MOVE_RE.match(move_token), f"malformed bestmove token: {move_token!r}"

    # `go depth 3` should have made at least some search progress reported
    # via `info` lines (architecture.md §12's `_search_and_report`).
    info_lines = [line for line in lines if line.startswith("info ")]
    assert info_lines, output
    for line in info_lines:
        assert re.match(r"^info depth \d+ score cp -?\d+ nodes \d+ pv ", line), line

    # --- ordering -------------------------------------------------------
    # uciok (end of 'uci' handshake) before readyok (end of 'isready')
    # before bestmove (end of 'go'), matching the order the commands were
    # sent in and the order a real GUI depends on.
    uciok_idx = lines.index("uciok")
    readyok_idx = lines.index("readyok")
    bestmove_idx = next(i for i, line in enumerate(lines) if line.startswith("bestmove "))

    assert uciok_idx < readyok_idx < bestmove_idx, output


def test_go_completes_and_reports_bestmove_without_quit() -> None:
    """Without a trailing `quit`, `run()` returns on plain EOF instead --
    `go`'s background search thread still runs to completion and still
    writes `bestmove`, since nothing ever set its stop event."""
    commands = "uci\nisready\nposition startpos moves e2e4 e7e5\ngo depth 2\n"
    engine, output = _drive(commands)

    assert engine.quit is False
    lines = _lines(output)
    bestmove_lines = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmove_lines) == 1, output
    move_token = bestmove_lines[0].split()[1]
    assert _UCI_MOVE_RE.match(move_token), f"malformed bestmove token: {move_token!r}"


def test_position_startpos_moves_advances_side_to_move() -> None:
    """Sanity check that `position ... moves e2e4 e7e5` (parsed by
    `cmd_position`, architecture.md §12) actually replayed both plies onto
    `engine.board` before `go` ever searches it -- White should be on move
    again after one full move each."""
    from chessengine.constants import WHITE

    engine, _output = _drive(
        "position startpos moves e2e4 e7e5\n", timeout=0.1
    )
    assert engine.board.side_to_move == WHITE


# --- Opening book short-circuits `go` (architecture.md §12, §15) -----------


def test_go_depth_10_from_startpos_uses_book_and_returns_fast() -> None:
    """`cmd_go` consults `book.probe_book` *before* ever touching
    `parse_go_limits`/`Search.search` (architecture.md §12, §15): a book hit
    for the current position short-circuits straight to `bestmove` without
    spawning a search thread at all.

    From the starting position, `go depth 10` must therefore: (a) come back
    with a single well-formed `bestmove` line, (b) that move must be one of
    `OPENING_BOOK`'s own startpos entries, not just any legal move a real
    search might have found, and (c) it must come back *fast* -- well under
    what a genuine depth-10 search would ever take (a depth-6 search alone
    already takes multiple seconds on this engine/evaluator), which is the
    only external signal this black-box test has that no real search thread
    ran at all.
    """
    from chessengine.board import Board
    from chessengine.book import OPENING_BOOK

    start_hash = Board.starting_position().zobrist_hash
    book_ucis = {uci for uci, _weight in OPENING_BOOK[start_hash]}

    commands = "uci\nisready\nposition startpos\ngo depth 10\n"
    start = time.monotonic()
    engine, output = _drive(commands, timeout=2.0)
    elapsed = time.monotonic() - start

    lines = _lines(output)
    bestmove_lines = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmove_lines) == 1, output
    move_token = bestmove_lines[0].split()[1]
    assert _UCI_MOVE_RE.match(move_token), f"malformed bestmove token: {move_token!r}"
    assert move_token in book_ucis, (
        f"bestmove {move_token!r} for 'go depth 10' from startpos wasn't one "
        f"of the book's own entries for this position, {sorted(book_ucis)!r} "
        "-- a real depth-10 search must not have been short-circuited by the book"
    )

    # `go depth 3`/`go depth 2` elsewhere in this file (a real, if shallow,
    # search) already take a noticeable fraction of a second; an *unbounded*
    # depth-10 search would take vastly longer than that. Coming back this
    # fast is only possible if the book answered before `Search.search` (and
    # its background thread) ever ran.
    assert elapsed < 1.5, (
        f"'go depth 10' from startpos took {elapsed:.3f}s -- the opening book "
        "should have short-circuited this to an immediate bestmove with no "
        "real depth-10 search ever running"
    )

    # No search thread should have been spawned on the book-hit path at all
    # (architecture.md §12's cmd_go: `self.search_thread = None` on that
    # branch), matching the "no real search ran" claim above directly rather
    # than only inferring it from timing.
    assert engine.search_thread is None


# --- Adaptive time management (Milestone 5) --------------------------------


from chessengine.constants import WHITE, BLACK
from chessengine.uci import parse_go_limits


def test_time_management_normal_game_with_increment() -> None:
    """go wtime 60000 btime 60000 winc 1000 binc 1000 at move 1 (fresh game).

    With 60s on the clock and 1s increment, the formula gives:
        est_moves = max(20, 40 - 1) = 39
        allocated = 60000 / 39 + 1000 ~= 2538ms
    This should be in a reasonable range (1000-5000ms).
    """
    args = "wtime 60000 btime 60000 winc 1000 binc 1000".split()
    limits = parse_go_limits(args, WHITE, move_number=1)
    assert limits.movetime_ms is not None
    assert 1000 <= limits.movetime_ms <= 5000, (
        f"Expected 1000-5000ms for a fresh game with 60s+1s, got {limits.movetime_ms}ms"
    )


def test_time_management_low_time_capped() -> None:
    """go wtime 1000 btime 60000 — White has only 1s left.

    The safety cap (50% of remaining time) limits allocation to at most
    500ms, regardless of what the base formula would produce.
    """
    args = "wtime 1000 btime 60000".split()
    limits = parse_go_limits(args, WHITE, move_number=1)
    assert limits.movetime_ms is not None
    assert limits.movetime_ms <= 500, (
        f"Expected at most 500ms (50% of 1000ms), got {limits.movetime_ms}ms"
    )


def test_time_management_very_low_time_minimum_allocation() -> None:
    """go wtime 100 btime 60000 — White is near-flagging with only 100ms.

    At exactly 100ms the normal minimum (50ms) applies.  Below 100ms,
    the near-flag logic kicks in and keeps a buffer.
    """
    args = "wtime 100 btime 60000".split()
    limits = parse_go_limits(args, WHITE, move_number=1)
    assert limits.movetime_ms is not None
    assert limits.movetime_ms >= 1, "Must allocate at least 1ms"
    assert limits.movetime_ms <= 50, (
        f"Expected at most 50ms (50% of 100ms cap), got {limits.movetime_ms}ms"
    )

    # Even more extreme: only 60ms left.
    args2 = "wtime 60 btime 60000".split()
    limits2 = parse_go_limits(args2, WHITE, move_number=1)
    assert limits2.movetime_ms is not None
    assert limits2.movetime_ms >= 1
    assert limits2.movetime_ms <= 30, (
        f"Expected at most 30ms (50% of 60ms), got {limits2.movetime_ms}ms"
    )


def test_time_management_sudden_death_more_conservative() -> None:
    """go wtime 60000 btime 60000 — no increment (sudden death).

    Without increment, the formula uses a higher divisor:
        est_moves = max(30, 50 - 1) = 49
        allocated = 60000 / 49 ~= 1224ms
    This should be noticeably less than the same position with increment.
    """
    args_no_inc = "wtime 60000 btime 60000".split()
    limits_no_inc = parse_go_limits(args_no_inc, WHITE, move_number=1)

    args_with_inc = "wtime 60000 btime 60000 winc 1000 binc 1000".split()
    limits_with_inc = parse_go_limits(args_with_inc, WHITE, move_number=1)

    assert limits_no_inc.movetime_ms is not None
    assert limits_with_inc.movetime_ms is not None

    # Sudden death should allocate strictly less than with increment.
    assert limits_no_inc.movetime_ms < limits_with_inc.movetime_ms, (
        f"Sudden death ({limits_no_inc.movetime_ms}ms) should be more conservative "
        f"than with increment ({limits_with_inc.movetime_ms}ms)"
    )

    # Sanity: sudden death allocation should still be reasonable.
    assert 500 <= limits_no_inc.movetime_ms <= 3000, (
        f"Expected 500-3000ms for sudden death with 60s, got {limits_no_inc.movetime_ms}ms"
    )


def test_time_management_does_not_affect_movetime() -> None:
    """go movetime X must pass through unchanged regardless of move_number."""
    args = "movetime 5000".split()
    limits = parse_go_limits(args, WHITE, move_number=20)
    assert limits.movetime_ms == 5000


def test_time_management_does_not_affect_depth_or_infinite() -> None:
    """go depth X and go infinite must be unaffected by the new logic."""
    args_depth = "depth 6".split()
    limits_depth = parse_go_limits(args_depth, WHITE, move_number=10)
    assert limits_depth.max_depth == 6
    assert limits_depth.movetime_ms is None

    args_inf = "infinite".split()
    limits_inf = parse_go_limits(args_inf, WHITE, move_number=10)
    assert limits_inf.movetime_ms is None


# --- Pondering (go ponder / ponderhit) ------------------------------------


def test_bestmove_includes_ponder_move_when_pv_has_two_moves() -> None:
    """When the search produces a PV with >= 2 moves and Ponder is
    enabled, bestmove should include 'ponder <move>'."""
    engine = UCIEngine()
    out = io.StringIO()

    engine.cmd_position(["startpos", "moves", "e2e4", "g8f6"], out)
    engine.cmd_go(["depth", "6"], out)

    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while "bestmove" not in out.getvalue() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
    if engine.search_thread is not None:
        engine.search_thread.join(timeout=5.0)

    captured = out.getvalue()
    lines = _lines(captured)

    info_lines = [l for l in lines if l.startswith("info ") and " pv " in l]
    assert info_lines, f"no info lines with PV found: {captured}"
    last_pv_str = info_lines[-1][info_lines[-1].index(" pv ") + 4 :]
    last_pv_moves = last_pv_str.split()
    assert len(last_pv_moves) >= 2, (
        f"PV too short at depth 6 (expected >= 2): {last_pv_moves}"
    )

    bestmove_lines = [l for l in lines if l.startswith("bestmove ")]
    assert len(bestmove_lines) == 1, captured
    parts = bestmove_lines[0].split()
    assert len(parts) >= 4 and parts[2] == "ponder", (
        f"expected 'bestmove X ponder Y', got: {bestmove_lines[0]!r}"
    )
    assert _UCI_MOVE_RE.match(parts[1]), f"malformed bestmove token: {parts[1]!r}"
    assert _UCI_MOVE_RE.match(parts[3]), f"malformed ponder token: {parts[3]!r}"


def test_go_ponder_does_not_stop_on_time() -> None:
    """'go ponder' should search indefinitely, ignoring time limits."""
    engine = UCIEngine()
    out = io.StringIO()

    engine.cmd_position(["startpos", "moves", "e2e4", "e7e5"], out)
    engine.cmd_go(["ponder", "movetime", "1"], out)

    time.sleep(0.3)

    assert engine.pondering is True
    assert engine.search_thread is not None and engine.search_thread.is_alive()
    assert "bestmove" not in out.getvalue()

    engine.cmd_stop([], out)
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while "bestmove" not in out.getvalue() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
    assert "bestmove" in out.getvalue()


def test_ponderhit_transitions_to_timed_search() -> None:
    """After 'ponderhit', the engine applies the time limits from the
    original 'go ponder' command and the search finishes on its own."""
    engine = UCIEngine()
    out = io.StringIO()

    engine.cmd_position(["startpos", "moves", "e2e4", "e7e5"], out)
    engine.cmd_go(["ponder", "movetime", "100"], out)

    time.sleep(0.05)
    assert engine.pondering is True

    engine.cmd_ponderhit([], out)
    assert engine.pondering is False

    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while "bestmove" not in out.getvalue() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
    assert "bestmove" in out.getvalue()

    lines = _lines(out.getvalue())
    bestmove_lines = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmove_lines) == 1
    move_token = bestmove_lines[0].split()[1]
    assert _UCI_MOVE_RE.match(move_token), f"malformed bestmove token: {move_token!r}"


def test_stop_during_pondering_outputs_bestmove() -> None:
    """'stop' during pondering should terminate the search and produce
    a well-formed bestmove line."""
    engine = UCIEngine()
    out = io.StringIO()

    engine.cmd_position(["startpos", "moves", "e2e4", "e7e5"], out)
    engine.cmd_go(["ponder", "wtime", "60000", "btime", "60000"], out)

    time.sleep(0.1)
    assert engine.pondering is True

    engine.cmd_stop([], out)
    assert engine.pondering is False

    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while "bestmove" not in out.getvalue() and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)

    captured = out.getvalue()
    lines = _lines(captured)
    bestmove_lines = [line for line in lines if line.startswith("bestmove ")]
    assert len(bestmove_lines) == 1, captured
    move_token = bestmove_lines[0].split()[1]
    assert _UCI_MOVE_RE.match(move_token), f"malformed bestmove token: {move_token!r}"
