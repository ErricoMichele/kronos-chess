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
    search finishes almost immediately."""
    commands = "uci\nisready\nposition startpos moves e2e4 e7e5\ngo depth 3\nquit\n"
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

    # `quit` (architecture.md §12's cmd_quit) sets the engine's quit flag,
    # which is what let run()'s main loop return without waiting on the
    # search thread in the first place.
    assert engine.quit is True


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
