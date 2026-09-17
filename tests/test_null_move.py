"""Null-move machinery tests (architecture.md §6, §9, §13, §15): the board-level
`make_null_move`/`unmake_null_move` primitive and search's null-move pruning
that's built on top of it.

Four things, matching the Milestone 5 null-move extension's own risk profile:

1. **Board-level round-trip** (mirrors `test_make_unmake_roundtrip.py`'s
   exactness pattern): for a battery of §14-style positions,
   `make_null_move` touches *only* `side_to_move`, `ep_square`, and the
   corresponding component of `zobrist_hash` -- `pieces`/`occupied_co`/
   `occupied`/`mailbox`/`castling_rights`/`halfmove_clock`/
   `fullmove_number`/`material_pst_score` must be bit-for-bit identical
   immediately after `make_null_move`, not just after the round trip -- and
   `unmake_null_move` restores every field exactly.

2. **The zugzwang guard.** `Search._negamax` must never even attempt a null
   move when the side to move has no non-pawn material (`_has_non_pawn_material`,
   architecture.md §9's "standard zugzwang-avoidance guard"). This is checked
   two ways: (a) a call-count spy on `Board.make_null_move` proves the guard
   actually suppresses every attempt across a whole king-and-pawn subtree
   (and, as a sanity check on the spy itself, that the same spy *does* see
   calls in an ordinary position with material to spare); (b) a genuine
   scoring/best-move regression, built by actually disabling the guard
   (monkeypatching `_has_non_pawn_material` to always report "yes, there's
   non-pawn material") on a real opposition-based king-and-pawn zugzwang
   position and showing the search's answer changes -- and, critically,
   changes *away* from what the same search produces with null-move pruning
   turned off entirely (a valid reference: alpha-beta plus quiescence,
   without the null-move heuristic layered on top, is exactly what
   `test_search.py`'s own differential oracle already established agrees
   with full-width minimax at these depths). The guarded, as-shipped search
   matches that reference exactly; the guard-disabled search does not.

3. **The existing mate-in-N suite.** `test_search_tactics.py` already
   exercises `Search._negamax` (null-move pruning included, since it's on
   by default) end-to-end and must keep passing -- that's the Verify
   phase's job, not duplicated here. As a cheap extra safety net, one of
   its mate-in-1 fixtures is re-solved directly in this file too (see
   `test_null_move_pruning_does_not_break_a_known_mate_in_1` below).

4. **Basic sanity**: a normal, non-endgame search at depth 5-6 on a busy
   middlegame position still returns a legal move and completes cleanly
   with null-move pruning active (the default, unmodified `Search`).
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.constants import file_of
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits
from chessengine import search as search_mod
from chessengine.zobrist import ZOBRIST_EP_FILE, ZOBRIST_SIDE

# --- §14-style FEN battery (reused from test_make_unmake_roundtrip.py) -------

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
PERFT3_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
PERFT4_FEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
PERFT5_FEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
PERFT6_FEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
# En passant set (make_null_move's other documented branch: clearing ep_square
# and un-hashing ZOBRIST_EP_FILE for it).
EP_WHITE_TO_MOVE_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 4"
EP_BLACK_TO_MOVE_FEN = "rnbqkbnr/ppp1pppp/8/8/3pP3/5N2/PPPP1PPP/RNBQKB1R b KQkq e3 0 3"

NULL_MOVE_BATTERY_FENS = [
    STARTPOS_FEN,
    KIWIPETE_FEN,
    PERFT3_FEN,
    PERFT4_FEN,
    PERFT5_FEN,
    PERFT6_FEN,
    EP_WHITE_TO_MOVE_FEN,
    EP_BLACK_TO_MOVE_FEN,
]


# --- 1. Board-level make_null_move / unmake_null_move round trip -------------


def _snapshot(board: Board) -> dict:
    """Every field `make_null_move`/`unmake_null_move` could conceivably
    touch, copied deeply enough that later mutation of `board` cannot
    retroactively change the snapshot (mirrors
    `test_make_unmake_roundtrip.py`'s `_snapshot`)."""
    return {
        "pieces": [row[:] for row in board.pieces],
        "occupied_co": board.occupied_co[:],
        "occupied": board.occupied,
        "mailbox": board.mailbox[:],
        "side_to_move": board.side_to_move,
        "castling_rights": board.castling_rights,
        "ep_square": board.ep_square,
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
        "zobrist_hash": board.zobrist_hash,
        "material_pst_score": board.material_pst_score,
        "history_len": len(board.history),
        "position_history": board.position_history[:],
        "null_move_history_len": len(board.null_move_history),
    }


# Fields make_null_move's own docstring promises are untouched -- checked
# immediately after make_null_move, before any unmake, so an accidental
# mutation can't hide behind unmake_null_move happening to restore it.
_UNTOUCHED_BY_NULL_MOVE = (
    "pieces",
    "occupied_co",
    "occupied",
    "mailbox",
    "castling_rights",
    "halfmove_clock",
    "fullmove_number",
    "material_pst_score",
    "history_len",
    "position_history",
)


@pytest.mark.parametrize("fen", NULL_MOVE_BATTERY_FENS)
def test_make_null_move_touches_only_side_ep_and_hash(fen: str) -> None:
    board = parse_fen(fen)
    before = _snapshot(board)

    board.make_null_move()
    after_make = _snapshot(board)

    for field in _UNTOUCHED_BY_NULL_MOVE:
        assert after_make[field] == before[field], (
            f"make_null_move touched {field!r}, which its own contract says it "
            f"must leave alone (fen={fen!r}): before={before[field]!r} "
            f"after={after_make[field]!r}"
        )

    # The three fields it *does* touch, checked exactly, not just "changed".
    assert after_make["side_to_move"] == 1 - before["side_to_move"], (
        f"side_to_move did not flip (fen={fen!r})"
    )
    assert after_make["ep_square"] is None, (
        f"make_null_move must always clear ep_square, even if it was already "
        f"None (fen={fen!r}), got {after_make['ep_square']!r}"
    )
    assert after_make["null_move_history_len"] == before["null_move_history_len"] + 1, (
        f"null_move_history must grow by exactly one entry per make_null_move (fen={fen!r})"
    )
    expected_hash = before["zobrist_hash"]
    if before["ep_square"] is not None:
        expected_hash ^= ZOBRIST_EP_FILE[file_of(before["ep_square"])]
    expected_hash ^= ZOBRIST_SIDE
    assert after_make["zobrist_hash"] == expected_hash, (
        f"zobrist_hash after make_null_move does not match the exact expected "
        f"EP-file-unhash (if any) + side-to-move-hash formula (fen={fen!r}): "
        f"expected={expected_hash:#018x} got={after_make['zobrist_hash']:#018x}"
    )

    board.unmake_null_move()
    after_unmake = _snapshot(board)
    assert after_unmake == before, (
        f"unmake_null_move did not restore every field exactly (fen={fen!r}): "
        f"before={before!r} after={after_unmake!r}"
    )
    assert board.to_fen() == fen, "unmake_null_move must restore the exact starting FEN"


@pytest.mark.parametrize("fen", [STARTPOS_FEN, EP_WHITE_TO_MOVE_FEN, EP_BLACK_TO_MOVE_FEN])
def test_make_null_move_round_trips_when_chained_two_deep(fen: str) -> None:
    """Two `make_null_move`s in a row (legal at the board level -- only
    `Search`'s `null_ok` guard forbids back-to-back null moves, and that's a
    search-level policy, not something `Board` itself enforces or needs to)
    followed by two `unmake_null_move`s must restore the exact starting
    position, exercising `null_move_history` as a genuine LIFO stack rather
    than a single-slot cache."""
    board = parse_fen(fen)
    before = _snapshot(board)

    board.make_null_move()
    mid = _snapshot(board)
    assert mid["side_to_move"] == 1 - before["side_to_move"]

    board.make_null_move()
    after_two = _snapshot(board)
    assert after_two["side_to_move"] == before["side_to_move"], (
        "side_to_move must flip back after a second null move"
    )
    assert len(board.null_move_history) == 2

    board.unmake_null_move()
    assert _snapshot(board) == mid, "first unmake must restore the one-null-move state exactly"

    board.unmake_null_move()
    after_both_unmade = _snapshot(board)
    assert after_both_unmade == before
    assert board.to_fen() == fen
    assert len(board.null_move_history) == 0


# --- 2. Zugzwang: the non-pawn-material guard --------------------------------
#
# A pure king-and-pawn endgame exhibiting the textbook "opposition" zugzwang:
# whichever side is *forced to move* is worse off than if it could pass --
# exactly the situation null-move pruning's zugzwang-avoidance guard
# (`_has_non_pawn_material`, search.py) exists to protect against. Neither
# side has a knight/bishop/rook/queen anywhere in this FEN, so the guard is
# "live" at literally every node reachable from it -- there is no possible
# node in this subtree where null-move pruning is legitimately allowed to
# fire, which is what makes it such a clean regression fixture.
_ZUGZWANG_FEN = "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1"  # Black king e8, White king e6 + pawn e5, Black to move

# Deep enough to clear NULL_MOVE_MIN_DEPTH (3) with room to spare for the
# reduced (depth - 1 - NULL_MOVE_REDUCTION) recursive call to also do real
# work; both verified (see this file's history / the PR that added it) to
# reproduce an actual best-move regression when the guard is disabled, not
# just a score wobble.
_ZUGZWANG_DEPTHS = [6, 7]


def _search_zugzwang(fen: str, depth: int):
    board = parse_fen(fen)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=depth))
    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    return result


@pytest.mark.parametrize("depth", _ZUGZWANG_DEPTHS)
def test_zugzwang_guard_prevents_a_real_misjudgment(depth: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling the zugzwang guard (forcing `_has_non_pawn_material` to
    always claim "yes") on a pure king-and-pawn zugzwang position changes
    the search's answer -- and changes it *away* from what the identical
    search produces with null-move pruning switched off altogether, which is
    the correct reference here: alpha-beta + quiescence without the
    null-move heuristic layered on top is exactly the machinery
    `test_search.py`'s differential oracle already validates against
    full-width minimax, so "guarded == null-move-disabled" is "guarded ==
    correct", and "guard-disabled != null-move-disabled" is a genuine,
    demonstrated misjudgment, not merely 'a different number'.
    """
    guarded = _search_zugzwang(_ZUGZWANG_FEN, depth)

    # Reference: null-move pruning switched off entirely (NULL_MOVE_MIN_DEPTH
    # raised out of reach of any depth used in this file), so the search
    # falls back to plain alpha-beta + quiescence for every node.
    monkeypatch.setattr(search_mod, "NULL_MOVE_MIN_DEPTH", 10_000)
    reference = _search_zugzwang(_ZUGZWANG_FEN, depth)
    monkeypatch.undo()

    assert guarded.score_cp == reference.score_cp, (
        f"the as-shipped, guarded search should never attempt a null move anywhere "
        f"in this all-king-and-pawn subtree, so it must score identically to "
        f"null-move pruning being switched off entirely, at depth {depth}: "
        f"guarded={guarded.score_cp} reference={reference.score_cp}"
    )
    assert guarded.best_move == reference.best_move, (
        f"guarded={move_to_uci(guarded.best_move)!r} vs "
        f"reference={move_to_uci(reference.best_move)!r} at depth {depth}"
    )

    # Now actually disable the guard: `_has_non_pawn_material` always reports
    # "yes", so null-move pruning is attempted even though neither side ever
    # has anything but a king (and, for White, one pawn).
    monkeypatch.setattr(search_mod, "_has_non_pawn_material", lambda board, color: True)
    unguarded = _search_zugzwang(_ZUGZWANG_FEN, depth)

    assert (unguarded.score_cp, unguarded.best_move) != (reference.score_cp, reference.best_move), (
        f"disabling the zugzwang guard was expected to actually corrupt the result "
        f"at depth {depth} (that's the whole point of this fixture) but "
        f"unguarded={unguarded.score_cp}/{move_to_uci(unguarded.best_move)!r} matched "
        f"the null-move-disabled reference {reference.score_cp}/"
        f"{move_to_uci(reference.best_move)!r} anyway -- this position may no longer "
        f"demonstrate the regression and needs a replacement fixture."
    )


def test_zugzwang_guard_suppresses_every_null_move_attempt_via_spy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct spy on `Board.make_null_move`: across the *entire* search tree
    reachable from a pure king-and-pawn position, the guarded, as-shipped
    `Search` must never call it even once, since neither side ever has
    non-pawn material at any node.
    """
    original_make_null_move = Board.make_null_move
    calls = {"count": 0}

    def spy(self: Board) -> None:
        calls["count"] += 1
        return original_make_null_move(self)

    monkeypatch.setattr(Board, "make_null_move", spy)

    board = parse_fen(_ZUGZWANG_FEN)
    search = Search(default_evaluator())
    search.search(board, SearchLimits(max_depth=7))

    assert calls["count"] == 0, (
        f"Board.make_null_move was called {calls['count']} time(s) while searching a "
        f"position where the side to move never has more than a king and pawns at any "
        f"reachable node -- the zugzwang guard should have suppressed every attempt"
    )
    assert board.to_fen() == _ZUGZWANG_FEN


def test_null_move_spy_actually_detects_calls_in_an_ordinary_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check on the spy mechanism itself: in an ordinary position with
    plenty of non-pawn material for the side to move, at a depth clearing
    `NULL_MOVE_MIN_DEPTH`, null-move pruning *should* fire at least once --
    proving the previous test's zero count reflects the guard actually doing
    something, not merely the spy failing to observe calls at all.
    """
    original_make_null_move = Board.make_null_move
    calls = {"count": 0}

    def spy(self: Board) -> None:
        calls["count"] += 1
        return original_make_null_move(self)

    monkeypatch.setattr(Board, "make_null_move", spy)

    board = parse_fen(KIWIPETE_FEN)
    search = Search(default_evaluator())
    search.search(board, SearchLimits(max_depth=4))

    assert calls["count"] > 0, (
        "expected at least one null-move attempt while searching a busy middlegame "
        "position with non-pawn material to spare -- if this is zero, the spy itself "
        "isn't observing calls and the zugzwang test above would be vacuous"
    )
    assert board.to_fen() == KIWIPETE_FEN


# --- 3. The existing mate-in-N suite must keep passing -----------------------
#
# `test_search_tactics.py` is the authoritative mate-in-N regression suite
# (architecture.md §13, §15) and already exercises `Search._negamax` with
# null-move pruning active by default -- that file passing is the real gate
# here and is not duplicated in this one. As a cheap, self-contained extra
# check (not a replacement), one of its mate-in-1 fixtures is re-solved
# directly below: a back-rank mate is a good one to keep here specifically
# because the checkmate is delivered right at the root, where §9's null-move
# code explicitly never fires (`ply > 0` guard) -- so this also incidentally
# double-checks that guard.


def test_null_move_pruning_does_not_break_a_known_mate_in_1() -> None:
    # Back-rank mate: Black's own f7/g7/h7 pawns block every escape square,
    # and nothing can interpose on or capture the rook's 8th-rank check.
    # 1.Ra8# (same fixture as test_search_tactics.py's "mate-in-1-back-rank").
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=6))

    assert board.to_fen() == fen
    assert result.best_move != NULL_MOVE
    legal_moves = generate_legal_moves(board)
    assert result.best_move in legal_moves, (
        f"{move_to_uci(result.best_move)!r} isn't even legal in {fen!r}"
    )

    board.make_move(result.best_move)
    try:
        is_immediate_mate = not generate_legal_moves(board) and board.in_check()
        assert is_immediate_mate, (
            f"{move_to_uci(result.best_move)!r} does not deliver immediate checkmate "
            f"from {fen!r} with null-move pruning active (score was {result.score_cp} "
            f"at depth {result.depth})"
        )
    finally:
        board.unmake_move()
    assert board.to_fen() == fen


# --- 4. Basic sanity: a normal middlegame search still works -----------------


@pytest.mark.parametrize("depth", [5, 6])
def test_normal_middlegame_search_completes_with_null_move_pruning_active(depth: int) -> None:
    """A busy middlegame position (Kiwipete, architecture.md §14) at depth
    5-6 must still return a legal move and complete without raising, with
    null-move pruning active (the default, unmodified `Search`) -- the basic
    "this doesn't blow anything up in ordinary play" sanity check.
    """
    fen = KIWIPETE_FEN
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=depth))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE
    assert result.depth == depth
    legal_moves = generate_legal_moves(board)
    assert result.best_move in legal_moves, (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )
    assert result.nodes > 0
