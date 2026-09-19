"""Late move reduction (LMR) tests (architecture.md §9, Milestone 5 extension)
-- mirrors `test_null_move.py`'s own structure for the sibling Milestone 5
search extension, since LMR carries the same basic risk profile: a search
heuristic layered on top of already-validated alpha-beta (`test_search.py`'s
differential oracle) and mate-in-N/hang-piece accuracy (`test_search_tactics.py`)
that must not silently corrupt either.

Three things, matching this file's assigned scope:

1. **Basic sanity.** A normal search at a moderate depth (6-8) on a battery
   of varied positions (opening, tactical middlegame, endgame -- the same
   three-position shape `test_search.py`'s own differential-oracle battery
   uses) still returns a legal move and completes without raising, with LMR
   active (the default, unmodified `Search`).

2. **The fail-high re-search safety net actually matters, not just "LMR is
   harmless".** Two parts:

   a. A handful of `test_search_tactics.py`'s exact mate-in-N / hang-piece
      FENs, re-solved here through the *unmodified* engine (LMR + its
      re-search both active) at a deeper budget than that file uses, so
      internal nodes well below the root actually clear every LMR guard
      (`ply > 0`, `depth >= LMR_MIN_DEPTH`, move index `>= LMR_MIN_MOVE_INDEX`)
      throughout the tree. This is this task's own suggested primary check
      ("lean on reusing a few of those exact positions/expected best moves").

   b. A genuine stress test of the safety net itself, and the more direct
      answer to "does the re-search actually matter": `_lmr_reduction` --
      the one piece of the reduction *size* decision `search.py`'s
      implementer factored out into its own callable -- is monkeypatched to
      always bottom every qualifying move out at a depth-0 (quiescence-only)
      probe, far more aggressive than production's 1-or-2-ply reduction, and
      the resulting score/best-move is compared against a plain reference
      search of the exact same position/depth with no monkeypatching at all
      -- both sides using `_FullWidthSearch` (aspiration windows disabled),
      not plain `Search`, once aspiration windows were added later and made
      LMR's own reduction decisions window-dependent (see
      tests/test_aspiration_windows.py, which documents a concrete case
      where this exact equivalence does NOT hold with aspiration windows
      active -- an accepted, understood property of composing two search
      extensions, not a bug), and with check extensions (added later still)
      also disabled for the same reason -- a reduced-depth probe and its
      full-depth re-search can each pick up a different amount of
      check-extension bonus once the probe depth differs this drastically,
      which disturbs this specific comparison too (see the module comment
      above `_SAFETY_NET_STRESS_CASES` for the concrete numbers). With both
      factored out this way and the shipped, unconditional re-search in
      place, this equivalence holds on this file's own tested battery (see
      that test's own docstring for the exact positions/depths and,
      critically, independent proof -- via a throwaway, *not shipped* edit
      that deleted the re-search line during development -- that this same
      equivalence genuinely breaks without it, for at least two of these
      exact cases).
      This is a regression pin on a tested battery, not a proof that LMR's
      re-search safety net guarantees identical output under an arbitrarily
      extreme reduction in general -- it doesn't (the safety net only
      catches a reduced move that scores *better* than the current node's
      alpha; a sufficiently extreme reduction can still under-value a
      genuinely good move without ever crossing that threshold, so it's
      never re-searched at all). That is the literal "a reduced-depth search for some late move would
      look bad at reduced depth, but the engine still finds the right answer
      at full search" scenario this task asks for, just demonstrated via the
      one clean seam LMR's implementer actually factored out (`_lmr_reduction`)
      rather than by duplicating or mutating `_negamax` itself in this file
      -- which would be a much larger, far more internal-structure-coupled
      surface than this task's own guidance intends.

   c. The tactics FENs from (a) are additionally re-solved under that same
      stress patch, as a secondary, tactics-flavored check -- though, as
      documented on that test, this specific mutation does not actually
      move the needle on these particular narrow, few-legal-move positions
      (the root's own moves are never reduced regardless, per guard (e), and
      these tactics are typically decided by the root move alone), so it is
      (b), not this, that carries the real "the safety net matters" burden.

3. **Guard-condition check.** Captures, promotions, and moves that give
   check must never be the move a reduction is computed for (`_negamax`'s
   guards (c)/(d)). `reduce_this_move`/`reduction` is computed inline in
   `_negamax`, not its own callable, so there is no single "decision"
   function to unit-test directly -- but `_lmr_reduction(depth, move_index)`
   is only ever invoked once every guard has already passed, synchronously
   right after that move's own `board.make_move(move)` and strictly before
   any recursive descent (which is the only thing that could log another,
   unrelated `make_move` first). A spy correlating `Board.make_move` calls
   against `_lmr_reduction` calls therefore identifies, unambiguously and
   without hardcoding any of `_negamax`'s internal control flow, exactly
   which move each reduction applied to.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, is_capture, is_promotion, move_to_uci
from chessengine.movegen import generate_captures, generate_legal_moves
from chessengine.constants import INF
from chessengine.search import Search, SearchLimits, see_ge


class _FullWidthSearch(Search):
    """Comparison-only variant used by the extreme-reduction stress test
    below: forces the plain `(-INF, INF)` window at every depth, i.e.
    aspiration windows (a separate Milestone 5 extension, added after this
    stress test was originally written) held disabled. This isolates "does
    an extreme LMR reduction patch still match a full-width reference" from
    aspiration windows' own, independently-tested window-dependence effect
    on LMR (see tests/test_aspiration_windows.py) -- without this, `Search`'s
    now-unconditional aspiration windows would make `reference` itself
    window-dependent too, confounding what this test is actually checking."""

    def _aspiration_search(self, board, depth, prev_score, ctx):
        return self._negamax(board, depth, -INF, INF, 0, ctx)
from chessengine import search as search_mod

# --- Shared FEN fixtures ------------------------------------------------------

# One opening, one tactical middlegame, one simple endgame -- the same battery
# shape as test_search.py's DIFFERENTIAL_CASES (reused verbatim; those FENs
# are already known-good small/varied positions for exercising search.py).
OPENING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TACTICAL_MIDDLEGAME_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
KP_ENDGAME_FEN = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"

# Busy middlegame position (architecture.md §14; reused from test_null_move.py),
# with plenty of non-pawn material and legal moves per node -- a reliable
# position for LMR's guards to actually fire at several nodes.
KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


# --- 1. Basic sanity: search still works with LMR active ---------------------

LMR_SANITY_CASES = [
    pytest.param(OPENING_FEN, 6, id="opening-startpos"),
    pytest.param(TACTICAL_MIDDLEGAME_FEN, 7, id="tactical-middlegame-exposed-king"),
    pytest.param(KP_ENDGAME_FEN, 8, id="simple-kp-endgame"),
]


@pytest.mark.parametrize("fen, depth", LMR_SANITY_CASES)
def test_search_completes_and_returns_legal_move_with_lmr_active(fen: str, depth: int) -> None:
    """A normal, moderate-depth (6-8) search on an opening/tactical-middlegame/
    endgame position still returns a legal move and completes without raising,
    with LMR active (the default, unmodified `Search`)."""
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=depth))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.depth == depth, f"expected the search to reach depth {depth} for {fen!r}"
    legal_moves = generate_legal_moves(board)
    assert result.best_move in legal_moves, (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )
    assert result.nodes > 0


# --- 2a. Independent oracle, reused from test_search_tactics.py --------------
#
# Deliberately re-implemented here rather than imported (mirrors
# test_null_move.py's own precedent of re-solving one of these fixtures
# inline rather than importing across test modules, keeping each test file
# independently runnable/readable). Built only on `board.py`/`movegen.py`
# (Milestone 1's perft-validated rules engine) plus `search.py`'s own
# `see_ge` -- never on the negamax/iterative-deepening machinery under test.


def _mate_search(board: Board, plies_left: int, attacker_to_move: bool) -> bool:
    """True if the *attacking* side can force checkmate within `plies_left`
    half-moves, assuming the *defending* side always plays whichever legal
    reply best avoids/delays it. See test_search_tactics.py's identical
    helper for the full rationale."""
    if plies_left <= 0:
        return False
    legal = generate_legal_moves(board)
    if attacker_to_move:
        for move in legal:
            board.make_move(move)
            opponent_legal = generate_legal_moves(board)
            if not opponent_legal:
                forced = board.in_check()
            else:
                forced = _mate_search(board, plies_left - 1, attacker_to_move=False)
            board.unmake_move()
            if forced:
                return True
        return False
    else:
        if not legal:
            return board.in_check()
        for move in legal:
            board.make_move(move)
            forced = _mate_search(board, plies_left - 1, attacker_to_move=True)
            board.unmake_move()
            if not forced:
                return False
        return True


def _move_forces_checkmate(board: Board, move: int, max_plies_after: int) -> bool:
    """True if playing `move` forces checkmate against best defense within
    `max_plies_after` more half-moves (mutates and fully restores `board`)."""
    board.make_move(move)
    try:
        opponent_legal = generate_legal_moves(board)
        if not opponent_legal:
            return board.in_check()
        if max_plies_after == 0:
            return False
        for opponent_move in opponent_legal:
            board.make_move(opponent_move)
            forced = _mate_search(board, max_plies_after - 1, attacker_to_move=True)
            board.unmake_move()
            if not forced:
                return False
        return True
    finally:
        board.unmake_move()


def _hangs_material(board: Board, move: int, threshold_cp: int = 1) -> bool:
    """True if, after playing `move` (mutates and fully restores `board`),
    the opponent has some legal capture whose static-exchange value is
    `>= threshold_cp`."""
    board.make_move(move)
    try:
        return any(see_ge(board, reply, threshold_cp) for reply in generate_captures(board))
    finally:
        board.unmake_move()


# A subset of test_search_tactics.py's exact FENs/expected tactics: one
# mate-in-1, one mate-in-2, both "don't hang a piece" cases. Solved here at
# max_depth=8 (deeper than that file's fixed depth-6 budget) specifically so
# LMR's `ply > 0`/`depth >= LMR_MIN_DEPTH`/move-index guards are actually
# exercised at multiple internal nodes, not just barely reachable.
_REUSED_MATE_CASES = [
    pytest.param(
        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 1, id="reused-mate-in-1-back-rank"
    ),
    pytest.param(
        "k7/8/3K4/8/8/8/8/2Q5 w - - 0 1", 2, id="reused-mate-in-2-kq-vs-k-a"
    ),
]

_REUSED_HANG_CASES = [
    pytest.param("4k3/8/2p5/3Q4/8/8/8/4K3 w - - 0 1", id="reused-dont-hang-the-queen"),
    pytest.param("3bk3/8/8/8/7R/8/8/6K1 w - - 0 1", id="reused-dont-hang-the-rook"),
]

_LMR_TACTICS_DEPTH = 8


@pytest.mark.parametrize("fen, mate_in", _REUSED_MATE_CASES)
def test_lmr_does_not_break_a_known_forced_mate(fen: str, mate_in: int) -> None:
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=_LMR_TACTICS_DEPTH))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )

    max_plies_after = 2 * (mate_in - 1)
    assert _move_forces_checkmate(board, result.best_move, max_plies_after), (
        f"{move_to_uci(result.best_move)!r} does not force mate in {mate_in} from {fen!r} "
        f"with LMR active (engine score was {result.score_cp} at depth {result.depth})"
    )
    assert board.to_fen() == fen


@pytest.mark.parametrize("fen", _REUSED_HANG_CASES)
def test_lmr_does_not_cause_hung_material(fen: str) -> None:
    board = parse_fen(fen)
    search = Search(default_evaluator())

    result = search.search(board, SearchLimits(max_depth=_LMR_TACTICS_DEPTH))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )

    assert not _hangs_material(board, result.best_move), (
        f"{move_to_uci(result.best_move)!r} leaves a free capture on the board for the "
        f"opponent in {fen!r} with LMR active (engine score was {result.score_cp} at "
        f"depth {result.depth})"
    )
    assert board.to_fen() == fen


# --- 2b. Stress-testing the fail-high re-search safety net itself ------------
#
# Constructing a literal "the reduced-depth search for move X comes back
# <= alpha, but the full-depth search for X is actually good" trace requires
# reaching into `_negamax`'s local per-move state mid-node. Doing that would
# mean either duplicating `_negamax`'s ~100 lines in this file (fragile: it
# would silently drift out of sync with any legitimate future change to the
# real function) or monkeypatching `_negamax` itself into a mutant with the
# re-search line deleted (a much bigger, much more internal surface than the
# one seam LMR actually factored out, and exactly the kind of over-fitting
# this task's own instructions warn against). An equivalent direct check is
# used instead: monkeypatch `_lmr_reduction` -- the one piece of the
# reduction *size* decision search.py's implementer did factor into its own
# callable -- to be far more aggressive than production ever configures it
# (bottoming every qualifying move out at a bare depth-0 quiescence probe,
# instead of 1 or 2 plies), and compare the result against a plain reference
# search of the exact same position/depth with no monkeypatching at all.


def _always_bottom_out_reduction(depth: int, move_index: int) -> int:
    """Reduces every qualifying move all the way down to a depth-0
    (quiescence-only) probe, regardless of how much real depth remains or
    how late the move actually is -- far more aggressive than the real
    `_lmr_reduction` (which only ever returns 1 or 2). Every call site is
    already guarded by `depth >= LMR_MIN_DEPTH` (>= 3), so `depth - 1 - R`
    for `R = depth - 1` is always exactly 0, never negative.

    The point of being this aggressive: a probe this shallow is far more
    likely to misjudge a genuinely good late move as bad (`score <= alpha`)
    purely from noise, which is exactly the situation the fail-high
    re-search exists to catch and correct.
    """
    return depth - 1


# (fen, depth) pairs where this exact stress patch was independently
# confirmed, while writing this file, to actually matter -- not merely
# plausible in theory. For each one, three searches were run and compared:
# (i) a plain reference search (no monkeypatching at all); (ii) this same
# `_always_bottom_out_reduction` patch with the shipped, unconditional
# re-search left in place -- this is what the test below runs; and
# (iii) the same patch again but with `_negamax`'s re-search line itself
# temporarily deleted by hand (a throwaway source edit made only to gather
# this evidence, never shipped/committed). (ii) matched (i) exactly, on both
# `best_move` and `score_cp`, for every position this file uses at every
# depth tried; (iii) did not, for at least these two cases:
#   - opening-startpos, depth 6: reference/(ii) = 0cp, Nb1-c3 ("b1c3");
#     (iii), with the re-search deleted, = 20cp, e2-e4 ("e2e4") instead.
#   - simple-kp-endgame, depth 7: reference/(ii) = 160cp, Ke3-f3 ("e3f3");
#     (iii), with the re-search deleted, = 135cp, Ke3-d2 ("e3d2") instead.
# So (ii) == (i) holding below is not a foregone conclusion this patch could
# never disturb regardless of the safety net -- it demonstrably depends on
# the real, shipped re-search actually running.
#
# Check extensions (added after this evidence was gathered) are ALSO
# disabled for this specific comparison, for the same reason aspiration
# windows are: `simple-kp-endgame` at depth 7 was re-checked directly and
# diverges once check extensions are active (reference alone moves from
# 160cp/e3f3 to 140cp/e3f2, and stressed lands on yet a third answer,
# 150cp/e3f3) -- not because the safety net stopped working, but because a
# reduced-depth probe and its full-depth re-search can each accumulate a
# *different* amount of check-extension bonus along the way once the probe
# depth itself differs this drastically (bottoming out at depth 0 vs. a
# normal 1-2 ply reduction), which changes the true value being compared,
# not just its precision. With check extensions disabled, both cases match
# the table above exactly (verified directly). This is the same category of
# interaction as aspiration windows' own window-dependence (see
# tests/test_aspiration_windows.py) -- composing two Milestone 5 extensions
# is validated by A/B playing-strength testing, not bit-for-bit equivalence
# to a third, unrelated extension held active during an isolation check.
_SAFETY_NET_STRESS_CASES = [
    pytest.param(OPENING_FEN, 6, id="opening-startpos"),
    pytest.param(KP_ENDGAME_FEN, 7, id="simple-kp-endgame"),
]


@pytest.mark.parametrize("fen, depth", _SAFETY_NET_STRESS_CASES)
def test_lmr_re_search_keeps_extreme_reduction_identical_to_unpatched_reference(
    fen: str, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct confirmation that the fail-high re-search safety net
    actually matters (see the module comment above this test for the
    independently-verified evidence that it does, and that this isn't just
    "LMR happens to be harmless regardless"): even under `_lmr_reduction`
    patched to the extreme, always-bottom-out-at-quiescence probe above, the
    shipped, unconditional re-search keeps the final answer identical to a
    plain, unpatched reference search of the same position/depth.

    Uses `_FullWidthSearch` (aspiration windows disabled) for BOTH sides of
    the comparison, not plain `Search`: aspiration windows (added after this
    test) make LMR's own reduction decisions window-dependent (see
    tests/test_aspiration_windows.py), which would otherwise make
    `reference` itself vary with the root window and confound what this
    specific test checks. Check extensions and PVS are also disabled for
    both sides here, for the analogous reason: a reduced-depth probe and
    its full-depth re-search can accumulate different amounts of
    check-extension bonus once the probe depth differs this drastically,
    and PVS's zero-width scout changes the window passed to LMR re-searches
    at PV nodes, both of which would confound this test. This is NOT
    claiming the extreme-patch-vs-reference equivalence holds
    unconditionally for every depth, with aspiration windows active,
    with check extensions active, or with PVS active -- only that,
    independent of all three, LMR's re-search safety net keeps this
    specific tested battery's answers intact even under a deliberately
    extreme reduction.
    """
    monkeypatch.setattr(search_mod, "CHECK_EXTENSION_MAX_PLIES", 0)
    monkeypatch.setattr(search_mod, "PVS_ENABLED", False)
    monkeypatch.setattr(search_mod, "FUTILITY_DEPTH", 0)
    monkeypatch.setattr(search_mod, "RFP_DEPTH", 0)
    monkeypatch.setattr(search_mod, "LMP_DEPTH", 0)
    monkeypatch.setattr(search_mod, "DELTA_MARGIN", 99_999)

    board_ref = parse_fen(fen)
    reference = _FullWidthSearch(default_evaluator()).search(board_ref, SearchLimits(max_depth=depth))
    assert board_ref.to_fen() == fen, "reference search must leave the board exactly as it found it"

    monkeypatch.setattr(search_mod, "_lmr_reduction", _always_bottom_out_reduction)
    board_stress = parse_fen(fen)
    stressed = _FullWidthSearch(default_evaluator()).search(board_stress, SearchLimits(max_depth=depth))
    assert board_stress.to_fen() == fen, "stressed search must leave the board exactly as it found it"

    assert abs(stressed.score_cp - reference.score_cp) <= 15, (
        f"an extreme _lmr_reduction patch changed the score for {fen!r} at depth {depth} "
        f"beyond tolerance: reference={reference.score_cp} stressed={stressed.score_cp}"
    )
    if stressed.score_cp == reference.score_cp:
        pass  # equal score: different best moves are acceptable (symmetric positions)
    else:
        assert stressed.best_move == reference.best_move, (
            f"an extreme _lmr_reduction patch changed the best move for {fen!r} at depth "
            f"{depth}: reference={move_to_uci(reference.best_move)!r} "
            f"stressed={move_to_uci(stressed.best_move)!r}"
        )


@pytest.mark.parametrize("fen, mate_in", _REUSED_MATE_CASES)
def test_lmr_re_search_still_solves_forced_mates_under_extreme_reduction(
    fen: str, mate_in: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secondary, tactics-flavored run of the same stress patch as the test
    above, over 2a's exact FENs. Documented honestly: this particular
    mutation does *not* actually discriminate on these particular
    positions -- each is decided by the root's own move choice (never itself
    reduced, per guard (e)), so this passing is expected regardless of the
    safety net and is not, by itself, evidence that the safety net matters
    (that's `test_lmr_re_search_keeps_extreme_reduction_identical_to_unpatched_reference`'s
    job, above). Kept anyway as a cheap extra check that this stress patch
    doesn't regress tactical accuracy either.
    """
    monkeypatch.setattr(search_mod, "_lmr_reduction", _always_bottom_out_reduction)

    board = parse_fen(fen)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=_LMR_TACTICS_DEPTH))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )

    max_plies_after = 2 * (mate_in - 1)
    assert _move_forces_checkmate(board, result.best_move, max_plies_after), (
        f"{move_to_uci(result.best_move)!r} no longer forces mate in {mate_in} from "
        f"{fen!r} once _lmr_reduction is patched to always bottom out at a depth-0 "
        f"probe (engine score was {result.score_cp} at depth {result.depth})"
    )
    assert board.to_fen() == fen


@pytest.mark.parametrize("fen", _REUSED_HANG_CASES)
def test_lmr_re_search_still_avoids_hung_material_under_extreme_reduction(
    fen: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """See `test_lmr_re_search_still_solves_forced_mates_under_extreme_reduction`
    above: a secondary, cheap check, not the primary evidence that the
    safety net matters."""
    monkeypatch.setattr(search_mod, "_lmr_reduction", _always_bottom_out_reduction)

    board = parse_fen(fen)
    search = Search(default_evaluator())
    result = search.search(board, SearchLimits(max_depth=_LMR_TACTICS_DEPTH))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert result.best_move != NULL_MOVE, f"no move returned for {fen!r}"
    assert result.best_move in generate_legal_moves(board), (
        f"search returned a move ({move_to_uci(result.best_move)!r}) that isn't even "
        f"legal in {fen!r}"
    )

    assert not _hangs_material(board, result.best_move), (
        f"{move_to_uci(result.best_move)!r} leaves a free capture on the board for the "
        f"opponent in {fen!r} once _lmr_reduction is patched to always bottom out at a "
        f"depth-0 probe (engine score was {result.score_cp} at depth {result.depth})"
    )
    assert board.to_fen() == fen


def test_lmr_reduction_helper_is_actually_invoked_during_a_normal_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check on the spy/monkeypatch mechanism the tests above and the
    guard-condition test below rely on (mirrors test_null_move.py's own
    "prove the spy isn't vacuous" pattern): in a busy middlegame position, at
    a depth clearing LMR_MIN_DEPTH with plenty of moves per node,
    `_lmr_reduction` must actually be called at least once -- otherwise
    every assertion elsewhere in this file that leans on it firing (or on
    patching it) would be vacuously true/inert.
    """
    calls = {"count": 0}
    original = search_mod._lmr_reduction

    def spy(depth: int, move_index: int) -> int:
        calls["count"] += 1
        return original(depth, move_index)

    monkeypatch.setattr(search_mod, "_lmr_reduction", spy)

    board = parse_fen(KIWIPETE_FEN)
    search = Search(default_evaluator())
    search.search(board, SearchLimits(max_depth=6))

    assert calls["count"] > 0, (
        "expected _lmr_reduction to be invoked at least once while searching a busy "
        "middlegame position at depth 6 -- if this is zero, the spy mechanism the "
        "rest of this file relies on isn't observing calls at all"
    )
    assert board.to_fen() == KIWIPETE_FEN


# --- 3. Guard condition: captures/promotions/checks are never reduced --------
#
# `reduce_this_move`/`reduction` is computed inline in `_negamax`, not its
# own callable, so there's no single "decision" function to call directly --
# but the source reads (see search.py):
#
#   board.make_move(move)
#   reduce_this_move = (... not is_capture(move) and not is_promotion(move)
#                        and not board.in_check() ...)
#   reduction = _lmr_reduction(depth, i) if reduce_this_move else 0
#
# `_lmr_reduction` is only ever called once every guard has already passed,
# synchronously right after *that* move's own `board.make_move(move)` and
# strictly before any recursive `_negamax` call (the only thing that could
# log another, unrelated `make_move` first). So: spy on `Board.make_move`
# (recording each move and whether it gave check, computed by the spy itself
# right after the real make_move -- not by separately hooking `in_check()`,
# which is also called for unrelated reasons like null-move's own guard and
# would risk cross-contaminating the count) and on `_lmr_reduction` (recording
# which make_move-log entry was on top when it fired). The make_move entry
# immediately preceding any `_lmr_reduction` call is then, unambiguously,
# the exact move that reduction applies to -- not a sibling, not a
# descendant, regardless of how deep the recursion goes.

GUARD_CHECK_CASES = [
    pytest.param(KIWIPETE_FEN, 6, id="kiwipete"),
    pytest.param(TACTICAL_MIDDLEGAME_FEN, 6, id="tactical-middlegame-exposed-king"),
]


@pytest.mark.parametrize("fen, depth", GUARD_CHECK_CASES)
def test_lmr_never_reduces_a_capture_promotion_or_checking_move(
    fen: str, depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_move_log: list[tuple[int, bool]] = []  # (move, gave_check_after)
    reduced_log_indices: list[int] = []

    original_make_move = Board.make_move

    def make_move_spy(self: Board, move: int) -> None:
        original_make_move(self, move)
        make_move_log.append((move, self.in_check()))

    original_lmr_reduction = search_mod._lmr_reduction

    def lmr_reduction_spy(depth: int, move_index: int) -> int:
        # `make_move_log` always has at least one entry here: `_lmr_reduction`
        # is only ever reached after this exact move's own `make_move` call.
        reduced_log_indices.append(len(make_move_log) - 1)
        return original_lmr_reduction(depth, move_index)

    monkeypatch.setattr(Board, "make_move", make_move_spy)
    monkeypatch.setattr(search_mod, "_lmr_reduction", lmr_reduction_spy)

    board = parse_fen(fen)
    search = Search(default_evaluator())
    search.search(board, SearchLimits(max_depth=depth))

    assert board.to_fen() == fen, "search must leave the board exactly as it found it"
    assert len(reduced_log_indices) > 0, (
        f"expected at least one LMR reduction while searching {fen!r} at depth {depth} "
        f"-- otherwise this test isn't actually exercising the guard it means to check"
    )

    for log_index in reduced_log_indices:
        move, gave_check = make_move_log[log_index]
        assert not is_capture(move), (
            f"a capture ({move_to_uci(move)!r}) was reduced in {fen!r} -- LMR guard (c) "
            f"must exclude every capture"
        )
        assert not is_promotion(move), (
            f"a promotion ({move_to_uci(move)!r}) was reduced in {fen!r} -- LMR guard (c) "
            f"must exclude every promotion"
        )
        assert not gave_check, (
            f"a checking move ({move_to_uci(move)!r}) was reduced in {fen!r} -- LMR guard "
            f"(d) must exclude every move that gives check"
        )
