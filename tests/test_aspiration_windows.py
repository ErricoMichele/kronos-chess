"""Aspiration window tests (architecture.md §9.3, Milestone 5 extension).

Aspiration windows are, in ISOLATION (null-move pruning and LMR both held
disabled), a provably exact speed optimization: `Search._aspiration_search`
re-searches a depth with a widening window until a result lands strictly
inside it, which is exactly `_negamax`'s own criterion for "this is the
real, provable minimax value," not a bound (see the ASPIRATION_* constants
block in search.py for the full argument). Test 1 below verifies exactly
that isolated claim, across several positions and depths.

That claim does NOT extend, unmodified, to the full shipped `Search`
(aspiration + LMR + null-move pruning all active together): LMR's own
reduction decision depends on whether a reduced-depth probe fails high
against the CURRENT node's `beta`, which is itself a function of the window
handed down from the root -- and aspiration windows deliberately vary that
root window across attempts. So LMR's own search tree is not invariant to
the window it's given, and the FULL system is not guaranteed to be
bit-identical to a full-width search of the same depth. Test 2 documents a
concrete, reproducible example of this (a king+pawn endgame that diverges
with LMR active, but matches exactly once LMR/null-move pruning are
disabled for the same comparison) -- not as a bug, but as an accepted,
understood property of composing an exact optimization with an
already-heuristic one, the same way real engines validate this combination
via A/B playing-strength testing (tests/match_harness.py) rather than
bit-for-bit reproducibility. Tests 3-4 cover what the full, shipped system
*is* actually expected to guarantee: legal moves, no exceptions, and known
tactics still solved.
"""

from __future__ import annotations

import pytest

from chessengine import search as search_mod
from chessengine.constants import INF
from chessengine.evaluate import default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import NULL_MOVE, move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits

OPENING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLEGAME_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
ENDGAME_FEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
KP_ENDGAME_FEN = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"


class _FullWidthSearch(Search):
    """Comparison-only variant: every depth uses the plain `(-INF, INF)`
    window `Search.search` used before aspiration windows existed, instead
    of ever calling `_aspiration_search`. `search.py` itself is never
    edited."""

    def _aspiration_search(self, board, depth, prev_score, ctx):
        return self._negamax(board, depth, -INF, INF, 0, ctx)


@pytest.fixture
def _lmr_and_null_move_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disables LMR (guard (a): move index never qualifies) and null-move
    pruning (guard (b): depth never qualifies) for the duration of a test,
    isolating aspiration windows' own behavior from either heuristic."""
    monkeypatch.setattr(search_mod, "LMR_MIN_MOVE_INDEX", 10_000)
    monkeypatch.setattr(search_mod, "NULL_MOVE_MIN_DEPTH", 10_000)


# --- 1. Aspiration windows are exact, in isolation from LMR/null-move -------


@pytest.mark.parametrize(
    "fen",
    [
        pytest.param(OPENING_FEN, id="opening-startpos"),
        pytest.param(MIDDLEGAME_FEN, id="kiwipete-middlegame"),
        pytest.param(ENDGAME_FEN, id="tactical-endgame"),
        pytest.param(KP_ENDGAME_FEN, id="kp-endgame"),
    ],
)
@pytest.mark.parametrize(
    "depth",
    [
        4,
        5,
        pytest.param(6, marks=pytest.mark.slow),
        pytest.param(7, marks=pytest.mark.slow),
    ],
)
def test_aspiration_matches_full_width_exactly_when_isolated(
    fen: str, depth: int, _lmr_and_null_move_disabled: None
) -> None:
    """With LMR and null-move pruning both disabled, aspiration-enabled
    search must return the IDENTICAL score and best move as a full-width
    search at every depth -- the core provable-exactness claim (see the
    module docstring above)."""
    board_aspiration = parse_fen(fen)
    aspiration_result = Search(default_evaluator()).search(
        board_aspiration, SearchLimits(max_depth=depth)
    )

    board_full_width = parse_fen(fen)
    full_width_result = _FullWidthSearch(default_evaluator()).search(
        board_full_width, SearchLimits(max_depth=depth)
    )

    assert aspiration_result.score_cp == full_width_result.score_cp, (
        f"aspiration windows changed the score at {fen!r} depth {depth}: "
        f"aspiration={aspiration_result.score_cp} full_width={full_width_result.score_cp}"
    )
    assert aspiration_result.best_move == full_width_result.best_move, (
        f"aspiration windows changed the best move at {fen!r} depth {depth}: "
        f"aspiration={move_to_uci(aspiration_result.best_move)!r} "
        f"full_width={move_to_uci(full_width_result.best_move)!r}"
    )


# --- 2. Documented: with LMR active, exact equivalence is not guaranteed ----


@pytest.mark.slow  # depth-7 searches with LMR/null-move pruning disabled are expensive
def test_aspiration_can_diverge_from_full_width_once_lmr_is_active() -> None:
    """Documents a concrete, reproducible case (found during development,
    not a hypothetical) where the FULL shipped `Search` (aspiration + LMR +
    null-move pruning all active, exactly as `default_evaluator()`/`Search`
    ship) diverges from a full-width search of the same depth: LMR's own
    reduction decisions depend on the window handed down from the root,
    which aspiration windows deliberately vary -- see the ASPIRATION_*
    constants block in search.py for the full explanation. This is not a
    correctness bug (both results are the product of a valid, if heuristic,
    search) -- it's why the shipped combination is validated by an A/B
    playing-strength match (tests/match_harness.py), not by asserting
    bit-for-bit equivalence to full-width search once LMR is in play (that
    stronger claim is what test 1 above verifies, ONLY holds with LMR/
    null-move pruning disabled, and this test is the pinned-down evidence of
    why it can't be strengthened to the full, shipped configuration).

    If a future change to LMR or aspiration windows happens to make this
    specific example stop diverging, that's fine (it only demonstrates that
    divergence CAN happen, not that it always must at this exact position/
    depth) -- but the two "isolated" assertions below (both must still hold)
    are what actually matter: disabling LMR/null-move pruning must still
    make the SAME comparison match exactly, confirming the divergence really
    is attributable to LMR's window-dependence and not a hidden bug
    elsewhere.
    """
    depth = 7

    board_aspiration = parse_fen(KP_ENDGAME_FEN)
    aspiration_result = Search(default_evaluator()).search(
        board_aspiration, SearchLimits(max_depth=depth)
    )
    board_full_width = parse_fen(KP_ENDGAME_FEN)
    full_width_result = _FullWidthSearch(default_evaluator()).search(
        board_full_width, SearchLimits(max_depth=depth)
    )
    # Both are legal, well-formed results regardless of whether they agree.
    for result, board in ((aspiration_result, board_aspiration), (full_width_result, board_full_width)):
        assert result.best_move != NULL_MOVE
        assert result.best_move in generate_legal_moves(parse_fen(KP_ENDGAME_FEN))

    # The isolated (LMR/null-move pruning disabled) comparison, at the exact
    # same position/depth, MUST still match exactly -- this is what confirms
    # the divergence above (if it still reproduces) is attributable to LMR's
    # window-dependence specifically, not some other, unrelated bug.
    search_mod_orig_lmr = search_mod.LMR_MIN_MOVE_INDEX
    search_mod_orig_null = search_mod.NULL_MOVE_MIN_DEPTH
    search_mod.LMR_MIN_MOVE_INDEX = 10_000
    search_mod.NULL_MOVE_MIN_DEPTH = 10_000
    try:
        board_iso_a = parse_fen(KP_ENDGAME_FEN)
        iso_aspiration = Search(default_evaluator()).search(board_iso_a, SearchLimits(max_depth=depth))
        board_iso_b = parse_fen(KP_ENDGAME_FEN)
        iso_full_width = _FullWidthSearch(default_evaluator()).search(
            board_iso_b, SearchLimits(max_depth=depth)
        )
    finally:
        search_mod.LMR_MIN_MOVE_INDEX = search_mod_orig_lmr
        search_mod.NULL_MOVE_MIN_DEPTH = search_mod_orig_null

    assert iso_aspiration.score_cp == iso_full_width.score_cp, (
        "with LMR/null-move pruning disabled, aspiration windows must still match "
        "full-width exactly at this same position/depth -- if this fails, the earlier "
        "divergence is NOT attributable to LMR's window-dependence and needs "
        "re-investigation as a genuine aspiration-window bug"
    )
    assert iso_aspiration.best_move == iso_full_width.best_move


# --- 3. The full, shipped system: legal moves, no exceptions ----------------


@pytest.mark.parametrize(
    "fen",
    [
        pytest.param(OPENING_FEN, id="opening-startpos"),
        pytest.param(MIDDLEGAME_FEN, id="kiwipete-middlegame"),
        pytest.param(ENDGAME_FEN, id="tactical-endgame"),
        pytest.param(KP_ENDGAME_FEN, id="kp-endgame"),
    ],
)
def test_full_system_returns_legal_move_without_raising(fen: str) -> None:
    """The full, shipped `Search` (aspiration + LMR + null-move pruning all
    active) must return a legal move and complete without raising, on a
    battery of varied positions, at a depth deep enough to exercise all
    three extensions together."""
    board = parse_fen(fen)
    result = Search(default_evaluator()).search(board, SearchLimits(max_depth=5))
    assert result.best_move != NULL_MOVE
    assert result.best_move in generate_legal_moves(parse_fen(fen))


# --- 4. The full, shipped system still solves known tactics -----------------


@pytest.mark.parametrize(
    "fen, expected_uci",
    [
        pytest.param("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", id="mate-in-1-back-rank"),
    ],
)
def test_full_system_still_solves_a_known_mate_in_1(fen: str, expected_uci: str) -> None:
    """A cheap smoke check that aspiration windows, combined with LMR and
    null-move pruning exactly as shipped, don't cost the engine an
    elementary forced mate it must always find."""
    board = parse_fen(fen)
    result = Search(default_evaluator()).search(board, SearchLimits(max_depth=4))
    assert move_to_uci(result.best_move) == expected_uci
