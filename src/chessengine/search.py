"""Negamax + alpha-beta search with iterative deepening (architecture.md §9).

Per the module-boundary DAG (architecture.md §11), `search.py` depends on
`constants`, `board`, `movegen`, `evaluate`, `transposition`, and `move` —
it is the only module that depends on `movegen.py` + `evaluate.py` +
`transposition.py` together. It also depends on `attacks.py` (for
`attackers_to`, used by `see_ge`, §5.4/§9.5); `attacks.py` sits below
`board.py`/`movegen.py` in the DAG, so this creates no cycle.

**Milestone 4 scope.** This module implements §9.1 (negamax + alpha-beta),
§9.2 (TT usage), §9.3 (iterative deepening / time control), §9.4 (move
ordering: TT move, MVV-LVA, killers, history), §9.5 (quiescence search and
SEE-gated capture pruning), and §9.6 (draws). At `depth == 0`, `_negamax`
hands off to `_quiescence` rather than returning a plain static-eval leaf,
so a side to move is never evaluated mid-capture-sequence (the horizon
effect).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import threading

from .attacks import attackers_to
from .bitboard import iter_bits
from .board import Board
from .constants import (
    BISHOP,
    DRAW_SCORE,
    INF,
    KNIGHT,
    MATE_SCORE,
    MAX_PLY,
    NO_PIECE,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    piece_type_of,
)
from .evaluate import PIECE_VALUE, Evaluator
from .move import (
    EN_PASSANT,
    NULL_MOVE,
    PROMO_PIECE_OF,
    is_capture,
    is_promotion,
    move_flag,
    move_from,
    move_to,
)
from .movegen import generate_captures, generate_legal_moves
from .transposition import TTFlag, TranspositionTable, score_from_tt, score_to_tt

# --- Null-move pruning constants (architecture.md §9, Milestone 5 extension) -
#
# Standard null-move pruning: at an internal (non-root) node, try "passing"
# (the side to move makes no move) and search the opponent's reply at a
# reduced depth with a null (zero-width) window around `beta`. If even a free
# tempo isn't enough for the opponent to avoid a score >= beta, the real
# position is assumed to be at least that good too, and the node is pruned
# without generating/searching any real moves.
#
# Gated per architecture.md §15's rule for search extensions ("an A/B match
# at a fixed time control against the immediately prior version that it must
# not lose measurable strength against"): tests/match_harness.py's
# play_match_searches, same evaluator both sides, SearchLimits(movetime_ms=200),
# ply_cap=120 (large enough for games to actually reach checkmate/repetition/
# 50-move conclusions rather than all drawing by cap, unlike a first attempt
# at ply_cap=24 which was inconclusive for exactly that reason), over a
# 12-position/24-game battery: NMP-enabled scored 12.5 vs NMP-disabled's 11.5
# -- a modest but genuine, non-negative edge (comfortably clears the "must
# not lose strength" bar; a small margin is expected and typical for NMP at
# equal time, not a red flag).
NULL_MOVE_MIN_DEPTH = 3  # guard (b): only try null-move at depth >= this
NULL_MOVE_REDUCTION = 2  # "R": reduced search is at depth - 1 - R

# --- Late move reductions (LMR) (architecture.md §9, Milestone 5 extension) -
#
# Standard LMR: moves searched late in an already-ordered move list
# (architecture.md §9.4 -- TT move, MVV-LVA, killers, history) are, by
# construction, exactly the ones move ordering rated *least* promising.
# Rather than pay a full-depth search for every one of them, a "late" QUIET
# move is first probed at a reduced depth using the exact same
# (negated/swapped) alpha-beta window a full-depth search of that move would
# have used. If that reduced probe still comes back better than `alpha` (a
# "fail high" relative to the reduced search), the move might genuinely be
# good, so it is re-searched at the full, unreduced depth before its score is
# trusted. This re-search is what makes LMR safe: a reduction can only ever
# *cost* extra work (via the re-search), it can never silently produce a
# wrong score, unlike null-move pruning above, which trusts a cutoff outright.
#
# Every guard below is standard and conservative -- each one excludes a move
# the reduction itself could plausibly get wrong:
#   (a) `i >= LMR_MIN_MOVE_INDEX`: never reduce the first few moves in the
#       list -- those are precisely the ones TT-move/MVV-LVA/killers/history
#       ordering already front-loaded as most promising, so reducing them
#       would throw away the point of move ordering.
#   (b) `depth >= LMR_MIN_DEPTH`: too shallow otherwise for a reduced search
#       to mean anything. Combined with guard (e) below (never at the root),
#       this also structurally keeps LMR out of `test_search.py`'s depth-3,
#       ply-0-rooted differential-oracle test: depth only reaches
#       `LMR_MIN_DEPTH` (3) at ply 0, which guard (e) excludes, and every
#       node below that has depth <= 2 -- exactly the same "guard combo can
#       never simultaneously hold within this test" property the null-move
#       section above already relies on.
#   (c) the move is quiet (no capture, no promotion): a capture or promotion
#       is exactly the kind of forcing, material-swinging move LMR must not
#       risk underestimating -- MVV-LVA/SEE (§9.4/§9.5) already handle
#       captures' ordering and quiescence already extends them at the
#       horizon, so they never need this heuristic's help.
#   (d) neither side is "in check" across the move: not in check before the
#       move (a move played while in check is answering a forced, narrow set
#       of evasions, not a genuinely "late/unpromising" try) and the move
#       does not itself give check (checked, for free, via `board.in_check()`
#       from the opponent's now-to-move perspective right after making the
#       move) -- a checking move opens a forcing line that a reduced search
#       is exactly the wrong tool to risk misjudging, the same concern check
#       extensions exist to address.
#   (e) `ply > 0`: never at the root -- the root must always produce a real,
#       fully-searched best move (mirrors the null-move section's own
#       rationale above), and, as guard (b) notes, this is also what keeps
#       LMR out of the shallow differential-oracle test.
#
# Reduction size: a small fixed R=1 for an ordinary late move, stepping up to
# R=2 only once a move is both very late (`i >= LMR_DEEP_MOVE_INDEX`) and
# there is plenty of remaining depth to spend the extra reduction on
# (`depth >= LMR_DEEP_DEPTH`) -- the standard "reduce more, the later and the
# deeper" shape most textbook LMR tables use, kept here as a two-tier step
# function rather than a full log-scaled table for simplicity/auditability.
# Given guards (b)/the two depth thresholds above, `depth - 1 - R` is always
# >= 1 (never negative, and never so small it skips straight past depth 0
# without at least one more real ply of search): at the minimum qualifying
# depth (3), R is always 1 (R=2 requires depth >= 6), giving 3 - 1 - 1 = 1;
# at the minimum depth where R=2 applies (6), 6 - 1 - 2 = 3.
#
# Gated per architecture.md §15's rule for search extensions, the same way
# null-move pruning above is: tests/match_harness.py's play_match_searches,
# same evaluator both sides, `SearchLimits(movetime_ms=200)`, `ply_cap=120`
# (large enough for games to reach a real conclusion rather than drawing by
# cap -- a smaller preliminary check at ply_cap=60 was run first during
# implementation and is superseded by this larger, more decisive one), over
# a 12-position/24-game battery (the null-move gate's own battery, plus
# several more varied positions): LMR-enabled scored 13.5 vs LMR-disabled's
# 10.5 -- a clear, non-negative edge (a larger margin than null-move
# pruning's own +1, comfortably clearing the "must not lose strength" bar).
LMR_MIN_MOVE_INDEX = 4  # guard (a): moves with index < this are never reduced
LMR_MIN_DEPTH = 3  # guard (b): no reduction below this remaining depth
LMR_DEEP_MOVE_INDEX = 8  # "very late" threshold for the deeper reduction
LMR_DEEP_DEPTH = 6  # "plenty of remaining depth" threshold for the deeper reduction
LMR_BASE_REDUCTION = 1  # R for an ordinary qualifying late move
LMR_DEEP_REDUCTION = 2  # R once a move is both very late and depth is large


def _lmr_reduction(depth: int, move_index: int) -> int:
    """`R` for a move that has already passed every LMR guard:
    `LMR_DEEP_REDUCTION` once it's both very late and there's plenty of
    depth left to spend the extra reduction on, `LMR_BASE_REDUCTION`
    otherwise. See the constants block above for the full rationale."""
    if depth >= LMR_DEEP_DEPTH and move_index >= LMR_DEEP_MOVE_INDEX:
        return LMR_DEEP_REDUCTION
    return LMR_BASE_REDUCTION


# --- Aspiration windows (architecture.md §9.3, Milestone 5 extension) ------
#
# Standard aspiration windows: once iterative deepening has a previous
# completed depth's score to work with, the next (deeper) iteration is first
# tried with a NARROW window centered on that score (`Search._aspiration_
# search`, called from `Search.search`'s own loop below) instead of the full
# `(-INF, INF)` window every depth used before this. Successive
# iterative-deepening scores usually move smoothly from one depth to the
# next, so a window only `ASPIRATION_INITIAL_DELTA` centipawns wide around
# the last depth's score still contains the truth almost all the time -- and
# a narrower window lets alpha-beta cut off far more of the tree to *prove*
# that, for the identical final answer, in less time.
#
# In ISOLATION -- i.e. plain alpha-beta + TT, with null-move pruning/LMR
# both held disabled -- this is a pure speed optimization, exact rather than
# a heuristic: `_negamax`'s `(alpha, beta)` window is advisory, never
# authoritative, so handing it a too-narrow window can only ever make it
# return an inexact *bound* (`score <= alpha`, a "fail low", or `score >=
# beta`, a "fail high"; see its own `TTFlag` classification at the bottom of
# `_negamax`, `EXACT` iff `alpha_orig < best_score < beta`), never a wrong
# exact score. `Search._aspiration_search` treats exactly those two cases as
# "not yet trustworthy": it re-searches the SAME depth with a wider window
# and repeats until a result lands strictly inside its own window, which is
# precisely `_negamax`'s own criterion for "this is the real, provable
# minimax value" rather than a bound -- only then is the result fed back as
# `prev_score` for the next depth's window and reported as that depth's
# score. Doubling `ASPIRATION_INITIAL_DELTA` on every retry (both edges of
# the window widen together, still centered on the same `prev_score`)
# guarantees this terminates: `delta` cannot double forever without the
# window's edges being clamped to `-INF`/`INF` (see `_aspiration_search`),
# at which point the "re-search" is, by construction, an ordinary
# full-width search. `tests/test_aspiration_windows.py` verifies this exact
# isolated claim directly (LMR/null-move pruning both disabled for the
# comparison), and it holds with zero exceptions found.
#
# IMPORTANT caveat once LMR is back in the picture (as it always is in the
# shipped `Search`, never actually run in isolation): LMR's own reduction
# decision depends on whether a reduced-depth probe's score fails high
# against the CURRENT node's `beta` (see the LMR section above) -- and
# `beta` at any node is a function of the window handed down from its
# ancestors, ultimately from the root's own `(alpha, beta)`. Since aspiration
# windows deliberately vary the ROOT's window across attempts (that's the
# entire point), they can change which nodes' late moves happen to fail high
# and get a full-depth re-search versus which ones don't -- i.e. LMR's own
# search tree is not itself invariant to the window it's given. This means
# the FULL system (aspiration + LMR + null-move pruning together, exactly as
# shipped) is NOT guaranteed to be bit-identical to a full-width search of
# the same depth, even though aspiration windows alone provably are:
# confirmed empirically (`tests/test_aspiration_windows.py` documents a
# concrete king+pawn-endgame example that diverges at depths 6-8 with LMR
# enabled, yet matches exactly at every depth once LMR/null-move pruning are
# disabled for the comparison). This is not a bug in aspiration windows, LMR,
# or their combination -- it is an inherent property of composing an
# exactness-preserving optimization with an already-heuristic one, the same
# category of behavior real engines accept and validate via A/B playing-
# strength testing rather than bit-for-bit reproducibility.
#
# Mate scores need no special-casing: a sudden mate score appearing at some
# depth is just an ordinary fail-high (if positive, blowing past a narrow
# `beta`) or fail-low (if negative, blowing past a narrow `alpha`), caught by
# the exact same widen-and-retry loop as any other fail -- it costs one or
# two extra widening rounds to grow the window out to where the mate score
# actually lands, not a different code path.
#
# `ASPIRATION_MIN_DEPTH`: depths below this always use the full `(-INF,
# INF)` window (`Search.search` never calls `_aspiration_search` for them).
# Depth 1 has no previous completed depth's score to center a window on at
# all, and depth 2's score is still typically too unstable (move ordering
# itself is still stabilizing this early) to be worth narrowing around --
# both depths are cheap enough that a full-width search costs nothing
# meaningful anyway. Aspiration only starts paying for itself once there is
# a depth-(n-1) score that is actually a decent predictor of depth n's.
#
# Gated per architecture.md §15's rule for search extensions, the same way
# null-move pruning/LMR above are: tests/match_harness.py's
# play_match_searches, same evaluator both sides, SearchLimits(movetime_ms=200),
# ply_cap=120, over the same 12-position/24-game battery as the NMP/LMR
# gates: aspiration-enabled scored 17.0 vs aspiration-disabled's 7.0 -- a
# large, clear edge (bigger than either NMP's +1 or LMR's +3), consistent
# with aspiration windows being the closest thing to a "free" optimization
# of the three (provably exact in isolation, per this file's own tests,
# unlike NMP/LMR which are heuristic prunes by construction).
ASPIRATION_MIN_DEPTH = 3  # depths below this always use the full (-INF, INF) window
ASPIRATION_INITIAL_DELTA = 25  # centipawns; ~1/4 of a pawn, standard narrow starting half-width


# --- Check extensions (architecture.md §9, Milestone 5 extension) ----------
#
# When a move gives check, the side to move next has a much narrower reply
# set (only moves that get the king out of check are legal at all), and
# forced check sequences can hide a mate or a decisive material swing one
# ply beyond where a fixed-depth search would otherwise stop looking. The
# fix: search a move that gives check one ply DEEPER than normal
# (`depth - 1 + CHECK_EXTENSION_PLIES` instead of `depth - 1`), instead of
# treating it like any other quiet move.
#
# `gives_check` is computed once per move, right after `board.make_move`
# (it reflects the position from the new side to move's perspective, i.e.
# whether OUR move put THEM in check), and is reused for both the
# extension decision here and LMR's own guard above (`not
# board.in_check()`) -- a move that gives check is therefore never
# LMR-reduced in the first place, so a single move is never both reduced
# and extended in the same call.
#
# `ext_remaining` bounds cumulative extensions along any one path: it is
# threaded down through every recursive call (unlike `depth`, it is NOT
# reset per node) and decremented only when an extension is actually
# granted, specifically to rule out unbounded recursion from a long chain
# of only-check moves repeatedly cancelling `depth`'s own decrement.
# Once the budget hits zero, later checking moves on that same line are
# searched at the normal, unextended depth like any other move -- search
# remains correct either way (a missing extension only ever means a
# fixed, finite amount less lookahead in an extreme, contrived line, never
# a wrong score), so this budget is a performance/termination safeguard,
# not a correctness requirement. `CHECK_EXTENSION_MAX_PLIES` (16) is far
# more than any but the most pathological perpetual-check-shaped line
# would consume in practice, since the fifty-move-rule/repetition checks
# at the top of `_negamax` also bound any real perpetual-check line long
# before this budget could matter.
#
# Gated per architecture.md §15's rule for search extensions, the same way
# null-move pruning/LMR/aspiration windows above are: tests/match_harness.py's
# play_match_searches, same evaluator both sides, SearchLimits(movetime_ms=200),
# ply_cap=120, over a 12-position/24-game battery (the same 6 DEFAULT_POSITIONS
# plus 6 additional varied openings/middlegames used by the NMP/LMR/aspiration
# gates): check-extensions-enabled scored 13.5 vs disabled's 10.5 -- a clear,
# non-negative edge (the same +3 margin LMR's own gate showed, comfortably
# clearing the "must not lose measurable strength" bar).
CHECK_EXTENSION_PLIES = 1  # depth bonus applied to a move that gives check
CHECK_EXTENSION_MAX_PLIES = 16  # cumulative per-path budget; see rationale above


# --- Public search-parameter/result types (architecture.md §9.3) -----------


@dataclass
class SearchLimits:
    max_depth: int = 64
    movetime_ms: int | None = None
    nodes: int | None = None


@dataclass
class SearchInfo:  # one per completed depth, handed to on_info
    depth: int
    score_cp: int
    nodes: int
    pv: list[int]


@dataclass
class SearchResult:
    best_move: int
    score_cp: int
    depth: int
    nodes: int
    pv: list[int]


class _SearchCtx:
    """Per-call transient state. Not reused across `.search()` calls."""

    def __init__(
        self,
        limits: SearchLimits,
        deadline: float | None,
        stop_event: "threading.Event | None",
        extra_stop: "Callable[[], bool] | None",
    ) -> None:
        self.limits = limits
        self.deadline = deadline
        self.stop_event = stop_event
        self.extra_stop = extra_stop
        self.nodes = 0
        # Triangular PV array (architecture.md §9.3): `pv[ply]` holds the
        # principal variation from that ply downward, and `pv_length[ply]`
        # is the number of valid moves in it.  Updated inside `_negamax`
        # whenever a new best move is found at a given ply; the root's PV
        # (`pv[0][:pv_length[0]]`) is the complete, stable principal
        # variation for the most recently completed depth -- replacing the
        # old TT-walk approach (`_extract_pv`), which was fragile because
        # TT entries can be overwritten mid-search.
        self.pv: list[list[int]] = [[] for _ in range(MAX_PLY + 1)]
        self.pv_length: list[int] = [0] * (MAX_PLY + 1)

    def should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        if self.limits.nodes is not None and self.nodes >= self.limits.nodes:
            return True
        # Wall-clock checked every 2048 nodes, not every node, so
        # time.monotonic() overhead doesn't distort node counts at shallow depth.
        if self.deadline is not None and self.nodes % 2048 == 0 and time.monotonic() >= self.deadline:
            return True
        if self.extra_stop is not None and self.extra_stop():
            return True
        return False


# --- Search --------------------------------------------------------------


class Search:
    """Owns the search's persistent state: the transposition table, killer
    moves, and the history heuristic (architecture.md §9.1). Persistent
    across calls to `.search()` within one game; `new_game()` resets all of
    it for UCI `ucinewgame`."""

    def __init__(self, evaluator: Evaluator, tt_size_mb: int = 64) -> None:
        self.evaluator = evaluator
        self.tt = TranspositionTable(tt_size_mb)
        self.killers: list[list[int]] = [[NULL_MOVE, NULL_MOVE] for _ in range(MAX_PLY)]
        self.history: list[list[int]] = [[0] * 64 for _ in range(64)]  # [from][to]

    def new_game(self) -> None:
        """Called on UCI 'ucinewgame'. Stale TT/killer/history entries from
        a previous, unrelated game must not leak into this one."""
        self.tt.clear()
        self.killers = [[NULL_MOVE, NULL_MOVE] for _ in range(MAX_PLY)]
        self.history = [[0] * 64 for _ in range(64)]

    # --- Top-level entry point (architecture.md §9.3) ----------------------

    def search(
        self,
        board: Board,
        limits: SearchLimits,
        *,
        stop_event: "threading.Event | None" = None,
        extra_stop: "Callable[[], bool] | None" = None,
        on_info: "Callable[[SearchInfo], None] | None" = None,
    ) -> SearchResult:
        deadline = time.monotonic() + limits.movetime_ms / 1000 if limits.movetime_ms else None
        ctx = _SearchCtx(limits, deadline, stop_event, extra_stop)

        # Seed a legal fallback move (roughly ordered, so it's at least a
        # reasonable capture/central move rather than an arbitrary one)
        # *before* the first iteration runs. If a stop signal is already
        # set (or fires before depth 1's root loop completes even one
        # move), `_negamax` bails out having never reached `tt.store`, so
        # `_extract_pv` finds nothing — without this seed, `best.best_move`
        # would stay `NULL_MOVE` and get reported over UCI as an illegal
        # "bestmove a1a1". Empty only in a terminal position (no legal
        # moves), where `NULL_MOVE` is the only sensible answer anyway.
        root_moves = generate_legal_moves(board)
        fallback_move = self._order_moves(root_moves, board, NULL_MOVE, 0)[0] if root_moves else NULL_MOVE
        best = SearchResult(fallback_move, 0, 0, 0, [])

        # `prev_score` feeds `_aspiration_search`'s window center once depth
        # reaches `ASPIRATION_MIN_DEPTH` (see the ASPIRATION_* constants
        # block above); its initial value is never actually used as a window
        # center (depths below that threshold always take the full-window
        # branch instead), so 0 vs. anything else here makes no difference.
        prev_score = 0
        for depth in range(1, limits.max_depth + 1):
            if depth < ASPIRATION_MIN_DEPTH:
                # Too shallow for a previous depth's score to be a
                # meaningful window center yet -- searched exactly as every
                # depth was before aspiration windows existed: full width.
                score = self._negamax(board, depth, -INF, INF, 0, ctx)
            else:
                score = self._aspiration_search(board, depth, prev_score, ctx)
            if ctx.should_stop():
                break  # partial/unreliable result from an aborted depth: discard entirely
            # Read the PV from the triangular PV array (stable, built
            # inside _negamax as it runs) instead of the old TT-walk
            # approach (_extract_pv), which was fragile against TT
            # overwrites producing truncated or stale PV lines.
            pv = ctx.pv[0][:ctx.pv_length[0]]
            best = SearchResult(pv[0] if pv else best.best_move, score, depth, ctx.nodes, pv)
            prev_score = score
            if on_info is not None:
                on_info(SearchInfo(depth, score, ctx.nodes, pv))
            if abs(score) >= MATE_SCORE - 128:
                break  # forced mate found; no point searching deeper
        return best

    # --- Aspiration windows (architecture.md §9.3, Milestone 5 extension) ---

    def _aspiration_search(self, board: Board, depth: int, prev_score: int, ctx: _SearchCtx) -> int:
        """Depth `depth`'s real score, found by searching with a narrow
        window centered on `prev_score` (the previous completed depth's
        score) and re-searching this SAME depth with a wider window
        whenever the result isn't trustworthy yet -- see the ASPIRATION_*
        constants block above for the full correctness argument. Only ever
        called for `depth >= ASPIRATION_MIN_DEPTH`; `Search.search` uses the
        full `(-INF, INF)` window directly for every depth below that.

        A result is trustworthy exactly when it lands strictly inside the
        window it was searched with (`alpha < score < beta`): `_negamax`'s
        own TT-flag logic classifies anything else -- `score <= alpha` ("fail
        low") or `score >= beta` ("fail high") -- as a bound, not an exact
        value, so this loop treats those identically and never returns one
        as depth `depth`'s final score. Widening doubles `delta` and
        recenters both edges on the same `prev_score` (clamped so the window
        can never exceed the ordinary full `(-INF, INF)` bounds), which is
        what guarantees the loop terminates: enough doublings clamp both
        edges and the "re-search" degenerates into an ordinary full-width
        search, which cannot itself fail (a legal position always has a
        finite minimax value strictly between `-INF` and `INF`).

        Mid-re-search stop signals: `ctx.should_stop()` is checked
        immediately after every `_negamax` call, exactly where every other
        call site in this file checks it, and this method returns
        immediately (without widening further) the instant it's set. The
        returned value is never inspected in that case -- `Search.search`
        checks `ctx.should_stop()` again right after this method returns and
        discards the whole depth if so, the same contract every other
        aborted-iteration case in `.search()` already relies on -- so an
        untrustworthy score from a stopped mid-widening call can never be
        mistaken for depth `depth`'s real value, and the loop can never spin
        forever waiting for a window a stopped search will never fill in.

        Mate scores get no special case: a mate score simply fails high or
        low like any other out-of-window score and is caught by the same
        loop (see the ASPIRATION_* constants block above).
        """
        delta = ASPIRATION_INITIAL_DELTA
        alpha = max(prev_score - delta, -INF)
        beta = min(prev_score + delta, INF)
        while True:
            score = self._negamax(board, depth, alpha, beta, 0, ctx)
            if ctx.should_stop():
                return score  # discarded by the caller; see the docstring above
            if alpha < score < beta:
                return score  # lands strictly inside the window: a real, exact score
            # Fail low (score <= alpha) or fail high (score >= beta): not
            # trustworthy yet. Double delta and re-center the (still
            # symmetric) window on the same prev_score, then re-search this
            # SAME depth from scratch with the wider window.
            delta *= 2
            alpha = max(prev_score - delta, -INF)
            beta = min(prev_score + delta, INF)

    # --- Negamax core (architecture.md §9.1) --------------------------------

    def _negamax(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        ctx: _SearchCtx,
        null_ok: bool = True,
        ext_remaining: int | None = None,
    ) -> int:
        if ext_remaining is None:
            # Resolved dynamically (not a plain default-parameter value) so
            # that, exactly like `NULL_MOVE_MIN_DEPTH`/`LMR_MIN_DEPTH` above,
            # tests can monkeypatch `CHECK_EXTENSION_MAX_PLIES` on the module
            # to disable check extensions for isolation -- a default bound at
            # function-definition time would not observe that patch.
            ext_remaining = CHECK_EXTENSION_MAX_PLIES
        ctx.nodes += 1
        ctx.pv_length[ply] = 0  # no PV yet at this ply; updated below if a best move is found
        if ctx.should_stop():
            return 0  # discarded: caller checks ctx.should_stop()

        if board.is_fifty_move_draw() or board.is_repetition_draw():
            return DRAW_SCORE

        alpha_orig = alpha
        entry = self.tt.probe(board.zobrist_hash)
        tt_move = entry.best_move if entry is not None else NULL_MOVE
        # No TT-based early return/bound-narrowing at the root (`ply == 0`),
        # only ever used above for move-ordering's `tt_move` hint. This is
        # standard practice, and became load-bearing once aspiration windows
        # (Milestone 5 extension) started passing a NARROW window into the
        # root: `_aspiration_search` re-searches the SAME position at the
        # SAME depth repeatedly (once per widening attempt) whenever a
        # search fails low/high, and each attempt's own `store` leaves an
        # UPPERBOUND/LOWERBOUND entry at that exact depth behind. Without
        # this guard, the *next* (wider-window) attempt's TT probe finds
        # that entry (`entry.depth >= depth` trivially holds -- same
        # depth), narrows alpha/beta with it, and can short-circuit straight
        # to `return score` before ever generating a single root move --
        # returning last attempt's stale bound rather than a real result
        # for the new window. That stale value can easily land strictly
        # inside the new, wider window (`alpha < score < beta`), which is
        # exactly the trustworthiness test `_aspiration_search` uses to
        # accept a score as final -- so the aspiration loop would accept a
        # bound left over from a too-narrow window as if it were this
        # depth's genuine, fully-searched value. (Verified empirically: a
        # king+pawn endgame at depth 6-7 diverged from a full-width
        # reference before this guard was added, and matched after.) Every
        # other node (`ply > 0`) keeps the TT cutoff exactly as before --
        # only the root is special, precisely because it's the one node
        # `_aspiration_search` deliberately re-queries at an unchanged depth.
        if ply > 0 and entry is not None and entry.depth >= depth:
            score = score_from_tt(entry.score, ply)
            if entry.flag == TTFlag.EXACT:
                return score
            if entry.flag == TTFlag.LOWERBOUND:
                alpha = max(alpha, score)
            elif entry.flag == TTFlag.UPPERBOUND:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        if depth == 0:
            # Quiescence search (architecture.md §9.5), not a plain
            # static-eval leaf: extends the leaf with captures until the
            # position is quiet, to avoid misjudging a hanging piece mid
            # capture-sequence (the horizon effect).
            return self._quiescence(board, alpha, beta, ply, ctx)

        # --- Null-move pruning (Milestone 5 extension; architecture.md §9) ---
        #
        # Never at the root (`ply > 0`): the root must always produce a real
        # best move to play, which a null-move cutoff cannot supply. All four
        # guards must hold, or we fall straight through to normal move
        # generation:
        #   (a) side to move is not in check — a null move while in check is
        #       illegal/unsound (it would "answer" the check by doing nothing);
        #   (b) depth >= NULL_MOVE_MIN_DEPTH — too shallow otherwise for the
        #       reduced search below to mean anything;
        #   (c) the side to move has at least one non-pawn, non-king piece —
        #       the standard zugzwang-avoidance guard, since null-move pruning
        #       is unsound in pure king+pawn endgames where passing can
        #       genuinely be the only reasonable try;
        #   (d) `null_ok` — this node was not itself reached via a null move
        #       (no two null moves back to back).
        us = board.side_to_move
        if (
            ply > 0
            and null_ok
            and depth >= NULL_MOVE_MIN_DEPTH
            and not board.in_check()
            and _has_non_pawn_material(board, us)
        ):
            board.make_null_move()
            null_score = -self._negamax(
                board,
                depth - 1 - NULL_MOVE_REDUCTION,
                -beta,
                -beta + 1,
                ply + 1,
                ctx,
                null_ok=False,  # guard (d) for the child: no back-to-back null moves
                ext_remaining=ext_remaining,
            )
            board.unmake_null_move()
            if ctx.should_stop():
                return 0
            # Fail-hard cutoff: return `beta` itself, never `null_score`. A
            # score this reduced/null-window search produces is only ever
            # trusted as a >=/< beta signal, not as this node's real score —
            # in particular a mate-range score here (>= MATE_SCORE - 128 in
            # magnitude) would be an artifact of the opponent getting a free
            # extra move, not a real, reachable mate line through an actual
            # move, so it must never leak out of this function (which would
            # corrupt this node's TT entry and mate-distance reporting, per
            # score_to_tt/score_from_tt, §9.2). Such scores are treated as an
            # untrustworthy signal and skipped rather than used for a cutoff.
            if null_score >= beta and abs(null_score) < MATE_SCORE - 128:
                return beta

        moves = generate_legal_moves(board)
        if not moves:
            return -MATE_SCORE + ply if board.in_check() else DRAW_SCORE

        self._order_moves(moves, board, tt_move, ply)

        # Whether the side to move *here* is in check, computed once and
        # reused by every move's LMR guard (d) below, rather than
        # re-deriving the pre-move half of that check per move.
        in_check_before = board.in_check()

        best_score, best_move = -INF, moves[0]
        for i, move in enumerate(moves):
            board.make_move(move)
            gives_check = board.in_check()  # does this move itself give check?

            # --- Late move reductions (Milestone 5 extension; see the
            # LMR_* constants block above for the full guard-by-guard
            # rationale). Every guard must hold or this move is searched at
            # the normal, unreduced depth like any other move.
            reduce_this_move = (
                ply > 0
                and i >= LMR_MIN_MOVE_INDEX
                and depth >= LMR_MIN_DEPTH
                and not in_check_before
                and not is_capture(move)
                and not is_promotion(move)
                and not gives_check
            )
            reduction = _lmr_reduction(depth, i) if reduce_this_move else 0

            # --- Check extensions (Milestone 5 extension; see the
            # CHECK_EXTENSION_* constants block above). Mutually exclusive
            # with the reduction above by construction (`reduce_this_move`
            # already requires `not gives_check`), so a move is never both
            # reduced and extended.
            extend = CHECK_EXTENSION_PLIES if (gives_check and ext_remaining > 0) else 0
            child_ext_remaining = ext_remaining - extend

            score = -self._negamax(
                board,
                depth - 1 - reduction + extend,
                -beta,
                -alpha,
                ply + 1,
                ctx,
                ext_remaining=child_ext_remaining,
            )
            if reduction and not ctx.should_stop() and score > alpha:
                # The reduced search still looks like it might beat alpha:
                # never trust that outright -- re-search this exact move at
                # the full, unreduced depth before accepting its score. This
                # re-search is what makes LMR safe: a reduction only ever
                # costs extra work here, it can never silently produce a
                # wrong score (unlike the null-move cutoff above, which is
                # trusted outright).
                score = -self._negamax(
                    board, depth - 1, -beta, -alpha, ply + 1, ctx, ext_remaining=child_ext_remaining
                )

            board.unmake_move()
            if ctx.should_stop():
                return 0
            if score > best_score:
                best_score, best_move = score, move
                # Update the triangular PV: this ply's PV is now [move]
                # followed by the child's PV (from ply+1).
                ctx.pv[ply] = [move] + ctx.pv[ply + 1][:ctx.pv_length[ply + 1]]
                ctx.pv_length[ply] = 1 + ctx.pv_length[ply + 1]
            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_cutoff(move, depth, ply)  # killers/history, §9.4
                break

        flag = (
            TTFlag.EXACT
            if alpha_orig < best_score < beta
            else TTFlag.LOWERBOUND
            if best_score >= beta
            else TTFlag.UPPERBOUND
        )
        self.tt.store(board.zobrist_hash, depth, score_to_tt(best_score, ply), flag, best_move)
        return best_score

    # --- Quiescence search (architecture.md §9.5) ---------------------------

    def _quiescence(
        self, board: Board, alpha: int, beta: int, ply: int, ctx: _SearchCtx
    ) -> int:
        """Extends a leaf node with captures only, until the position is
        "quiet" (no immediately-winning captures left to make), to avoid the
        classic horizon effect of stopping mid-capture-sequence and
        misjudging a hanging piece.

        `stand_pat` (the static evaluation, as if the side to move could
        "stand pat" and decline every capture) both bounds the search and is
        the fallback score when no capture improves on it. Captures are
        ordered the same way `_negamax` orders moves (TT move — always
        `NULL_MOVE` here, since quiescence nodes aren't TT-probed — then
        MVV-LVA) and gated by `see_ge(board, move, 0)`, which skips captures
        whose estimated exchange is clearly losing rather than paying to
        search them out.
        """
        ctx.nodes += 1
        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        for move in self._order_moves(generate_captures(board), board, NULL_MOVE, ply):
            if not see_ge(board, move, 0):  # SEE-based pruning: skip clearly-losing captures
                continue
            board.make_move(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1, ctx)
            board.unmake_move()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    # --- Move ordering (architecture.md §9.4) -------------------------------

    def _order_moves(self, moves: list[int], board: Board, tt_move: int, ply: int) -> list[int]:
        """Sorts `moves` in place, highest priority first:

        1. The TT's stored `best_move` for this position, if any.
        2. Captures ranked by MVV-LVA (`victim_value * 16 - attacker_value`).
        3. Killer moves: up to two quiet moves per ply that most recently
           caused a beta cutoff at this ply in a sibling node.
        4. History heuristic (`self.history[frm][to]`) as the tiebreak.
        """
        killers = self.killers[ply]

        def key(move: int) -> tuple[int, int]:
            if move == tt_move:
                return (3, 0)
            if is_capture(move):
                return (2, _mvv_lva_score(board, move))
            if move == killers[0] or move == killers[1]:
                return (1, 0)
            return (0, self.history[move_from(move)][move_to(move)])

        moves.sort(key=key, reverse=True)
        return moves

    def _record_cutoff(self, move: int, depth: int, ply: int) -> None:
        """Called on a beta cutoff: updates killers/history only for quiet
        moves — captures already get MVV-LVA ordering and don't need history
        bookkeeping."""
        if is_capture(move):
            return
        killers = self.killers[ply]
        if killers[0] != move:
            killers[1] = killers[0]
            killers[0] = move
        self.history[move_from(move)][move_to(move)] += depth * depth

    # --- Principal variation extraction (architecture.md §9.3) --------------
    #
    # NOTE: `_extract_pv` is no longer called from the main search loop --
    # the triangular PV array (`ctx.pv[0][:ctx.pv_length[0]]`) replaced it
    # as the primary PV source (Milestone 5). Kept here for
    # debugging/testing (e.g. comparing TT-based vs. triangular PV
    # stability).

    def _extract_pv(self, board: Board, depth: int) -> list[int]:
        """Walks the TT from the current position following each node's
        stored `best_move`, making/unmaking as it goes. Can be shorter than
        `depth`, or slightly unstable across iterations if TT entries were
        overwritten mid-line — a known, accepted Milestone-2/3 limitation
        (a triangular PV array is the standard Milestone 5 fix)."""
        pv: list[int] = []
        made = 0
        seen_keys: set[int] = set()
        try:
            while len(pv) < depth:
                entry = self.tt.probe(board.zobrist_hash)
                if entry is None or entry.best_move == NULL_MOVE:
                    break
                if board.zobrist_hash in seen_keys:
                    break  # guard against a cycle of TT entries
                seen_keys.add(board.zobrist_hash)
                move = entry.best_move
                if move not in generate_legal_moves(board):
                    break  # stale/mismatched TT entry: stop rather than apply a bogus move
                board.make_move(move)
                made += 1
                pv.append(move)
        finally:
            for _ in range(made):
                board.unmake_move()
        return pv


def _has_non_pawn_material(board: Board, color: int) -> bool:
    """True if `color` has at least one knight, bishop, rook, or queen —
    null-move pruning's standard zugzwang-avoidance guard (architecture.md
    §9): with only pawns and a king left, "passing" can genuinely be the best
    or only reasonable try, so a null-move cutoff there would be unsound."""
    p = board.pieces[color]
    return (p[KNIGHT] | p[BISHOP] | p[ROOK] | p[QUEEN]) != 0


def see_ge(board: Board, move: int, threshold: int) -> bool:
    """Static Exchange Evaluation (architecture.md §9.5): estimates the net
    material swing, in centipawns, of playing `move` and then letting both
    sides recapture on its destination square, cheapest attacker first,
    for as long as doing so still helps the side to move — and returns
    whether that estimate is `>= threshold`.

    Built on `attacks.attackers_to` (architecture.md §5.4): at each step of
    the simulated exchange, the attacker list for the square is recomputed
    against a scratch occupancy bitboard with already-"used" pieces
    cleared. Because `attackers_to` re-derives sliding attacks by
    ray-scanning against whatever occupancy it is given, this naturally
    exposes an x-ray attacker behind a piece that has just been "captured"
    off, with no separate x-ray bookkeeping needed.

    Deliberately conservative/simplified per §9.5: it does not check
    whether a defender is itself pinned to its king (an absolutely pinned
    piece cannot legally recapture, but is still counted here as if it
    could), and it does not model a pawn promoting mid-exchange (only the
    move being evaluated itself gets promotion's value swing). A too-
    conservative or even buggy SEE only ever costs quiescence search some
    speed — searching a capture it needn't have, or skipping one it could
    have kept — never correctness, since it never changes which moves are
    legal, only which ones quiescence bothers to search.
    """
    frm, to, flag = move_from(move), move_to(move), move_flag(move)
    us = board.side_to_move
    them = 1 - us

    if flag == EN_PASSANT:
        captured_sq = to - 8 if us == WHITE else to + 8
        gain = PIECE_VALUE[PAWN]
    else:
        captured_sq = to
        victim = board.mailbox[to]
        gain = PIECE_VALUE[piece_type_of(victim)] if victim != NO_PIECE else 0

    attacker_value = PIECE_VALUE[piece_type_of(board.mailbox[frm])]
    if flag in PROMO_PIECE_OF:
        # The pawn becomes the promoted piece on `to`: both its own
        # material swing and its new (higher) recapture value are part of
        # this move, not of a later step in the exchange.
        promoted_value = PIECE_VALUE[PROMO_PIECE_OF[flag]]
        gain += promoted_value - attacker_value
        attacker_value = promoted_value

    # Scratch occupancy for the simulated exchange: the moving piece has
    # left `frm` (and, for en passant, the captured pawn is gone from a
    # square that differs from `to`); everything else starts as the real
    # board's occupancy.
    occ = board.occupied & ~(1 << frm)
    if flag == EN_PASSANT:
        occ &= ~(1 << captured_sq)

    return gain - _see_recapture(board, to, them, occ, attacker_value) >= threshold


def _see_recapture(board: Board, sq: int, side: int, occ: int, hanging_value: int) -> int:
    """The best net material `side` can gain by recapturing on `sq` with its
    cheapest available attacker (given the scratch occupancy `occ`, and
    `hanging_value` = the value of whatever currently sits on `sq` and would
    be captured), assuming both sides always keep recapturing
    cheapest-attacker-first for as long as it still helps them.

    Returns 0 if `side` has no attacker left on `sq`, or if recapturing
    would only make things worse for `side` — a side is never forced to
    continue a losing exchange, so a losing recapture simply isn't taken.
    """
    attackers = attackers_to(board, sq, side, occupied=occ) & occ
    if not attackers:
        return 0
    least_sq = min(iter_bits(attackers), key=lambda s: PIECE_VALUE[piece_type_of(board.mailbox[s])])
    least_value = PIECE_VALUE[piece_type_of(board.mailbox[least_sq])]
    occ_after = occ & ~(1 << least_sq)
    net = hanging_value - _see_recapture(board, sq, 1 - side, occ_after, least_value)
    return max(0, net)


def _mvv_lva_score(board: Board, move: int) -> int:
    """`victim_value * 16 - attacker_value`, from the same `PIECE_VALUE`
    table evaluation uses (architecture.md §9.4), so "queen takes pawn"
    sorts after "pawn takes queen." En passant's victim is a pawn even
    though its captured square differs from `to`."""
    attacker_type = piece_type_of(board.mailbox[move_from(move)])
    flag = move_flag(move)
    if flag == EN_PASSANT:
        victim_type = PAWN
    else:
        victim_type = piece_type_of(board.mailbox[move_to(move)])
    return PIECE_VALUE[victim_type] * 16 - PIECE_VALUE[attacker_type]
