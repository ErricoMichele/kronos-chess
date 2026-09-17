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
from .constants import DRAW_SCORE, INF, MATE_SCORE, MAX_PLY, NO_PIECE, PAWN, WHITE, piece_type_of
from .evaluate import PIECE_VALUE, Evaluator
from .move import (
    EN_PASSANT,
    NULL_MOVE,
    PROMO_PIECE_OF,
    is_capture,
    move_flag,
    move_from,
    move_to,
)
from .movegen import generate_captures, generate_legal_moves
from .transposition import TTFlag, TranspositionTable, score_from_tt, score_to_tt

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

        for depth in range(1, limits.max_depth + 1):
            score = self._negamax(board, depth, -INF, INF, 0, ctx)
            if ctx.should_stop():
                break  # partial/unreliable result from an aborted depth: discard entirely
            pv = self._extract_pv(board, depth)
            best = SearchResult(pv[0] if pv else best.best_move, score, depth, ctx.nodes, pv)
            if on_info is not None:
                on_info(SearchInfo(depth, score, ctx.nodes, pv))
            if abs(score) >= MATE_SCORE - 128:
                break  # forced mate found; no point searching deeper
        return best

    # --- Negamax core (architecture.md §9.1) --------------------------------

    def _negamax(
        self, board: Board, depth: int, alpha: int, beta: int, ply: int, ctx: _SearchCtx
    ) -> int:
        ctx.nodes += 1
        if ctx.should_stop():
            return 0  # discarded: caller checks ctx.should_stop()

        if board.is_fifty_move_draw() or board.is_repetition_draw():
            return DRAW_SCORE

        alpha_orig = alpha
        entry = self.tt.probe(board.zobrist_hash)
        tt_move = entry.best_move if entry is not None else NULL_MOVE
        if entry is not None and entry.depth >= depth:
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

        moves = generate_legal_moves(board)
        if not moves:
            return -MATE_SCORE + ply if board.in_check() else DRAW_SCORE

        self._order_moves(moves, board, tt_move, ply)

        best_score, best_move = -INF, moves[0]
        for move in moves:
            board.make_move(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, ctx)
            board.unmake_move()
            if ctx.should_stop():
                return 0
            if score > best_score:
                best_score, best_move = score, move
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
