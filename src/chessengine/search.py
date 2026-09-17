"""Negamax + alpha-beta search with iterative deepening (architecture.md §9).

Per the module-boundary DAG (architecture.md §11), `search.py` depends on
`constants`, `board`, `movegen`, `evaluate`, `transposition`, and `move` —
it is the only module that depends on `movegen.py` + `evaluate.py` +
`transposition.py` together.

**Milestone 2 scope.** This module implements §9.1 (negamax + alpha-beta),
§9.2 (TT usage), §9.3 (iterative deepening / time control), §9.4 (move
ordering: TT move, MVV-LVA, killers, history), and §9.6 (draws). Quiescence
search (§9.5, `_quiescence`/`see_ge`) is explicitly out of scope until
Milestone 4: at `depth == 0` this module calls `self.evaluator.evaluate(board)`
directly, a plain static-eval leaf, not a quiescence search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import threading

from .board import Board
from .constants import DRAW_SCORE, INF, MATE_SCORE, MAX_PLY, PAWN, piece_type_of
from .evaluate import PIECE_VALUE, Evaluator
from .move import EN_PASSANT, NULL_MOVE, is_capture, move_flag, move_from, move_to
from .movegen import generate_legal_moves
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
        best = SearchResult(NULL_MOVE, 0, 0, 0, [])
        for depth in range(1, limits.max_depth + 1):
            score = self._negamax(board, depth, -INF, INF, 0, ctx)
            if ctx.should_stop() and depth > 1:
                break  # partial/unreliable result from an aborted depth: discard
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
            # Milestone 2 scope: a plain static-eval leaf, not quiescence
            # (quiescence/_see_ge are Milestone 4, architecture.md §9.5).
            return self.evaluator.evaluate(board)

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
