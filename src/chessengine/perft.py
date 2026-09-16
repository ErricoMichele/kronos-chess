"""Perft (performance test / path-enumeration test): the move-generator
correctness oracle (architecture.md §13, §14).

`perft(board, depth)` counts the number of leaf nodes reachable from
`board` by playing exactly `depth` plies of *legal* moves, with no pruning
or evaluation of any kind — every leaf at the target depth is counted once,
regardless of the position it represents (checkmate, stalemate, or an
ordinary position). Matched exactly against the published node counts in
§14, this is the single most important correctness gate in the project: a
wrong count means some rule (pin, check evasion, castling right bookkeeping,
en passant, promotion) was generated incorrectly somewhere in `movegen.py`.

`divide(board, depth)` is perft's standard debugging companion: it reports
the perft count broken down by root move, so a mismatch against a reference
value can be bisected — a human (or another perft-capable reference engine)
compares the per-move breakdown, finds the one root move whose subtree count
disagrees, then re-runs `divide` one ply deeper *inside* that subtree, and so
on, until the exact offending move is isolated.

Per the module-boundary DAG (architecture.md §11), `perft.py` depends only
on `board` and `movegen` — nothing below it in the DAG ever imports
`perft.py`.
"""

from __future__ import annotations

from .board import Board
from .move import move_to_uci
from .movegen import generate_legal_moves


def perft(board: Board, depth: int) -> int:
    """Count leaf nodes reachable from `board` after exactly `depth` plies
    of legal moves.

    `depth <= 0` counts the current position itself as the single leaf
    (the standard perft(0) == 1 base case). At `depth == 1`, the count is
    just the number of legal moves — no need to make/unmake each one only
    to immediately count its (trivially size-1) sub-perft, so that case is
    special-cased purely for speed, never for correctness (the general
    recursive case below would compute the exact same number).
    """
    if depth <= 0:
        return 1

    moves = generate_legal_moves(board)
    if depth == 1:
        return len(moves)

    nodes = 0
    for move in moves:
        board.make_move(move)
        nodes += perft(board, depth - 1)
        board.unmake_move()
    return nodes


def divide(board: Board, depth: int) -> dict[str, int]:
    """Perft broken down by root move: `{uci_move: subtree_node_count}`.

    Each root move is played once, its subtree counted via `perft` at
    `depth - 1`, then unmade — so the returned counts sum to
    `perft(board, depth)`. Two same-position moves that are otherwise
    indistinguishable in UCI notation cannot both occur among a position's
    legal moves (UCI's `from` + `to` + promotion letter always uniquely
    identifies one legal move in a given position, per `move_to_uci`/
    `move_from_uci`), so the dict never silently drops or overwrites a
    root move's count.

    `depth <= 0` returns an empty breakdown: there is no "per root move"
    split of a zero-ply search, only the single leaf `perft` itself counts.
    """
    if depth <= 0:
        return {}

    breakdown: dict[str, int] = {}
    for move in generate_legal_moves(board):
        board.make_move(move)
        breakdown[move_to_uci(move)] = perft(board, depth - 1)
        board.unmake_move()
    return breakdown
