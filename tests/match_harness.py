"""Reusable engine-vs-engine self-play match harness for Milestone 5 A/B
evaluation gates (architecture.md §10.2, §15).

§15's Milestone 5 exit criteria require that "each new evaluation term
passes its own unit test **and** an engine-vs-engine match (self-play at
fixed depth, or vs. a fixed reference opponent) shows a non-negative trend
before being kept, since a term 'correct' in isolation can still lose
strength through interaction effects." This module is the one, generic
implementation of that match machinery, shared by every such gate (mobility,
king safety, pawn structure, and any later search extension gated the same
way per §15's last bullet) rather than being reimplemented per term.

This is a plain helper module, not a test file itself (no `test_` prefix,
so pytest never tries to collect it) -- other test files do
`from match_harness import play_match, MatchResult, DEFAULT_POSITIONS`.

Genericity: nothing here is hardcoded to one specific evaluation term.
`play_match` takes any two objects satisfying `evaluate.py`'s one-method
`Evaluator` protocol (architecture.md §10) -- e.g. `default_evaluator()`
for the baseline vs. a `CompositeEvaluator` with a new term's weight turned
on -- and reports which one scored better across a fixed battery of
starting positions, playing both colors of each so first-move asymmetry
cancels out.

Kept deliberately fast/shallow (depth 2-3, a modest ply cap): this is meant
to give a fast, fixed-depth CI A/B *signal* ("did this term help or hurt"),
not a rigorous Elo estimate -- that distinction is what §15 itself asks for
("self-play at fixed depth"), and is why `SearchLimits(max_depth=2)` or
`max_depth=3` is the expected caller-supplied `limits` value, not a long
time control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chessengine.board import Board
from chessengine.constants import BLACK, WHITE
from chessengine.evaluate import Evaluator
from chessengine.fen import parse_fen
from chessengine.move import move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits

# --- Default position battery (architecture.md §14) -------------------------
#
# Reusing the six §14 perft reference positions gives a small, already-
# vetted-for-variety battery for free: the standard startpos, a busy
# middlegame with both castling rights intact (Kiwipete), a near-empty,
# highly tactical king-and-pawn/pin position, two asymmetric
# castling-rights/promotion-heavy middlegames, and a fully-developed,
# roughly balanced middlegame. Between them they exercise quiet, tactical,
# simplified, and asymmetric material/king-safety situations without
# reaching for an unrelated position set.

DEFAULT_POSITIONS: list[str] = [
    # 1. Startpos
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 2. Kiwipete: busy middlegame, both sides can still castle either way
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # 3. Position 3: near-empty board, tactical king/pawn/rook position
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    # 4. Position 4: asymmetric, one side already castled, promotion-heavy
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    # 5. Position 5: asymmetric material and castling rights
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    # 6. Position 6: fully developed, roughly balanced middlegame
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


# --- Single-game playout ------------------------------------------------------


def play_game(
    white_search: Search,
    black_search: Search,
    board: Board,
    limits: SearchLimits,
    ply_cap: int = 200,
) -> str:
    """Play one game to completion (or until `ply_cap` plies have been
    played), alternately calling `white_search.search(...)`/
    `black_search.search(...)` and `board.make_move(...)`.

    `board` is mutated in place and left at the game's final position.
    Returns `"1-0"`, `"0-1"`, or `"1/2-1/2"`.

    Termination, checked before each move is generated (mirrors the
    self-play smoke test in `tests/test_search.py`):
    - no legal moves + `board.in_check()`: checkmate, the side to move
      (which has no legal moves) loses;
    - no legal moves + not in check: stalemate, a draw;
    - `board.is_fifty_move_draw()` or `board.is_repetition_draw()`: a draw;
    - `ply_cap` plies played with the game still going: a draw (the
      fixed-depth/short-cap regime this harness runs under has no
      obligation to ever reach a "real" game-ending result).
    """
    for _ in range(ply_cap):
        legal_moves = generate_legal_moves(board)
        if not legal_moves:
            if board.in_check():
                # Side to move is checkmated: the other side wins.
                return "0-1" if board.side_to_move == WHITE else "1-0"
            return "1/2-1/2"  # stalemate
        if board.is_fifty_move_draw() or board.is_repetition_draw():
            return "1/2-1/2"

        mover = white_search if board.side_to_move == WHITE else black_search
        move = mover.search(board, limits).best_move
        if move not in legal_moves:
            # Search's contract (architecture.md §9.3) guarantees a legal
            # move whenever one exists; surface a loud, specific failure
            # rather than silently playing something else if that contract
            # is ever violated.
            raise RuntimeError(
                f"search returned a move ({move_to_uci(move)!r}) not in the "
                f"legal move list for position {board.to_fen()!r}"
            )
        board.make_move(move)
    return "1/2-1/2"  # ply cap reached: treated as a draw


# --- Match result -------------------------------------------------------------


@dataclass
class MatchResult:
    """Accumulated outcome of a `play_match` run.

    `score_a`/`score_b` use standard chess scoring (1 win / 0.5 draw / 0
    loss) summed across *every* game engine A (resp. B) played, regardless
    of which color it had that game -- so a caller can compare them
    directly to see which evaluator/search configuration came out ahead.
    """

    score_a: float = 0.0
    score_b: float = 0.0
    games: list[dict[str, Any]] = field(default_factory=list)


_RESULT_SCORES = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0), "1/2-1/2": (0.5, 0.5)}


# --- Full match: every position, both colors -----------------------------------


def play_match(
    evaluator_a: Evaluator,
    evaluator_b: Evaluator,
    positions: list[str],
    limits: SearchLimits,
    ply_cap: int = 200,
) -> MatchResult:
    """Play every FEN in `positions` twice: once with a fresh
    `Search(evaluator_a)` as White vs. a fresh `Search(evaluator_b)` as
    Black, and once with the colors swapped, so first-move asymmetry
    cancels out across the pair. Accumulates every game into one
    `MatchResult`.

    A fresh `Search` (fresh TT, killers, history) is constructed for each
    color in each game, so no state leaks between games or between colors
    -- each game is an independent sample.
    """
    result = MatchResult()
    for fen in positions:
        for a_plays_white in (True, False):
            board = parse_fen(fen)
            white_evaluator = evaluator_a if a_plays_white else evaluator_b
            black_evaluator = evaluator_b if a_plays_white else evaluator_a
            white_search = Search(white_evaluator)
            black_search = Search(black_evaluator)

            outcome = play_game(white_search, black_search, board, limits, ply_cap)

            white_score, black_score = _RESULT_SCORES[outcome]
            a_score, b_score = (
                (white_score, black_score) if a_plays_white else (black_score, white_score)
            )
            result.score_a += a_score
            result.score_b += b_score
            result.games.append(
                {
                    "fen": fen,
                    "white": "a" if a_plays_white else "b",
                    "result": outcome,
                }
            )
    return result
