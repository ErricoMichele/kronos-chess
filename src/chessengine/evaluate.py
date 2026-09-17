"""Evaluation interface: the `Evaluator` protocol, material+PST (the only
enabled term through Milestone 4), and `CompositeEvaluator`'s extension path
(architecture.md §10).

Per the module-boundary DAG (architecture.md §11), `evaluate.py` depends only
on `constants` and `board` (read-only queries: piece bitboards, side to
move) — never on `movegen.py` or `search.py`. That is what makes evaluation
terms unit-testable against a bare `Board` with no search machinery
involved, and what lets `search.py` depend on the one-method `Evaluator`
protocol without ever depending on a concrete evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .attacks import KNIGHT_ATTACKS, bishop_attacks, queen_attacks, rook_attacks
from .bitboard import iter_bits, popcount
from .board import Board
from .constants import BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE


class Evaluator(Protocol):
    def evaluate(self, board: Board) -> int:
        """Centipawns, from the side-to-move's perspective (negamax
        convention: positive is good for whoever is to move)."""
        ...


TermFn = Callable[[Board], int]  # a pure evaluation term, side-to-move-relative centipawns


# --- 10.1 Material + PST (the only enabled term through Milestone 4) --------
#
# Material values in centipawns, plus one 64-entry piece-square table (PST)
# per piece type, defined from White's point of view with index 0 = a1 and
# index 63 = h8 (matching board indexing, §3.1); Black's lookup mirrors the
# square vertically via `sq ^ 56` (flips the rank, keeps the file) instead of
# maintaining a second table.
#
# The row order below runs **rank 1 to rank 8, top to bottom**, matching
# a1=0 indexing — the opposite of how these tables are usually printed in
# chess references (rank 8 first). Transcribing a reference table's printed
# row order directly into an a1=0 array is a classic off-by-mirror bug (it
# rewards White's pawns for sitting on their *starting* rank instead of
# their promotion rank); each table below has been reversed from its
# conventional rank-8-first printing into rank-1-first order.

PIECE_VALUE = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}

PAWN_PST = (
      0,   0,   0,   0,   0,   0,   0,   0,   # rank 1
      5,  10,  10, -20, -20,  10,  10,   5,   # rank 2 (starting rank)
      5,  -5, -10,   0,   0, -10,  -5,   5,   # rank 3
      0,   0,   0,  20,  20,   0,   0,   0,   # rank 4
      5,   5,  10,  25,  25,  10,   5,   5,   # rank 5
     10,  10,  20,  30,  30,  20,  10,  10,   # rank 6
     50,  50,  50,  50,  50,  50,  50,  50,   # rank 7 (one step from promotion)
      0,   0,   0,   0,   0,   0,   0,   0,   # rank 8
)

KNIGHT_PST = (
    -50, -40, -30, -30, -30, -30, -40, -50,   # rank 1
    -40, -20,   0,   5,   5,   0, -20, -40,   # rank 2
    -30,   5,  10,  15,  15,  10,   5, -30,   # rank 3
    -30,   0,  15,  20,  20,  15,   0, -30,   # rank 4
    -30,   5,  15,  20,  20,  15,   5, -30,   # rank 5
    -30,   0,  10,  15,  15,  10,   0, -30,   # rank 6
    -40, -20,   0,   0,   0,   0, -20, -40,   # rank 7
    -50, -40, -30, -30, -30, -30, -40, -50,   # rank 8
)

BISHOP_PST = (
    -20, -10, -10, -10, -10, -10, -10, -20,   # rank 1
    -10,   5,   0,   0,   0,   0,   5, -10,   # rank 2
    -10,  10,  10,  10,  10,  10,  10, -10,   # rank 3
    -10,   0,  10,  10,  10,  10,   0, -10,   # rank 4
    -10,   5,   5,  10,  10,   5,   5, -10,   # rank 5
    -10,   0,   5,  10,  10,   5,   0, -10,   # rank 6
    -10,   0,   0,   0,   0,   0,   0, -10,   # rank 7
    -20, -10, -10, -10, -10, -10, -10, -20,   # rank 8
)

ROOK_PST = (
      0,   0,   0,   5,   5,   0,   0,   0,   # rank 1
     -5,   0,   0,   0,   0,   0,   0,  -5,   # rank 2
     -5,   0,   0,   0,   0,   0,   0,  -5,   # rank 3
     -5,   0,   0,   0,   0,   0,   0,  -5,   # rank 4
     -5,   0,   0,   0,   0,   0,   0,  -5,   # rank 5
     -5,   0,   0,   0,   0,   0,   0,  -5,   # rank 6
      5,  10,  10,  10,  10,  10,  10,   5,   # rank 7
      0,   0,   0,   0,   0,   0,   0,   0,   # rank 8
)

QUEEN_PST = (
    -20, -10, -10,  -5,  -5, -10, -10, -20,   # rank 1
    -10,   0,   5,   0,   0,   0,   0, -10,   # rank 2
    -10,   5,   5,   5,   5,   5,   0, -10,   # rank 3
      0,   0,   5,   5,   5,   5,   0,  -5,   # rank 4
     -5,   0,   5,   5,   5,   5,   0,  -5,   # rank 5
    -10,   0,   5,   5,   5,   5,   0, -10,   # rank 6
    -10,   0,   0,   0,   0,   0,   0, -10,   # rank 7
    -20, -10, -10,  -5,  -5, -10, -10, -20,   # rank 8
)

KING_PST = (
     20,  30,  10,   0,   0,  10,  30,  20,   # rank 1
     20,  20,   0,   0,   0,   0,  20,  20,   # rank 2
    -10, -20, -20, -20, -20, -20, -20, -10,   # rank 3
    -20, -30, -30, -40, -40, -30, -30, -20,   # rank 4
    -30, -40, -40, -50, -50, -40, -40, -30,   # rank 5
    -30, -40, -40, -50, -50, -40, -40, -30,   # rank 6
    -30, -40, -40, -50, -50, -40, -40, -30,   # rank 7
    -30, -40, -40, -50, -50, -40, -40, -30,   # rank 8
)

PST = {
    PAWN: PAWN_PST,
    KNIGHT: KNIGHT_PST,
    BISHOP: BISHOP_PST,
    ROOK: ROOK_PST,
    QUEEN: QUEEN_PST,
    KING: KING_PST,
}


def _side_score(board: Board, color: int) -> int:
    score, mirror = 0, color == BLACK
    for ptype in range(6):
        table = PST[ptype]
        for sq in iter_bits(board.pieces[color][ptype]):
            score += PIECE_VALUE[ptype] + (table[sq ^ 56] if mirror else table[sq])
    return score


def material_pst_term(board: Board) -> int:
    white_score = _side_score(board, WHITE) - _side_score(board, BLACK)
    return white_score if board.side_to_move == WHITE else -white_score


# --- Mobility (Milestone 5, enabled via Weights.mobility after its A/B gate) -
#
# "Mobility" for a color is the total count of squares attacked by that
# color's knights/bishops/rooks/queens (pawns and king are excluded: pawn
# "attacks" are captures rather than mobility in the usual sense, and king
# attacks are near-constant and already the concern of a separate king-safety
# term). This calls attacks.py's attack-table primitives directly — the
# knight leaper table plus bishop_attacks/rook_attacks/queen_attacks against
# the board's full `occupied` bitboard — rather than movegen.py's
# pseudo-legal/legal move generation, because movegen.py only ever generates
# moves for `board.side_to_move`; this term must be computable for *both*
# colors regardless of whose turn it is (following the same DAG-respecting
# pattern as material_pst_term, just reaching one module further upstream to
# attacks.py, which is still strictly before evaluate.py in the dependency
# order per architecture.md §11: constants/bitboard -> attacks -> board ->
# ... -> evaluate).
#
# Scale: 3 centipawns per extra attacked square, the middle of the commonly
# used ~2-5 cp/square range for a simple attacked-square-count mobility term
# (finer per-piece-type mobility tables exist, e.g. PeSTO-style, but a flat
# per-square value is a reasonable, easily-explainable starting point). This
# scale was deliberately not hand-tuned before being validated: an A/B
# self-play match (tests/match_harness.py, DEFAULT_POSITIONS, depth 3) against
# a material+PST-only baseline scored 9.0/12 vs 3.0/12 at Weights.mobility=1.0,
# a clear non-negative trend, so it's kept at that weight below.
MOBILITY_CP_PER_SQUARE = 3


def _side_mobility(board: Board, color: int) -> int:
    occ = board.occupied
    p = board.pieces[color]
    total = 0
    for sq in iter_bits(p[KNIGHT]):
        total += popcount(KNIGHT_ATTACKS[sq])
    for sq in iter_bits(p[BISHOP]):
        total += popcount(bishop_attacks(sq, occ))
    for sq in iter_bits(p[ROOK]):
        total += popcount(rook_attacks(sq, occ))
    for sq in iter_bits(p[QUEEN]):
        total += popcount(queen_attacks(sq, occ))
    return total


def mobility_term(board: Board) -> int:
    white_mobility_squares = _side_mobility(board, WHITE) - _side_mobility(board, BLACK)
    white_score = white_mobility_squares * MOBILITY_CP_PER_SQUARE
    return white_score if board.side_to_move == WHITE else -white_score


# --- 10.2 `CompositeEvaluator` and the extension path -----------------------


@dataclass
class Weights:
    material_pst: float = 1.0
    # Enabled at 1.0 per an A/B self-play match against a material+PST-only
    # baseline (tests/match_harness.py, DEFAULT_POSITIONS, depth 3): the
    # mobility-enabled candidate scored 9.0/12 vs the baseline's 3.0/12, a
    # clear non-negative (in fact strongly positive) trend per the
    # architecture.md §10.2/§15 gate.
    mobility: float = 1.0
    king_safety: float = 0.0
    pawn_structure: float = 0.0


class CompositeEvaluator:
    """Sums independently-testable term functions by weight. Adding a term
    is: write a new pure `*_term(board) -> int` function, register it here,
    add one `Weights` field — search.py never changes."""

    def __init__(self, terms: dict[str, TermFn], weights: Weights | None = None) -> None:
        self.terms = terms
        self.weights = weights or Weights()

    def evaluate(self, board: Board) -> int:
        total = 0
        for name, term_fn in self.terms.items():
            w = getattr(self.weights, name, 0.0)
            if w:
                total += int(w * term_fn(board))
        return total


def default_evaluator() -> CompositeEvaluator:
    """Material+PST and mobility are both enabled. `mobility_term` passed its
    A/B gate (see `Weights.mobility`'s docstring comment above) and is kept
    at its validated weight; king_safety/pawn_structure remain unimplemented
    (Weights defaults to 0.0 for those) until their own terms and A/B gates
    land."""
    return CompositeEvaluator(
        terms={"material_pst": material_pst_term, "mobility": mobility_term},
        weights=Weights(material_pst=1.0, mobility=1.0),
    )
