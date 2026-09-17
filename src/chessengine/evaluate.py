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

from . import attacks
from .attacks import KING_ATTACKS, KNIGHT_ATTACKS, bishop_attacks, queen_attacks, rook_attacks
from .bitboard import iter_bits, popcount
from .board import Board
from .constants import (
    BISHOP,
    BISHOP_PST,
    BLACK,
    FILE_MASK,
    KING,
    KING_PST,
    KNIGHT,
    KNIGHT_PST,
    PAWN,
    PAWN_PST,
    PIECE_VALUE,
    PST,
    QUEEN,
    QUEEN_PST,
    RANK_MASK,
    ROOK,
    ROOK_PST,
    WHITE,
    file_of,
    rank_of,
)


class Evaluator(Protocol):
    def evaluate(self, board: Board) -> int:
        """Centipawns, from the side-to-move's perspective (negamax
        convention: positive is good for whoever is to move)."""
        ...


TermFn = Callable[[Board], int]  # a pure evaluation term, side-to-move-relative centipawns


# --- 10.1 Material + PST (the only enabled term through Milestone 4) --------
#
# PIECE_VALUE/PAWN_PST/KNIGHT_PST/BISHOP_PST/ROOK_PST/QUEEN_PST/KING_PST/PST
# live in constants.py (not here) so board.py's make_move/unmake_move hot
# path can read them for the incremental `material_pst_score` running total
# (architecture.md §10.1's closing note) without board.py depending on
# evaluate.py, which would invert the module DAG (§11). Re-imported here
# (see the `from .constants import (...)` above) purely so existing
# `from chessengine.evaluate import KNIGHT_PST`-style call sites (e.g.
# tests/test_evaluate.py) keep working unchanged.


def _side_score(board: Board, color: int) -> int:
    """From-scratch total, independent of `board.material_pst_score`'s
    incremental bookkeeping. Kept as the differential-test oracle for the
    incremental running total (architecture.md §10.1's closing note) — never
    called from `material_pst_term` itself any more."""
    score, mirror = 0, color == BLACK
    for ptype in range(6):
        table = PST[ptype]
        for sq in iter_bits(board.pieces[color][ptype]):
            score += PIECE_VALUE[ptype] + (table[sq ^ 56] if mirror else table[sq])
    return score


def material_pst_term(board: Board) -> int:
    """O(1) read of the incremental running total `board.py` maintains
    inside `_add_piece`/`_remove_piece` (architecture.md §10.1's closing
    note) — same White-relative-then-negamax-flipped contract `_side_score`
    used to compute from scratch every call."""
    return board.material_pst_score if board.side_to_move == WHITE else -board.material_pst_score


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


# --- King safety (Milestone 5, enabled after tuning + A/B gate) ---------------
#
# Two standard, simple components, combined into one term:
#
#   (a) Pawn shield: count of the king's *own* pawns on the 3 squares
#       directly in front of the king — its file and the two adjacent
#       files, one rank towards the enemy (rank+1 for White, rank-1 for
#       Black) — clipped at the board edge, so a king on the a-file or
#       h-file only ever has 2 possible shield squares, not 3.
#
#   (b) King-zone attackers: count of enemy pieces attacking any square in
#       the king's immediate 8-neighborhood ("king zone"), i.e.
#       `KING_ATTACKS[king_sq]` — the same leaper table attacks.py already
#       builds for king moves — via `attacks.attackers_to`, one of
#       attacks.py's own attacked-square-query primitives (§5.4), against
#       the board's actual occupancy. This is the same DAG-respecting
#       pattern `mobility_term` follows above: attacks.py's primitives are
#       called directly, never movegen.py's side-to-move-bound
#       pseudo-legal/legal move generation, so this term stays computable
#       for *both* colors regardless of whose turn it is.
#
# Scale: +15 cp per shield pawn, -10 cp per enemy piece attacking a
# king-zone square (a flat penalty per attacking piece, not weighted by
# attacker value — a simple, easily-explainable starting point, same spirit
# as MOBILITY_CP_PER_SQUARE above; a piece attacking several king-zone
# squares at once is counted once per square it attacks, which is standard
# for this kind of "attack units" heuristic). Originally 20 cp/attacker,
# which proved too aggressive in A/B testing (many pieces naturally attack
# king-zone squares in normal middlegame positions, creating noisy, large
# penalties); retuned to 10 cp/attacker, which passed its A/B gate
# (12.5/22 vs baseline's 9.5/22 at depth 3).
KING_SHIELD_CP_PER_PAWN = 15
KING_ZONE_CP_PER_ATTACKER = 10


def _king_safety_score(board: Board, color: int) -> int:
    king_sq = board.king_square(color)
    king_file, king_rank = file_of(king_sq), rank_of(king_sq)

    shield_rank = king_rank + 1 if color == WHITE else king_rank - 1
    shield = 0
    if 0 <= shield_rank < 8:
        own_pawns = board.pieces[color][PAWN]
        for f in (king_file - 1, king_file, king_file + 1):
            if 0 <= f < 8:
                sq = shield_rank * 8 + f
                if own_pawns & (1 << sq):
                    shield += 1

    enemy = 1 - color
    zone_attackers = 0
    for sq in iter_bits(KING_ATTACKS[king_sq]):
        zone_attackers += popcount(attacks.attackers_to(board, sq, enemy))

    return shield * KING_SHIELD_CP_PER_PAWN - zone_attackers * KING_ZONE_CP_PER_ATTACKER


def king_safety_term(board: Board) -> int:
    white_score = _king_safety_score(board, WHITE) - _king_safety_score(board, BLACK)
    return white_score if board.side_to_move == WHITE else -white_score


# --- Pawn structure (implemented, gated off via Weights.pawn_structure until --
# --- its own A/B match) ------------------------------------------------------
#
# Three standard, simple components, combined into one term, computed
# directly from `board.pieces[color][PAWN]` bitboards plus FILE_MASK/
# RANK_MASK (constants.py) -- no movegen.py dependency, same DAG-respecting
# pattern as mobility_term/king_safety_term above (this term doesn't even
# need to reach as far as attacks.py: file/rank masks are enough).
#
#   (a) Doubled pawns: on a given file, every friendly pawn beyond the
#       first is "doubled" -- they block each other's advance and, between
#       them, defend fewer squares than two pawns on separate files would.
#   (b) Isolated pawns: a pawn with no friendly pawn on either adjacent
#       file can never be defended by another pawn for the rest of the
#       game -- a permanent structural weakness, independent of anything
#       else on the board.
#   (c) Passed pawns: a pawn with no enemy pawn anywhere on its own file or
#       either adjacent file, from its current rank all the way to its
#       promotion square, can only ever be stopped by a piece, never by a
#       pawn -- the single most important structural feature in the
#       endgame. The bonus is scaled by how many squares remain to
#       promotion, since a passed pawn's danger grows sharply, not
#       linearly, as it advances (a 7th-rank passer is often worth close
#       to a minor piece; a 2nd-rank one is a long-term asset at most).
#
# Scales (centipawns; simple starting points in the same spirit as
# MOBILITY_CP_PER_SQUARE/KING_SHIELD_CP_PER_PAWN above -- not hand-tuned,
# pending an A/B self-play gate via tests/match_harness.py before
# Weights.pawn_structure moves off 0.0):
#   - DOUBLED_PAWN_PENALTY_CP: -12 cp per pawn beyond the first on a file.
#     Kept small: doubled pawns are a real but mild weakness, and this
#     already penalizes each extra pawn on the file individually, so a
#     triple-doubled file naturally costs proportionally more without a
#     separate multiplier.
#   - ISOLATED_PAWN_PENALTY_CP: -15 cp per isolated pawn. A little larger
#     than the doubled penalty -- an isolated pawn is a standing target for
#     the whole game, not just a local inefficiency.
#   - PASSED_PAWN_BONUS_BY_DISTANCE: indexed by squares-still-to-travel to
#     the promotion square (0 = the pawn's next push promotes it), so the
#     bonus grows sharply near promotion rather than growing linearly with
#     rank.
DOUBLED_PAWN_PENALTY_CP = 12
ISOLATED_PAWN_PENALTY_CP = 15
PASSED_PAWN_BONUS_BY_DISTANCE = (200, 150, 100, 60, 35, 20, 10, 0)
# index 0 = one square from promotion ... index 6 = a pawn still on its own
# starting rank; index 7 is unused in practice (kept so any distance in
# [0, 7] safely indexes the table) since a pawn cannot stand on the
# promotion rank itself without having already promoted.

# Precomputed once at import time (same style as attacks.py's RAY_ATTACKS):
# _AHEAD_RANK_MASK[color][r] = union of RANK_MASK for every rank strictly
# between rank r and that color's promotion rank, exclusive of r itself.
_AHEAD_RANK_MASK: dict[int, list[int]] = {WHITE: [0] * 8, BLACK: [0] * 8}
for _r in range(8):
    for _rr in range(_r + 1, 8):
        _AHEAD_RANK_MASK[WHITE][_r] |= RANK_MASK[_rr]
    for _rr in range(0, _r):
        _AHEAD_RANK_MASK[BLACK][_r] |= RANK_MASK[_rr]
del _r, _rr


def _side_pawn_structure(board: Board, color: int) -> int:
    pawns = board.pieces[color][PAWN]
    enemy_pawns = board.pieces[1 - color][PAWN]
    ahead_rank_mask = _AHEAD_RANK_MASK[color]
    score = 0

    # (a) Doubled pawns.
    for f in range(8):
        count = popcount(pawns & FILE_MASK[f])
        if count > 1:
            score -= (count - 1) * DOUBLED_PAWN_PENALTY_CP

    for sq in iter_bits(pawns):
        f, r = file_of(sq), rank_of(sq)
        adjacent_files = 0
        if f > 0:
            adjacent_files |= FILE_MASK[f - 1]
        if f < 7:
            adjacent_files |= FILE_MASK[f + 1]

        # (b) Isolated pawns: no friendly pawn on either adjacent file.
        if not (pawns & adjacent_files):
            score -= ISOLATED_PAWN_PENALTY_CP

        # (c) Passed pawns: no enemy pawn on this file or an adjacent file,
        # anywhere between this pawn and its promotion square.
        span_files = FILE_MASK[f] | adjacent_files
        if not (enemy_pawns & span_files & ahead_rank_mask[r]):
            distance = (7 - r) if color == WHITE else r
            score += PASSED_PAWN_BONUS_BY_DISTANCE[distance]

    return score


def pawn_structure_term(board: Board) -> int:
    white_score = _side_pawn_structure(board, WHITE) - _side_pawn_structure(board, BLACK)
    return white_score if board.side_to_move == WHITE else -white_score


# --- Endgame mop-up (Milestone 5, gated off via Weights.endgame_mopup until --
# --- its own A/B gate) -------------------------------------------------------
#
# A classic "mop-up" heuristic (architecture.md §15: "simple endgame
# heuristics ... driving a lone king to the edge with mating material").
# It only ever activates in a genuine lone-king-vs-mating-material endgame:
# one color has no pawns and no pieces at all beyond its king (a bare king),
# and the *other* color has no pawns and exactly enough material to force
# mate alone against a bare king -- K+Q, K+R, K+B+B, or K+B+N (deliberately
# **not** K+N+N, which cannot force mate unaided even with the defending
# king reduced to a bare king). Anything else -- a normal material-rich
# position, or a genuinely drawn "insufficient material" ending like
# K+N+N vs K -- returns 0 immediately via two cheap all-zero/popcount
# checks, before any of the distance math below ever runs, so this term
# costs nothing outside its one narrow, explicitly-detected shape.
#
# When the shape *is* present, the bonus for the side with mating material
# combines two standard mating-technique components:
#   (a) how far the lone enemy king is from the center, via a
#       center-Manhattan-distance table (0 at the 4 center squares, rising
#       to 6 in each corner) -- computed directly from file_of/rank_of
#       rather than a literal 64-entry table.
#   (b) how close the friendly king is to the enemy king (Chebyshev
#       distance, 0..7) -- the "box the king in" technique; smaller is
#       better, so this contributes (7 - distance).
#
# Scale (centipawns; a simple starting point in the same spirit as
# MOBILITY_CP_PER_SQUARE/KING_SHIELD_CP_PER_PAWN above -- not yet
# hand-tuned or A/B-gated, hence Weights.endgame_mopup starts at 0.0):
# 10 cp per unit of center-distance (max 6 * 10 = 60 cp for an enemy king
# pinned in a corner) plus 10 cp per unit of (7 - king_distance) (max
# 7 * 10 = 70 cp for the two kings standing adjacent), for a maximum
# combined bonus of 130 cp -- comparable in size to the other terms above,
# nudging the mating side toward the technique without ever swamping the
# material difference itself.
MOPUP_CENTER_CP_PER_UNIT = 10
MOPUP_KING_DISTANCE_CP_PER_UNIT = 10


def _is_bare_king(board: Board, color: int) -> bool:
    """True if `color` has no pawns and no pieces at all other than its king."""
    p = board.pieces[color]
    return p[PAWN] == 0 and p[KNIGHT] == 0 and p[BISHOP] == 0 and p[ROOK] == 0 and p[QUEEN] == 0


def _has_lone_mating_material(board: Board, color: int) -> bool:
    """True if `color` has no pawns and exactly enough material to force
    mate alone against a bare enemy king: one queen, one rook, or two minor
    pieces including at least one bishop (K+B+B or K+B+N). K+N+N is
    deliberately excluded -- two bare knights cannot force mate alone."""
    p = board.pieces[color]
    if p[PAWN] != 0:
        return False
    knights, bishops = popcount(p[KNIGHT]), popcount(p[BISHOP])
    rooks, queens = popcount(p[ROOK]), popcount(p[QUEEN])
    if queens == 1 and rooks == 0 and knights == 0 and bishops == 0:
        return True
    if rooks == 1 and queens == 0 and knights == 0 and bishops == 0:
        return True
    if queens == 0 and rooks == 0 and bishops >= 1 and knights + bishops == 2:
        return True
    return False


def _mopup_bonus_for(board: Board, mating_color: int) -> int:
    """Centipawn bonus for `mating_color`. Only called once the caller has
    already confirmed `mating_color` holds lone mating material against
    `1 - mating_color`'s bare king, so no shape-detection happens here."""
    lone_color = 1 - mating_color
    enemy_king = board.king_square(lone_color)
    enemy_file, enemy_rank = file_of(enemy_king), rank_of(enemy_king)
    # Center-Manhattan distance: 0 at the 4 center squares (d4/d5/e4/e5),
    # rising to 6 in each corner.
    center_dist = max(3 - enemy_file, enemy_file - 4) + max(3 - enemy_rank, enemy_rank - 4)

    friendly_king = board.king_square(mating_color)
    friendly_file, friendly_rank = file_of(friendly_king), rank_of(friendly_king)
    king_dist = max(abs(friendly_file - enemy_file), abs(friendly_rank - enemy_rank))

    return (
        center_dist * MOPUP_CENTER_CP_PER_UNIT
        + (7 - king_dist) * MOPUP_KING_DISTANCE_CP_PER_UNIT
    )


def endgame_mopup_term(board: Board) -> int:
    white_bare, black_bare = _is_bare_king(board, WHITE), _is_bare_king(board, BLACK)
    if white_bare == black_bare:
        return 0  # both bare (dead draw) or neither bare: not this endgame shape

    mating_color = BLACK if white_bare else WHITE
    if not _has_lone_mating_material(board, mating_color):
        return 0

    bonus = _mopup_bonus_for(board, mating_color)
    white_score = bonus if mating_color == WHITE else -bonus
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
    # Enabled at 1.0 after tuning KING_ZONE_CP_PER_ATTACKER from 20 to 10
    # (the original 20cp/attacker was too aggressive for normal middlegame
    # positions, where many pieces naturally attack king-zone squares). The
    # first two A/B attempts with the original untuned weights (shield=15,
    # zone=20) came back as a 6.0/12 tie and then a 10.5-9.5 loss on a
    # larger sample. With the retuned zone penalty (shield=15, zone=10), a
    # 22-game match (DEFAULT_POSITIONS + 5 additional positions, depth 3)
    # scored the king-safety-enabled candidate 12.5/22 vs the baseline's
    # 9.5/22 -- a clear +3.0 non-negative trend, comparable to
    # pawn_structure's own +4.0 margin on the same scale.
    king_safety: float = 1.0
    # Enabled at 1.0: an A/B self-play match against a material+PST+mobility
    # baseline (tests/match_harness.py, DEFAULT_POSITIONS' 6 positions plus 5
    # additional hand-built FENs -- a doubled/isolated-pawn middlegame, a
    # passed-pawn K+P endgame, a clean symmetric control middlegame, an IQP
    # middlegame, and that doubled/isolated position's color-flipped mirror
    # -- 11 positions x 2 colors = 22 games, depth 3) scored the
    # pawn-structure-enabled candidate 13.0/22 vs the baseline's 9.0/22.
    # Broken down by position (both colors of each position paired), the
    # candidate was clearly ahead on 5 of the 11 positions, the baseline
    # clearly ahead on only 2, and 4 tied -- a distributed, non-negative
    # trend (not one outlier position skewing the total), and a real margin
    # rather than the near-tie/1-point-out-of-20 "muddle" that sank
    # king_safety above. Kept at 1.0 per architecture.md §10.2/§15.
    pawn_structure: float = 1.0
    # Enabled at 3.0 per a *functional* gate rather than the usual opening-
    # position A/B match: a generic opening/middlegame battery (like the one
    # `mobility`/`pawn_structure` above used) essentially never reaches a
    # lone-king-vs-mating-material endgame in a few plies at shallow depth,
    # so it would never exercise this term at all. Instead, 3 hand-built
    # KQvK/KRvK FENs with both kings placed far apart (opposite corners or
    # far edges) and the mating side to move were played out via
    # `tests/match_harness.py`'s `play_game` in full self-play (same
    # `CompositeEvaluator` instance for both `white_search` and
    # `black_search`, so the lone king's defense gets the same mop-up
    # awareness as the mating side's attack) at a deliberately shallow,
    # weak `SearchLimits(max_depth=2)` (also re-checked at `max_depth=3`),
    # `ply_cap=200`, comparing `endgame_mopup=0.0` (material_pst+mobility+
    # pawn_structure only, i.e. the pre-this-change default) against
    # `endgame_mopup=3.0`:
    #   - depth=2: without mop-up, all 3 positions stalled -- drawn by
    #     repetition or the 50-move rule (plies 46/94/100) with the mating
    #     side making little to no progress (e.g. king_dist/center_dist
    #     barely moving). With mop-up=3.0, all 3 positions reached actual,
    #     verified checkmate (0 legal moves + in_check) well inside the ply
    #     cap: KQvK corner-to-corner at ply 67, KRvK corner-to-corner at ply
    #     13, KQvK far-edges at ply 19.
    #   - depth=3: same pattern -- without mop-up, all 3 again just drew
    #     (plies 80/100/105, no progress); with mop-up=3.0, all 3 again
    #     reached verified checkmate (plies 27/73/41).
    # That is a clean, decisive functional improvement (checkmate reached in
    # 6/6 mop-up-enabled runs across both depths vs. 0/6 without it, not
    # merely "meaningfully closer") -- a far stronger signal than the
    # architecture.md §10.2/§15 "non-negative trend" bar. (A smaller weight
    # of 1.0 was tried first per that section's suggestion and only reached
    # mate in 1 of 3 positions, with one game losing the queen to a
    # depth-2-shallow-search blunder; 3.0 was the value that produced the
    # clean 3/3 result, and since this term is 0 outside the narrow bare-
    # king-vs-mating-material shape it's gated to, a larger weight here
    # doesn't risk distorting evaluation anywhere else.)
    endgame_mopup: float = 3.0


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
    """All five evaluation terms enabled: material+PST, mobility,
    king_safety, pawn_structure, and endgame_mopup. Each passed its own A/B
    self-play gate (see each `Weights.*` field's docstring comment for the
    exact match results). `endgame_mopup` passed a *functional* gate
    instead (a generic opening-position A/B match can't exercise a term that
    only ever activates in a bare-king-vs-mating-material endgame -- see
    `Weights.endgame_mopup`'s docstring comment for the exact
    positions/results) and is enabled at weight 3.0."""
    return CompositeEvaluator(
        terms={
            "material_pst": material_pst_term,
            "mobility": mobility_term,
            "king_safety": king_safety_term,
            "pawn_structure": pawn_structure_term,
            "endgame_mopup": endgame_mopup_term,
        },
        weights=Weights(
            material_pst=1.0,
            mobility=1.0,
            king_safety=1.0,
            pawn_structure=1.0,
            endgame_mopup=3.0,
        ),
    )
