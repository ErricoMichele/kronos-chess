"""Evaluation term unit tests (architecture.md §10, §13's `test_evaluate.py`
gate).

Everything here is exercised against a bare `chessengine.board.Board` built
directly from FEN. Per the module-boundary DAG (architecture.md §11),
`evaluate.py` depends only on `constants` and `board` -- never on
`movegen.py` or `search.py` -- and these tests deliberately mirror that
boundary: no move generation, no search, no `Search`/`TranspositionTable`
machinery anywhere below. That is exactly what makes a PST-mirroring bug (or
a `CompositeEvaluator` wiring bug) visible without a search-shaped test
harness first having to be right.

Three properties are checked:

1. **Color-flip symmetry** (`material_pst_term` and `default_evaluator()`):
   evaluating a position and evaluating a test-only color-flipped mirror of
   it must give exactly negated scores (see `_color_flip_pieces` below for
   why the mirror deliberately leaves `side_to_move` untouched).
2. **Material term sanity**: a FEN with an extra White queen scores
   `material_pst_term` about +900 relative to the same FEN without it.
3. **PST sanity**: a knight on a central square scores at least as well as
   one on a corner square, from White's perspective on an otherwise-empty
   board.

Plus direct `CompositeEvaluator` unit tests (weighting, zero-weight
exclusion, term wiring) and a check that `default_evaluator()` is wired the
way architecture.md §10.2 says it must be: `material_pst` weight 1.0,
every other term's weight 0.0.
"""

from __future__ import annotations

from chessengine.board import Board
from chessengine.constants import NO_PIECE, WHITE, color_of, piece_type_of
from chessengine.evaluate import (
    DOUBLED_PAWN_PENALTY_CP,
    ISOLATED_PAWN_PENALTY_CP,
    KING_SHIELD_CP_PER_PAWN,
    KING_ZONE_CP_PER_ATTACKER,
    MOPUP_CENTER_CP_PER_UNIT,
    MOPUP_KING_DISTANCE_CP_PER_UNIT,
    PASSED_PAWN_BONUS_BY_DISTANCE,
    CompositeEvaluator,
    KNIGHT_PST,
    QUEEN_PST,
    Weights,
    default_evaluator,
    endgame_mopup_term,
    king_safety_term,
    material_pst_term,
    mobility_term,
    pawn_structure_term,
)
from chessengine.fen import STARTPOS_FEN, parse_fen

# --- Test-only mirroring helper (no search machinery involved) --------------


def _color_flip_pieces(board: Board) -> Board:
    """Build a fresh `Board` with every piece's color inverted and its
    square mirrored vertically (`sq ^ 56` -- the exact same rank-flip
    `evaluate._side_score` itself uses to look up Black's PST value,
    architecture.md §10.1), leaving `side_to_move` untouched.

    This is a test-only helper (never imported by `src/`), analogous to the
    "deliberately naive, test-only" oracle helpers architecture.md §13 calls
    for elsewhere (e.g. the optional naive legal-move oracle) -- a second,
    independent way to construct "the same material, mirrored" that a real
    evaluator bug can't accidentally satisfy too.

    Why `side_to_move` is deliberately left alone, not flipped: nothing
    written down in this repo requires it to flip. `material_pst_term`
    returns a *side-to-move-relative* score (the negamax convention,
    architecture.md §10) -- internally it computes a white-referenced total
    and negates it when Black is to move. That white-referenced total is
    exactly anti-symmetric under a pure color swap + vertical mirror (this
    is the property under test): swap every piece's color and mirror its
    square, and the white-referenced total negates. If this helper also
    flipped `side_to_move`, that would fold in a *second* sign flip from the
    side-to-move convention, which exactly cancels the first and turns this
    into an equality -- silently hiding exactly the class of bug this test
    exists to catch (e.g. a missing/wrong `sq ^ 56` for Black in
    `_side_score`). Holding `side_to_move` fixed is what makes "mirroring
    the pieces negates the score" a clean, always-true invariant instead of
    one that depends on which side happened to be on move.
    """
    flipped = Board()
    for sq in range(64):
        code = board.mailbox[sq]
        if code == NO_PIECE:
            continue
        color, ptype = color_of(code), piece_type_of(code)
        flipped._add_piece((1 - color) * 6 + ptype, sq ^ 56)
    flipped.side_to_move = board.side_to_move
    return flipped


# A handful of positions spanning startpos, a busy middlegame (Kiwipete),
# a near-empty endgame (§14 position 3), and an already-asymmetric position
# with Black to move -- so the symmetry property is checked against more
# than one convenient, possibly-already-symmetric case.
SYMMETRY_FENS = (
    STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
)


# --- 1. Color-flip symmetry ---------------------------------------------------


def test_material_pst_term_negates_under_color_flip_mirror() -> None:
    """`material_pst_term(mirror) == -material_pst_term(original)` for every
    sample position -- catches a PST mirrored incorrectly for Black (or any
    other color-asymmetric bug in `_side_score`) without any search
    machinery involved (architecture.md §13)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        mirror = _color_flip_pieces(board)
        assert material_pst_term(mirror) == -material_pst_term(board), (
            f"color-flip symmetry broken for {fen!r}: "
            f"material_pst_term(original)={material_pst_term(board)}, "
            f"material_pst_term(mirror)={material_pst_term(mirror)}"
        )


def test_material_pst_term_double_mirror_restores_original_score() -> None:
    """Mirroring twice is the identity transform on piece placement, so it
    must also be the identity on the score (a second, independent check on
    top of the plain negation above)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        double_mirror = _color_flip_pieces(_color_flip_pieces(board))
        assert material_pst_term(double_mirror) == material_pst_term(board)


def test_default_evaluator_negates_under_color_flip_mirror() -> None:
    """The same color-flip symmetry, through `CompositeEvaluator`/
    `default_evaluator()` rather than calling `material_pst_term` directly
    -- confirms the composite wiring (weights, term lookup) doesn't
    introduce its own color asymmetry."""
    evaluator = default_evaluator()
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        mirror = _color_flip_pieces(board)
        assert evaluator.evaluate(mirror) == -evaluator.evaluate(board), (
            f"default_evaluator() color-flip symmetry broken for {fen!r}"
        )


def test_default_evaluator_agrees_with_material_pst_plus_mobility_plus_pawn_structure() -> None:
    """`default_evaluator()` enables `material_pst`, `mobility`, and
    `pawn_structure`, all at weight 1.0 (the latter two each validated by
    their own A/B self-play match, see `Weights.mobility`'s and
    `Weights.pawn_structure`'s docstring comments in evaluate.py).
    `king_safety` is registered but disabled at weight 0.0 (its own A/B
    match did not show a non-negative trend on a fair-sized sample, see
    `Weights.king_safety`'s docstring comment) and so must contribute
    nothing -- output must equal the exact sum of just the three enabled
    terms, on every sample position, not just correlate with it."""
    evaluator = default_evaluator()
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        assert evaluator.evaluate(board) == (
            material_pst_term(board) + mobility_term(board) + pawn_structure_term(board)
        )


# --- 2. Material term sanity: an extra queen is worth about +900 ------------


def test_extra_white_queen_scores_roughly_plus_900() -> None:
    """A FEN with an extra White queen must score `material_pst_term`
    about +900 relative to the same FEN without it (architecture.md §10.2's
    own example of an independently-testable term property).

    The extra queen is placed on d2, a square whose `QUEEN_PST` entry is
    exactly 0 (verified below), so the *only* contribution the extra piece
    makes is its raw material value -- the "roughly" in the assertion's
    tolerance is there for robustness against a future PST retune, not
    because this particular square is expected to carry any PST swing.
    """
    assert QUEEN_PST[11] == 0  # d2 == rank 2 (index 8..15), file d (offset 3) -> index 11

    fen_without_queen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    fen_with_queen = "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1"  # White queen added on d2

    score_without = material_pst_term(parse_fen(fen_without_queen))
    score_with = material_pst_term(parse_fen(fen_with_queen))

    diff = score_with - score_without
    assert abs(diff - 900) <= 30, (
        f"extra White queen changed material_pst_term by {diff}, expected roughly +900"
    )


def test_extra_black_queen_scores_roughly_minus_900_for_white_to_move() -> None:
    """The mirror image of the above: an extra *Black* queen, with White
    still to move, should be worth roughly -900 to the side to move
    (White) -- confirms the sign, not just the magnitude, of the material
    term."""
    fen_without_queen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    fen_with_black_queen = "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1"  # Black queen on d2

    score_without = material_pst_term(parse_fen(fen_without_queen))
    score_with = material_pst_term(parse_fen(fen_with_black_queen))

    diff = score_with - score_without
    assert abs(diff - (-900)) <= 30, (
        f"extra Black queen changed material_pst_term by {diff}, expected roughly -900"
    )


# --- 3. PST sanity: a central knight beats a cornered one -------------------


def test_knight_on_central_square_scores_at_least_as_well_as_on_corner() -> None:
    """A White knight on a central square (d4) must score `material_pst_term`
    at least as well as the same knight on a corner square (a1), on an
    otherwise-empty (just the two kings) board, from White's own
    perspective (White to move in both FENs, so `material_pst_term`'s sign
    is directly White's own)."""
    fen_knight_central = "4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"  # White knight on d4
    fen_knight_corner = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1"  # White knight on a1

    score_central = material_pst_term(parse_fen(fen_knight_central))
    score_corner = material_pst_term(parse_fen(fen_knight_corner))

    assert score_central >= score_corner, (
        f"central knight (d4) scored {score_central}, corner knight (a1) scored "
        f"{score_corner} -- expected the central square to score at least as well"
    )
    # KNIGHT_PST itself must be the reason (not some incidental cancellation
    # elsewhere): the raw table entries alone already show the same ordering.
    assert KNIGHT_PST[27] >= KNIGHT_PST[0]  # d4 == index 27, a1 == index 0


# --- CompositeEvaluator unit tests (no search machinery involved) -----------


def test_composite_evaluator_sums_weighted_terms() -> None:
    """`CompositeEvaluator.evaluate` sums each registered term, scaled by
    its `Weights` field, exactly (architecture.md §10.2) -- checked here
    with small hand-written fake terms rather than `material_pst_term`, so
    this test is purely about the composite's own arithmetic."""

    def term_a(_board: Board) -> int:
        return 10

    def term_b(_board: Board) -> int:
        return -4

    evaluator = CompositeEvaluator(
        terms={"material_pst": term_a, "mobility": term_b},
        weights=Weights(material_pst=2.0, mobility=0.5),
    )
    board = parse_fen(STARTPOS_FEN)

    # int(2.0 * 10) + int(0.5 * -4) == 20 + (-2) == 18
    assert evaluator.evaluate(board) == 18


def test_composite_evaluator_zero_weight_excludes_term_entirely() -> None:
    """A term registered with weight 0.0 must contribute nothing -- and per
    `CompositeEvaluator.evaluate`'s own `if w:` guard, must not even be
    *called*, so a term that would raise or that depends on state the test
    never set up is safely skipped."""

    def poisoned_term(_board: Board) -> int:
        raise AssertionError("a zero-weight term must never be called")

    def material_term(_board: Board) -> int:
        return 123

    evaluator = CompositeEvaluator(
        terms={"material_pst": material_term, "mobility": poisoned_term},
        weights=Weights(material_pst=1.0, mobility=0.0),
    )
    board = parse_fen(STARTPOS_FEN)

    assert evaluator.evaluate(board) == 123


def test_composite_evaluator_defaults_to_weights_of_one_material_pst() -> None:
    """Constructing a `CompositeEvaluator` with no explicit `weights` must
    fall back to a plain `Weights()` (material_pst=1.0, everything else
    0.0), not to all-zero or some other default."""
    evaluator = CompositeEvaluator(terms={"material_pst": material_pst_term})
    board = parse_fen(STARTPOS_FEN)

    assert evaluator.evaluate(board) == material_pst_term(board)


def test_composite_evaluator_ignores_unregistered_weights_fields() -> None:
    """A nonzero weight on a name that has no matching entry in `terms`
    (e.g. `king_safety` before Milestone 5 implements the term) must be
    silently inert -- `evaluate` only iterates `self.terms`, never
    `self.weights`' own fields."""
    evaluator = CompositeEvaluator(
        terms={"material_pst": material_pst_term},
        weights=Weights(material_pst=1.0, king_safety=999.0),
    )
    board = parse_fen(STARTPOS_FEN)

    assert evaluator.evaluate(board) == material_pst_term(board)


def test_default_evaluator_enables_material_pst_mobility_and_pawn_structure() -> None:
    """`default_evaluator()` enables `material_pst`, `mobility`,
    `pawn_structure`, and `endgame_mopup` at nonzero weight (`mobility`/
    `pawn_structure` each validated by their own A/B self-play match,
    architecture.md §10.2/§15's gate -- see `Weights.mobility`'s and
    `Weights.pawn_structure`'s docstring comments; `endgame_mopup` validated
    by its own *functional* gate instead, since a generic opening-position
    A/B match can't exercise a term that only activates in a bare-king-vs-
    mating-material endgame -- see `Weights.endgame_mopup`'s docstring
    comment for the exact positions/results). `king_safety` is registered
    (`king_safety_term` is implemented and unit-tested below) but stays at
    weight 0.0: its own A/B match did not show a non-negative trend on a
    fair-sized sample (see `Weights.king_safety`'s docstring comment) --
    pinned down field by field so a future term accidentally left enabled
    early (or a validated one accidentally left disabled/enabled) would
    fail this test."""
    evaluator = default_evaluator()

    assert set(evaluator.terms) == {
        "material_pst",
        "mobility",
        "king_safety",
        "pawn_structure",
        "endgame_mopup",
    }
    assert evaluator.terms["material_pst"] is material_pst_term
    assert evaluator.terms["mobility"] is mobility_term
    assert evaluator.terms["king_safety"] is king_safety_term
    assert evaluator.terms["pawn_structure"] is pawn_structure_term
    assert evaluator.terms["endgame_mopup"] is endgame_mopup_term

    assert evaluator.weights.material_pst == 1.0
    assert evaluator.weights.mobility == 1.0
    assert evaluator.weights.king_safety == 0.0
    assert evaluator.weights.pawn_structure == 1.0
    assert evaluator.weights.endgame_mopup == 3.0


# --- mobility_term ------------------------------------------------------
#
# `mobility_term` (architecture.md §10.1's Milestone 5 extension path, wired
# into `default_evaluator()` at weight 1.0 after passing its A/B gate -- see
# `evaluate.py`'s own module docstring above `mobility_term`) counts total
# attacked squares for knights/bishops/rooks/queens only (pawns and king are
# deliberately excluded), scaled by `MOBILITY_CP_PER_SQUARE`. These tests
# reuse exactly the same `_color_flip_pieces`/`SYMMETRY_FENS` pattern used
# for `material_pst_term` above, rather than inventing a second, parallel
# mirroring convention.


# --- 1. Color-flip symmetry --------------------------------------------------


def test_mobility_term_negates_under_color_flip_mirror() -> None:
    """`mobility_term(mirror) == -mobility_term(original)` for every sample
    position -- the same color-flip symmetry checked for `material_pst_term`
    above, now for the mobility term (catches e.g. a knight/bishop/rook/queen
    attack count computed for the wrong color, or a mirror that doesn't
    flip occupancy consistently)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        mirror = _color_flip_pieces(board)
        assert mobility_term(mirror) == -mobility_term(board), (
            f"color-flip symmetry broken for {fen!r}: "
            f"mobility_term(original)={mobility_term(board)}, "
            f"mobility_term(mirror)={mobility_term(mirror)}"
        )


def test_mobility_term_double_mirror_restores_original_score() -> None:
    """Mirroring twice is the identity transform on piece placement, so it
    must also be the identity on `mobility_term` (a second, independent
    check on top of the plain negation above)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        double_mirror = _color_flip_pieces(_color_flip_pieces(board))
        assert mobility_term(double_mirror) == mobility_term(board)


def test_mobility_term_zero_for_symmetric_startpos() -> None:
    """The starting position is perfectly mirror-symmetric -- both sides
    have identical knights/bishops/rooks/queens on identical (mirrored)
    squares with identical (mirrored) blockers -- so `mobility_term` must
    score exactly 0, not just "small" or "roughly balanced"."""
    assert mobility_term(parse_fen(STARTPOS_FEN)) == 0


# --- 2. Directional sanity: developed/open vs. boxed-in-by-own-pawns --------


def test_mobility_term_favors_developed_open_side_over_boxed_in_side() -> None:
    """A position with the same minor-piece material on both sides, but
    White's knight/bishop developed to open, central/long-diagonal squares
    versus Black's knight/bishop still on the back rank and boxed in by
    Black's own pawns, must score `mobility_term` clearly positive (favoring
    White, the more mobile side, who is also the side to move here -- the
    negamax convention makes a positive score mean "good for the side to
    move," §10).

    Position (White to move):
        8  n . b . k . . .
        7  . p . p . . . .
        6  . . . . . . . .
        5  . . . . . . . .
        4  . . . N . . . .
        3  . . . . . . . .
        2  . . . . . . B .
        1  . . . . K . . .
           a b c d e f g h

    White: Nd4 (central -- all 8 knight-attack squares on-board) and Bg2
    (fianchettoed on the long diagonal, only blocked far down it by Black's
    own b7 pawn) = 8 + 8 = 16 attacked squares.
    Black: Na8 (cornered -- only 2 of the 8 knight-attack squares are
    on-board) and Bc8 (still on the back rank, boxed in immediately by its
    own b7/d7 pawns one square out on each open diagonal) = 2 + 2 = 4
    attacked squares.

    Expected diff: (16 - 4) * MOBILITY_CP_PER_SQUARE(=3) = +36, exactly --
    checked exactly (not just ">0") since every attacked square above was
    counted by hand and independently confirmed against `_side_mobility`.
    """
    fen = "n1b1k3/1p1p4/8/8/3N4/8/6B1/4K3 w - - 0 1"
    board = parse_fen(fen)

    score = mobility_term(board)
    assert score > 0, (
        f"expected mobility_term to favor the developed/open White side, got {score}"
    )
    assert score == 36, f"expected mobility_term == +36 exactly, got {score}"


def test_mobility_term_favors_developed_open_side_regardless_of_side_to_move() -> None:
    """The same position as above but with the side to move flipped to
    Black: White is still objectively the more mobile side, so `mobility_term`
    (relative to the side to move, i.e. Black here) must flip sign to
    negative -- confirms the directional result isn't an artifact of which
    side happens to be on move."""
    fen = "n1b1k3/1p1p4/8/8/3N4/8/6B1/4K3 b - - 0 1"
    board = parse_fen(fen)

    score = mobility_term(board)
    assert score < 0, (
        f"expected mobility_term to disfavor Black (the side to move, and the "
        f"less mobile side) here, got {score}"
    )
    assert score == -36, f"expected mobility_term == -36 exactly, got {score}"


# --- king_safety_term ---------------------------------------------------
#
# `king_safety_term` (Milestone 5, gated off via `Weights.king_safety` until
# its own A/B match -- see the term's module comment and `Weights.king_safety`
# docstring in evaluate.py) combines a pawn-shield count (own pawns on the 3
# squares directly in front of the king) with a king-zone-attacker count
# (enemy pieces attacking any of the 8 squares around the king), scaled by
# `KING_SHIELD_CP_PER_PAWN` and `KING_ZONE_CP_PER_ATTACKER` respectively.
# These tests reuse exactly the same `_color_flip_pieces`/`SYMMETRY_FENS`
# pattern used for `material_pst_term`/`mobility_term` above, rather than
# inventing a third, parallel mirroring convention.


# --- 1. Color-flip symmetry --------------------------------------------------


def test_king_safety_term_negates_under_color_flip_mirror() -> None:
    """`king_safety_term(mirror) == -king_safety_term(original)` for every
    sample position -- the same color-flip symmetry checked for
    `material_pst_term`/`mobility_term` above, now for the king-safety term
    (catches e.g. a pawn-shield direction computed for the wrong color, or a
    king-zone-attacker count computed against the wrong `by_color`)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        mirror = _color_flip_pieces(board)
        assert king_safety_term(mirror) == -king_safety_term(board), (
            f"color-flip symmetry broken for {fen!r}: "
            f"king_safety_term(original)={king_safety_term(board)}, "
            f"king_safety_term(mirror)={king_safety_term(mirror)}"
        )


def test_king_safety_term_double_mirror_restores_original_score() -> None:
    """Mirroring twice is the identity transform on piece placement, so it
    must also be the identity on `king_safety_term` (a second, independent
    check on top of the plain negation above)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        double_mirror = _color_flip_pieces(_color_flip_pieces(board))
        assert king_safety_term(double_mirror) == king_safety_term(board)


def test_king_safety_term_zero_for_symmetric_startpos() -> None:
    """The starting position is perfectly mirror-symmetric -- both kings have
    an identical (mirrored) 3-pawn shield in front of them, and neither king's
    zone is attacked by anything (every piece is blocked by its own pawn rank)
    -- so `king_safety_term` must score exactly 0, not just "small" or
    "roughly balanced"."""
    assert king_safety_term(parse_fen(STARTPOS_FEN)) == 0


# --- 2. Directional sanity: castled-and-shielded vs. exposed-in-the-center --


def test_king_safety_term_favors_castled_shielded_king_over_exposed_king() -> None:
    """A position where White's king is safely tucked away on g1 behind an
    intact 3-pawn shield (f2/g2/h2) and totally unbothered, versus Black's
    king stranded on e5 in the center with no pawn shield at all and four
    White pieces bearing down on its immediate 8-square king zone, must score
    `king_safety_term` clearly positive (favoring White, the safer king, who
    is also the side to move here -- the negamax convention makes a positive
    score mean "good for the side to move," §10).

    Position (White to move):
        8  . . . . . . . .
        7  . . . . . . . .
        6  . . . . . . . .
        5  . . . . k . . Q
        4  . . N . . . . .
        3  . . . . . . . .
        2  . B . . . P P P
        1  . . . . R . K .
           a b c d e f g h

    White: Kg1 with an intact 3-pawn shield on f2/g2/h2 (shield = 3, +45 cp)
    and zero enemy pieces attacking its king zone (f1/h1/f2/g2/h2) -- White's
    own `_king_safety_score` is exactly 3*15 - 0*20 = +45.

    Black: Ke5 with no pawn shield at all (shield = 0) and four White pieces
    each attacking one distinct square of its 8-square king zone
    (d4/e4/f4/d5/f5/d6/e6/f6): Bb2->d4, Re1->e4 (blocked by the king itself,
    same as a real check would be), Qh5->f5 (blocked from going further by
    the king), and Nc4->d6. That's 4 attackers, so Black's own
    `_king_safety_score` is exactly 0*15 - 4*20 = -80.

    `king_safety_term` = white_score - black_score = 45 - (-80) = +125,
    checked exactly (not just ">0"), with every shield pawn and every
    attacker above counted by hand and independently confirmed against
    `_king_safety_score`.
    """
    fen = "8/8/8/4k2Q/2N5/8/1B3PPP/4R1K1 w - - 0 1"
    board = parse_fen(fen)

    score = king_safety_term(board)
    assert score > 0, (
        f"expected king_safety_term to favor White's castled, shielded king "
        f"over Black's exposed, attacked king, got {score}"
    )
    assert score == 125, f"expected king_safety_term == +125 exactly, got {score}"

    # Sanity-check the hand-derived components directly, so a passing test
    # can't be an accident of unrelated cancellation.
    assert KING_SHIELD_CP_PER_PAWN == 15
    assert KING_ZONE_CP_PER_ATTACKER == 20


def test_king_safety_term_favors_castled_shielded_king_regardless_of_side_to_move() -> None:
    """The same position as above but with the side to move flipped to
    Black: White's king is still objectively the safer one, so
    `king_safety_term` (relative to the side to move, i.e. Black here) must
    flip sign to negative -- confirms the directional result isn't an
    artifact of which side happens to be on move."""
    fen = "8/8/8/4k2Q/2N5/8/1B3PPP/4R1K1 b - - 0 1"
    board = parse_fen(fen)

    score = king_safety_term(board)
    assert score < 0, (
        f"expected king_safety_term to disfavor Black (the side to move, and "
        f"the side with the exposed, attacked king) here, got {score}"
    )
    assert score == -125, f"expected king_safety_term == -125 exactly, got {score}"


# --- pawn_structure_term --------------------------------------------------
#
# `pawn_structure_term` (implemented, enabled via `Weights.pawn_structure`
# after passing its own A/B match -- see the term's module comment and
# `Weights.pawn_structure` docstring in evaluate.py) combines three
# components computed per side directly from `board.pieces[color][PAWN]`:
# doubled-pawn penalties, isolated-pawn penalties, and distance-scaled
# passed-pawn bonuses. These tests reuse exactly the same
# `_color_flip_pieces`/`SYMMETRY_FENS` pattern used for
# `material_pst_term`/`mobility_term`/`king_safety_term` above, rather than
# inventing a fourth, parallel mirroring convention.


# --- 1. Color-flip symmetry --------------------------------------------------


def test_pawn_structure_term_negates_under_color_flip_mirror() -> None:
    """`pawn_structure_term(mirror) == -pawn_structure_term(original)` for
    every sample position -- the same color-flip symmetry checked for
    `material_pst_term`/`mobility_term`/`king_safety_term` above, now for the
    pawn-structure term (catches e.g. a doubled/isolated/passed check
    computed against the wrong color's pawns, or a passed-pawn distance
    computed with the wrong promotion direction for Black)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        mirror = _color_flip_pieces(board)
        assert pawn_structure_term(mirror) == -pawn_structure_term(board), (
            f"color-flip symmetry broken for {fen!r}: "
            f"pawn_structure_term(original)={pawn_structure_term(board)}, "
            f"pawn_structure_term(mirror)={pawn_structure_term(mirror)}"
        )


def test_pawn_structure_term_double_mirror_restores_original_score() -> None:
    """Mirroring twice is the identity transform on piece placement, so it
    must also be the identity on `pawn_structure_term` (a second, independent
    check on top of the plain negation above)."""
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        double_mirror = _color_flip_pieces(_color_flip_pieces(board))
        assert pawn_structure_term(double_mirror) == pawn_structure_term(board)


def test_pawn_structure_term_zero_for_startpos() -> None:
    """The starting position has, on both sides, all 8 files occupied by
    exactly one pawn each: no file ever has more than one pawn (no doubling),
    every pawn has a same-rank neighbor on an adjacent file (no isolation --
    even the a- and h-file pawns have their one adjacent file occupied), and
    every pawn is blocked from ever passing by the opponent's mirrored pawn
    two ranks further up its own file (no passed pawns). So both sides'
    `_side_pawn_structure` is exactly 0 and `pawn_structure_term` must be
    exactly 0, not just "small" or "roughly balanced"."""
    assert pawn_structure_term(parse_fen(STARTPOS_FEN)) == 0


# --- 2. Doubled pawns: a doubled file is clearly worse than a clean one -----


def test_pawn_structure_term_penalizes_doubled_pawn() -> None:
    """A FEN with a doubled White pawn on the d-file (d2 and d3), with every
    pawn already defended from isolation by a pawn on an adjacent file (e2)
    and every pawn blocked from passing by Black's c7/d7/e7 pawns -- so
    doubling is the *only* structural feature in play -- must score
    `pawn_structure_term` clearly worse for White than the same position
    with the extra d-file pawn removed.

    Position with the doubled pawn (White to move):
        8  . . . . k . . .
        7  . . . p p p . .
        6  . . . . . . . .
        5  . . . . . . . .
        4  . . . . . . . .
        3  . . . P . . . .
        2  . . . P P P . .
        1  . . . . K . . .
           a b c d e f g h

    White's d-file has 2 pawns (d2, d3): exactly one "extra" pawn beyond the
    first, so `_side_pawn_structure` docks exactly
    `DOUBLED_PAWN_PENALTY_CP` (=12) once for that file. Neither d-pawn is
    isolated (e2 sits on the adjacent e-file) and none of White's pawns are
    passed (Black's c7/d7/e7 block the d/e/f files). Black's own structure
    (c7/d7/e7, no doubling, no isolation, blocked from passing by White's
    d/e/f-file pawns) is identical in both FENs, so it only has to cancel
    out, not equal any particular value.
    """
    fen_doubled = "4k3/3ppp2/8/8/8/3P4/3PPP2/4K3 w - - 0 1"
    fen_fixed = "4k3/3ppp2/8/8/8/8/3PPP2/4K3 w - - 0 1"  # extra d3 pawn removed

    score_doubled = pawn_structure_term(parse_fen(fen_doubled))
    score_fixed = pawn_structure_term(parse_fen(fen_fixed))

    assert score_doubled < score_fixed, (
        f"expected the doubled d-file to score clearly worse for White, got "
        f"doubled={score_doubled}, fixed={score_fixed}"
    )
    # Hand-verified exactly: doubling is the only difference between the two
    # FENs, and it costs precisely one DOUBLED_PAWN_PENALTY_CP.
    assert score_fixed == 0, f"expected the clean structure to score exactly 0, got {score_fixed}"
    assert score_doubled == -DOUBLED_PAWN_PENALTY_CP, (
        f"expected the doubled structure to score exactly -{DOUBLED_PAWN_PENALTY_CP}, "
        f"got {score_doubled}"
    )


# --- 3. Isolated pawns: an isolated pawn is clearly worse than a supported one


def test_pawn_structure_term_penalizes_isolated_pawn() -> None:
    """A FEN with a lone, isolated White d-pawn (no White pawn on the c- or
    e-file) must score `pawn_structure_term` clearly worse for White than
    the same position with a pawn added on e2 -- which gives the d-pawn (and
    the new e-pawn) a same-rank neighbor, eliminating the isolation.

    Position with the isolated pawn (White to move):
        8  . . . . k . . .
        7  . . p p p . . .
        6  . . . . . . . .
        5  . . . . . . . .
        4  . . . . . . . .
        3  . . . . . . . .
        2  . . . P . . . .
        1  . . . . K . . .
           a b c d e f g h

    White's only pawn (d2) has no friendly pawn on the c- or e-file, so it is
    isolated: `_side_pawn_structure` docks exactly `ISOLATED_PAWN_PENALTY_CP`
    (=15). It is not doubled (only one pawn on the d-file) and not passed
    (Black's c7/d7/e7 block the c/d/e files it would have to cross). Adding a
    White pawn on e2 gives d2 an adjacent-file neighbor (no longer isolated)
    and the new e2 pawn also has d2 as its own neighbor (not isolated
    either); e2 is likewise not passed (blocked by Black's d7/e7). Black's
    own structure (c7/d7/e7, unaffected either way) only has to cancel out,
    not equal any particular value.
    """
    fen_isolated = "4k3/2ppp3/8/8/8/8/3P4/4K3 w - - 0 1"
    fen_fixed = "4k3/2ppp3/8/8/8/8/3PP3/4K3 w - - 0 1"  # e2 pawn added

    score_isolated = pawn_structure_term(parse_fen(fen_isolated))
    score_fixed = pawn_structure_term(parse_fen(fen_fixed))

    assert score_isolated < score_fixed, (
        f"expected the isolated d-pawn to score clearly worse for White, got "
        f"isolated={score_isolated}, fixed={score_fixed}"
    )
    # Hand-verified exactly: isolation is the only difference between the
    # two FENs, and it costs precisely one ISOLATED_PAWN_PENALTY_CP.
    assert score_fixed == 0, f"expected the supported structure to score exactly 0, got {score_fixed}"
    assert score_isolated == -ISOLATED_PAWN_PENALTY_CP, (
        f"expected the isolated structure to score exactly -{ISOLATED_PAWN_PENALTY_CP}, "
        f"got {score_isolated}"
    )


# --- 4. Passed pawns: a passed pawn scores better, and more advanced is better


def test_pawn_structure_term_favors_more_advanced_passed_pawn() -> None:
    """Two FENs with an otherwise-identical White structure -- a fixed,
    unadvanced e2 pawn (itself passed too, since Black has no pawns at all)
    plus one White pawn on the d-file that is unambiguously passed (no Black
    pawns anywhere on the board, let alone on the c/d/e files ahead of it) --
    differing *only* in how far that d-pawn has advanced (d3 vs. d6), must
    both score `pawn_structure_term` clearly positive for White (a passed
    pawn is a clear structural asset), and the more advanced pawn (d6, only
    2 squares from promotion) must score strictly higher than the less
    advanced one (d3, 5 squares from promotion) -- `PASSED_PAWN_BONUS_BY_DISTANCE`
    grows as a passer nears promotion, so distance 2 (bonus 100) must beat
    distance 5 (bonus 20).

    Neither pawn is ever doubled (one pawn per file) or isolated (the d-pawn
    always has e2 as an adjacent-file neighbor, and e2 always has the d-pawn
    as its own), in either FEN, so the score difference between the two FENs
    is exactly the passed-pawn bonus difference for the d-pawn alone
    (e2's own passed bonus, from its fixed, unadvanced position, is
    identical in both FENs and only has to cancel out of the *comparison*,
    not vanish from either score individually).
    """
    fen_less_advanced = "6k1/8/8/8/8/3P4/4P3/6K1 w - - 0 1"  # passed d-pawn on d3
    fen_more_advanced = "6k1/8/3P4/8/8/8/4P3/6K1 w - - 0 1"  # passed d-pawn on d6

    score_less_advanced = pawn_structure_term(parse_fen(fen_less_advanced))
    score_more_advanced = pawn_structure_term(parse_fen(fen_more_advanced))

    assert score_less_advanced > 0, (
        f"expected a clear passed pawn to score clearly positive for White, "
        f"got {score_less_advanced}"
    )
    assert score_more_advanced > 0, (
        f"expected a clear passed pawn to score clearly positive for White, "
        f"got {score_more_advanced}"
    )
    assert score_more_advanced > score_less_advanced, (
        f"expected the more advanced passed pawn (d6) to score higher than the "
        f"less advanced one (d3), got more_advanced={score_more_advanced}, "
        f"less_advanced={score_less_advanced}"
    )
    # Hand-verified exactly: e2's own passed bonus (distance 6 -> 10 cp) is
    # identical in both FENs, so the only difference is the d-pawn's own
    # passed bonus at distance 5 (d3, 20 cp) vs. distance 2 (d6, 100 cp).
    assert score_less_advanced == 10 + PASSED_PAWN_BONUS_BY_DISTANCE[5] == 30
    assert score_more_advanced == 10 + PASSED_PAWN_BONUS_BY_DISTANCE[2] == 110


# --- endgame_mopup_term ------------------------------------------------------
#
# `endgame_mopup_term` (Milestone 5, gated off via `Weights.endgame_mopup`
# until its own A/B match -- see the term's module comment and
# `Weights.endgame_mopup` docstring in evaluate.py) activates only in a
# genuine lone-king-vs-mating-material endgame (one bare king vs. K+Q, K+R,
# K+B+B, or K+B+N -- explicitly not K+N+N) and, only then, rewards driving
# the lone enemy king toward the edge/corner and the friendly king closer to
# it. These tests use their own FEN set (`ENDGAME_MOPUP_FENS`), since none of
# `SYMMETRY_FENS` above happens to contain a bare king.

ENDGAME_MOPUP_FENS = (
    "7k/8/8/8/8/8/8/4K2Q w - - 0 1",  # K+Q vs bare K, enemy king in the corner
    "8/8/8/4k3/8/8/8/4K2Q w - - 0 1",  # K+Q vs bare K, enemy king in the center
    "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",  # K+R vs bare K
    "4k3/8/8/8/8/8/8/2B1K1N1 w - - 0 1",  # K+B+N vs bare K
)


def test_endgame_mopup_term_negates_under_color_flip_mirror() -> None:
    """`endgame_mopup_term(mirror) == -endgame_mopup_term(original)` for
    every sample endgame position -- the same color-flip symmetry checked
    for the other terms above, now for the mop-up term. `_color_flip_pieces`
    only flips ranks (`sq ^ 56`), under which both the center-Manhattan-
    distance and king-Chebyshev-distance components are themselves
    invariant, so mirroring must exactly swap which color the bonus favors
    without changing its magnitude."""
    for fen in ENDGAME_MOPUP_FENS:
        board = parse_fen(fen)
        mirror = _color_flip_pieces(board)
        assert endgame_mopup_term(mirror) == -endgame_mopup_term(board), (
            f"color-flip symmetry broken for {fen!r}: "
            f"endgame_mopup_term(original)={endgame_mopup_term(board)}, "
            f"endgame_mopup_term(mirror)={endgame_mopup_term(mirror)}"
        )


def test_endgame_mopup_term_double_mirror_restores_original_score() -> None:
    """Mirroring twice is the identity transform on piece placement, so it
    must also be the identity on `endgame_mopup_term` (a second, independent
    check on top of the plain negation above)."""
    for fen in ENDGAME_MOPUP_FENS:
        board = parse_fen(fen)
        double_mirror = _color_flip_pieces(_color_flip_pieces(board))
        assert endgame_mopup_term(double_mirror) == endgame_mopup_term(board)


def test_endgame_mopup_term_zero_for_startpos() -> None:
    """Neither side is remotely close to a bare king in the starting
    position, so the shape-detection guard must return exactly 0 --
    cheaply, without ever reaching the distance math."""
    assert endgame_mopup_term(parse_fen(STARTPOS_FEN)) == 0


def test_endgame_mopup_term_zero_when_neither_side_is_bare_king() -> None:
    """A materially unbalanced but non-endgame position (extra queen for
    White, but Black still has a knight, i.e. neither side is a bare king)
    must score exactly 0 -- the term must not fire outside its one narrow,
    explicitly-detected shape."""
    fen = "3nk3/8/8/8/8/8/8/3QK3 w - - 0 1"  # White K+Q, Black K+N: neither is bare
    assert endgame_mopup_term(parse_fen(fen)) == 0


def test_endgame_mopup_term_zero_for_insufficient_material_knn_vs_k() -> None:
    """K+N+N vs. a bare king is well-known *insufficient* material to force
    mate unaided -- explicitly excluded from `_has_lone_mating_material` --
    so this must score exactly 0 even though one side is a bare king and the
    other holds two minor pieces."""
    fen = "4k3/8/8/8/8/8/8/2N1K1N1 w - - 0 1"
    assert endgame_mopup_term(parse_fen(fen)) == 0


def test_endgame_mopup_term_zero_for_extra_piece_beyond_lone_mating_material() -> None:
    """K+Q+N vs. a bare king has *more* than the lone-mating-material shape
    (an extra knight beyond the queen) -- `_has_lone_mating_material` must
    reject it, scoring exactly 0, since the detection is for the exact
    K+Q/K+R/K+B+B/K+B+N shapes only, not merely 'has a queen or rook'."""
    fen = "4k3/8/8/8/8/8/8/2N1K2Q w - - 0 1"
    assert endgame_mopup_term(parse_fen(fen)) == 0


def test_endgame_mopup_term_favors_cornered_enemy_king_over_centralized_one() -> None:
    """With the same K+Q vs. bare-K material and an identical friendly king
    square (e8 in both FENs), a lone enemy king in the corner (h8,
    center-Manhattan-distance 6) must score a strictly larger bonus than the
    same lone king on a central square (e5, center-Manhattan-distance 0) --
    checked exactly via `MOPUP_CENTER_CP_PER_UNIT`, not just '>'."""
    fen_corner = "4K2k/8/8/8/8/8/8/7Q w - - 0 1"  # White Ke8, Black king h8 (corner), Qh1
    fen_central = "4K3/8/8/4k3/8/8/8/7Q w - - 0 1"  # White Ke8, Black king e5 (center), Qh1

    score_corner = endgame_mopup_term(parse_fen(fen_corner))
    score_central = endgame_mopup_term(parse_fen(fen_central))

    assert score_corner > score_central, (
        f"expected the cornered enemy king to score a larger bonus, got "
        f"corner={score_corner}, central={score_central}"
    )
    # Hand-verified exactly: the friendly king (e8) is Chebyshev-distance 3
    # from both h8 and e5, so the king-distance component is identical in
    # both FENs and the entire difference is the center-distance component:
    # h8's center-distance is 6, e5's is 0, a difference of
    # 6 * MOPUP_CENTER_CP_PER_UNIT.
    assert score_corner - score_central == 6 * MOPUP_CENTER_CP_PER_UNIT


def test_endgame_mopup_term_favors_closer_friendly_king() -> None:
    """With the same K+R vs. bare-K material and the same lone enemy king
    square (e8 in both FENs), a friendly king standing closer (Chebyshev
    distance) to the enemy king must score a strictly larger bonus than one
    standing farther away -- checked exactly via
    `MOPUP_KING_DISTANCE_CP_PER_UNIT`."""
    fen_far = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"  # White king e1, Chebyshev dist 7 from e8
    fen_close = "4k3/8/4K3/8/8/8/8/R7 w - - 0 1"  # White king e6, Chebyshev dist 2 from e8

    score_far = endgame_mopup_term(parse_fen(fen_far))
    score_close = endgame_mopup_term(parse_fen(fen_close))

    assert score_close > score_far, (
        f"expected the closer friendly king to score a larger bonus, got "
        f"close={score_close}, far={score_far}"
    )
    # Hand-verified exactly: e8's center-distance (3) is identical in both
    # FENs (the lone king never moves), so the entire difference is the
    # king-distance component: e1->e8 is Chebyshev distance 7, e6->e8 is
    # distance 2, a difference of 5 * MOPUP_KING_DISTANCE_CP_PER_UNIT.
    assert score_close - score_far == 5 * MOPUP_KING_DISTANCE_CP_PER_UNIT


def test_endgame_mopup_term_favors_mating_side_regardless_of_side_to_move() -> None:
    """The same K+Q vs. bare-K position with Black to move instead of White:
    White is still objectively the mating side, so `endgame_mopup_term`
    (relative to the side to move, i.e. Black here) must flip sign to
    negative -- confirms the directional result isn't an artifact of which
    side happens to be on move."""
    fen_white_to_move = "7k/8/8/8/8/8/8/4K2Q w - - 0 1"
    fen_black_to_move = "7k/8/8/8/8/8/8/4K2Q b - - 0 1"

    score_white_to_move = endgame_mopup_term(parse_fen(fen_white_to_move))
    score_black_to_move = endgame_mopup_term(parse_fen(fen_black_to_move))

    assert score_white_to_move > 0
    assert score_black_to_move == -score_white_to_move


# --- Additional isolated endgame_mopup_term checks --------------------------
#
# The tests above already exercise `endgame_mopup_term`'s color-flip
# symmetry, its zero-guard on non-matching shapes, and its two directional
# components using a mix of K+Q and K+R material. The tests below add a
# handful of further, narrowly-scoped checks -- a normal (non-endgame)
# middlegame FEN, an isolated single-position K+Q-vs-K symmetry check,
# K+Q-vs-K-specific versions of both directional components, and a K+R-vs-K+R
# ("mating material on both sides") zero check -- each self-contained and
# independently hand-verified, reusing the same `_color_flip_pieces`/FEN
# patterns used throughout this file.


def test_endgame_mopup_term_zero_for_startpos_and_normal_middlegame() -> None:
    """`endgame_mopup_term` must be exactly 0 both on the starting position
    and on a normal, material-balanced middlegame FEN (Kiwipete) -- neither
    is remotely close to the lone-king-vs-mating-material shape this term
    targets, so the bare-king guard in `endgame_mopup_term` must return 0
    immediately for both, without ever reaching the distance math."""
    assert endgame_mopup_term(parse_fen(STARTPOS_FEN)) == 0

    middlegame_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    assert endgame_mopup_term(parse_fen(middlegame_fen)) == 0


def test_endgame_mopup_term_kqvk_negates_under_color_flip_mirror() -> None:
    """A dedicated, single-position version of the color-flip symmetry
    check: on a K+Q-vs-bare-K position, `endgame_mopup_term(mirror)` must be
    the exact negative of `endgame_mopup_term(original)`. The mating side's
    bonus is nonzero here (confirmed below), so this also rules out a
    trivial "both sides score 0" pass."""
    fen = "7k/8/8/8/8/8/8/4K2Q w - - 0 1"  # White K+Q vs bare Black king in the corner
    board = parse_fen(fen)
    mirror = _color_flip_pieces(board)

    score = endgame_mopup_term(board)
    assert score != 0, "expected a nonzero mop-up bonus for this K+Q-vs-bare-K position"
    assert endgame_mopup_term(mirror) == -score, (
        f"color-flip symmetry broken for {fen!r}: original={score}, "
        f"mirror={endgame_mopup_term(mirror)}"
    )


def test_endgame_mopup_term_kqvk_favors_cornered_enemy_king_over_centralized() -> None:
    """Two K+Q-vs-bare-K FENs differing only in the lone (Black) king's
    square -- a8 (corner) versus d5 (one of the 4 center squares) -- with the
    mating White king (d8) and queen (h1) held fixed, must score the
    cornered case strictly higher.

    The White king (d8) is deliberately Chebyshev-distance 3 from *both*
    a8 (same rank, file distance 3) and d5 (same file, rank distance 3), so
    the king-distance component of the bonus is identical in both FENs and
    the entire difference is the center-distance component: a8's
    center-Manhattan-distance is 6 (a genuine corner), d5's is 0, a
    difference of 6 * MOPUP_CENTER_CP_PER_UNIT -- checked exactly.
    """
    fen_corner = "k2K4/8/8/8/8/8/8/7Q w - - 0 1"  # Black king a8 (corner), White Kd8, Qh1
    fen_central = "3K4/8/8/3k4/8/8/8/7Q w - - 0 1"  # Black king d5 (center), White Kd8, Qh1

    score_corner = endgame_mopup_term(parse_fen(fen_corner))
    score_central = endgame_mopup_term(parse_fen(fen_central))

    assert score_corner > score_central, (
        f"expected the cornered enemy king to score a larger bonus, got "
        f"corner={score_corner}, central={score_central}"
    )
    assert score_corner - score_central == 6 * MOPUP_CENTER_CP_PER_UNIT
    assert score_corner == 100 and score_central == 40


def test_endgame_mopup_term_kqvk_favors_closer_friendly_king() -> None:
    """Two K+Q-vs-bare-K FENs differing only in the mating White king's
    square -- e1 (far) versus e6 (close) -- with the lone Black king (e8)
    and White queen (a1) held fixed, must score the closer-king case
    strictly higher.

    e8's center-Manhattan-distance (3) is identical in both FENs (the lone
    king never moves), so the entire difference is the king-distance
    component: e1->e8 is Chebyshev distance 7, e6->e8 is distance 2, a
    difference of 5 * MOPUP_KING_DISTANCE_CP_PER_UNIT -- checked exactly.
    """
    fen_far = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"  # Black king e8, White Ke1 (far), Qa1
    fen_close = "4k3/8/4K3/8/8/8/8/Q7 w - - 0 1"  # Black king e8, White Ke6 (close), Qa1

    score_far = endgame_mopup_term(parse_fen(fen_far))
    score_close = endgame_mopup_term(parse_fen(fen_close))

    assert score_close > score_far, (
        f"expected the closer friendly king to score a larger bonus, got "
        f"close={score_close}, far={score_far}"
    )
    assert score_close - score_far == 5 * MOPUP_KING_DISTANCE_CP_PER_UNIT
    assert score_far == 30 and score_close == 80


def test_endgame_mopup_term_zero_for_krvkr_mating_material_on_both_sides() -> None:
    """A K+R-vs-K+R endgame -- mating material on *both* sides, not the
    lone-king-vs-mating-material shape this term targets -- must score
    exactly 0: `_is_bare_king` is False for both colors here, so
    `endgame_mopup_term`'s own guard (`white_bare == black_bare`) must return
    0 immediately, confirming the detection is specific to the intended
    shape and doesn't fire just because mating material is present
    somewhere on the board."""
    fen = "4k2r/8/8/8/8/8/8/R3K3 w - - 0 1"  # White K+R, Black K+R
    assert endgame_mopup_term(parse_fen(fen)) == 0
