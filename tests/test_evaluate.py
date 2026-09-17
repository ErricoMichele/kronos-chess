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
    CompositeEvaluator,
    KNIGHT_PST,
    QUEEN_PST,
    Weights,
    default_evaluator,
    material_pst_term,
    mobility_term,
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


def test_default_evaluator_agrees_with_material_pst_plus_mobility() -> None:
    """`default_evaluator()` enables `material_pst` and `mobility`, both at
    weight 1.0 (the latter validated by an A/B self-play match, see
    `Weights.mobility`'s docstring comment in evaluate.py) -- so its output
    must equal the exact sum of the two terms, on every sample position, not
    just correlate with it."""
    evaluator = default_evaluator()
    for fen in SYMMETRY_FENS:
        board = parse_fen(fen)
        assert evaluator.evaluate(board) == material_pst_term(board) + mobility_term(board)


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


def test_default_evaluator_enables_material_pst_and_mobility_only() -> None:
    """`default_evaluator()` enables `material_pst` and `mobility` (the
    latter validated by an A/B self-play match against a material+PST-only
    baseline, architecture.md §10.2/§15's gate); `king_safety` and
    `pawn_structure` remain unimplemented and stay at 0.0 -- pinned down
    field by field so a future term accidentally left enabled early (or a
    validated one accidentally left disabled) would fail this test."""
    evaluator = default_evaluator()

    assert set(evaluator.terms) == {"material_pst", "mobility"}
    assert evaluator.terms["material_pst"] is material_pst_term
    assert evaluator.terms["mobility"] is mobility_term

    assert evaluator.weights.material_pst == 1.0
    assert evaluator.weights.mobility == 1.0
    assert evaluator.weights.king_safety == 0.0
    assert evaluator.weights.pawn_structure == 0.0


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
