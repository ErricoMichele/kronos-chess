"""Milestone 4 exit-criteria tests for quiescence search
(architecture.md §9.5, §13, §15).

§13 calls for "hand-built positions where a naive fixed-depth search
(quiescence disabled) misjudges a hanging piece one ply past the horizon,
and the quiescence-enabled search correctly does not." §15's Milestone 4
exit criteria restate the same requirement: `test_search_quiescence.py`
green, with hand-built horizon-effect positions resolved correctly.

**The horizon effect, concretely.** Each position below gives the side to
move exactly one capturing move: taking a defended piece that nets a *bad*
trade once the opponent recaptures (a rook for a defended knight; a knight
for a defended pawn). Immediately after that capture -- before the
recapture -- the position's raw material snapshot looks like a clean win
(a piece was just taken "for free"). A search that stops exactly there and
reports that snapshot as the move's value is fooled: the very next ply
gives the material back and then some. This is the textbook horizon
effect: stopping mid-exchange and misjudging a piece that is about to be
recaptured.

**No quiescence on/off switch exists.** `Search._negamax` (search.py,
Milestone 4) always hands `depth == 0` off to `Search._quiescence` -- there
is no Milestone-2-style plain-static-eval leaf left to flip back on. Per
the task, the quiescence-*disabled* side of the comparison is therefore
simulated the simplest correct way: by calling `Search.evaluator.evaluate`
directly on the horizon position, which is exactly the value a
quiescence-disabled `_negamax` would have returned unchanged at that leaf.
That naive value is compared against:

1. `Search._quiescence` called directly on that *same* horizon board (the
   "shallow depth reaching that same horizon" the task describes) -- the
   most direct apples-to-apples comparison of naive-vs-corrected scoring of
   one specific node.
2. The real, public `Search.search(...)` entry point run from the position
   *before* the bad capture, at a tiny fixed depth -- proving the effect
   disappears end-to-end, not just when a private method is poked directly.

Every position here is deliberately sparse (a handful of pieces) so the
capturing move is the *only* capture available to the side to move at the
root, which keeps the "what should happen" reasoning in each assertion
unambiguous, and keeps every search in this file done in well under a
second.
"""

from __future__ import annotations

import pytest

from chessengine.board import Board
from chessengine.constants import INF
from chessengine.evaluate import Evaluator, default_evaluator
from chessengine.fen import parse_fen
from chessengine.move import move_to_uci
from chessengine.movegen import generate_legal_moves
from chessengine.search import Search, SearchLimits, _SearchCtx

# --- Hand-built horizon-effect positions ------------------------------------
#
# (case id, root FEN, the one bad capturing move available at the root, in
# UCI notation). White is always the side to move and the side about to be
# fooled by the horizon effect; Black always holds the defended piece and
# the follow-up recapture.
#
# Case 1: a rook takes a knight that a pawn defends (Rxe5). White is already
# up a rook+pawn vs. knight+pawn (+170ish) before the capture. Right after
# Rxe5 the snapshot looks like White just won a whole knight outright; after
# ...dxe5 the rook is gone for a mere knight, and White's pre-existing edge
# has been traded away for nothing.
#
# Case 2: a knight takes a pawn that another pawn defends (Nxd5). Same
# shape with different piece types, so the effect isn't an artifact of one
# particular piece: right after Nxd5 White looks like it won a free pawn;
# after ...exd5 the knight itself is gone for a single pawn.
HORIZON_CASES = [
    pytest.param(
        "6k1/8/3p4/4n3/8/8/P7/4R1K1 w - - 0 1",
        "e1e5",
        id="rook-takes-knight-defended-by-pawn",
    ),
    pytest.param(
        "6k1/8/4p3/3p4/8/2N5/P7/6K1 w - - 0 1",
        "c3d5",
        id="knight-takes-pawn-defended-by-pawn",
    ),
]


def _find_move(board: Board, uci: str) -> int:
    """Resolve a UCI move string against `board`'s own legal move list,
    exactly like the rest of the suite's `move_to_uci`-based lookups
    (see e.g. `test_search.py`'s use of the same encoding)."""
    for move in generate_legal_moves(board):
        if move_to_uci(move) == uci:
            return move
    raise AssertionError(f"{uci!r} is not a legal move in {board.to_fen()!r}")


def _naive_move_score(evaluator: Evaluator, board: Board, move: int) -> int:
    """The score a quiescence-*disabled* search would report for `move`,
    from the mover's point of view: make the move, take the static
    evaluation at that leaf exactly as it stands (no follow-up captures
    considered), negate back across the one ply played (negamax
    convention), and restore `board` exactly as found.
    """
    board.make_move(move)
    try:
        return -evaluator.evaluate(board)
    finally:
        board.unmake_move()


# --- 1. Quiescence disabled: static eval alone is fooled ---------------------


@pytest.mark.parametrize("root_fen, capture_uci", HORIZON_CASES)
def test_static_eval_alone_misjudges_the_hanging_piece(root_fen: str, capture_uci: str) -> None:
    """The quiescence-disabled side of the comparison. A fixed-depth search
    that bottomed out on a bare `evaluator.evaluate(board)` call at the
    horizon -- the Milestone-2 shape architecture.md §15 describes, before
    `_negamax`'s `depth == 0` case was wired to `_quiescence` -- would report
    exactly `_naive_move_score` for the capturing move: the raw post-capture
    snapshot, with no visibility into Black's immediate recapture.
    """
    evaluator = default_evaluator()
    root = parse_fen(root_fen)
    root_fen_before = root.to_fen()

    root_static = evaluator.evaluate(root)  # White's baseline edge, before any trade
    capture = _find_move(root, capture_uci)
    naive_move_score = _naive_move_score(evaluator, root, capture)

    assert root.to_fen() == root_fen_before  # _naive_move_score must restore the board

    # The hand-built position must actually demonstrate the effect: judged
    # by static eval alone, the capture must look like a large, clean
    # material win for White -- far better than White's pre-capture
    # baseline -- even though (per this file's cases) it is really a bad
    # trade once Black recaptures.
    assert naive_move_score > 300, (
        f"{capture_uci!r} in {root_fen!r} should look like a clear material "
        f"win when judged by static eval alone (got {naive_move_score}); "
        "the position doesn't exercise the horizon effect this test needs."
    )
    assert naive_move_score > root_static + 50, (
        f"the naive post-capture score ({naive_move_score}) should look "
        f"substantially better for White than the pre-capture baseline "
        f"({root_static}) -- otherwise there's no illusory gain for "
        "quiescence to later correct."
    )


# --- 2. Quiescence enabled: the same horizon node, corrected -----------------


@pytest.mark.parametrize("root_fen, capture_uci", HORIZON_CASES)
def test_quiescence_corrects_the_same_horizon_node(root_fen: str, capture_uci: str) -> None:
    """The identical horizon node as above, but scored the way
    `Search._quiescence` actually scores a depth-0 leaf (architecture.md
    §9.5): captures are kept alive past the naive cutoff, gated by
    `see_ge`, until the position is quiet. Black's immediate recapture here
    is unambiguously favorable (it wins back the capturing piece for far
    less than it cost), so quiescence finds it and the corrected score for
    the same node is far lower for White than the naive snapshot -- the
    piece placed en prise by the capturing move was never really won.
    """
    evaluator = default_evaluator()
    search = Search(evaluator)
    root = parse_fen(root_fen)
    capture = _find_move(root, capture_uci)

    root.make_move(capture)
    horizon_fen = root.to_fen()
    naive_move_score = -evaluator.evaluate(root)

    ctx = _SearchCtx(SearchLimits(max_depth=1), deadline=None, stop_event=None, extra_stop=None)
    horizon_quiescence_score = search._quiescence(root, -INF, INF, 1, ctx)
    corrected_move_score = -horizon_quiescence_score

    # `_quiescence` must leave the horizon board exactly as it found it --
    # it only makes/unmakes moves internally while exploring captures.
    assert root.to_fen() == horizon_fen

    assert corrected_move_score < naive_move_score - 200, (
        f"quiescence should substantially correct the naive, pre-recapture "
        f"score ({naive_move_score}) downward once the follow-up recapture "
        f"is accounted for; got a corrected score of {corrected_move_score}, "
        "which is not meaningfully lower -- the horizon effect was not "
        "actually resolved."
    )
    assert corrected_move_score < 100, (
        f"corrected score {corrected_move_score} still looks like a clean "
        "material win for White; Black's recapture should have equalized "
        "(or worsened) it instead."
    )


# --- 3. The real, public entry point: Search.search --------------------------


@pytest.mark.parametrize("root_fen, capture_uci", HORIZON_CASES)
def test_search_does_not_hang_the_piece(root_fen: str, capture_uci: str) -> None:
    """The end-to-end check the Milestone 4 exit criteria actually ask for:
    `Search.search` (architecture.md §9.3), which since Milestone 4 always
    resolves its depth-0 leaves through `_quiescence` rather than a bare
    static eval (architecture.md §9.1, §9.5). Even at a tiny fixed depth of
    1 ply, it must not be fooled into thinking the capturing move wins
    material for free: it must neither choose that move as `best_move` nor
    report anywhere close to the naive, pre-recapture score for the
    position as a whole.
    """
    evaluator = default_evaluator()
    root = parse_fen(root_fen)
    root_fen_before = root.to_fen()
    capture = _find_move(root, capture_uci)

    # The naive, quiescence-disabled score for the bad capture -- the
    # misjudged number the real search below must not reproduce.
    naive_move_score = _naive_move_score(evaluator, root, capture)
    assert root.to_fen() == root_fen_before

    search = Search(evaluator)
    result = search.search(root, SearchLimits(max_depth=1))

    assert result.best_move != capture, (
        f"Search.search chose the losing capture {capture_uci!r} as its "
        f"best move in {root_fen!r} ({move_to_uci(result.best_move)!r} was "
        "expected instead) -- it hung material to a follow-up recapture "
        "that quiescence should have found."
    )
    assert result.score_cp < naive_move_score - 50, (
        f"Search.search's reported score ({result.score_cp}) is too close "
        f"to the naive, horizon-blind score for the bad capture "
        f"({naive_move_score}); it should reflect the corrected, "
        "post-recapture material picture instead, not the illusory "
        "pre-recapture gain."
    )

    # `Search.search` must also leave the board exactly as it found it.
    assert root.to_fen() == root_fen_before
