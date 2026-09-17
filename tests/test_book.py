"""`test_book.py` (architecture.md §15, Milestone 5): the opening-book gate.

Four things, per the milestone's exit criteria:

1. `probe_book` returns a *legal* move for the starting position, and that
   move is one of the UCI strings this book's own startpos entry actually
   lists (not merely "some legal move that happens to also be legal").
2. `probe_book` returns `None` once the position is clearly outside this
   book's demonstration repertoire -- both a standard, well-known mid-game
   FEN (Kiwipete) and a short line that leaves the repertoire almost
   immediately and is then walked several plies deeper.
3. Walking a few plies through the book (probe, make the move, probe again,
   ...) stays legal and consistent at every step, for one complete opening
   line end to end.
4. A sanity check on `OPENING_BOOK`'s *data* itself, not just `probe_book`'s
   lookup logic: every `(uci, weight)` pair stored under every key decodes
   (via `movegen.move_from_uci`) to an actually legal move in the position
   that key's hash corresponds to. Since a bare Zobrist hash can't be turned
   back into a `Board`, this replays the book from `Board.starting_position()`
   exactly the way book.py's own generation script (see its module
   docstring) built the table in the first place, reconstructing the right
   board for every keyed hash along the way.
"""

from __future__ import annotations

from chessengine.board import Board
from chessengine.book import OPENING_BOOK, probe_book
from chessengine.fen import parse_fen
from chessengine.move import move_to_uci
from chessengine.movegen import generate_legal_moves, move_from_uci

# Kiwipete: a standard, well-known tactical mid-game test FEN, nowhere near
# any opening theory this book's demonstration repertoire (architecture.md
# §15's "after 1.e4 ...", "after 1.d4 ...", etc.) could plausibly cover.
_KIWIPETE_FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# A short line that leaves this book's repertoire almost immediately --
# 1...Nf6 (the Alekhine Defense) is not one of this book's four listed
# replies to 1.e4 (e7e5 / c7c5 / e7e6 / c7c6) -- and is then walked several
# plies further, well past anything the demonstration repertoire covers.
_OUT_OF_BOOK_LINE = ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"]

# The book's own highest-weight choice at every node (ties broken by
# first-listed entry, per book.py's docstring) is a fully deterministic line:
# 1.e4 e5 2.Nf3 Nc6 3.Bb5 (the Ruy Lopez), five plies deep, after which this
# book has no entry for the resulting position.
_EXPECTED_GREEDY_LINE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]


# --- 1. probe_book on the starting position ---------------------------------


def test_probe_book_returns_legal_startpos_move() -> None:
    board = Board.starting_position()
    move = probe_book(board)

    assert move is not None, "startpos must be in this book's repertoire"

    legal_moves = generate_legal_moves(board)
    assert move in legal_moves, (
        f"probe_book returned {move_to_uci(move)!r} for startpos, which isn't "
        f"even legal there: {sorted(move_to_uci(m) for m in legal_moves)!r}"
    )

    book_ucis = {uci for uci, _weight in OPENING_BOOK[board.zobrist_hash]}
    assert move_to_uci(move) in book_ucis, (
        f"probe_book returned {move_to_uci(move)!r}, which isn't one of the "
        f"startpos entry's own UCI strings {sorted(book_ucis)!r}"
    )

    # And per book.py's own tie-break rule (highest weight wins), this must
    # be the max-weight entry: 1.e4.
    assert move_to_uci(move) == "e2e4"


# --- 2. probe_book returns None once clearly out of book ---------------------


def test_probe_book_returns_none_for_kiwipete() -> None:
    """Kiwipete is a standard mid-game test position with no relation to any
    opening line -- a textbook "definitely not in an opening book" case."""
    board = parse_fen(_KIWIPETE_FEN)
    assert probe_book(board) is None


def test_probe_book_returns_none_several_plies_out_of_book() -> None:
    board = Board.starting_position()
    for uci in _OUT_OF_BOOK_LINE:
        board.make_move(move_from_uci(board, uci))

    assert probe_book(board) is None, (
        f"probe_book found a hit after {_OUT_OF_BOOK_LINE!r}, a line well "
        "outside this book's demonstration repertoire"
    )


# --- 3. Walking a full opening line through the book -------------------------


def test_walking_a_full_opening_line_stays_legal_and_consistent() -> None:
    """Make a book move, probe again, make that move, probe again, ... for one
    complete opening line, asserting at every ply that the move `probe_book`
    hands back is actually legal in the position it was just probed from.
    The walk necessarily ends once the position falls off the book's
    repertoire -- nothing here pretends the book is infinite."""
    board = Board.starting_position()
    played: list[str] = []

    for _ply in range(20):  # generous cap; the real line below is 5 plies deep
        move = probe_book(board)
        if move is None:
            break
        legal_moves = generate_legal_moves(board)
        assert move in legal_moves, (
            f"probe_book returned an illegal move after {played!r} "
            f"(fen {board.to_fen()!r}): {move_to_uci(move)!r} not in "
            f"{sorted(move_to_uci(m) for m in legal_moves)!r}"
        )
        played.append(move_to_uci(move))
        board.make_move(move)

    assert played == _EXPECTED_GREEDY_LINE, (
        f"expected the book's deterministic greedy line {_EXPECTED_GREEDY_LINE!r}, "
        f"got {played!r}"
    )
    assert len(played) >= 4, "expected at least a full opening line, not a single ply"


# --- 4. Sanity-check the OPENING_BOOK data itself, not just the lookup ------


def test_every_book_move_decodes_to_a_legal_move_on_its_own_position() -> None:
    """Every `(uci, weight)` pair stored under every `OPENING_BOOK` key must
    decode, via `movegen.move_from_uci`, to a move that's actually legal in
    the position that key's hash corresponds to.

    There is no way to go from a bare Zobrist hash back to a `Board`, so this
    reconstructs the right board for each keyed hash the same way book.py's
    own generation script (see its module docstring) built the table in the
    first place: starting from `Board.starting_position()`, recursively walk
    every `(uci, weight)` entry, make the move, and recurse into whatever
    entry (if any) exists for the resulting position.
    """
    visited: set[int] = set()

    def walk(board: Board) -> None:
        entries = OPENING_BOOK.get(board.zobrist_hash)
        if entries is None or board.zobrist_hash in visited:
            return
        visited.add(board.zobrist_hash)
        for uci, _weight in entries:
            try:
                move = move_from_uci(board, uci)
            except ValueError as exc:
                raise AssertionError(
                    f"OPENING_BOOK entry {uci!r} at hash {board.zobrist_hash} "
                    f"(fen {board.to_fen()!r}) does not decode to a legal move: {exc}"
                ) from exc
            assert move in generate_legal_moves(board), (
                f"OPENING_BOOK entry {uci!r} at hash {board.zobrist_hash} "
                f"(fen {board.to_fen()!r}) decoded to a move not present in "
                "this position's own legal move list"
            )
            board.make_move(move)
            walk(board)
            board.unmake_move()

    walk(Board.starting_position())

    # Every key in OPENING_BOOK must have been reached this way -- otherwise
    # either the table has an orphaned/unreachable entry, or this test's
    # replay silently failed to retrace one of the book's own branches, and
    # either way some entry would have escaped the per-move check above
    # entirely without this catching it.
    assert visited == set(OPENING_BOOK.keys()), (
        f"reachable-from-startpos keys ({len(visited)}) don't match "
        f"OPENING_BOOK's own keys ({len(OPENING_BOOK)}) -- "
        f"missing: {set(OPENING_BOOK.keys()) - visited!r}"
    )
