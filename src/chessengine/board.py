"""The `Board` class: bitboard + mailbox position state, and make/unmake move
(architecture.md §3.2, §6).

Per the module-boundary DAG (architecture.md §11), `board.py` depends only on
`constants`, `bitboard`, `attacks`, `zobrist`, and `move` — never on `fen.py`
or `movegen.py` at module scope. `fen.py` depends on `board.py` (it
constructs and reads a `Board`), so importing it here at module scope would
create a cycle; `Board.starting_position()` therefore imports it lazily,
inside the method body, the one documented exception to "imports happen at
module scope" (§3.2, §8, §11).

`pieces`/`occupied_co`/`occupied`/`mailbox` are redundant encodings of the
same position, kept in sync as invariants by `make_move`/`unmake_move` alone
(single-writer discipline) — nothing else in this module, or any other
module, ever mutates them directly. `_add_piece`/`_remove_piece`/
`_move_piece` are the only places that do, which is what makes the
occupied/mailbox/zobrist invariants (architecture.md §13) checkable at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import attacks
from .bitboard import lsb_index
from .constants import (
    A1,
    A8,
    BLACK,
    CASTLE_BK,
    CASTLE_BQ,
    CASTLE_WK,
    CASTLE_WQ,
    D1,
    D8,
    E1,
    E8,
    F1,
    F8,
    H1,
    H8,
    KING,
    NO_PIECE,
    PAWN,
    PIECE_VALUE,
    PST,
    ROOK,
    WHITE,
    color_of,
    file_of,
    piece_type_of,
)
from .move import (
    DOUBLE_PAWN_PUSH,
    EN_PASSANT,
    KING_CASTLE,
    PROMO_PIECE_OF,
    QUEEN_CASTLE,
    move_flag,
    move_from,
    move_to,
)
from .zobrist import ZOBRIST_CASTLING, ZOBRIST_EP_FILE, ZOBRIST_PIECE, ZOBRIST_SIDE

# --- Castling bookkeeping tables (architecture.md §5.7, §6) ------------------
#
# `make_move`/`unmake_move` are the only code that actually moves the rook on
# a castle or strips castling rights on a king/rook move-or-capture, so these
# two tables live here rather than in `movegen.py` — `board.py` may not
# import `movegen.py` (movegen depends on board, per §11's DAG, not the other
# way around). `movegen.py`'s own castling-generation tables
# (`CASTLE_EMPTY_MASK`, `CASTLE_KING_PATH`, `CASTLE_RIGHT_BIT`) are a
# separate concern (which squares must be empty/unattacked to *generate* the
# move) and belong there instead.

CASTLE_ROOK_MOVE = {
    (KING_CASTLE, WHITE): (H1, F1),
    (QUEEN_CASTLE, WHITE): (A1, D1),
    (KING_CASTLE, BLACK): (H8, F8),
    (QUEEN_CASTLE, BLACK): (A8, D8),
}

CASTLE_SPOILER = [0] * 64
CASTLE_SPOILER[E1] = CASTLE_WK | CASTLE_WQ
CASTLE_SPOILER[A1] = CASTLE_WQ
CASTLE_SPOILER[H1] = CASTLE_WK
CASTLE_SPOILER[E8] = CASTLE_BK | CASTLE_BQ
CASTLE_SPOILER[A8] = CASTLE_BQ
CASTLE_SPOILER[H8] = CASTLE_BK

_PIECE_CHARS = "PNBRQK"  # indexed by piece_type_of(...): pawn/knight/bishop/rook/queen/king


@dataclass(slots=True)
class UndoInfo:
    """Everything `make_move` cannot cheaply re-derive from the move encoding
    alone, captured before the move is applied so `unmake_move` can restore
    it exactly (architecture.md §3.2)."""

    move: int  # encoded move (§4)
    captured_piece: int  # piece code 0..11, or NO_PIECE
    captured_square: int  # differs from move's `to` only for en passant
    castling_rights: int  # rights *before* this move
    ep_square: int | None  # en-passant target square *before* this move
    halfmove_clock: int  # 50-move counter *before* this move
    zobrist_hash: int  # full hash *before* this move
    material_pst_score: int  # White-relative material+PST total *before* this move
    pawn_hash: int  # pawn-only Zobrist hash *before* this move


class Board:
    __slots__ = (
        "pieces",
        "occupied_co",
        "occupied",
        "mailbox",
        "side_to_move",
        "castling_rights",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
        "zobrist_hash",
        "history",
        "position_history",
        "material_pst_score",
        "null_move_history",
        "pawn_hash",
    )

    def __init__(self) -> None:
        self.pieces: list[list[int]] = [[0] * 6 for _ in range(2)]  # [color][piece_type] -> bitboard
        self.occupied_co: list[int] = [0, 0]  # per-color union
        self.occupied: int = 0  # union of both colors
        self.mailbox: list[int] = [NO_PIECE] * 64  # square -> piece code
        self.side_to_move: int = WHITE
        self.castling_rights: int = 0  # 4-bit mask (§3.1)
        self.ep_square: int | None = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.zobrist_hash: int = 0
        self.history: list[UndoInfo] = []
        self.position_history: list[int] = []  # hashes seen, for repetition
        self.material_pst_score: int = 0  # White's PIECE_VALUE+PST total minus Black's (§10.1)
        self.pawn_hash: int = 0  # pawn-only Zobrist hash (for pawn structure caching)
        # Dedicated undo stack for make_null_move/unmake_null_move (§6), kept
        # fully separate from `history`/`UndoInfo` so null-move bookkeeping
        # can never disturb the already-perft-verified real move machinery.
        # Each entry is (ep_square_before, zobrist_hash_before).
        self.null_move_history: list[tuple[int | None, int]] = []

    @staticmethod
    def starting_position() -> "Board":
        from . import fen  # lazy import: fen.py depends

        return fen.parse_fen(fen.STARTPOS_FEN)  # on board.py, not vice versa

    # --- Queries --------------------------------------------------------

    def king_square(self, color: int) -> int:
        return lsb_index(self.pieces[color][KING])

    def in_check(self, color: int | None = None) -> bool:
        c = self.side_to_move if color is None else color
        return attacks.checkers(self, c) != 0  # delegates to attacks.checkers

    def is_fifty_move_draw(self) -> bool:
        return self.halfmove_clock >= 100

    def is_repetition_draw(self) -> bool:
        """Treats the current position recurring once before (its 2nd
        occurrence in tracked history) as a draw — stricter than the
        official threefold rule, a standard, safe simplification for the
        search's internal draw detection (§9.6)."""
        if not self.position_history:
            return False
        current = self.position_history[-1]
        return self.position_history[:-1].count(current) >= 1

    # --- Mutators (§6) ----------------------------------------------------

    def make_move(self, move: int) -> None:
        frm, to, flag = move_from(move), move_to(move), move_flag(move)
        us, them = self.side_to_move, 1 - self.side_to_move
        moving_piece = self.mailbox[frm]
        moving_type = piece_type_of(moving_piece)

        captured_piece, captured_sq = NO_PIECE, to
        if flag == EN_PASSANT:
            captured_sq = to - 8 if us == WHITE else to + 8
            captured_piece = self.mailbox[captured_sq]
        elif flag & 0x4:  # CAPTURE or a *_CAP promotion
            captured_piece = self.mailbox[to]

        self.history.append(
            UndoInfo(
                move,
                captured_piece,
                captured_sq,
                self.castling_rights,
                self.ep_square,
                self.halfmove_clock,
                self.zobrist_hash,
                self.material_pst_score,
                self.pawn_hash,
            )
        )

        if captured_piece != NO_PIECE:
            self._remove_piece(captured_piece, captured_sq)
        self._move_piece(moving_piece, frm, to)

        if flag in PROMO_PIECE_OF:
            self._remove_piece(moving_piece, to)
            self._add_piece(us * 6 + PROMO_PIECE_OF[flag], to)
        elif flag in (KING_CASTLE, QUEEN_CASTLE):
            rook_from, rook_to = CASTLE_ROOK_MOVE[(flag, us)]
            self._move_piece(us * 6 + ROOK, rook_from, rook_to)

        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights]
        self.castling_rights &= ~(CASTLE_SPOILER[frm] | CASTLE_SPOILER[to])
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights]

        if self.ep_square is not None:
            self.zobrist_hash ^= ZOBRIST_EP_FILE[file_of(self.ep_square)]
        self.ep_square = (frm + to) // 2 if flag == DOUBLE_PAWN_PUSH else None
        if self.ep_square is not None:
            self.zobrist_hash ^= ZOBRIST_EP_FILE[file_of(self.ep_square)]

        self.halfmove_clock = (
            0 if (moving_type == PAWN or captured_piece != NO_PIECE) else self.halfmove_clock + 1
        )
        if us == BLACK:
            self.fullmove_number += 1
        self.side_to_move = them
        self.zobrist_hash ^= ZOBRIST_SIDE
        self.position_history.append(self.zobrist_hash)

    def unmake_move(self) -> None:
        undo = self.history.pop()
        self.position_history.pop()
        move, flag = undo.move, move_flag(undo.move)
        frm, to = move_from(move), move_to(move)
        self.side_to_move = 1 - self.side_to_move
        us = self.side_to_move

        if flag in PROMO_PIECE_OF:
            self._remove_piece(us * 6 + PROMO_PIECE_OF[flag], to)
            self._add_piece(us * 6 + PAWN, frm)
        else:
            self._move_piece(self.mailbox[to], to, frm)

        if flag in (KING_CASTLE, QUEEN_CASTLE):
            rook_from, rook_to = CASTLE_ROOK_MOVE[(flag, us)]
            self._move_piece(us * 6 + ROOK, rook_to, rook_from)

        if undo.captured_piece != NO_PIECE:
            self._add_piece(undo.captured_piece, undo.captured_square)

        self.castling_rights = undo.castling_rights
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove_clock
        self.zobrist_hash = undo.zobrist_hash
        self.material_pst_score = undo.material_pst_score
        self.pawn_hash = undo.pawn_hash
        if us == BLACK:
            self.fullmove_number -= 1

    # --- Null move (search-internal only, §15) -----------------------------
    #
    # A "null move" passes the turn without moving any piece — it exists
    # purely for search's null-move pruning and must never be used during
    # real game play (it is not a legal chess move). It therefore gets its
    # own tiny, dedicated undo stack (`null_move_history`) instead of
    # `history`/`UndoInfo`: real moves and null moves are bookkept through
    # completely separate code paths, so a null-move bug can never corrupt
    # the already-perft-verified real make_move/unmake_move machinery, and
    # vice versa.

    def make_null_move(self) -> None:
        """Pass the turn with no piece moved.

        Only `side_to_move`, `ep_square`, and the corresponding components of
        `zobrist_hash` change — mirroring exactly what a real move does to
        those fields when en passant expires and the side to move flips.
        `pieces`/`occupied_co`/`occupied`/`mailbox`/`castling_rights`/
        `material_pst_score` are untouched, since nothing moved.

        `halfmove_clock` is deliberately left unincremented. A null move is
        neither a pawn move nor a capture, so incrementing it would be
        modeling something that didn't happen; and since this method is
        search-internal only (always paired with `unmake_null_move` before
        control returns to anything that could act on a real fifty-move
        draw), there is no observable difference either way. We choose not
        to touch it at all, the simplest option that keeps this method's
        contract to "only side/ep/hash change."

        Does not push onto `position_history`: the resulting position is a
        search-internal fiction never actually reached in the game, so it
        must never feed repetition detection.
        """
        self.null_move_history.append((self.ep_square, self.zobrist_hash))
        if self.ep_square is not None:
            self.zobrist_hash ^= ZOBRIST_EP_FILE[file_of(self.ep_square)]
            self.ep_square = None
        self.side_to_move = 1 - self.side_to_move
        self.zobrist_hash ^= ZOBRIST_SIDE

    def unmake_null_move(self) -> None:
        """Undo the most recent `make_null_move`, restoring `side_to_move`,
        `ep_square`, and `zobrist_hash` exactly (the only fields it changed)."""
        ep_square_before, zobrist_hash_before = self.null_move_history.pop()
        self.side_to_move = 1 - self.side_to_move
        self.ep_square = ep_square_before
        self.zobrist_hash = zobrist_hash_before

    # --- Private single-writer helpers (§6) --------------------------------
    #
    # These are the *only* places `pieces`, `occupied_co`, `occupied`,
    # `mailbox`, the piece-square component of `zobrist_hash`, and
    # `material_pst_score` are ever mutated, which is what keeps the
    # invariants in §13 checkable.

    def _add_piece(self, piece_code: int, sq: int) -> None:
        color, ptype, bit = color_of(piece_code), piece_type_of(piece_code), 1 << sq
        self.pieces[color][ptype] |= bit
        self.occupied_co[color] |= bit
        self.occupied |= bit
        self.mailbox[sq] = piece_code
        self.zobrist_hash ^= ZOBRIST_PIECE[color][ptype][sq]
        if ptype == PAWN:
            self.pawn_hash ^= ZOBRIST_PIECE[color][ptype][sq]
        contribution = PIECE_VALUE[ptype] + (PST[ptype][sq ^ 56] if color == BLACK else PST[ptype][sq])
        self.material_pst_score += contribution if color == WHITE else -contribution

    def _remove_piece(self, piece_code: int, sq: int) -> None:
        color, ptype, bit = color_of(piece_code), piece_type_of(piece_code), 1 << sq
        self.pieces[color][ptype] &= ~bit
        self.occupied_co[color] &= ~bit
        self.occupied &= ~bit
        self.mailbox[sq] = NO_PIECE
        self.zobrist_hash ^= ZOBRIST_PIECE[color][ptype][sq]
        if ptype == PAWN:
            self.pawn_hash ^= ZOBRIST_PIECE[color][ptype][sq]
        contribution = PIECE_VALUE[ptype] + (PST[ptype][sq ^ 56] if color == BLACK else PST[ptype][sq])
        self.material_pst_score -= contribution if color == WHITE else -contribution

    def _move_piece(self, piece_code: int, frm: int, to: int) -> None:
        self._remove_piece(piece_code, frm)
        self._add_piece(piece_code, to)

    # --- Debug / IO ---------------------------------------------------------

    def to_fen(self) -> str:
        from . import fen  # lazy import, same reason as starting_position()

        return fen.board_to_fen(self)

    def __str__(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            cells = []
            for file in range(8):
                code = self.mailbox[rank * 8 + file]
                if code == NO_PIECE:
                    cells.append(".")
                else:
                    ch = _PIECE_CHARS[piece_type_of(code)]
                    cells.append(ch if color_of(code) == WHITE else ch.lower())
            rows.append(f"{rank + 1}  " + " ".join(cells))
        rows.append("   a b c d e f g h")
        return "\n".join(rows)
