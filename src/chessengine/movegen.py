"""Legal chess move generation (architecture.md §5.5-§5.10).

Pseudo-legal generation per piece type, plus the machinery that turns
pseudo-legal moves into exactly the legal ones: explicit pin masks
(§5.5), check-evasion masks (§5.6), castling legality (§5.7), en passant
including its discovered-check edge case (§5.8), and promotion (§5.9).

Per the module-boundary DAG (architecture.md §11), `movegen.py` depends on
`constants`, `bitboard`, `attacks`, `board`, and `move` — nothing below it
in the DAG ever imports `movegen.py` (in particular `board.py` never
imports this module), so importing `Board` directly here (not merely under
`TYPE_CHECKING`) creates no cycle.

No magic bitboards (architecture.md §5.1): sliding move generation is built
entirely on `attacks.sliding_attacks`/`bishop_attacks`/`rook_attacks`, the
classical ray-scanning primitives.

The legal-move generator is pseudo-legal generation + a fast filter for the
common case (pin masks, check-evasion masks, and explicit king-move safety
checks), with one deliberate, narrow exception: every en passant candidate
is legality-checked via a make/unmake + `attacks.is_attacked` round trip,
because it is the one move where two pieces leave the board simultaneously
and can expose a horizontal discovered check that neither the pin table nor
the evasion mask accounts for (§5.8).
"""

from __future__ import annotations

from . import attacks
from .attacks import (
    DIAG_DIRS,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    ORTHO_DIRS,
    PAWN_ATTACKS,
    POSITIVE_DIRS,
    RAY_ATTACKS,
    SQUARES_BETWEEN,
    sliding_attacks,
)
from .bitboard import BB_ALL, iter_bits, lsb_index, msb_index
from .board import Board
from .constants import (
    B1,
    B8,
    BISHOP,
    BLACK,
    C1,
    C8,
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
    G1,
    G8,
    KNIGHT,
    PAWN,
    QUEEN,
    RANK_MASK,
    ROOK,
    SQUARE_NAMES,
    WHITE,
    piece_type_of,
)
from .move import (
    CAPTURE,
    DOUBLE_PAWN_PUSH,
    EN_PASSANT,
    KING_CASTLE,
    PROMO_BISHOP,
    PROMO_BISHOP_CAP,
    PROMO_KNIGHT,
    PROMO_KNIGHT_CAP,
    PROMO_PIECE_OF,
    PROMO_QUEEN,
    PROMO_QUEEN_CAP,
    PROMO_ROOK,
    PROMO_ROOK_CAP,
    QUEEN_CASTLE,
    QUIET,
    encode_move,
    is_capture,
    is_promotion,
    move_flag,
    move_from,
    move_to,
)

# --- Pins (architecture.md §5.5) ---------------------------------------------


def pinned_pieces(board: Board, color: int) -> dict[int, int]:
    """{pinned_square: allowed_destination_mask}. A pinned piece may only
    move within its returned mask (the squares between the king and the
    pinner, plus the pinner's own square).

    Computed once per `generate_legal_moves` call by ray-scanning outward
    from the king in all eight directions: the first blocker on a ray must
    be an own piece (otherwise there is nothing to pin), and the next
    blocker beyond it must be an enemy slider that actually attacks along
    that ray direction (a bishop/queen on a diagonal, a rook/queen on a
    file/rank) for the first blocker to be pinned.
    """
    king_sq = board.king_square(color)
    enemy = 1 - color
    occ, own = board.occupied, board.occupied_co[color]
    result: dict[int, int] = {}
    for d in range(8):
        ray = RAY_ATTACKS[d][king_sq]
        blockers = ray & occ
        if not blockers:
            continue
        first = lsb_index(blockers) if d in POSITIVE_DIRS else msb_index(blockers)
        if not (own >> first) & 1:
            continue  # nearest piece on this ray is the enemy's: no pin
        blockers_beyond = RAY_ATTACKS[d][first] & occ
        if not blockers_beyond:
            continue
        second = lsb_index(blockers_beyond) if d in POSITIVE_DIRS else msb_index(blockers_beyond)
        if not (board.occupied_co[enemy] >> second) & 1:
            continue
        pt = piece_type_of(board.mailbox[second])
        slides_this_way = (
            pt == QUEEN or (pt == ROOK and d in ORTHO_DIRS) or (pt == BISHOP and d in DIAG_DIRS)
        )
        if slides_this_way:
            result[first] = SQUARES_BETWEEN[king_sq][second] | (1 << second)
    return result


# --- Check evasion masks (architecture.md §5.6) ------------------------------


def evasion_masks(board: Board, color: int) -> tuple[int, int]:
    """(capture_mask, push_mask): a non-king move is legal only if its
    destination is in (capture_mask | push_mask). Both are 'all squares'
    when not in check.
    """
    ch = attacks.checkers(board, color)
    if ch == 0:
        return BB_ALL, BB_ALL
    if ch.bit_count() >= 2:
        return 0, 0  # double check: only king moves are legal
    checker_sq = lsb_index(ch)
    checker_type = piece_type_of(board.mailbox[checker_sq])
    king_sq = board.king_square(color)
    push_mask = SQUARES_BETWEEN[king_sq][checker_sq] if checker_type in (BISHOP, ROOK, QUEEN) else 0
    return ch, push_mask


# --- Castling (architecture.md §5.7) -----------------------------------------
#
# Precomputed per (color, side) lookup tables. Applying a castle (moving the
# rook, spoiling rights) is `board.py`'s concern (§6); these tables are only
# about deciding whether the move may be *generated* at all.

CASTLE_EMPTY_MASK = {  # squares that must be empty
    (WHITE, KING_CASTLE): (1 << F1) | (1 << G1),
    (WHITE, QUEEN_CASTLE): (1 << B1) | (1 << C1) | (1 << D1),
    (BLACK, KING_CASTLE): (1 << F8) | (1 << G8),
    (BLACK, QUEEN_CASTLE): (1 << B8) | (1 << C8) | (1 << D8),
}
CASTLE_KING_PATH = {  # king's start/pass-through/landing squares; none may be attacked
    (WHITE, KING_CASTLE): (E1, F1, G1),
    (WHITE, QUEEN_CASTLE): (E1, D1, C1),
    (BLACK, KING_CASTLE): (E8, F8, G8),
    (BLACK, QUEEN_CASTLE): (E8, D8, C8),
}
CASTLE_RIGHT_BIT = {
    (WHITE, KING_CASTLE): CASTLE_WK,
    (WHITE, QUEEN_CASTLE): CASTLE_WQ,
    (BLACK, KING_CASTLE): CASTLE_BK,
    (BLACK, QUEEN_CASTLE): CASTLE_BQ,
}


def generate_castling_moves(board: Board, color: int, in_check: bool, moves: list[int]) -> None:
    """Append any legal castling moves for `color` to `moves`. A castle is
    legal only if: the right hasn't been lost, the squares between king and
    rook are empty, and the king is not currently in check, does not pass
    through, and does not land on an attacked square (it may never move
    through or into check, per the rules of chess)."""
    if in_check:
        return
    enemy = 1 - color
    for side in (KING_CASTLE, QUEEN_CASTLE):
        if not (board.castling_rights & CASTLE_RIGHT_BIT[(color, side)]):
            continue
        if board.occupied & CASTLE_EMPTY_MASK[(color, side)]:
            continue
        if any(attacks.is_attacked(board, sq, enemy) for sq in CASTLE_KING_PATH[(color, side)]):
            continue
        to_sq = CASTLE_KING_PATH[(color, side)][2]
        moves.append(encode_move(CASTLE_KING_PATH[(color, side)][0], to_sq, side))


# --- Per-piece pseudo-legal generators (architecture.md §5.8-§5.10) ----------

_PROMO_QUIET_FLAGS = (PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT)
_PROMO_CAP_FLAGS = (PROMO_QUEEN_CAP, PROMO_ROOK_CAP, PROMO_BISHOP_CAP, PROMO_KNIGHT_CAP)


def _emit_pawn_promotions(frm: int, to: int, capture: bool, moves: list[int]) -> None:
    """A pawn move landing on the back rank is expanded into four moves, one
    per promotion piece (architecture.md §5.9)."""
    for flag in _PROMO_CAP_FLAGS if capture else _PROMO_QUIET_FLAGS:
        moves.append(encode_move(frm, to, flag))


def _gen_pawn_moves(board: Board, color: int, moves: list[int]) -> None:
    occ = board.occupied
    enemy_occ = board.occupied_co[1 - color]
    if color == WHITE:
        push_dir = 8
        start_rank_mask = RANK_MASK[1]  # rank 2
        promo_rank_mask = RANK_MASK[7]  # rank 8
    else:
        push_dir = -8
        start_rank_mask = RANK_MASK[6]  # rank 7
        promo_rank_mask = RANK_MASK[0]  # rank 1

    for frm in iter_bits(board.pieces[color][PAWN]):
        # Single push, and the double push it may unlock.
        to = frm + push_dir
        if 0 <= to < 64 and not (occ >> to) & 1:
            if (1 << to) & promo_rank_mask:
                _emit_pawn_promotions(frm, to, capture=False, moves=moves)
            else:
                moves.append(encode_move(frm, to, QUIET))
                if (1 << frm) & start_rank_mask:
                    to2 = to + push_dir
                    if not (occ >> to2) & 1:
                        moves.append(encode_move(frm, to2, DOUBLE_PAWN_PUSH))

        # Captures (ordinary and promoting).
        for to in iter_bits(PAWN_ATTACKS[color][frm] & enemy_occ):
            if (1 << to) & promo_rank_mask:
                _emit_pawn_promotions(frm, to, capture=True, moves=moves)
            else:
                moves.append(encode_move(frm, to, CAPTURE))

        # En passant (architecture.md §5.8): never lands on the back rank,
        # so no interaction with promotion.
        if board.ep_square is not None and (PAWN_ATTACKS[color][frm] >> board.ep_square) & 1:
            moves.append(encode_move(frm, board.ep_square, EN_PASSANT))


def _gen_knight_moves(board: Board, color: int, moves: list[int]) -> None:
    own = board.occupied_co[color]
    enemy_occ = board.occupied_co[1 - color]
    for frm in iter_bits(board.pieces[color][KNIGHT]):
        targets = KNIGHT_ATTACKS[frm] & ~own
        for to in iter_bits(targets):
            flag = CAPTURE if (enemy_occ >> to) & 1 else QUIET
            moves.append(encode_move(frm, to, flag))


def _gen_king_moves(board: Board, color: int, moves: list[int]) -> None:
    own = board.occupied_co[color]
    enemy_occ = board.occupied_co[1 - color]
    frm = board.king_square(color)
    for to in iter_bits(KING_ATTACKS[frm] & ~own):
        flag = CAPTURE if (enemy_occ >> to) & 1 else QUIET
        moves.append(encode_move(frm, to, flag))
    in_check = attacks.is_attacked(board, frm, 1 - color)
    generate_castling_moves(board, color, in_check, moves)


def _gen_sliding_moves(
    board: Board, color: int, piece_type: int, dirs: tuple[int, ...], moves: list[int]
) -> None:
    own = board.occupied_co[color]
    enemy_occ = board.occupied_co[1 - color]
    occ = board.occupied
    for frm in iter_bits(board.pieces[color][piece_type]):
        targets = sliding_attacks(frm, occ, dirs) & ~own
        for to in iter_bits(targets):
            flag = CAPTURE if (enemy_occ >> to) & 1 else QUIET
            moves.append(encode_move(frm, to, flag))


# --- Putting it together (architecture.md §5.10) -----------------------------


def generate_pseudo_legal_moves(board: Board) -> list[int]:
    """Moves obeying piece movement rules, not yet filtered for leaving
    one's own king in check."""
    color = board.side_to_move
    moves: list[int] = []
    _gen_pawn_moves(board, color, moves)
    _gen_knight_moves(board, color, moves)
    _gen_king_moves(board, color, moves)  # includes castling
    _gen_sliding_moves(board, color, BISHOP, DIAG_DIRS, moves)
    _gen_sliding_moves(board, color, ROOK, ORTHO_DIRS, moves)
    _gen_sliding_moves(board, color, QUEEN, ORTHO_DIRS + DIAG_DIRS, moves)
    return moves


def generate_legal_moves(board: Board) -> list[int]:
    """Exactly the legal moves in `board`'s current position.

    Pseudo-legal moves are filtered by: an explicit make/unmake fallback for
    en passant (the one case that can expose a discovered check neither the
    pin table nor the evasion mask accounts for, §5.8); an explicit
    king-move safety check against the occupancy with the king itself
    removed (so a slider's attack through the king's own square is counted,
    §5.10); the pin mask for every other piece; and the check-evasion mask
    for the rest.
    """
    color = board.side_to_move
    king_sq = board.king_square(color)
    capture_mask, push_mask = evasion_masks(board, color)
    pins = pinned_pieces(board, color)
    legal: list[int] = []
    for m in generate_pseudo_legal_moves(board):
        frm, to, flag = move_from(m), move_to(m), move_flag(m)
        if flag == EN_PASSANT:
            board.make_move(m)  # §5.8 fallback
            ok = not attacks.is_attacked(board, board.king_square(color), 1 - color)
            board.unmake_move()
            if ok:
                legal.append(m)
            continue
        if frm == king_sq:
            if flag in (KING_CASTLE, QUEEN_CASTLE):
                legal.append(m)  # already fully validated by generate_castling_moves
                continue
            occ_without_king = board.occupied & ~(1 << king_sq)
            if attacks.attackers_to(board, to, 1 - color, occupied=occ_without_king) == 0:
                legal.append(m)
            continue
        if frm in pins and not (pins[frm] >> to) & 1:
            continue  # pinned piece moving off its pin line
        if not ((capture_mask | push_mask) >> to) & 1:
            continue  # doesn't capture the checker or block the check
        legal.append(m)
    return legal


def generate_captures(board: Board) -> list[int]:
    """Subset used by quiescence search (§9.5): captures, en passant, and
    capturing/quiet promotions (a quiet promotion to queen is treated as
    'noisy enough' to search in quiescence even without a capture)."""
    return [m for m in generate_legal_moves(board) if is_capture(m) or is_promotion(m)]


# --- UCI move resolution ------------------------------------------------------

_PROMO_LETTER_OF_PIECE = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}


def move_from_uci(board: Board, uci: str) -> int:
    """Resolve a UCI move string (e.g. "e2e4", "e7e8q") against the current
    position's legal move list.

    Lives here, not in `move.py` (§4, §11): resolving a UCI string (which
    carries no flag bits) requires matching against the legal move list,
    and `move.py` may not depend on `movegen.py` without creating an import
    cycle. Resolving against the generator rather than re-deriving a flag
    from scratch means this can never construct a move the generator itself
    would not produce.
    """
    frm, to = SQUARE_NAMES.index(uci[0:2]), SQUARE_NAMES.index(uci[2:4])
    promo = uci[4] if len(uci) == 5 else None
    for m in generate_legal_moves(board):
        if move_from(m) != frm or move_to(m) != to:
            continue
        if promo is None and not is_promotion(m):
            return m
        if (
            promo is not None
            and is_promotion(m)
            and _PROMO_LETTER_OF_PIECE[PROMO_PIECE_OF[move_flag(m)]] == promo
        ):
            return m
    raise ValueError(f"illegal or unknown move: {uci}")
