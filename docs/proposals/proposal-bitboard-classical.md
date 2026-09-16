# Proposal: Classical Bitboard Chess Engine

- **Status:** Draft proposal
- **Package:** `src/chessengine/`
- **Author:** Francesco Errico (francesco.errico@relatech.com)
- **Date:** 2026-09-16

## 1. Goals & Non-Goals

**Goals**

- A from-scratch Python chess engine: one 64-bit bitboard per (color, piece
  type), classical ray-scanning attack generation (no magic bitboards, no
  external dependencies), make/unmake move with a full undo stack, negamax
  (minimax reformulated with alpha-beta), iterative deepening, a
  Zobrist-keyed transposition table, a material + piece-square-table (PST)
  evaluation, and a UCI protocol handler so the engine can be dropped into
  any standard chess GUI (Arena, CuteChess, `xboard`, etc.).
- Code that is **approachable**: every data structure and function below is
  specified concretely enough to implement directly, favoring readability
  and correctness over squeezing out maximum NPS (nodes per second).
- Perft-validated correctness as the top-level acceptance bar for move
  generation, including the notoriously fiddly rules: en passant, castling
  (through/into check), promotion, pins, and discovered checks.

**Non-goals (v1)**

- Magic bitboards / PEXT sliding attacks (explicitly out of scope; classical
  ray-scanning is chosen for approachability).
- NNUE / learned evaluation, opening book, endgame tablebases.
- Multi-threaded (lazy-SMP) search. The engine is single-threaded; `go` runs
  on one background thread so the UCI stdin loop stays responsive to
  `stop`/`quit`.
- Quiescence search and advanced pruning (null-move, LMR, aspiration
  windows) are **not** part of the v1 contract requested here, but the
  search module is structured so they are additive (see §12).

---

## 2. Board Representation

### 2.1 Square indexing

Little-endian rank-file (LERF) mapping, the conventional bitboard layout:
square index `sq = rank * 8 + file`, so `a1 = 0`, `h1 = 7`, `a8 = 56`,
`h8 = 63`. Bit `i` of a bitboard corresponds to square `i`.

`constants.py`:

```python
# Square constants, a1..h8, generated once:
SQUARE_NAMES = [f + r for r in "12345678" for f in "abcdefgh"]
A1, B1, C1, D1, E1, F1, G1, H1, \
A2, B2, C2, D2, E2, F2, G2, H2, \
A3, B3, C3, D3, E3, F3, G3, H3, \
A4, B4, C4, D4, E4, F4, G4, H4, \
A5, B5, C5, D5, E5, F5, G5, H5, \
A6, B6, C6, D6, E6, F6, G6, H6, \
A7, B7, C7, D7, E7, F7, G7, H7, \
A8, B8, C8, D8, E8, F8, G8, H8 = range(64)

FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H = range(8)
RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 = range(8)

def file_of(sq: int) -> int: return sq & 7
def rank_of(sq: int) -> int: return sq >> 3

FILE_MASK = [0x0101010101010101 << f for f in range(8)]
RANK_MASK = [0xFF << (8 * r) for r in range(8)]
NOT_FILE_A, NOT_FILE_H = ~FILE_MASK[FILE_A] & 0xFFFFFFFFFFFFFFFF, ~FILE_MASK[FILE_H] & 0xFFFFFFFFFFFFFFFF

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
NO_PIECE = 12  # mailbox sentinel; piece_code = color * 6 + piece_type otherwise

def piece_type_of(code: int) -> int: return code % 6
def color_of(code: int) -> int: return code // 6

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8
```

Bitboards are plain Python `int`s masked to 64 bits (`& 0xFFFFFFFFFFFFFFFF`
after any shift that could overflow left, e.g. `<< 1` near file H). Python's
arbitrary-precision ints mean there is no risk of silent 32-bit truncation
that plagues this trick in C, but every left-shift-based generator must
still mask explicitly or it will accumulate garbage high bits.

### 2.2 `Board` class (`board.py`)

```python
@dataclass
class UndoInfo:
    move: int                 # encoded move (see §3)
    captured_piece: int       # piece code, or NO_PIECE
    captured_square: int      # differs from `to` only for en passant
    castling_rights: int      # rights *before* this move
    ep_square: int | None     # en-passant target square *before* this move
    halfmove_clock: int       # 50-move counter *before* this move
    zobrist_hash: int         # full hash *before* this move

class Board:
    def __init__(self) -> None:
        # pieces[color][piece_type] -> bitboard
        self.pieces: list[list[int]] = [[0] * 6 for _ in range(2)]
        self.occupied_co: list[int] = [0, 0]      # per-color occupancy
        self.occupied: int = 0                    # union of both colors
        self.mailbox: list[int] = [NO_PIECE] * 64  # square -> piece code
        self.side_to_move: int = WHITE
        self.castling_rights: int = 0              # 4-bit mask, see constants
        self.ep_square: int | None = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.zobrist_hash: int = 0
        self.history: list[UndoInfo] = []          # move-undo stack
        self.position_history: list[int] = []      # hashes, for repetition

    @staticmethod
    def starting_position() -> "Board": ...        # builds via fen.parse_fen(STARTPOS_FEN)

    # Queries
    def king_square(self, color: int) -> int: ...
    def in_check(self, color: int | None = None) -> bool: ...
    def is_fifty_move_draw(self) -> bool: return self.halfmove_clock >= 100
    def is_repetition_draw(self) -> bool: ...        # see §7.6

    # Mutators (§5)
    def make_move(self, move: int) -> None: ...
    def unmake_move(self) -> None: ...

    # Debug / IO
    def to_fen(self) -> str: ...                     # delegates to fen.py
    def __str__(self) -> str: ...                    # ASCII board for debugging
```

`occupied_co`/`occupied` are redundant with `pieces` but are kept as
maintained invariants (updated in lockstep inside `make_move`/`unmake_move`)
because sliding-attack generation and pseudo-legal generation both need
"all pieces" and "own/enemy pieces" on nearly every call; recomputing them
by OR-ing 6 bitboards on every query would be wasteful. `mailbox` exists so
"what piece, if any, sits on square X" is an O(1) array lookup instead of a
12-bitboard scan — needed constantly by move generation, SEE-style capture
logic, and `make_move`/`unmake_move`.

**Invariant** (checked by tests, §11): `self.occupied == self.occupied_co[WHITE] | self.occupied_co[BLACK]`,
and for every color/piece, `popcount(pieces[c][p])` bits of `mailbox` equal
`color*6+p` — i.e. the three representations never disagree.

---

## 3. Move Encoding

A move is packed into a 16-bit integer: 6 bits `from`, 6 bits `to`, 4 bits
`flag`. This is the classical scheme used by chess-programming-wiki-style
engines (e.g. Stockfish uses a near-identical layout in `Move`).

`move.py`:

```python
# 4-bit move flags
QUIET             = 0x0
DOUBLE_PAWN_PUSH   = 0x1
KING_CASTLE        = 0x2
QUEEN_CASTLE       = 0x3
CAPTURE            = 0x4
EN_PASSANT         = 0x5
# 0x6, 0x7 unused/reserved
PROMO_KNIGHT       = 0x8
PROMO_BISHOP       = 0x9
PROMO_ROOK         = 0xA
PROMO_QUEEN        = 0xB
PROMO_KNIGHT_CAP   = 0xC
PROMO_BISHOP_CAP   = 0xD
PROMO_ROOK_CAP     = 0xE
PROMO_QUEEN_CAP    = 0xF

PROMO_PIECE_OF = {  # flag -> piece type, for the 8 promotion flags
    PROMO_KNIGHT: KNIGHT, PROMO_KNIGHT_CAP: KNIGHT,
    PROMO_BISHOP: BISHOP, PROMO_BISHOP_CAP: BISHOP,
    PROMO_ROOK: ROOK,     PROMO_ROOK_CAP: ROOK,
    PROMO_QUEEN: QUEEN,   PROMO_QUEEN_CAP: QUEEN,
}

def encode_move(frm: int, to: int, flag: int = QUIET) -> int:
    return frm | (to << 6) | (flag << 12)

def move_from(m: int) -> int: return m & 0x3F
def move_to(m: int) -> int: return (m >> 6) & 0x3F
def move_flag(m: int) -> int: return (m >> 12) & 0xF
def is_capture(m: int) -> bool: return bool(move_flag(m) & 0x4)  # covers CAPTURE, EN_PASSANT, and all *_CAP promotions
def is_promotion(m: int) -> bool: return bool(move_flag(m) & 0x8)

NULL_MOVE = 0  # a1a1 quiet; never legal, used as a sentinel ("no move")

def move_to_uci(m: int) -> str:
    frm, to, flag = move_from(m), move_to(m), move_flag(m)
    s = SQUARE_NAMES[frm] + SQUARE_NAMES[to]
    if flag in PROMO_PIECE_OF:
        s += "nbrq"[[KNIGHT, BISHOP, ROOK, QUEEN].index(PROMO_PIECE_OF[flag])]
    return s

def move_from_uci(board: Board, uci: str) -> int:
    """Resolve a UCI-format move string against legal moves in `board`
    (needed because UCI strings carry no flag info — e.g. 'e1g1' must be
    matched against the generated KING_CASTLE move, not synthesized)."""
    frm = SQUARE_NAMES.index(uci[0:2])
    to = SQUARE_NAMES.index(uci[2:4])
    promo = uci[4] if len(uci) == 5 else None
    promo_letter = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
    for m in movegen.generate_legal_moves(board):
        if move_from(m) != frm or move_to(m) != to:
            continue
        if promo is None:
            if not is_promotion(m):
                return m
        elif is_promotion(m) and promo_letter[PROMO_PIECE_OF[move_flag(m)]] == promo:
            return m
    raise ValueError(f"illegal or unknown move: {uci}")
```

`move_from_uci` resolves the move against the legal move list rather than
re-deriving a flag from scratch, so it can never construct a move the
generator itself would not produce — one canonical source of truth for
"what flag does this move have."

A move never needs to record the moving piece or captured piece itself —
`make_move` looks the moving piece up via `board.mailbox[frm]`, and the
captured piece (if any) via `board.mailbox[to]` (or the en-passant square,
see §5) at apply time. This keeps the encoding tiny and immutable regardless
of board state, which matters because move lists are plain
`list[int]`s that can be freely copied/sorted/filtered.

---

## 4. Move Generation

### 4.1 Attack tables & sliding attacks (`attacks.py`)

All tables are built once at import time.

```python
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING_DELTAS   = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

def _leaper_table(deltas: list[tuple[int, int]]) -> list[int]:
    table = [0] * 64
    for sq in range(64):
        f, r = file_of(sq), rank_of(sq)
        bb = 0
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= 1 << (nr * 8 + nf)
        table[sq] = bb
    return table

KNIGHT_ATTACKS = _leaper_table(KNIGHT_DELTAS)
KING_ATTACKS   = _leaper_table(KING_DELTAS)

# Pawn attacks are asymmetric per color.
PAWN_ATTACKS = [_leaper_table([(1, 1), (-1, 1)]),   # WHITE captures NE/NW
                _leaper_table([(1, -1), (-1, -1)])]  # BLACK captures SE/SW

# 8 ray directions, indices chosen so "positive" dirs increase the square
# index along the ray and "negative" dirs decrease it (LERF property).
NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST = range(8)
DIR_FILE_RANK_DELTA = {
    NORTH: (0, 1), NORTH_EAST: (1, 1), EAST: (1, 0), SOUTH_EAST: (1, -1),
    SOUTH: (0, -1), SOUTH_WEST: (-1, -1), WEST: (-1, 0), NORTH_WEST: (-1, 1),
}
POSITIVE_DIRS = (NORTH, NORTH_EAST, EAST, NORTH_WEST)   # nearest blocker = lowest set bit
NEGATIVE_DIRS = (SOUTH, SOUTH_EAST, SOUTH_WEST, WEST)   # nearest blocker = highest set bit
ORTHO_DIRS = (NORTH, EAST, SOUTH, WEST)
DIAG_DIRS  = (NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST)

def _build_ray_attacks() -> list[list[int]]:
    """RAY_ATTACKS[d][sq] = bitboard of all squares from sq to the board
    edge along direction d, ignoring occupancy. File wraparound is handled
    by walking file/rank deltas and bounds-checking, not by raw index
    arithmetic (which would wrap around the a/h files)."""
    rays = [[0] * 64 for _ in range(8)]
    for sq in range(64):
        f0, r0 = file_of(sq), rank_of(sq)
        for d in range(8):
            df, dr = DIR_FILE_RANK_DELTA[d]
            bb, f, r = 0, f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                bb |= 1 << (r * 8 + f)
                f += df
                r += dr
            rays[d][sq] = bb
    return rays

RAY_ATTACKS = _build_ray_attacks()

# lsb_index/msb_index are bit-scan primitives owned by bitboard.py (shown in
# full in §4.9, alongside pop_lsb/popcount/iter_bits) and imported here and
# by movegen.py (§4.4/4.5) -- attacks.py does not redefine them.
from .bitboard import lsb_index, msb_index

def sliding_attacks(sq: int, occupied: int, dirs: tuple[int, ...]) -> int:
    """Classical ray-scanning sliding attack generation (no magic
    bitboards): union the full ray in each direction, then chop off
    everything beyond the first blocker."""
    attacks = 0
    for d in dirs:
        ray = RAY_ATTACKS[d][sq]
        attacks |= ray
        blockers = ray & occupied
        if blockers:
            blocker_sq = lsb_index(blockers) if d in POSITIVE_DIRS else msb_index(blockers)
            attacks &= ~RAY_ATTACKS[d][blocker_sq]
    return attacks

def bishop_attacks(sq: int, occupied: int) -> int: return sliding_attacks(sq, occupied, DIAG_DIRS)
def rook_attacks(sq: int, occupied: int) -> int: return sliding_attacks(sq, occupied, ORTHO_DIRS)
def queen_attacks(sq: int, occupied: int) -> int: return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)
```

`sliding_attacks` is O(4) ray unions plus at most 4 bit-scans per call — no
precomputed 64×2^n magic tables, no PEXT, at the cost of being roughly an
order of magnitude slower than magic bitboards. That trade is explicitly
accepted (see §12) in exchange for code any contributor can read start to
finish without a magic-number lookup table.

`SQUARES_BETWEEN[a][b]` (used for check-blocking and pin masks, §4.4/4.5) is
also precomputed once:

```python
def _build_between() -> list[list[int]]:
    between = [[0] * 64 for _ in range(64)]
    for a in range(64):
        for d in range(8):
            ray = RAY_ATTACKS[d][a]
            for b_sq in iter_bits(ray):
                # squares strictly between a and b_sq along this ray
                between[a][b_sq] = ray & ~RAY_ATTACKS[d][b_sq] & ~(1 << b_sq)
    return between

SQUARES_BETWEEN = _build_between()  # 0 for any (a, b) not on a shared rank/file/diagonal
```

### 4.2 `attacked_by` / `attackers_to` (`attacks.py`)

```python
def attackers_to(board: "Board", sq: int, by_color: int, occupied: int | None = None) -> int:
    """Bitboard of every `by_color` piece attacking `sq`, given an explicit
    occupancy (so callers can probe 'what if the king weren't there',
    §4.6)."""
    occ = board.occupied if occupied is None else occupied
    p = board.pieces[by_color]
    attackers = PAWN_ATTACKS[1 - by_color][sq] & p[PAWN]
    attackers |= KNIGHT_ATTACKS[sq] & p[KNIGHT]
    attackers |= KING_ATTACKS[sq] & p[KING]
    attackers |= bishop_attacks(sq, occ) & (p[BISHOP] | p[QUEEN])
    attackers |= rook_attacks(sq, occ) & (p[ROOK] | p[QUEEN])
    return attackers

def is_attacked(board: "Board", sq: int, by_color: int) -> bool:
    return attackers_to(board, sq, by_color) != 0
```

`PAWN_ATTACKS[1 - by_color][sq]` is the standard trick: "does an enemy pawn
attack `sq`" is the same lookup as "which squares would a pawn of the
*opposite* color attack from `sq`."

### 4.3 Check detection (`attacks.py`)

`checkers` lives in `attacks.py` (it only needs `attackers_to` and
`Board.king_square`, not the move generator), so `board.py`'s `in_check`
method — declared in §2.2 — can call straight into it as an ordinary
import, keeping the `board → attacks` dependency direction from §10 intact
(no reaching back into `board.py` from outside the class):

```python
# attacks.py
def checkers(board: "Board", color: int) -> int:
    return attackers_to(board, board.king_square(color), 1 - color)

# board.py
from . import attacks

class Board:
    ...
    def in_check(self, color: int | None = None) -> bool:
        c = self.side_to_move if color is None else color
        return attacks.checkers(self, c) != 0
```

### 4.4 Pins (`movegen.py`)

Computed once per `generate_legal_moves` call via ray-scanning from the
king outward — this is what lets pinned pieces be restricted without
falling back to make/unmake for the common case:

```python
def pinned_pieces(board: "Board", color: int) -> dict[int, int]:
    """Returns {pinned_square: allowed_destination_mask}. A pinned piece may
    only move within `allowed_destination_mask` (the squares between the
    king and the pinner, plus the pinner's square itself)."""
    king_sq = board.king_square(color)
    enemy = 1 - color
    occ = board.occupied
    own = board.occupied_co[color]
    result: dict[int, int] = {}
    for d in range(8):
        ray = RAY_ATTACKS[d][king_sq]
        blockers = ray & occ
        if not blockers:
            continue
        first = lsb_index(blockers) if d in POSITIVE_DIRS else msb_index(blockers)
        if not (own >> first) & 1:
            continue  # nearest piece on this ray is enemy's -> no pin possible here
        # RAY_ATTACKS[d][first] is, by construction, exactly "every square
        # strictly beyond `first` along direction d" -- i.e. where a
        # pinning slider would have to sit.
        blockers_beyond = RAY_ATTACKS[d][first] & occ
        if not blockers_beyond:
            continue
        second = lsb_index(blockers_beyond) if d in POSITIVE_DIRS else msb_index(blockers_beyond)
        if not (board.occupied_co[enemy] >> second) & 1:
            continue
        pt = piece_type_of(board.mailbox[second])
        slides_this_way = pt == QUEEN or (pt == ROOK and d in ORTHO_DIRS) or (pt == BISHOP and d in DIAG_DIRS)
        if slides_this_way:
            result[first] = SQUARES_BETWEEN[king_sq][second] | (1 << second)
    return result
```

Eight directions scanned per call, O(1) bit tricks per direction — cheap
enough to call at the top of every `generate_legal_moves`, no need to cache
across plies.

### 4.5 Check evasion (capture/block masks) (`movegen.py`)

```python
def evasion_masks(board: "Board", color: int) -> tuple[int, int]:
    """Returns (capture_mask, push_mask): a non-king move is only legal if
    its destination is in (capture_mask | push_mask). Both default to
    'all squares' when not in check."""
    ALL = 0xFFFFFFFFFFFFFFFF
    king_sq = board.king_square(color)
    ch = checkers(board, color)
    n = ch.bit_count()
    if n == 0:
        return ALL, ALL
    if n >= 2:
        return 0, 0  # double check: only king moves are legal
    checker_sq = lsb_index(ch)
    capture_mask = ch
    checker_type = piece_type_of(board.mailbox[checker_sq])
    push_mask = SQUARES_BETWEEN[king_sq][checker_sq] if checker_type in (BISHOP, ROOK, QUEEN) else 0
    return capture_mask, push_mask
```

### 4.6 Castling (`movegen.py`)

```python
# Precomputed per side/direction (built once, not shown for both colors):
CASTLE_KING_FROM = {WHITE: E1, BLACK: E8}
CASTLE_EMPTY_MASK = {          # squares that must be empty
    (WHITE, KING_CASTLE): (1 << F1) | (1 << G1),
    (WHITE, QUEEN_CASTLE): (1 << B1) | (1 << C1) | (1 << D1),
    (BLACK, KING_CASTLE): (1 << F8) | (1 << G8),
    (BLACK, QUEEN_CASTLE): (1 << B8) | (1 << C8) | (1 << D8),
}
CASTLE_KING_PATH = {           # squares the king passes through, incl. start/end;
    (WHITE, KING_CASTLE): (E1, F1, G1),   # none may be attacked
    (WHITE, QUEEN_CASTLE): (E1, D1, C1),
    (BLACK, KING_CASTLE): (E8, F8, G8),
    (BLACK, QUEEN_CASTLE): (E8, D8, C8),
}
CASTLE_RIGHT_BIT = {(WHITE, KING_CASTLE): CASTLE_WK, (WHITE, QUEEN_CASTLE): CASTLE_WQ,
                     (BLACK, KING_CASTLE): CASTLE_BK, (BLACK, QUEEN_CASTLE): CASTLE_BQ}
CASTLE_ROOK_MOVE = {  # (from, to) for the rook, indexed by move flag
    (KING_CASTLE, WHITE): (H1, F1), (QUEEN_CASTLE, WHITE): (A1, D1),
    (KING_CASTLE, BLACK): (H8, F8), (QUEEN_CASTLE, BLACK): (A8, D8),
}

def generate_castling_moves(board: "Board", color: int, in_check: bool, moves: list[int]) -> None:
    if in_check:
        return  # cannot castle out of check
    enemy = 1 - color
    for side in (KING_CASTLE, QUEEN_CASTLE):
        bit = CASTLE_RIGHT_BIT[(color, side)]
        if not (board.castling_rights & bit):
            continue
        if board.occupied & CASTLE_EMPTY_MASK[(color, side)]:
            continue
        if any(is_attacked(board, sq, enemy) for sq in CASTLE_KING_PATH[(color, side)]):
            continue
        to_sq = G1 if (color, side) == (WHITE, KING_CASTLE) else \
                C1 if (color, side) == (WHITE, QUEEN_CASTLE) else \
                G8 if (color, side) == (BLACK, KING_CASTLE) else C8
        moves.append(encode_move(CASTLE_KING_FROM[color], to_sq, side))
```

`CASTLE_RIGHT_BIT` values are `0b1111`-style flags on `board.castling_rights`,
maintained via a spoiler table applied in `make_move` (§5):

```python
CASTLE_SPOILER = [0] * 64
CASTLE_SPOILER[E1] = CASTLE_WK | CASTLE_WQ
CASTLE_SPOILER[A1] = CASTLE_WQ
CASTLE_SPOILER[H1] = CASTLE_WK
CASTLE_SPOILER[E8] = CASTLE_BK | CASTLE_BQ
CASTLE_SPOILER[A8] = CASTLE_BQ
CASTLE_SPOILER[H8] = CASTLE_BK
```

On every move, `board.castling_rights &= ~(CASTLE_SPOILER[frm] | CASTLE_SPOILER[to])`
handles all four ways rights are lost: king moves, either rook moves, or
either rook is *captured* on its home square (the `to` square lookup covers
captures automatically, with no special-casing needed).

### 4.7 En passant (`movegen.py`)

Generated as part of pawn-capture generation: if `board.ep_square is not
None` and `PAWN_ATTACKS[color][pawn_sq] & (1 << board.ep_square)`, emit
`encode_move(pawn_sq, board.ep_square, EN_PASSANT)`. The captured pawn's
square is **not** `ep_square` — it is `ep_square - 8` (white capturing) or
`ep_square + 8` (black capturing); `make_move` computes this explicitly
(§5), it must never be assumed to equal `to`.

`ep_square` itself is set only by a double pawn push, to the *skipped*
square (`(frm + to) // 2`), and is reset to `None` on every other move —
this is exactly the one-ply lifetime en passant has in the real rules, so
no extra bookkeeping is needed beyond "clear it unless this move sets it."

**Discovered check edge case.** En passant is the one move where *two*
pieces leave the board simultaneously (the capturing pawn's origin square
stays empty, but both the moving pawn's destination bookkeeping *and* the
captured pawn's square are vacated). This can expose a horizontal
discovered check along the 4th/5th rank that neither the pin table (§4.4,
which only tracks single blockers) nor the evasion mask (§4.5) accounts
for. Rather than special-case this rank-scan, en passant moves are
**always legality-checked by the make/unmake + `is_attacked` fallback**
(§4.9) regardless of the pin table's verdict — there are at most one or
two en passant moves in any position, so this costs nothing measurable
and sidesteps a well-known class of engine bugs.

### 4.8 Promotion (`movegen.py`)

A pawn move whose destination rank is `RANK_8` (white) or `RANK_1` (black)
is expanded into four moves — one per promotion piece — reusing the
capture/quiet flag distinction: `PROMO_QUEEN`/`PROMO_ROOK`/`PROMO_BISHOP`/`PROMO_KNIGHT`
for a quiet promotion, `PROMO_QUEEN_CAP` etc. for a capturing promotion
(including en passant is impossible here since en passant never lands on
the back rank). `make_move` reads `PROMO_PIECE_OF[flag]` to know which
piece to place on `to` instead of the pawn.

### 4.9 Putting it together (`movegen.py`)

```python
def generate_pseudo_legal_moves(board: "Board") -> list[int]:
    """All moves obeying piece movement rules, NOT yet filtered for leaving
    one's own king in check. Used internally by generate_legal_moves and by
    perft's optional 'unfiltered' mode for debugging."""
    color = board.side_to_move
    moves: list[int] = []
    _gen_pawn_moves(board, color, moves)
    _gen_knight_moves(board, color, moves)
    _gen_king_moves(board, color, moves)          # includes castling
    _gen_sliding_moves(board, color, BISHOP, DIAG_DIRS, moves)
    _gen_sliding_moves(board, color, ROOK, ORTHO_DIRS, moves)
    _gen_sliding_moves(board, color, QUEEN, ORTHO_DIRS + DIAG_DIRS, moves)
    return moves

def generate_legal_moves(board: "Board") -> list[int]:
    color = board.side_to_move
    king_sq = board.king_square(color)
    in_check = is_attacked(board, king_sq, 1 - color)
    capture_mask, push_mask = evasion_masks(board, color)
    pins = pinned_pieces(board, color)
    legal: list[int] = []
    for m in generate_pseudo_legal_moves(board):
        frm, to, flag = move_from(m), move_to(m), move_flag(m)
        if flag == EN_PASSANT:
            # Always resolved by the expensive-but-correct fallback (§4.7).
            board.make_move(m)
            ok = not is_attacked(board, board.king_square(color), 1 - color)
            board.unmake_move()
            if ok:
                legal.append(m)
            continue
        if frm == king_sq:
            if flag in (KING_CASTLE, QUEEN_CASTLE):
                legal.append(m)  # already fully validated in generate_castling_moves
                continue
            # King steps must not land on an attacked square; the king's own
            # bit is removed from occupancy first, so it cannot "shadow" a
            # slider that is checking it along the direction of travel.
            occ_without_king = board.occupied & ~(1 << king_sq)
            if attackers_to(board, to, 1 - color, occupied=occ_without_king) == 0:
                legal.append(m)
            continue
        if frm in pins and not (pins[frm] >> to) & 1:
            continue  # pinned piece moving off its pin line
        if not ((capture_mask | push_mask) >> to) & 1:
            continue  # doesn't capture the (sole) checker or block the check
        legal.append(m)
    return legal
```

Per-piece generators (`_gen_pawn_moves`, `_gen_knight_moves`,
`_gen_king_moves`, `_gen_sliding_moves`) all follow the same shape: iterate
set bits of `board.pieces[color][piece]` with the bit-scan helpers below,
look up (or compute) the attack bitboard, mask off squares occupied by the
mover's own pieces, and emit one `encode_move(...)` per destination bit
(splitting into quiet/capture, and further into four promotions where
applicable).

`bitboard.py` bit-scan helpers used throughout §4:

```python
def lsb_index(bb: int) -> int: return (bb & -bb).bit_length() - 1
def msb_index(bb: int) -> int: return bb.bit_length() - 1
def pop_lsb(bb: int) -> tuple[int, int]:
    idx = lsb_index(bb)
    return idx, bb & (bb - 1)
def popcount(bb: int) -> int: return bb.bit_count()   # native since Python 3.10
def iter_bits(bb: int):
    while bb:
        idx, bb = pop_lsb(bb)
        yield idx
```

---

## 5. Make / Unmake Move (`board.py`)

`Board.make_move` / `Board.unmake_move` mutate the four synchronized
representations (`pieces`, `occupied_co`, `occupied`, `mailbox`) plus
`zobrist_hash`, and push/pop a `UndoInfo` (§2.2) that captures everything
not trivially reversible from the move encoding alone.

```python
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

    undo = UndoInfo(move, captured_piece, captured_sq, self.castling_rights,
                     self.ep_square, self.halfmove_clock, self.zobrist_hash)
    self.history.append(undo)

    if captured_piece != NO_PIECE:
        self._remove_piece(captured_piece, captured_sq)

    self._move_piece(moving_piece, frm, to)

    if flag in PROMO_PIECE_OF:
        self._remove_piece(moving_piece, to)
        promoted = us * 6 + PROMO_PIECE_OF[flag]
        self._add_piece(promoted, to)
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

    self.halfmove_clock = 0 if (moving_type == PAWN or captured_piece != NO_PIECE) \
                             else self.halfmove_clock + 1
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
        promoted = us * 6 + PROMO_PIECE_OF[flag]
        self._remove_piece(promoted, to)
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
    if us == BLACK:
        self.fullmove_number -= 1
```

`_add_piece`, `_remove_piece`, `_move_piece` are small private helpers that
keep `pieces`/`occupied_co`/`occupied`/`mailbox`/`zobrist_hash` (the
piece-square component only) in sync in one place, e.g.:

```python
def _add_piece(self, piece_code: int, sq: int) -> None:
    color, ptype = color_of(piece_code), piece_type_of(piece_code)
    bit = 1 << sq
    self.pieces[color][ptype] |= bit
    self.occupied_co[color] |= bit
    self.occupied |= bit
    self.mailbox[sq] = piece_code
    self.zobrist_hash ^= ZOBRIST_PIECE[color][ptype][sq]

def _remove_piece(self, piece_code: int, sq: int) -> None:
    color, ptype = color_of(piece_code), piece_type_of(piece_code)
    bit = 1 << sq
    self.pieces[color][ptype] &= ~bit
    self.occupied_co[color] &= ~bit
    self.occupied &= ~bit
    self.mailbox[sq] = NO_PIECE
    self.zobrist_hash ^= ZOBRIST_PIECE[color][ptype][sq]

def _move_piece(self, piece_code: int, frm: int, to: int) -> None:
    self._remove_piece(piece_code, frm)
    self._add_piece(piece_code, to)
```

`UndoInfo` storing the *full previous hash* (rather than re-deriving it
from reversed XORs) trades 8 bytes of extra memory per stack frame for
"unmake is always exactly correct, trivially," which matches the
approachability goal — the alternative (recomputing castling/ep XORs in
reverse) is a common source of unmake bugs.

---

## 6. Zobrist Hashing (`zobrist.py`)

```python
import random

_rng = random.Random(0xC0FFEE)  # fixed seed: reproducible hashes across runs/tests

ZOBRIST_PIECE = [[[_rng.getrandbits(64) for _ in range(64)] for _ in range(6)] for _ in range(2)]
ZOBRIST_SIDE = _rng.getrandbits(64)
ZOBRIST_CASTLING = [_rng.getrandbits(64) for _ in range(16)]   # indexed by the 4-bit rights mask directly
ZOBRIST_EP_FILE = [_rng.getrandbits(64) for _ in range(8)]

def compute_hash(board: "Board") -> int:
    """From-scratch hash, independent of the incremental bookkeeping in
    make_move/unmake_move. Used only by tests (§11) to catch incremental
    hashing bugs, and to seed a Board built directly from FEN."""
    h = 0
    for color in (WHITE, BLACK):
        for ptype in range(6):
            for sq in iter_bits(board.pieces[color][ptype]):
                h ^= ZOBRIST_PIECE[color][ptype][sq]
    h ^= ZOBRIST_CASTLING[board.castling_rights]
    if board.ep_square is not None:
        h ^= ZOBRIST_EP_FILE[file_of(board.ep_square)]
    if board.side_to_move == BLACK:
        h ^= ZOBRIST_SIDE
    return h
```

---

## 7. Search Architecture (`search.py`)

### 7.1 Negamax formulation of minimax + alpha-beta

The engine implements alpha-beta as **negamax**: at every node the score is
relative to the side to move, and each recursive call negates and swaps
`(alpha, beta)`. This is mathematically identical to minimax with
alpha-beta pruning — it is the standard reformulation nearly every engine
uses because it removes the "maximize or minimize depending on whose turn
it is" branch, not a different algorithm.

```python
INF = 1_000_000
MATE_SCORE = 100_000
DRAW_SCORE = 0

def alpha_beta(board: Board, depth: int, alpha: int, beta: int, ply: int,
                tt: "TranspositionTable", ctx: "SearchContext") -> int:
    ctx.nodes += 1
    if ctx.should_stop():
        return 0  # value discarded: caller checks ctx.stopped after returning

    if board.is_fifty_move_draw() or board.is_repetition_draw():
        return DRAW_SCORE

    tt_key = board.zobrist_hash
    entry = tt.probe(tt_key)
    tt_move = NULL_MOVE
    if entry is not None:
        tt_move = entry.best_move
        if entry.depth >= depth:
            score = score_from_tt(entry.score, ply)
            if entry.flag == TTFlag.EXACT:
                return score
            if entry.flag == TTFlag.LOWERBOUND:
                alpha = max(alpha, score)
            elif entry.flag == TTFlag.UPPERBOUND:
                beta = min(beta, score)
            if alpha >= beta:
                return score

    if depth == 0:
        return evaluate(board)

    moves = generate_legal_moves(board)
    if not moves:
        return -MATE_SCORE + ply if board.in_check() else DRAW_SCORE

    order_moves(moves, board, tt_move, ply, ctx)

    best_score = -INF
    best_move = moves[0]
    original_alpha = alpha
    for move in moves:
        board.make_move(move)
        score = -alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, tt, ctx)
        board.unmake_move()
        if ctx.should_stop():
            return 0
        if score > best_score:
            best_score, best_move = score, move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            ctx.record_cutoff(move, ply, board.side_to_move)  # killers/history, §7.4
            break

    flag = (TTFlag.EXACT if original_alpha < best_score < beta else
            TTFlag.LOWERBOUND if best_score >= beta else
            TTFlag.UPPERBOUND)
    tt.store(tt_key, depth, score_to_tt(best_score, ply), flag, best_move)
    return best_score
```

### 7.2 Iterative deepening driver

```python
@dataclass
class SearchLimits:
    max_depth: int = 64
    move_time_ms: int | None = None
    nodes: int | None = None

@dataclass
class SearchResult:
    best_move: int
    score: int
    depth: int
    nodes: int
    pv: list[int]

class SearchContext:
    def __init__(self, limits: SearchLimits, deadline: float | None):
        self.limits = limits
        self.deadline = deadline          # time.monotonic() cutoff, or None
        self.nodes = 0
        self.stop_flag = threading.Event()  # set()/is_set() from the UCI `stop` handler
        self.killers: list[list[int]] = [[NULL_MOVE, NULL_MOVE] for _ in range(128)]
        self.history: list[list[int]] = [[0] * 64 for _ in range(64)]

    def should_stop(self) -> bool:
        if self.stop_flag.is_set():
            return True
        if self.limits.nodes is not None and self.nodes >= self.limits.nodes:
            return True
        # Wall-clock is checked every 2048 nodes, not every node, so
        # time.monotonic() overhead doesn't distort NPS at shallow depth.
        if self.deadline is not None and self.nodes % 2048 == 0:
            return time.monotonic() >= self.deadline
        return False

def iterative_deepening(board: Board, limits: SearchLimits, tt: "TranspositionTable",
                         stop_flag: threading.Event, report=lambda r: None) -> SearchResult:
    deadline = time.monotonic() + limits.move_time_ms / 1000 if limits.move_time_ms else None
    ctx = SearchContext(limits, deadline)
    ctx.stop_flag = stop_flag
    best = SearchResult(NULL_MOVE, 0, 0, 0, [])
    for depth in range(1, limits.max_depth + 1):
        score = alpha_beta(board, depth, -INF, INF, 0, tt, ctx)
        if ctx.should_stop() and depth > 1:
            break  # partial/unreliable result from an aborted depth is discarded
        pv = extract_pv(board, tt, depth)
        best = SearchResult(pv[0] if pv else best.best_move, score, depth, ctx.nodes, pv)
        report(best)   # emits a UCI `info ...` line, §9
        if abs(score) >= MATE_SCORE - 128:
            break       # found a forced mate; no point searching deeper
    return best

def extract_pv(board: Board, tt: "TranspositionTable", max_len: int) -> list[int]:
    """Walks the TT from the current position following each node's stored
    best_move, making/unmaking as it goes. Simple, but can be shorter than
    `max_len` or slightly unstable across depths if TT entries were
    overwritten (replacement scheme, §7.3) mid-line; a triangular PV array
    threaded through alpha_beta is the standard fix and is a natural v2
    addition (§12) if PV stability becomes an issue."""
    pv, undone = [], 0
    for _ in range(max_len):
        entry = tt.probe(board.zobrist_hash)
        if entry is None or entry.best_move == NULL_MOVE:
            break
        pv.append(entry.best_move)
        board.make_move(entry.best_move)
        undone += 1
    for _ in range(undone):
        board.unmake_move()
    return pv
```

### 7.3 Transposition table (`transposition.py`)

```python
class TTFlag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

@dataclass
class TTEntry:
    key: int          # full zobrist key (stored, not just the low bits used
                       # as the index) so index collisions are detected and
                       # rejected rather than returning a wrong-position hit
    depth: int
    score: int
    flag: "TTFlag"
    best_move: int

class TranspositionTable:
    def __init__(self, size_mb: int = 64) -> None:
        entry_count = (size_mb * 1024 * 1024) // 40   # ~40 bytes/entry incl. list overhead
        self.size = 1 << (entry_count.bit_length() - 1)  # round down to a power of two
        self.mask = self.size - 1
        self.table: list[TTEntry | None] = [None] * self.size

    def probe(self, key: int) -> TTEntry | None:
        entry = self.table[key & self.mask]
        return entry if entry is not None and entry.key == key else None

    def store(self, key: int, depth: int, score: int, flag: "TTFlag", best_move: int) -> None:
        idx = key & self.mask
        existing = self.table[idx]
        # Depth-preferred replacement: only overwrite a same-key or
        # shallower-or-equal-depth entry, so deep analysis isn't evicted by
        # a shallow re-search of the same slot.
        if existing is None or existing.key == key or depth >= existing.depth:
            self.table[idx] = TTEntry(key, depth, score, flag, best_move)

    def clear(self) -> None:
        self.table = [None] * self.size
```

**Mate-distance adjustment.** A mate score found N plies below the current
search root must be stored/retrieved relative to *this* node, not the
root, or a mate that is actually 2 plies further away (because it was
reached via a transposition) gets misreported as 2 plies closer:

```python
def score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 128:
        return score + ply
    if score <= -MATE_SCORE + 128:
        return score - ply
    return score

def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 128:
        return score - ply
    if score <= -MATE_SCORE + 128:
        return score + ply
    return score
```

### 7.4 Move ordering

`order_moves(moves, board, tt_move, ply, ctx)` sorts (or partitions) the
move list in place, highest priority first, using:

1. The TT's stored `best_move` for this position, if any — searched first.
2. Captures, ranked by MVV-LVA: `victim_value * 16 - attacker_value`
   (looked up from the same material table as evaluation, §8), so
   "queen takes pawn" sorts after "pawn takes queen."
3. Killer moves: up to two quiet moves per ply (`ctx.killers[ply]`) that
   most recently caused a beta cutoff at this ply in a sibling node.
4. History heuristic: `ctx.history[frm][to]`, incremented by `depth * depth`
   on every beta cutoff from a quiet move, used as the tiebreak for
   remaining quiet moves.

`ctx.record_cutoff(move, ply, side)` (called from the alpha-beta loop on a
beta cutoff) updates killers/history only for quiet moves — captures
already get their ordering from MVV-LVA and don't need history bookkeeping.

### 7.5 Time management / stop handling

`SearchContext.should_stop()` (§7.2) is the single choke point: it is
polled at the top of every `alpha_beta` call and checks, in cheap-to-
expensive order, an explicit `stop_flag` (set by the UCI thread on
`stop`/`quit`), a node budget, and — throttled to every 2048 nodes — a
wall-clock deadline computed from the UCI `go` command's time controls.
Search always runs on a background thread (§9) precisely so the main
thread reading stdin can set `stop_flag` promptly without waiting for the
current search call stack to unwind on its own.

### 7.6 Draws

```python
def is_repetition_draw(self) -> bool:
    """Treats the current position recurring once before (i.e. this would
    be its 2nd occurrence within the tracked history) as a draw. This is
    intentionally stricter than the official threefold rule so the search
    steers away from repeats a ply earlier instead of only at the exact
    legal draw claim -- a standard, safe simplification for the search's
    internal draw detection (it does not affect UCI-level game-result
    reporting, which is the GUI's responsibility)."""
    current = self.position_history[-1]
    return self.position_history[:-1].count(current) >= 1
```

---

## 8. Evaluation (`evaluate.py`)

Material values (centipawns), plus 64-entry piece-square tables (PST) per
piece type, defined from White's point of view with index 0 = a1 (matching
board indexing) and index 63 = h8; Black's PST lookup mirrors the square
vertically via `sq ^ 56` (flips the rank, keeps the file) rather than
maintaining a second table.

Note the row order below runs **rank 1 to rank 8, top to bottom** to match
that a1=0 indexing — the opposite of how these tables are usually printed
in chess references (rank 8 first, as White would view the board from
above). Transcribing a reference table's rows in printed order into an
a1=0 array is a classic off-by-mirror bug (it rewards White's pawns for
sitting on their *starting* rank 2 instead of rank 7, next to promotion);
double-check row-to-rank mapping against this indexing when filling in
`KNIGHT_PST`/`BISHOP_PST`/`ROOK_PST`/`QUEEN_PST`/`KING_PST`.

```python
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
# KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_PST: same shape,
# standard "PeSTO-style" tables; omitted here for brevity but each is a
# flat 64-int tuple indexed a1..h8 exactly like PAWN_PST.
PST = {PAWN: PAWN_PST, KNIGHT: KNIGHT_PST, BISHOP: BISHOP_PST,
        ROOK: ROOK_PST, QUEEN: QUEEN_PST, KING: KING_PST}

def _side_score(board: Board, color: int) -> int:
    score = 0
    mirror = color == BLACK
    for ptype in range(6):
        table = PST[ptype]
        for sq in iter_bits(board.pieces[color][ptype]):
            score += PIECE_VALUE[ptype]
            score += table[sq ^ 56] if mirror else table[sq]
    return score

def evaluate(board: Board) -> int:
    """Static evaluation in centipawns, from the side-to-move's
    perspective (required by the negamax search, §7.1)."""
    white_score = _side_score(board, WHITE) - _side_score(board, BLACK)
    return white_score if board.side_to_move == WHITE else -white_score
```

v1 recomputes the full sum at every leaf — simple and obviously correct.
An incremental running total updated inside `make_move`/`unmake_move`
(alongside the Zobrist hash) is a natural, low-risk optimization once
correctness is established (§12).

**Explicitly out of scope for v1** (listed so the evaluation's ceiling is
clear, not because they're hard to add later): bishop-pair bonus,
doubled/isolated/passed-pawn terms, rook-on-open-file, king safety /
pawn-shield, mobility, and a tapered midgame/endgame PST blend keyed off a
material-based game-phase estimate. `evaluate.py` isolates all evaluation
knowledge in this one file specifically so these can be added as pure
functions of `Board` without touching search or move generation.

---

## 9. UCI Protocol Handler (`uci.py`)

```python
class UCIEngine:
    def __init__(self) -> None:
        self.board = Board.starting_position()
        self.tt = TranspositionTable(size_mb=64)
        self.stop_flag = threading.Event()
        self.search_thread: threading.Thread | None = None
        self.quit = False

    def run(self, inp=sys.stdin, out=sys.stdout) -> None:
        for line in inp:
            self.handle_command(line.strip(), out)
            if self.quit:
                break

    def handle_command(self, line: str, out) -> None:
        if not line:
            return
        cmd, *args = line.split()
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is not None:
            handler(args, out)
        # unrecognized commands are silently ignored, per the UCI spec

    def cmd_uci(self, args, out) -> None:
        out.write("id name death-Token 0.1\n")
        out.write("id author Francesco Errico\n")
        out.write("option name Hash type spin default 64 min 1 max 1024\n")
        out.write("uciok\n"); out.flush()

    def cmd_isready(self, args, out) -> None:
        out.write("readyok\n"); out.flush()

    def cmd_setoption(self, args, out) -> None:
        # "setoption name Hash value 128"
        if "Hash" in args:
            mb = int(args[args.index("value") + 1])
            self.tt = TranspositionTable(size_mb=mb)

    def cmd_ucinewgame(self, args, out) -> None:
        self.board = Board.starting_position()
        self.tt.clear()

    def cmd_position(self, args, out) -> None:
        # "position [startpos | fen <FEN...>] [moves <m1> <m2> ...]"
        if args[0] == "startpos":
            self.board = Board.starting_position()
            rest = args[1:]
        else:  # args[0] == "fen"
            moves_idx = args.index("moves") if "moves" in args else len(args)
            fen = " ".join(args[1:moves_idx])
            self.board = fen_module.parse_fen(fen)
            rest = args[moves_idx:]
        if rest and rest[0] == "moves":
            for uci_move in rest[1:]:
                self.board.make_move(move_from_uci(self.board, uci_move))

    def cmd_go(self, args, out) -> None:
        limits = parse_go_limits(args, self.board.side_to_move)
        self.stop_flag = threading.Event()
        self.search_thread = threading.Thread(
            target=self._search_and_report, args=(limits, out), daemon=True)
        self.search_thread.start()

    def cmd_stop(self, args, out) -> None:
        self.stop_flag.set()
        if self.search_thread is not None:
            self.search_thread.join()

    def cmd_quit(self, args, out) -> None:
        self.stop_flag.set()
        self.quit = True

    def _search_and_report(self, limits: SearchLimits, out) -> None:
        def report(result: SearchResult) -> None:
            pv_str = " ".join(move_to_uci(m) for m in result.pv)
            out.write(f"info depth {result.depth} score cp {result.score} "
                      f"nodes {result.nodes} pv {pv_str}\n")
            out.flush()
        result = iterative_deepening(self.board, limits, self.tt, self.stop_flag, report)
        out.write(f"bestmove {move_to_uci(result.best_move)}\n"); out.flush()
```

`parse_go_limits(args, side_to_move)` (in `uci.py`) turns `go` tokens
(`depth`, `nodes`, `movetime`, `wtime`/`btime`/`winc`/`binc`, `infinite`)
into a `SearchLimits`; time-control math (allocate roughly
`remaining/movestogo_estimate + increment` per move) is intentionally
simple for v1 — a fixed fraction of remaining time plus the increment,
clamped to a minimum and to remaining time minus a safety buffer.

`cli.py` is the console-script entry point wired in `pyproject.toml`
(`chessengine = "chessengine.cli:main"`):

```python
def main() -> None:
    UCIEngine().run()
```

Search runs on a **background thread** specifically so `cmd_stop`/`cmd_quit`
handling on the main thread (still blocked reading stdin) is never delayed
by a long-running `alpha_beta` call — the only cross-thread state is
`self.stop_flag` (a `threading.Event`, safe to set from another thread) and
`self.tt` (read/written only by the search thread while a search is in
flight; the main thread must not mutate `self.board`/`self.tt` again until
`cmd_stop` has joined the thread, which `cmd_position`/`cmd_ucinewgame`
enforce implicitly since a compliant GUI always sends `stop` or waits for
`bestmove` before sending a new `position`).

---

## 10. Module Boundaries

| File | Owns |
|---|---|
| `src/chessengine/__init__.py` | Package version; re-exports `Board` for convenience (`from chessengine import Board`). |
| `constants.py` | Square/file/rank indices and names, color/piece-type constants, castling-right bit flags, file/rank bitmask tables. No behavior, just names and masks. |
| `bitboard.py` | Bit-twiddling primitives independent of chess semantics: `lsb_index`, `msb_index`, `pop_lsb`, `popcount`, `iter_bits`, a `print_bitboard(bb)` debug pretty-printer. |
| `attacks.py` | All precomputed attack tables (`KNIGHT_ATTACKS`, `KING_ATTACKS`, `PAWN_ATTACKS`, `RAY_ATTACKS`, `SQUARES_BETWEEN`), `sliding_attacks`/`bishop_attacks`/`rook_attacks`/`queen_attacks`, `attackers_to`/`is_attacked`/`checkers`. Depends on `bitboard.py` + `constants.py` only — deliberately not on `board.py`, so `board.py` can import it for `in_check` (§4.3) without a cycle. |
| `zobrist.py` | Random key tables and `compute_hash(board)` (from-scratch hash, used by tests and FEN loading). |
| `fen.py` | `parse_fen(fen: str) -> Board`, `board_to_fen(board: Board) -> str`, `STARTPOS_FEN`. The only place that knows FEN's textual grammar. |
| `board.py` | The `Board` class and `UndoInfo`: state, `make_move`/`unmake_move`, `king_square`/`in_check`, draw queries, `__str__`. Depends on `attacks.py` (for `in_check`), `zobrist.py`, `constants.py`. Does **not** know about search or evaluation. |
| `move.py` | Move encoding/decoding (`encode_move` and friends), the flag constants, `move_to_uci`/`move_from_uci`. Depends on `constants.py`; `move_from_uci` depends on `movegen.py` (to resolve against legal moves) — the one deliberate exception to an otherwise strict layering, documented here rather than hidden. |
| `movegen.py` | `generate_pseudo_legal_moves`, `generate_legal_moves`, `pinned_pieces`, `evasion_masks`, castling/en passant/promotion generation, `perft`/`divide`. Depends on `board.py`, `attacks.py`, `move.py`. |
| `evaluate.py` | `PIECE_VALUE`, all PSTs, `evaluate(board) -> int`. Depends only on `board.py` + `constants.py` — never on `search.py` or `movegen.py`. |
| `transposition.py` | `TTEntry`, `TTFlag`, `TranspositionTable`, `score_to_tt`/`score_from_tt`. Depends only on `constants.py`-level score constants (`MATE_SCORE`). |
| `search.py` | `SearchLimits`, `SearchResult`, `SearchContext`, `alpha_beta`, `iterative_deepening`, `extract_pv`, move ordering (`order_moves`, killers/history). Depends on `board.py`, `movegen.py`, `evaluate.py`, `transposition.py`. |
| `uci.py` | `UCIEngine`, `parse_go_limits`. The only module that touches `sys.stdin`/`sys.stdout` and `threading`. Depends on everything above. |
| `cli.py` | `main()` console-script entry point; wires `UCIEngine().run()` to real stdio. |

Dependency direction is strictly one-way: `constants → bitboard → attacks →
{zobrist, fen} → board → move → movegen → evaluate/transposition → search →
uci → cli`. Nothing below `board.py` imports anything above it, which is
what makes the perft suite (§11) able to exercise `board.py`/`movegen.py`
completely independently of search, evaluation, or UCI.

---

## 11. Testing Strategy

### 11.1 Perft as the correctness backbone

`perft(board, depth)` counts leaf nodes of the full game tree at `depth`
plies, walking through `make_move`/`unmake_move` exactly as search does —
it is the standard way to validate a move generator because a wrong node
count at some depth means *some* rule (check evasion, pin, en passant,
castling right bookkeeping, promotion) was generated incorrectly, and
`divide` (perft broken down by root move) bisects which one.

```python
def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    count = 0
    for move in generate_legal_moves(board):
        board.make_move(move)
        count += perft(board, depth - 1)
        board.unmake_move()
    return count

def divide(board: Board, depth: int) -> dict[str, int]:
    """perft count per root move, for bisecting a mismatch against a
    reference engine's own `divide`/`go perft` output."""
    return {
        move_to_uci(m): _perft_after(board, m, depth - 1)
        for m in generate_legal_moves(board)
    }

def _perft_after(board: Board, move: int, depth: int) -> int:
    board.make_move(move)
    n = perft(board, depth)
    board.unmake_move()
    return n
```

`tests/test_movegen_perft.py` runs `perft` against the standard reference
positions and depths below (the "CPW perft suite," used essentially
universally to validate chess move generators). **These counts must be
re-verified against the canonical Chess Programming Wiki "Perft Results"
page (or a trusted reference engine's own `perft`/`divide` output) before
being committed as hard `assert`s** — they are reproduced here from
well-established memory to scope the test plan, not as a substitute for
that verification step:

| # | FEN | Depths → expected node counts |
|---|---|---|
| 1 | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` (startpos) | 1:20, 2:400, 3:8902, 4:197281, 5:4865609, 6:119060324 |
| 2 (Kiwipete) | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1` | 1:48, 2:2039, 3:97862, 4:4085603, 5:193690690 |
| 3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | 1:14, 2:191, 3:2812, 4:43238, 5:674624, 6:11030083 |
| 4 | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | 1:6, 2:264, 3:9467, 4:422333, 5:15833292 |
| 5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 1:44, 2:1486, 3:62379, 4:2103487 |
| 6 | `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 1:46, 2:2079, 3:89890, 4:3894594 |

Position 1/2 exercise the general case at high volume; 2 (Kiwipete) is
specifically known for stressing castling-rights bookkeeping and pins; 3
stresses en passant and king-safety edge cases with few pieces; 4/5 stress
promotion (including underpromotion) combined with castling; 6 is a quiet
middlegame sanity check. Deeper depths (5–6 for positions with larger
trees) are marked `@pytest.mark.slow` and excluded from the default `pytest`
run; CI runs the full depth set on a schedule or on demand.

### 11.2 Supporting unit tests

- **`test_attacks.py`** — for a sample of squares and randomly generated
  occupancy bitboards, assert `sliding_attacks(...)` (ray-scanning) agrees
  with a deliberately naive reference (`for each of 8 directions, step one
  square at a time until off-board or blocked, collecting squares`,
  written independently and only for the test). This cross-checks the
  "chop off everything beyond the first blocker" bit-trick against an
  unoptimized ground truth.
- **`test_zobrist.py`** — after every `make_move`/`unmake_move` in a random
  legal game, `board.zobrist_hash == zobrist.compute_hash(board)`; also
  checks that `unmake_move` restores the exact pre-move hash (not just an
  equivalent one).
- **`test_fen.py`** — round-trip: `board_to_fen(parse_fen(fen)) == fen` for
  a battery of FENs covering all castling-rights combinations, an en
  passant square set, and non-standard `halfmove`/`fullmove` counters.
- **`test_board_invariants.py`** — a property-style test that plays random
  legal move sequences and asserts, after every `make_move` and every
  `unmake_move`, that `occupied == occupied_co[WHITE] | occupied_co[BLACK]`,
  each side has exactly one king bit set, and `mailbox` agrees with the
  bitboards square-by-square.
- **`test_transposition.py`** — searching a small suite of positions with
  the TT cleared vs. warm must return the same best move and score
  (modulo mate-distance adjustment), i.e. the TT is verified to be a pure
  optimization, never a source of a different answer.
- **`test_search_tactics.py`** — a short list of mate-in-1 and mate-in-2
  FENs that `iterative_deepening` must solve (correct `best_move`) within
  a small fixed depth/node budget; plus a "don't hang a queen" sanity
  check on a simple tactical position.
- **`test_uci.py`** — drives `UCIEngine.run()` over an in-memory
  `io.StringIO` pair (or a real subprocess talking over pipes) with
  `uci` / `isready` / `position startpos moves e2e4 e7e5` / `go depth 4`
  and asserts well-formed `uciok`/`readyok`/`info`/`bestmove` lines appear
  in the right order.

### 11.3 Debugging workflow for a perft mismatch

1. Run `divide(board, depth)` on the failing position and compare against
   a reference engine's `divide`/`go perft depth` output for the same
   position (any UCI engine, or `python-chess`, run purely as an oracle
   during development — not a runtime dependency of `chessengine`).
2. The first root move whose subtree count disagrees is searched one ply
   at a time by recursing `divide` into it, which bisects to the exact
   move (and therefore the exact special-case: promotion, castling, en
   passant, or a pin/check-evasion bug) responsible.

---

## 12. Key Tradeoffs & Future Extensions

- **Classical ray-scanning vs. magic bitboards.** Sliding attacks cost
  ~4 ray unions + up to 4 bit-scans per call instead of one array lookup;
  this is the deliberate approachability trade the proposal asks for.
  Because `bishop_attacks`/`rook_attacks`/`queen_attacks` in `attacks.py`
  are the *only* place magic bitboards would plug in, swapping to magics
  later (if NPS ever becomes the bottleneck) is a localized change that
  does not touch `movegen.py`, `board.py`, or `search.py`.
- **Pseudo-legal + targeted filtering vs. always make/unmake.** Pins and
  check evasion are handled with dedicated ray-scanning masks (§4.4/4.5)
  so the common case never pays for a make/unmake round-trip; en passant
  is the sole, deliberate exception (§4.7) because its discovered-check
  edge case is otherwise easy to get subtly wrong. This is a middle
  ground between "always make/unmake every pseudo-legal move" (simplest,
  slowest) and "encode every special case in bitmasks" (fastest,
  hardest to get right) — chosen because it keeps 99% of the generator
  simple while still being correct on the fiddly 1%.
- **Python `int` as bitboard.** Arbitrary precision means no manual 64-bit
  masking is needed for *reads*, but left-shifts that could set bit 64+
  (rare, e.g. some ray-construction intermediate steps) are avoided by
  construction in `_build_ray_attacks`/`_leaper_table` (bounds-checked via
  file/rank, never via raw shifting past the board edge) rather than by
  masking after the fact.
- **Full-hash TT verification, no lockless XOR scheme.** Because Python is
  single-process and the search itself is single-threaded (§7.5), the
  classic multi-threaded TT hazards (torn reads across threads) don't
  apply; storing the full 64-bit key per entry and comparing on probe is
  simplicity with no real downside here.
- **Recomputed evaluation vs. incremental.** `evaluate()` sums material+PST
  from scratch at every leaf. An incremental score (updated in
  `make_move`/`unmake_move` next to the Zobrist hash) is a natural,
  low-risk follow-up once perft and search are validated — deferred so
  `board.py`'s make/unmake stays focused on one job at a time.
- **No quiescence search in v1.** As specified, the search is exactly
  minimax/alpha-beta + iterative deepening + TT bottoming out at
  `evaluate()` on every leaf. This is known to suffer from the horizon
  effect on forcing capture sequences; quiescence search (searching
  captures only, past `depth == 0`, until the position is "quiet") is the
  natural first extension and slots in at the `if depth == 0:` line in
  `alpha_beta` without changing its signature.
- **Single-threaded search.** `SearchContext` is not shared across
  multiple search threads; adding lazy-SMP would require making the TT
  thread-safe (or accepting benign races on it, as most engines do) and
  giving each thread its own killers/history tables — a larger change,
  intentionally out of scope here.
- **Simple time management.** `parse_go_limits`'s allocation formula is a
  placeholder (a fixed fraction of remaining time); this is isolated in
  `uci.py` and can be replaced without touching `search.py`, whose
  `SearchLimits.move_time_ms` contract stays the same.
