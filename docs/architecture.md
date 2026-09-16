# death-Token Chess Engine — Architecture

- **Status:** Accepted (synthesized from three independent proposals)
- **Package:** `src/chessengine/`
- **Inputs:** `docs/proposals/proposal-bitboard-classical.md`,
  `docs/proposals/proposal-bitboard-magic.md`,
  `docs/proposals/proposal-hybrid-pragmatic.md`
- **Author:** Francesco Errico (francesco.errico@relatech.com)
- **Date:** 2026-09-16

## 0. How to use this document

This is the single architecture reference implementers build against. It
picks one concrete design per decision point (board representation, move
encoding, search, evaluation, module layout) and specifies it precisely
enough to implement directly: exact bit layouts, exact function signatures,
exact module boundaries. Where the three input proposals disagreed, §16
records the decision and why; the rest of the document states the *chosen*
design only, without hedging between alternatives.

The guiding priority order, in this order, for every decision below:

1. **Correctness** — a from-scratch chess engine lives or dies on legal move
   generation being exactly right (perft-provable), because every bug here
   silently corrupts search and evaluation in ways that are hard to detect
   later.
2. **Testability** — every layer must be testable in isolation: pure
   functions over an explicit `Board`, no hidden global state, no layer
   reaching into another's internals.
3. **Clean layering** — one-directional module dependencies (§11), so a
   rules-engine bug can never be a search bug in disguise, and vice versa.
4. **Performance** — real, but explicitly last. This is a pure-Python engine;
   it is not competing with C/C++ engines on nodes-per-second. Every
   performance-motivated choice below is justified against these criteria,
   not assumed by default, and premature micro-optimization is deferred to
   Milestone 5 behind stable interfaces.

## 1. Goals & non-goals

**Goals**

- A from-scratch, dependency-free (`dependencies = []`) Python chess engine:
  bitboard board representation, classical ray-scanning move generation,
  make/unmake with a full undo stack, negamax + alpha-beta search with
  iterative deepening, a Zobrist-keyed transposition table, quiescence
  search, an extensible material+PST evaluation, and a UCI protocol handler
  so the engine drops into any standard GUI (Arena, CuteChess, `xboard`).
- Perft-validated correctness as the non-negotiable, top-level acceptance
  bar for move generation, covering the fiddly rules: en passant, castling
  (through/into/out of check), promotion (incl. underpromotion), pins, and
  discovered checks.
- Code any contributor can read start to finish and trust: every hot path
  reduces to a small number of already-tested primitives, not a wall of
  special cases.

**Non-goals (v1, i.e. Milestones 1–4)**

- Magic bitboards / PEXT sliding attacks. Not ruled out forever — see §5.1
  and §5.11 for the concrete, low-risk seam to add them in Milestone 5 if
  profiling ever demands it — but not part of the initial design.
- NNUE / learned evaluation, endgame tablebases. Opening books and simple
  endgame heuristics are scoped into Milestone 5 only.
- Multi-threaded (lazy-SMP) search. Single-threaded negamax; `go` runs on a
  background thread purely so the UCI stdin loop stays responsive to
  `stop`/`quit` (§12).
- Aggressive pruning (null-move, LMR, aspiration windows, check extensions).
  Noted as optional, individually-gated Milestone 5 extensions, not part of
  the core contract.

## 2. Decision summary

| Decision point | Chosen design | Rationale (short) |
|---|---|---|
| Board representation | 12 bitboards (`pieces[color][piece_type]`) + occupancy unions + a maintained mailbox cache | Bitboards are the right substrate for Zobrist/TT and for a future magic-bitboard upgrade; mailbox makes "what's on this square" O(1) for the many call sites that need it (§3) |
| Sliding-attack generation | Classical ray-scanning (`RAY_ATTACKS` + first-blocker chop), **not** magic bitboards | Correctness/testability-first per project goals; ~4 ray unions + bit-scans per call is a modest cost dominated by Python's own per-call overhead anyway, and the alternative buys real speed at a real complexity and test-surface cost (§5.1) |
| Move encoding | Packed 16-bit int: 6 bits `from`, 6 bits `to`, 4 bits `flag` | Smallest scheme that still needs zero board lookups to decode; moving/captured piece are read from `mailbox` at apply time (§4) |
| Legal move generation | Pseudo-legal generation + explicit pin masks + check-evasion masks, with a make/unmake fallback reserved for the one case (en passant discovered check) that isn't cheaply maskable | Avoids paying a make/unmake round trip per pseudo-legal move in the common case, without hand-rolling the one genuinely fiddly edge case (§5) |
| Search | Negamax + alpha-beta, iterative deepening, Zobrist TT, quiescence with SEE-gated captures, MVV-LVA + killers + history ordering | Standard, well-understood architecture; TT must be provably not change results (§9.2), only speed |
| Time/stop control | Injectable stop-predicate (`threading.Event` for UCI, plain callable for tests) checked at a single choke point | Makes search interruption unit-testable without real wall-clock sleeps, while still supporting a real UCI `stop` command (§9.3) |
| Evaluation | `Evaluator` protocol + `CompositeEvaluator` over independently-testable term functions; material+PST is the only enabled term at first | Search depends on the one-method protocol, never a concrete evaluator — new terms are additive (§10) |
| Module layout | Flat `src/chessengine/*.py`, strict one-directional dependency DAG (§11) | Matches the existing flat package layout in `pyproject.toml`/`src/chessengine/__init__.py`; a DAG this shallow doesn't need subpackages to stay clean |

## 3. Board representation

### 3.1 Square indexing and bit layout

Little-endian rank-file (LERF), the standard bitboard convention:

```
square = rank * 8 + file        # rank, file in [0, 7]
a1 = 0   b1 = 1   ...  h1 = 7
a8 = 56  b8 = 57  ...  h8 = 63
```

Bit `i` of a 64-bit bitboard corresponds to square `i`. A bitboard is a
plain Python `int`, always kept in `[0, 2**64 - 1]`:

```python
# bitboard.py
BB_ALL = 0xFFFF_FFFF_FFFF_FFFF

def mask64(x: int) -> int:
    return x & BB_ALL
```

Python ints are arbitrary-precision, so reads never silently truncate, but
any construction that could set bit ≥ 64 (e.g. an intermediate step while
building ray tables) must be bounds-checked by construction (via file/rank
deltas, never raw `<< n` past the edge) — this is called out explicitly at
every site in §5 where it applies.

`constants.py` defines the shared vocabulary every other module may depend
on — enums, masks, and (since they have no natural home lower in the
dependency graph, see §11) the shared search-score constants:

```python
# constants.py — no behavior, only names, masks, and small pure helpers
WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
NO_PIECE = 12                     # mailbox sentinel

def piece_type_of(code: int) -> int: return code % 6
def color_of(code: int) -> int: return code // 6
def file_of(sq: int) -> int: return sq & 7
def rank_of(sq: int) -> int: return sq >> 3

FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H = range(8)
FILE_MASK = [0x0101010101010101 << f for f in range(8)]
RANK_MASK = [0xFF << (8 * r) for r in range(8)]

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

SQUARE_NAMES = [f + r for r in "12345678" for f in "abcdefgh"]  # index -> "e4" etc.

MAX_PLY = 128

# Search score constants (shared by transposition.py and search.py; homed
# here, not in either, so neither module depends on the other for a constant)
INF = 1_000_000
MATE_SCORE = 100_000
DRAW_SCORE = 0
```

### 3.2 `Board` class (`board.py`)

```python
# board.py
from dataclasses import dataclass

@dataclass(slots=True)
class UndoInfo:
    move: int                 # encoded move (§4)
    captured_piece: int       # piece code 0..11, or NO_PIECE
    captured_square: int      # differs from move's `to` only for en passant
    castling_rights: int      # rights *before* this move
    ep_square: int | None     # en-passant target square *before* this move
    halfmove_clock: int       # 50-move counter *before* this move
    zobrist_hash: int         # full hash *before* this move

class Board:
    __slots__ = (
        "pieces", "occupied_co", "occupied", "mailbox",
        "side_to_move", "castling_rights", "ep_square",
        "halfmove_clock", "fullmove_number",
        "zobrist_hash", "history", "position_history",
    )

    def __init__(self) -> None:
        self.pieces: list[list[int]] = [[0] * 6 for _ in range(2)]   # [color][piece_type] -> bitboard
        self.occupied_co: list[int] = [0, 0]        # per-color union
        self.occupied: int = 0                       # union of both colors
        self.mailbox: list[int] = [NO_PIECE] * 64    # square -> piece code
        self.side_to_move: int = WHITE
        self.castling_rights: int = 0                # 4-bit mask (§3.1)
        self.ep_square: int | None = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.zobrist_hash: int = 0
        self.history: list[UndoInfo] = []
        self.position_history: list[int] = []        # hashes seen, for repetition

    @staticmethod
    def starting_position() -> "Board":
        from . import fen                              # lazy import: fen.py depends
        return fen.parse_fen(fen.STARTPOS_FEN)          # on board.py, not vice versa

    # Queries
    def king_square(self, color: int) -> int: ...
    def in_check(self, color: int | None = None) -> bool: ...   # delegates to attacks.checkers
    def is_fifty_move_draw(self) -> bool: return self.halfmove_clock >= 100
    def is_repetition_draw(self) -> bool: ...                    # §9.6

    # Mutators (§6)
    def make_move(self, move: int) -> None: ...
    def unmake_move(self) -> None: ...

    # Debug / IO
    def to_fen(self) -> str: ...        # delegates to fen.board_to_fen(self)
    def __str__(self) -> str: ...       # ASCII board for debugging/tests
```

`occupied_co`/`occupied` and `mailbox` are redundant with `pieces` but are
maintained as invariants inside `make_move`/`unmake_move` (§6), never
recomputed on demand, because move generation and evaluation need "all
pieces"/"own pieces"/"what's on square X" on nearly every call. Recomputing
any of them by scanning 6–12 bitboards per query would be strictly worse for
both speed and (more importantly) for the "one obvious place this can go
wrong" testability goal — a single-writer discipline (only `make_move`/
`unmake_move` ever mutate these fields) is what makes the invariant checkable
at all.

**Invariants, enforced by tests (§13):**
`board.occupied == board.occupied_co[WHITE] | board.occupied_co[BLACK]`;
for every `(color, piece_type)`, the set bits of `board.pieces[color][piece_type]`
are exactly the squares where `board.mailbox[sq] == color * 6 + piece_type`;
each side has exactly one king bit set at all times.

## 4. Move encoding (`move.py`)

A move is a 16-bit packed `int`: 6 bits `from`, 6 bits `to`, 4 bits `flag`.

| Bits | Width | Field |
|---|---|---|
| 0–5 | 6 | `from` square (0–63) |
| 6–11 | 6 | `to` square (0–63) |
| 12–15 | 4 | `flag` |

Flags:

| Flag | Value | Meaning |
|---|---|---|
| `QUIET` | `0x0` | normal, non-capturing move |
| `DOUBLE_PAWN_PUSH` | `0x1` | sets `ep_square` |
| `KING_CASTLE` | `0x2` | O-O |
| `QUEEN_CASTLE` | `0x3` | O-O-O |
| `CAPTURE` | `0x4` | ordinary capture |
| `EN_PASSANT` | `0x5` | captured square ≠ `to` |
| `0x6`, `0x7` | — | reserved, unused |
| `PROMO_KNIGHT`/`BISHOP`/`ROOK`/`QUEEN` | `0x8`–`0xB` | quiet promotion |
| `PROMO_KNIGHT_CAP`/`BISHOP_CAP`/`ROOK_CAP`/`QUEEN_CAP` | `0xC`–`0xF` | capturing promotion |

The flag encoding is deliberately bit-patterned so two single-bit tests
answer the two questions callers ask most: `flag & 0x4` is set on `CAPTURE`,
`EN_PASSANT`, and all four `*_CAP` promotions (i.e. "is this a capture");
`flag & 0x8` is set on all eight promotion flags (i.e. "is this a
promotion").

```python
# move.py
PROMO_PIECE_OF = {
    PROMO_KNIGHT: KNIGHT, PROMO_KNIGHT_CAP: KNIGHT,
    PROMO_BISHOP: BISHOP, PROMO_BISHOP_CAP: BISHOP,
    PROMO_ROOK: ROOK,     PROMO_ROOK_CAP: ROOK,
    PROMO_QUEEN: QUEEN,   PROMO_QUEEN_CAP: QUEEN,
}

NULL_MOVE = 0   # a1a1 quiet; never a legal move, used as a "no move" sentinel

def encode_move(frm: int, to: int, flag: int = QUIET) -> int:
    return frm | (to << 6) | (flag << 12)

def move_from(m: int) -> int: return m & 0x3F
def move_to(m: int) -> int: return (m >> 6) & 0x3F
def move_flag(m: int) -> int: return (m >> 12) & 0xF
def is_capture(m: int) -> bool: return bool(move_flag(m) & 0x4)
def is_promotion(m: int) -> bool: return bool(move_flag(m) & 0x8)

def move_to_uci(m: int) -> str:
    """Pure string conversion — no Board needed, since a move's own fields
    are enough to print it."""
    s = SQUARE_NAMES[move_from(m)] + SQUARE_NAMES[move_to(m)]
    flag = move_flag(m)
    if flag in PROMO_PIECE_OF:
        s += "nbrq"[[KNIGHT, BISHOP, ROOK, QUEEN].index(PROMO_PIECE_OF[flag])]
    return s
```

A move never records the moving piece or captured piece itself — `make_move`
reads the moving piece from `board.mailbox[frm]` and the captured piece from
`board.mailbox[to]` (or the en-passant square, §6) at apply time. This keeps
moves immutable, comparable by plain integer equality (used for TT
best-move matching and killer-move tables), and trivially storable in flat
`list[int]`s.

**`move_from_uci(board, uci) -> int` lives in `movegen.py`, not `move.py`**,
because resolving a UCI string (which carries no flag bits) requires
matching against the legal move list — putting it in `move.py` would make
`move.py` depend on `movegen.py`, which would depend on `move.py`, an import
cycle. Keeping it in `movegen.py` (which already depends on `move.py`) keeps
the dependency graph in §11 acyclic with no exceptions:

```python
# movegen.py
def move_from_uci(board: "Board", uci: str) -> int:
    frm, to = SQUARE_NAMES.index(uci[0:2]), SQUARE_NAMES.index(uci[2:4])
    promo = uci[4] if len(uci) == 5 else None
    promo_letter = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
    for m in generate_legal_moves(board):
        if move_from(m) != frm or move_to(m) != to:
            continue
        if promo is None and not is_promotion(m):
            return m
        if promo is not None and is_promotion(m) and promo_letter[PROMO_PIECE_OF[move_flag(m)]] == promo:
            return m
    raise ValueError(f"illegal or unknown move: {uci}")
```

Resolving against the generator rather than re-deriving a flag from scratch
means `move_from_uci` can never construct a move the generator itself would
not produce — one canonical source of truth for "what flag does this move
have."

## 5. Attack generation and move generation strategy

### 5.1 Decision: classical ray-scanning, not magic bitboards

The magic-bitboard proposal's core argument for magics was speed: O(1)
table lookup versus ~4 ray unions + bit-scans per sliding-attack call. That
argument is real, but it is exactly the criterion the project's stated
priorities rank last. Weighed against the other three criteria:

- **Correctness/testability.** A classical ray-scanning `sliding_attacks`
  function is checkable by inspection and by a single property test against
  an even-more-naive one-square-at-a-time tracer (§13). Magic bitboards
  require an additional, non-obvious correctness burden: a magic-number
  search algorithm, a collision-freedom proof per magic, a generated data
  file that must be regenerated deterministically and verified at import
  time, and a fallback path for when that data file is stale or missing.
  None of that is hard, but all of it is *additional surface area* that has
  to be right before move generation can be trusted, for a payoff (NPS) the
  project has explicitly deprioritized.
- **Performance, honestly assessed.** In CPython, a 64-bit "bitboard" is a
  heap `int` object either way; the dominant cost of both approaches is
  Python's own per-call/per-bytecode overhead, not the arithmetic. Magic
  bitboards remain meaningfully faster in absolute terms, but the *relative*
  win is smaller here than in a C engine, and it does not change the
  asymptotic shape of the search (perft/search node counts are identical
  either way — only wall-clock per node changes).
- **A stable future seam.** `bishop_attacks(sq, occupied)`,
  `rook_attacks(sq, occupied)`, and `queen_attacks(sq, occupied)` in
  `attacks.py` are the *only* functions that would need to change to adopt
  magic bitboards later. Nothing in `movegen.py`, `board.py`, or `search.py`
  calls anything more specific than these three signatures. This is
  Milestone 5 work (§15), gated on profiling actually showing move
  generation is the bottleneck — not assumed up front.

**Decision: classical ray-scanning sliding attacks**, precomputed leaper
tables for knight/king/pawn, explicit pin and check-evasion masks so the
common case never pays for a make/unmake round trip, with one deliberate,
narrow exception (en passant discovered check, §5.8) resolved by make/unmake
because it is genuinely fiddly to mask correctly and vanishingly rare per
position (at most one or two candidate moves).

### 5.2 Leaper attack tables (`attacks.py`)

Built once at import time from bounds-checked file/rank deltas (never from
raw index arithmetic, which would wrap around the a/h files):

```python
# attacks.py
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
PAWN_ATTACKS = [_leaper_table([(1, 1), (-1, 1)]),    # WHITE: captures NE/NW
                _leaper_table([(1, -1), (-1, -1)])]  # BLACK: captures SE/SW
```

### 5.3 Sliding attacks via ray-scanning (`attacks.py`)

```python
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
    """RAY_ATTACKS[d][sq] = every square from sq to the board edge along
    direction d, ignoring occupancy. Bounds-checked via file/rank deltas."""
    rays = [[0] * 64 for _ in range(8)]
    for sq in range(64):
        f0, r0 = file_of(sq), rank_of(sq)
        for d in range(8):
            df, dr = DIR_FILE_RANK_DELTA[d]
            bb, f, r = 0, f0 + df, r0 + dr
            while 0 <= f < 8 and 0 <= r < 8:
                bb |= 1 << (r * 8 + f)
                f, r = f + df, r + dr
            rays[d][sq] = bb
    return rays

RAY_ATTACKS = _build_ray_attacks()

def sliding_attacks(sq: int, occupied: int, dirs: tuple[int, ...]) -> int:
    """Union the full ray in each direction, then chop off everything
    beyond the first blocker (blocker square itself stays attacked, since
    sliders can capture into it)."""
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

`SQUARES_BETWEEN[a][b]` (used for pin masks and check-block masks, §5.5/5.6)
is precomputed once the same way: `SQUARES_BETWEEN[a][b]` is 0 unless `a`
and `b` share a rank, file, or diagonal, in which case it is exactly the
squares strictly between them.

`bitboard.py` owns the bit-scan primitives used throughout this section and
by `movegen.py`:

```python
# bitboard.py
def lsb_index(bb: int) -> int: return (bb & -bb).bit_length() - 1
def msb_index(bb: int) -> int: return bb.bit_length() - 1
def pop_lsb(bb: int) -> tuple[int, int]:
    idx = lsb_index(bb)
    return idx, bb & (bb - 1)
def popcount(bb: int) -> int: return bb.bit_count()   # native, Python 3.10+
def iter_bits(bb: int):
    while bb:
        idx, bb = pop_lsb(bb)
        yield idx
```

### 5.4 Attacked-square queries and check detection (`attacks.py`)

```python
def attackers_to(board: "Board", sq: int, by_color: int, occupied: int | None = None) -> int:
    """Bitboard of every by_color piece attacking sq, given an explicit
    occupancy (so callers can probe 'what if this square were empty',
    needed for king-move legality, §5.10)."""
    occ = board.occupied if occupied is None else occupied
    p = board.pieces[by_color]
    attackers  = PAWN_ATTACKS[1 - by_color][sq] & p[PAWN]   # "attacked by enemy pawn" ==
    attackers |= KNIGHT_ATTACKS[sq] & p[KNIGHT]              # "pawn of the opposite color
    attackers |= KING_ATTACKS[sq] & p[KING]                  #  would attack from sq"
    attackers |= bishop_attacks(sq, occ) & (p[BISHOP] | p[QUEEN])
    attackers |= rook_attacks(sq, occ) & (p[ROOK] | p[QUEEN])
    return attackers

def is_attacked(board: "Board", sq: int, by_color: int) -> bool:
    return attackers_to(board, sq, by_color) != 0

def checkers(board: "Board", color: int) -> int:
    return attackers_to(board, board.king_square(color), 1 - color)
```

`board.py`'s `in_check` (§3.2) calls straight into `attacks.checkers` — this
keeps the `board → attacks` dependency direction from §11 intact (nothing in
`attacks.py` imports `board.py`).

### 5.5 Pins (`movegen.py`)

Computed once per `generate_legal_moves` call by ray-scanning outward from
the king — eight directions, O(1) bit tricks per direction, cheap enough to
call unconditionally instead of caching across plies:

```python
def pinned_pieces(board: "Board", color: int) -> dict[int, int]:
    """{pinned_square: allowed_destination_mask}. A pinned piece may only
    move within its returned mask (the squares between king and pinner,
    plus the pinner's own square)."""
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
            continue                          # nearest piece on this ray is enemy's: no pin
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

### 5.6 Check evasion masks (`movegen.py`)

```python
def evasion_masks(board: "Board", color: int) -> tuple[int, int]:
    """(capture_mask, push_mask): a non-king move is legal only if its
    destination is in (capture_mask | push_mask). Both are 'all squares'
    when not in check."""
    ch = attacks.checkers(board, color)
    if ch == 0:
        return BB_ALL, BB_ALL
    if ch.bit_count() >= 2:
        return 0, 0                          # double check: only king moves are legal
    checker_sq = lsb_index(ch)
    checker_type = piece_type_of(board.mailbox[checker_sq])
    king_sq = board.king_square(color)
    push_mask = SQUARES_BETWEEN[king_sq][checker_sq] if checker_type in (BISHOP, ROOK, QUEEN) else 0
    return ch, push_mask
```

### 5.7 Castling (`movegen.py`)

Precomputed per (color, side) lookup tables (built once, both colors):

```python
CASTLE_EMPTY_MASK = {                       # squares that must be empty
    (WHITE, KING_CASTLE): (1 << F1) | (1 << G1),
    (WHITE, QUEEN_CASTLE): (1 << B1) | (1 << C1) | (1 << D1),
    (BLACK, KING_CASTLE): (1 << F8) | (1 << G8),
    (BLACK, QUEEN_CASTLE): (1 << B8) | (1 << C8) | (1 << D8),
}
CASTLE_KING_PATH = {                        # king's start/pass-through/landing squares;
    (WHITE, KING_CASTLE): (E1, F1, G1),     # none may be attacked
    (WHITE, QUEEN_CASTLE): (E1, D1, C1),
    (BLACK, KING_CASTLE): (E8, F8, G8),
    (BLACK, QUEEN_CASTLE): (E8, D8, C8),
}
CASTLE_RIGHT_BIT = {(WHITE, KING_CASTLE): CASTLE_WK, (WHITE, QUEEN_CASTLE): CASTLE_WQ,
                     (BLACK, KING_CASTLE): CASTLE_BK, (BLACK, QUEEN_CASTLE): CASTLE_BQ}
CASTLE_ROOK_MOVE = {(KING_CASTLE, WHITE): (H1, F1), (QUEEN_CASTLE, WHITE): (A1, D1),
                     (KING_CASTLE, BLACK): (H8, F8), (QUEEN_CASTLE, BLACK): (A8, D8)}

def generate_castling_moves(board: "Board", color: int, in_check: bool, moves: list[int]) -> None:
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
```

Castling rights are maintained via a spoiler table applied on every move in
`make_move` (§6), which handles all four ways rights are lost (king moves,
either rook moves, either rook is *captured* on its home square) with one
lookup on both `frm` and `to`:

```python
CASTLE_SPOILER = [0] * 64
CASTLE_SPOILER[E1] = CASTLE_WK | CASTLE_WQ
CASTLE_SPOILER[A1] = CASTLE_WQ
CASTLE_SPOILER[H1] = CASTLE_WK
CASTLE_SPOILER[E8] = CASTLE_BK | CASTLE_BQ
CASTLE_SPOILER[A8] = CASTLE_BQ
CASTLE_SPOILER[H8] = CASTLE_BK
# applied as: board.castling_rights &= ~(CASTLE_SPOILER[frm] | CASTLE_SPOILER[to])
```

### 5.8 En passant (`movegen.py`)

Generated as part of pawn-capture generation: if `board.ep_square is not
None` and `PAWN_ATTACKS[color][pawn_sq] & (1 << board.ep_square)`, emit
`encode_move(pawn_sq, board.ep_square, EN_PASSANT)`. The captured pawn's
square is **not** `to` — it is `to - 8` (white capturing) or `to + 8` (black
capturing); `make_move` computes this explicitly (§6).

`ep_square` is set only by a double pawn push, to the *skipped* square
(`(frm + to) // 2`), and cleared on every other move — exactly the one-ply
lifetime en passant has under the real rules.

**Discovered-check edge case.** En passant is the one move where two pieces
leave the board simultaneously (the capturing pawn's origin, and the
captured pawn's square, which is *not* the destination square). This can
expose a horizontal discovered check along the 4th/5th rank that neither
the pin table (§5.5, which only tracks single blockers per ray) nor the
evasion mask (§5.6) accounts for. Rather than special-case this rank-scan,
**every en passant candidate move is legality-checked by the make/unmake +
`is_attacked` fallback**, regardless of what the pin table says — there are
at most one or two en passant candidates in any position, so this costs
nothing measurable and sidesteps a well-documented class of engine bugs.

### 5.9 Promotion (`movegen.py`)

A pawn move whose destination is rank 8 (white) or rank 1 (black) is
expanded into four moves, one per promotion piece, reusing the quiet/capture
distinction: `PROMO_QUEEN`/`ROOK`/`BISHOP`/`KNIGHT` for a quiet promotion,
the `*_CAP` variants for a capturing one (en passant never lands on the back
rank, so no interaction there). `make_move` reads `PROMO_PIECE_OF[flag]` to
know which piece to place on `to` instead of the pawn.

### 5.10 Putting it together (`movegen.py`)

```python
def generate_pseudo_legal_moves(board: "Board") -> list[int]:
    """Moves obeying piece movement rules, not yet filtered for leaving
    one's own king in check."""
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
    in_check = attacks.is_attacked(board, king_sq, 1 - color)
    capture_mask, push_mask = evasion_masks(board, color)
    pins = pinned_pieces(board, color)
    legal: list[int] = []
    for m in generate_pseudo_legal_moves(board):
        frm, to, flag = move_from(m), move_to(m), move_flag(m)
        if flag == EN_PASSANT:
            board.make_move(m)                                            # §5.8 fallback
            ok = not attacks.is_attacked(board, board.king_square(color), 1 - color)
            board.unmake_move()
            if ok:
                legal.append(m)
            continue
        if frm == king_sq:
            if flag in (KING_CASTLE, QUEEN_CASTLE):
                legal.append(m)          # already fully validated by generate_castling_moves
                continue
            occ_without_king = board.occupied & ~(1 << king_sq)
            if attacks.attackers_to(board, to, 1 - color, occupied=occ_without_king) == 0:
                legal.append(m)
            continue
        if frm in pins and not (pins[frm] >> to) & 1:
            continue                     # pinned piece moving off its pin line
        if not ((capture_mask | push_mask) >> to) & 1:
            continue                     # doesn't capture the checker or block the check
        legal.append(m)
    return legal

def generate_captures(board: "Board") -> list[int]:
    """Subset used by quiescence search (§9.5): captures, en-passant, and
    capturing/quiet promotions (a quiet promotion to queen is treated as
    'noisy enough' to search in quiescence even without a capture)."""
    return [m for m in generate_legal_moves(board) if is_capture(m) or is_promotion(m)]
```

Per-piece generators (`_gen_pawn_moves`, `_gen_knight_moves`,
`_gen_king_moves`, `_gen_sliding_moves`) all follow the same shape: iterate
set bits of `board.pieces[color][piece]` via `iter_bits`, look up the attack
bitboard, mask off the mover's own pieces, and emit one `encode_move(...)`
per destination bit (splitting quiet/capture, and further into four
promotions where applicable).

### 5.11 Future optimization seam: magic bitboards

If profiling in Milestone 5 shows sliding-attack generation is the
bottleneck (not assumed — measured), `bishop_attacks`/`rook_attacks` in
`attacks.py` can be replaced by magic-bitboard table lookups with the exact
same signatures (`(square: int, occupied: int) -> int`). No caller in
`movegen.py`, `board.py`, or `search.py` needs to change. The replacement
would additionally need: a relevant-occupancy mask per square, an offline
magic-number search (random search with a sparse-candidate bias is the
standard technique), a checked-in generated data module, and a
collision-freedom test against the classical `sliding_attacks` as the
oracle — i.e. the classical implementation in §5.3 does not get deleted, it
becomes the correctness oracle magics are validated against.

## 6. Make / unmake move (`board.py`)

`Board.make_move`/`Board.unmake_move` mutate the four synchronized
representations (`pieces`, `occupied_co`, `occupied`, `mailbox`) plus
`zobrist_hash`, and push/pop an `UndoInfo` (§3.2) capturing everything not
trivially reversible from the move encoding alone.

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
    elif flag & 0x4:                                   # CAPTURE or a *_CAP promotion
        captured_piece = self.mailbox[to]

    self.history.append(UndoInfo(move, captured_piece, captured_sq,
                                  self.castling_rights, self.ep_square,
                                  self.halfmove_clock, self.zobrist_hash))

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

    self.halfmove_clock = 0 if (moving_type == PAWN or captured_piece != NO_PIECE) else self.halfmove_clock + 1
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
    if us == BLACK:
        self.fullmove_number -= 1
```

`_add_piece`/`_remove_piece`/`_move_piece` are small private helpers that
keep all four representations plus the piece-square component of
`zobrist_hash` in sync in exactly one place:

```python
def _add_piece(self, piece_code: int, sq: int) -> None:
    color, ptype, bit = color_of(piece_code), piece_type_of(piece_code), 1 << sq
    self.pieces[color][ptype] |= bit
    self.occupied_co[color] |= bit
    self.occupied |= bit
    self.mailbox[sq] = piece_code
    self.zobrist_hash ^= ZOBRIST_PIECE[color][ptype][sq]

def _remove_piece(self, piece_code: int, sq: int) -> None:
    color, ptype, bit = color_of(piece_code), piece_type_of(piece_code), 1 << sq
    self.pieces[color][ptype] &= ~bit
    self.occupied_co[color] &= ~bit
    self.occupied &= ~bit
    self.mailbox[sq] = NO_PIECE
    self.zobrist_hash ^= ZOBRIST_PIECE[color][ptype][sq]

def _move_piece(self, piece_code: int, frm: int, to: int) -> None:
    self._remove_piece(piece_code, frm)
    self._add_piece(piece_code, to)
```

`UndoInfo` stores the full previous `zobrist_hash` (rather than re-deriving
it via reverse XORs) — 8 bytes of extra memory per stack frame buys "unmake
is always exactly correct, trivially," which matters more than the memory:
inverse-XOR unmake bugs are a classic, hard-to-spot source of engine bugs.

## 7. Zobrist hashing (`zobrist.py`)

```python
import random

_rng = random.Random(0xC0FFEE)   # fixed seed: reproducible hashes across runs/tests

ZOBRIST_PIECE = [[[_rng.getrandbits(64) for _ in range(64)] for _ in range(6)] for _ in range(2)]  # [color][piece][sq]
ZOBRIST_SIDE = _rng.getrandbits(64)
ZOBRIST_CASTLING = [_rng.getrandbits(64) for _ in range(16)]   # indexed by the 4-bit rights mask directly
ZOBRIST_EP_FILE = [_rng.getrandbits(64) for _ in range(8)]     # keyed by file only (rank is implied by side to move)

def compute_hash(board: "Board") -> int:
    """From-scratch hash, independent of make_move/unmake_move's incremental
    bookkeeping. Used only by tests (§13) as an oracle, and to seed a Board
    built directly from FEN — never called from a hot path."""
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

The en-passant key represents "en passant was *available* in this position"
(a property that affects legal moves), so it is XORed in when a double push
creates the target and XORed back out on the very next move, regardless of
whether the capture happened — otherwise two positions differing only in a
long-expired ep square would hash differently despite being legally
identical.

## 8. FEN (`fen.py`)

```python
# fen.py
STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def parse_fen(fen: str) -> "Board": ...        # the only place FEN grammar is parsed
def board_to_fen(board: "Board") -> str: ...   # the only place FEN grammar is serialized
```

`fen.py` depends on `board.py` (it constructs and reads a `Board`); `board.py`
never imports `fen.py` at module scope, only lazily inside
`Board.starting_position()` (§3.2), which is what keeps the dependency graph
in §11 a strict DAG despite `Board` wanting a `starting_position()`
convenience constructor.

## 9. Search architecture

### 9.1 Negamax with alpha-beta

Implemented as **negamax**: at every node the score is relative to the side
to move, and each recursive call negates and swaps `(alpha, beta)`. This is
the standard reformulation of minimax + alpha-beta that removes the
maximize-or-minimize branch, not a different algorithm — nearly every engine
uses it.

The search's persistent state (transposition table, killer moves, history
heuristic) is owned by a `Search` instance, not module-level globals or
free-floating parameters threaded through every call — this is what lets
`ucinewgame` reset exactly one object, and what lets a test construct a
fresh `Search` per test case with no cross-test leakage:

```python
# search.py
class Search:
    def __init__(self, evaluator: "Evaluator", tt_size_mb: int = 64) -> None:
        self.evaluator = evaluator
        self.tt = TranspositionTable(tt_size_mb)
        self.killers: list[list[int]] = [[NULL_MOVE, NULL_MOVE] for _ in range(MAX_PLY)]
        self.history: list[list[int]] = [[0] * 64 for _ in range(64)]   # [from][to]

    def new_game(self) -> None:
        """Called on UCI 'ucinewgame'. Stale TT/killer/history entries from
        a previous, unrelated game must not leak into this one."""
        self.tt.clear()
        self.killers = [[NULL_MOVE, NULL_MOVE] for _ in range(MAX_PLY)]
        self.history = [[0] * 64 for _ in range(64)]
```

The negamax core (a method, so it has `self.evaluator`/`self.tt`/
`self.killers`/`self.history` in scope without threading them through every
call):

```python
    def _negamax(self, board: Board, depth: int, alpha: int, beta: int, ply: int, ctx: "_SearchCtx") -> int:
        ctx.nodes += 1
        if ctx.should_stop():
            return 0                                   # discarded: caller checks ctx.stopped

        if board.is_fifty_move_draw() or board.is_repetition_draw():
            return DRAW_SCORE

        alpha_orig = alpha
        entry = self.tt.probe(board.zobrist_hash)
        tt_move = entry.best_move if entry is not None else NULL_MOVE
        if entry is not None and entry.depth >= depth:
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
            return self._quiescence(board, alpha, beta, ply, ctx)

        moves = generate_legal_moves(board)
        if not moves:
            return -MATE_SCORE + ply if board.in_check() else DRAW_SCORE

        self._order_moves(moves, board, tt_move, ply)

        best_score, best_move = -INF, moves[0]
        for move in moves:
            board.make_move(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, ctx)
            board.unmake_move()
            if ctx.should_stop():
                return 0
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_cutoff(move, depth, ply)      # killers/history, §9.4
                break

        flag = (TTFlag.EXACT if alpha_orig < best_score < beta else
                TTFlag.LOWERBOUND if best_score >= beta else TTFlag.UPPERBOUND)
        self.tt.store(board.zobrist_hash, depth, score_to_tt(best_score, ply), flag, best_move)
        return best_score
```

Checkmate scores `-MATE_SCORE + ply` (so a *shorter* mate scores higher in
magnitude, naturally steering the search toward faster mates with no special
casing); stalemate and the fifty-move/repetition draws score `DRAW_SCORE`.

### 9.2 Transposition table (`transposition.py`)

```python
# transposition.py
class TTFlag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

@dataclass(slots=True)
class TTEntry:
    key: int          # full 64-bit zobrist key, stored so index collisions are
                       # detected and rejected instead of returning a wrong hit
    depth: int
    score: int
    flag: TTFlag
    best_move: int

class TranspositionTable:
    def __init__(self, size_mb: int = 64) -> None:
        entry_count = (size_mb * 1024 * 1024) // 40     # ~40 bytes/entry incl. list overhead
        self.size = 1 << (entry_count.bit_length() - 1)  # round down to a power of two
        self.mask = self.size - 1
        self.table: list[TTEntry | None] = [None] * self.size

    def probe(self, key: int) -> TTEntry | None:
        entry = self.table[key & self.mask]
        return entry if entry is not None and entry.key == key else None

    def store(self, key: int, depth: int, score: int, flag: TTFlag, best_move: int) -> None:
        idx = key & self.mask
        existing = self.table[idx]
        # Depth-preferred replacement: only overwrite a same-key or
        # shallower-or-equal entry, so a shallow re-search never evicts a
        # deeper, more expensive-to-recompute one.
        if existing is None or existing.key == key or depth >= existing.depth:
            self.table[idx] = TTEntry(key, depth, score, flag, best_move)

    def clear(self) -> None:
        self.table = [None] * self.size
```

**Mate-distance adjustment.** A mate score found N plies below the *current*
node must be stored/retrieved relative to that node, not the search root, or
a mate reached via a transposition gets misreported as closer or further
than it really is from the root:

```python
def score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 128: return score + ply
    if score <= -MATE_SCORE + 128: return score - ply
    return score

def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_SCORE - 128: return score - ply
    if score <= -MATE_SCORE + 128: return score + ply
    return score
```

**Invariant, enforced by tests (§13): the TT must never change the search's
result, only its speed.** A search run with a real TT and the same search
run with a stub TT (`probe` always returns `None`, `store` a no-op) must
return the same score, and the same best move up to equal-score ties, at a
fixed small depth.

### 9.3 Iterative deepening and time management

```python
@dataclass
class SearchLimits:
    max_depth: int = 64
    movetime_ms: int | None = None
    nodes: int | None = None

@dataclass
class SearchInfo:            # one per completed depth, handed to on_info
    depth: int
    score_cp: int
    nodes: int
    pv: list[int]

@dataclass
class SearchResult:
    best_move: int
    score_cp: int
    depth: int
    nodes: int
    pv: list[int]

class _SearchCtx:
    """Per-call transient state. Not reused across .search() calls."""
    def __init__(self, limits: SearchLimits, deadline: float | None,
                 stop_event: "threading.Event | None", extra_stop: "Callable[[], bool] | None"):
        self.limits, self.deadline = limits, deadline
        self.stop_event, self.extra_stop = stop_event, extra_stop
        self.nodes = 0

    def should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        if self.limits.nodes is not None and self.nodes >= self.limits.nodes:
            return True
        # Wall-clock checked every 2048 nodes, not every node, so
        # time.monotonic() overhead doesn't distort node counts at shallow depth.
        if self.deadline is not None and self.nodes % 2048 == 0 and time.monotonic() >= self.deadline:
            return True
        if self.extra_stop is not None and self.extra_stop():
            return True
        return False
```

`extra_stop` is the testability seam: a unit test injects
`extra_stop=lambda: ctx_nodes_seen >= 100` or a fake-clock predicate and
asserts the search actually stops, with zero real wall-clock sleeps. `
stop_event` is the production seam: the UCI layer (§12) sets a real
`threading.Event` from the main thread on `stop`/`quit` while the search
runs on a background thread.

```python
    def search(self, board: Board, limits: SearchLimits, *,
               stop_event: "threading.Event | None" = None,
               extra_stop: "Callable[[], bool] | None" = None,
               on_info: "Callable[[SearchInfo], None] | None" = None) -> SearchResult:
        deadline = time.monotonic() + limits.movetime_ms / 1000 if limits.movetime_ms else None
        ctx = _SearchCtx(limits, deadline, stop_event, extra_stop)
        best = SearchResult(NULL_MOVE, 0, 0, 0, [])
        for depth in range(1, limits.max_depth + 1):
            score = self._negamax(board, depth, -INF, INF, 0, ctx)
            if ctx.should_stop() and depth > 1:
                break                       # partial/unreliable result from an aborted depth: discard
            pv = self._extract_pv(board, depth)
            best = SearchResult(pv[0] if pv else best.best_move, score, depth, ctx.nodes, pv)
            if on_info is not None:
                on_info(SearchInfo(depth, score, ctx.nodes, pv))
            if abs(score) >= MATE_SCORE - 128:
                break                       # forced mate found; no point searching deeper
        return best
```

Iterative deepening is not merely a time-management convenience: each
completed shallow pass populates the TT with best moves that dramatically
improve move ordering (and thus the alpha-beta cutoff rate) on the next,
deeper pass — the standard reason ID is *faster* than searching the target
depth directly, not only safer under a clock.

`_extract_pv` walks the TT from the current position following each node's
stored `best_move`, making/unmaking as it goes; it can be shorter than
`depth` or slightly unstable across iterations if TT entries were
overwritten mid-line (a known, accepted limitation — a triangular PV array
threaded through `_negamax` is the standard fix and is a natural Milestone 5
addition if PV stability becomes a visible problem).

### 9.4 Move ordering

`self._order_moves(moves, board, tt_move, ply)` sorts the move list in
place, highest priority first:

1. The TT's stored `best_move` for this position, if any — searched first.
2. Captures ranked by MVV-LVA: `victim_value * 16 - attacker_value` (from
   the same `PIECE_VALUE` table evaluation uses, §10), so "queen takes pawn"
   sorts after "pawn takes queen."
3. Killer moves: up to two quiet moves per ply (`self.killers[ply]`) that
   most recently caused a beta cutoff at this ply in a sibling node.
4. History heuristic: `self.history[frm][to]`, incremented by `depth * depth`
   on every beta cutoff from a quiet move, as the tiebreak for the rest.

`self._record_cutoff(move, depth, ply)` (called on a beta cutoff) updates
killers/history only for quiet moves — captures already get MVV-LVA
ordering and don't need history bookkeeping.

### 9.5 Quiescence search

Extends leaf nodes until the position is "quiet" (no immediately-winning
captures pending), avoiding the classic horizon effect (stopping mid-capture
sequence and misjudging a hanging piece):

```python
    def _quiescence(self, board: Board, alpha: int, beta: int, ply: int, ctx: "_SearchCtx") -> int:
        ctx.nodes += 1
        stand_pat = self.evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        for move in self._order_moves(generate_captures(board), board, NULL_MOVE, ply):
            if not see_ge(board, move, 0):        # SEE-based pruning: skip clearly-losing captures
                continue
            board.make_move(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1, ctx)
            board.unmake_move()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha
```

`stand_pat` (the static evaluation, as if the side to move could "stand pat"
and make no capture) both bounds the search and is the fallback score when
no capture improves on it. `see_ge(board, move, threshold) -> bool`
(Static Exchange Evaluation, in `search.py`) estimates a capture sequence on
one square using the sorted attacker-value list from the attack tables in
§5.4, and is used purely to prune obviously bad captures — a conservative or
even buggy SEE only costs speed, never correctness, since it never changes
which moves are legal, only which ones quiescence bothers to search.

### 9.6 Draws

```python
def is_repetition_draw(self) -> bool:
    """Treats the current position recurring once before (its 2nd
    occurrence in tracked history) as a draw — stricter than the official
    threefold rule, which is a standard, safe simplification for the
    search's internal draw detection (UCI-level game-result reporting is
    the GUI's responsibility, not this method's)."""
    current = self.position_history[-1]
    return self.position_history[:-1].count(current) >= 1
```

## 10. Evaluation interface (`evaluate.py`)

Search depends on a single-method protocol, never a concrete evaluator —
this is what lets material+PST ship first while mobility, king safety, and
pawn structure are added later as pure functions with zero changes to
`search.py`:

```python
# evaluate.py
class Evaluator(Protocol):
    def evaluate(self, board: "Board") -> int:
        """Centipawns, from the side-to-move's perspective (negamax
        convention: positive is good for whoever is to move)."""
        ...

TermFn = Callable[["Board"], int]   # a pure evaluation term, side-to-move-relative centipawns
```

### 10.1 Material + PST (the only enabled term through Milestone 4)

Material values in centipawns, plus one 64-entry piece-square table (PST)
per piece type, defined from White's point of view with index 0 = a1 and
index 63 = h8 (matching board indexing, §3.1); Black's lookup mirrors the
square vertically via `sq ^ 56` (flips the rank, keeps the file) instead of
maintaining a second table.

Note the row order below runs **rank 1 to rank 8, top to bottom**, matching
a1=0 indexing — the opposite of how these tables are usually printed in
chess references (rank 8 first). Transcribing a reference table's printed
row order directly into an a1=0 array is a classic off-by-mirror bug (it
rewards White's pawns for sitting on their *starting* rank instead of their
promotion rank) — double-check row-to-rank mapping when filling in
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
# KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_PST: same 64-int-tuple
# shape, standard "PeSTO-style" values, indexed a1..h8 exactly like PAWN_PST.
PST = {PAWN: PAWN_PST, KNIGHT: KNIGHT_PST, BISHOP: BISHOP_PST,
       ROOK: ROOK_PST, QUEEN: QUEEN_PST, KING: KING_PST}

def _side_score(board: "Board", color: int) -> int:
    score, mirror = 0, color == BLACK
    for ptype in range(6):
        table = PST[ptype]
        for sq in iter_bits(board.pieces[color][ptype]):
            score += PIECE_VALUE[ptype] + (table[sq ^ 56] if mirror else table[sq])
    return score

def material_pst_term(board: "Board") -> int:
    white_score = _side_score(board, WHITE) - _side_score(board, BLACK)
    return white_score if board.side_to_move == WHITE else -white_score
```

Milestone 2/3 recompute this sum from scratch at every leaf — simple and
obviously correct. An incremental running total, updated inside
`make_move`/`unmake_move` alongside the Zobrist hash, is a natural,
low-risk Milestone 5 optimization once correctness is established; it does
not change `Evaluator`'s contract.

### 10.2 `CompositeEvaluator` and the extension path

```python
@dataclass
class Weights:
    material_pst: float = 1.0
    mobility: float = 0.0           # 0.0 until Milestone 5 implements the term
    king_safety: float = 0.0
    pawn_structure: float = 0.0

class CompositeEvaluator:
    """Sums independently-testable term functions by weight. Adding a term
    is: write a new pure `*_term(board) -> int` function, register it here,
    add one Weights field — search.py never changes."""
    def __init__(self, terms: dict[str, TermFn], weights: Weights | None = None) -> None:
        self.terms = terms
        self.weights = weights or Weights()

    def evaluate(self, board: "Board") -> int:
        total = 0
        for name, term_fn in self.terms.items():
            w = getattr(self.weights, name, 0.0)
            if w:
                total += int(w * term_fn(board))
        return total

def default_evaluator() -> CompositeEvaluator:
    """The Milestone 2/3/4 evaluator: material+PST only."""
    return CompositeEvaluator(terms={"material_pst": material_pst_term}, weights=Weights(material_pst=1.0))
```

Each `*_term` function is pure (`Board -> int`, no hidden state, no
dependency on search or on other terms), independently unit-testable (e.g.
"a FEN with an extra queen for White scores `material_pst_term` roughly
+900"), and independently addable. **Milestone 5** adds `mobility_term`,
`king_safety_term`, and `pawn_structure_term` this same way — each as a new
function (in `evaluate.py`, or split into its own file, e.g. `mobility.py`,
if the file grows large; either way `Evaluator`'s one-method contract is
unaffected), registered into `default_evaluator()`'s `terms` dict with a
nonzero `Weights` field. A future NNUE-style or otherwise opaque evaluator
is a drop-in replacement for `CompositeEvaluator` with zero changes to
`search.py`, since both satisfy the same one-method `Evaluator` protocol.

## 11. Module boundaries

```
src/chessengine/
├── __init__.py         # package version; re-exports Board for `from chessengine import Board`
├── constants.py         # enums (color/piece), square/file/rank helpers & masks, castling flags,
│                        #   MAX_PLY, shared search-score constants (INF, MATE_SCORE, DRAW_SCORE).
│                        #   No behavior beyond tiny pure helpers. Deps: none.
├── bitboard.py           # lsb_index, msb_index, pop_lsb, popcount, iter_bits, print_bitboard (debug).
│                        #   Deps: none.
├── attacks.py            # KNIGHT_ATTACKS/KING_ATTACKS/PAWN_ATTACKS, RAY_ATTACKS, SQUARES_BETWEEN,
│                        #   sliding_attacks/bishop_attacks/rook_attacks/queen_attacks,
│                        #   attackers_to/is_attacked/checkers. Deps: constants, bitboard.
├── zobrist.py            # ZOBRIST_PIECE/SIDE/CASTLING/EP_FILE tables, compute_hash (test oracle).
│                        #   Deps: constants, bitboard.
├── move.py               # encode_move/move_from/move_to/move_flag/is_capture/is_promotion,
│                        #   PROMO_PIECE_OF, NULL_MOVE, move_to_uci. Deps: constants.
├── board.py              # Board, UndoInfo, make_move/unmake_move, king_square/in_check,
│                        #   is_fifty_move_draw/is_repetition_draw, to_fen/__str__.
│                        #   Deps: constants, bitboard, attacks, zobrist, move.
│                        #   (Board.starting_position() lazily imports fen — §3.2, §8.)
├── fen.py                # STARTPOS_FEN, parse_fen, board_to_fen. Deps: board, constants.
├── movegen.py            # generate_pseudo_legal_moves, generate_legal_moves, generate_captures,
│                        #   pinned_pieces, evasion_masks, generate_castling_moves, move_from_uci.
│                        #   Deps: constants, bitboard, attacks, board, move.
├── perft.py              # perft(board, depth), divide(board, depth). Deps: board, movegen.
├── transposition.py       # TTFlag, TTEntry, TranspositionTable, score_to_tt/score_from_tt.
│                        #   Deps: constants (MATE_SCORE only).
├── evaluate.py            # Evaluator protocol, PIECE_VALUE, all PSTs, material_pst_term,
│                        #   Weights, CompositeEvaluator, default_evaluator.
│                        #   Deps: constants, bitboard, board. Never movegen or search.
├── search.py              # SearchLimits, SearchInfo, SearchResult, Search (owns TT + killers +
│                        #   history), see_ge. Deps: constants, board, movegen, evaluate,
│                        #   transposition, move.
├── uci.py                # UCIEngine, parse_go_limits. The only module touching sys.stdin/stdout
│                        #   and threading. Deps: board, movegen, move, search, fen, constants.
└── cli.py                # main() console-script entry point. Deps: uci.
```

**Dependency direction is a strict DAG, no exceptions:**
`constants, bitboard` (leaves) → `attacks, zobrist, move` → `board` → `{fen,
movegen}` → `{evaluate, transposition, perft}` → `search` → `uci` → `cli`.
Concretely:

- `attacks.py`, `zobrist.py`, `move.py` depend only on `constants`/`bitboard`
  — none of them import `board.py`, which is what lets `board.py` import
  `attacks.py` for `in_check` with no cycle.
- `fen.py` depends on `board.py`; `board.py` never imports `fen.py` at
  module scope (only lazily, inside one method) — this is the one place a
  "convenience constructor wanting a lower-level helper" would otherwise
  create a cycle, and it's resolved by a documented lazy import rather than
  by an exception to the dependency rule.
- `movegen.py` depends on `board.py` and `move.py`; `move.py` never depends
  on `movegen.py` — `move_from_uci` (which does need the legal-move list)
  lives in `movegen.py`, not `move.py` (§4), specifically to avoid that
  cycle.
- `evaluate.py` depends on `board.py` only (read-only queries: piece
  bitboards, side to move) — never on `movegen.py` or `search.py`. This is
  what makes evaluation terms unit-testable against a bare `Board` with no
  search machinery involved.
- `search.py` is the only module that depends on `movegen.py` +
  `evaluate.py` + `transposition.py` together.
- `uci.py`/`cli.py` are the outermost, user-facing layer; nothing below them
  imports them.

This shallow, flat layout (no subpackages) matches the existing
`src/chessengine/` package and `pyproject.toml`; the DAG is enforceable in
CI cheaply (an import-linter config, or a small AST-walking test in
`tests/test_architecture.py` asserting no forbidden import edges exist) once
Milestone 1 stabilizes.

## 12. UCI protocol and CLI (`uci.py`, `cli.py`)

```python
# uci.py
class UCIEngine:
    def __init__(self) -> None:
        self.board = Board.starting_position()
        self.search = Search(default_evaluator(), tt_size_mb=64)
        self.stop_event = threading.Event()
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
            handler(args, out)         # unrecognized commands are silently ignored, per the UCI spec

    def cmd_uci(self, args, out) -> None:
        out.write("id name death-Token 0.1\nid author Francesco Errico\n")
        out.write("option name Hash type spin default 64 min 1 max 1024\n")
        out.write("uciok\n"); out.flush()

    def cmd_isready(self, args, out) -> None:
        out.write("readyok\n"); out.flush()

    def cmd_setoption(self, args, out) -> None:
        if "Hash" in args:
            self.search = Search(self.search.evaluator, tt_size_mb=int(args[args.index("value") + 1]))

    def cmd_ucinewgame(self, args, out) -> None:
        self.board = Board.starting_position()
        self.search.new_game()

    def cmd_position(self, args, out) -> None:
        if args[0] == "startpos":
            self.board, rest = Board.starting_position(), args[1:]
        else:  # args[0] == "fen"
            moves_idx = args.index("moves") if "moves" in args else len(args)
            self.board, rest = fen.parse_fen(" ".join(args[1:moves_idx])), args[moves_idx:]
        if rest and rest[0] == "moves":
            for uci_move in rest[1:]:
                self.board.make_move(movegen.move_from_uci(self.board, uci_move))

    def cmd_go(self, args, out) -> None:
        limits = parse_go_limits(args, self.board.side_to_move)
        self.stop_event = threading.Event()
        self.search_thread = threading.Thread(target=self._search_and_report, args=(limits, out), daemon=True)
        self.search_thread.start()

    def cmd_stop(self, args, out) -> None:
        self.stop_event.set()
        if self.search_thread is not None:
            self.search_thread.join()

    def cmd_quit(self, args, out) -> None:
        self.stop_event.set()
        self.quit = True

    def _search_and_report(self, limits: SearchLimits, out) -> None:
        def on_info(info: SearchInfo) -> None:
            pv_str = " ".join(move_to_uci(m) for m in info.pv)
            out.write(f"info depth {info.depth} score cp {info.score_cp} nodes {info.nodes} pv {pv_str}\n")
            out.flush()
        result = self.search.search(self.board, limits, stop_event=self.stop_event, on_info=on_info)
        out.write(f"bestmove {move_to_uci(result.best_move)}\n"); out.flush()
```

`parse_go_limits(args, side_to_move) -> SearchLimits` turns `go` tokens
(`depth`, `nodes`, `movetime`, `wtime`/`btime`/`winc`/`binc`, `infinite`)
into a `SearchLimits`. Time-control math (a fixed fraction of remaining time
plus increment, clamped to a safety minimum) is intentionally simple through
Milestone 4 and isolated entirely in `uci.py`, so it can be refined later
without touching `search.py`'s `SearchLimits` contract.

Search always runs on a **background thread** specifically so `cmd_stop`/
`cmd_quit` on the main thread (blocked reading stdin) is never delayed by a
long-running search call stack unwinding on its own. The only cross-thread
state is `self.stop_event` (a `threading.Event`, safe to set from another
thread) and `self.search` (mutated only by the search thread while a search
is in flight — a compliant GUI always sends `stop` or waits for `bestmove`
before a new `position`/`go`, which is what makes this safe without an
explicit lock).

```python
# cli.py
def main() -> None:
    UCIEngine().run()
```

## 13. Testing and correctness gates

These are the load-bearing tests; §15 states at which milestone each gate
must be green before the next milestone starts.

- **Perft** (`tests/test_perft.py`) — `perft(board, depth)` matched exactly
  against the published node counts in §14, for every listed position and
  depth. This is the single most important gate: a wrong count means some
  rule (pin, check evasion, castling right bookkeeping, en passant,
  promotion) was generated incorrectly, and `divide(board, depth)`
  (breakdown by root move) bisects a mismatch down to the exact offending
  move by recursing into whichever root move's subtree first diverges.
- **`test_attacks.py`** — for a sample of squares and many random occupancy
  bitboards, `sliding_attacks(...)` must agree with a deliberately naive
  one-square-at-a-time reference tracer written independently for the test.
- **`test_board_invariants.py`** — after every `make_move`/`unmake_move` in
  random legal games, assert `occupied == occupied_co[WHITE] |
  occupied_co[BLACK]`, `mailbox` agrees with the bitboards square-by-square,
  and each side has exactly one king bit.
- **`test_make_unmake_roundtrip.py`** — for every pseudo-legal move in a
  position, `make_move` then `unmake_move` restores every field (all 12
  piece bitboards, both occupancy unions, mailbox, castling rights, ep
  square, halfmove clock, side to move, zobrist hash) exactly.
- **`test_zobrist.py`** — after every move in a random legal game,
  `board.zobrist_hash == zobrist.compute_hash(board)`; `unmake_move`
  restores the exact prior hash, not merely an equivalent one.
- **`test_fen.py`** — `board_to_fen(parse_fen(fen)) == fen` for a battery of
  FENs covering all castling-rights combinations, an en passant square, and
  non-standard halfmove/fullmove counters.
- **`test_transposition.py`** — a search with the TT enabled must return the
  same score (and best move, up to equal-score ties) as the same search
  with a stub always-miss TT, at a fixed small depth (2–4) on a battery of
  tactical and quiet positions (§9.2's invariant). A dedicated mate-distance
  test: store a mate score found at `ply=5`, probe it back as if reached at
  `ply=2`, assert the corrected score reflects "mate in 3 from here." A
  replacement-policy test: fill a tiny TT past capacity and assert shallower
  entries are evicted by deeper ones but not vice versa.
- **`test_search_tactics.py`** — a short list of mate-in-1/2/3 and
  "don't hang a piece" FENs that `Search.search` must solve (correct
  `best_move`) within a small fixed depth/node budget.
- **`test_search_quiescence.py`** — hand-built positions where a naive
  fixed-depth search (quiescence disabled) misjudges a hanging piece one ply
  past the horizon, and the quiescence-enabled search correctly does not.
- **`test_evaluate.py`** — symmetry (`evaluate(board) ==
  evaluate(board.mirrored())` via a test-only horizontal color-flip helper)
  catches a PST mirrored incorrectly for Black without needing search at
  all; targeted term tests (e.g. an extra queen scores roughly +900).
- **`test_uci.py`** — drives `UCIEngine.run()` over an in-memory
  `io.StringIO` pair with `uci`/`isready`/`position startpos moves e2e4
  e7e5`/`go depth 4` and asserts well-formed `uciok`/`readyok`/`info`/
  `bestmove` lines appear in the right order.
- **Optional, recommended cross-check** — a deliberately naive, test-only
  "simulate every pseudo-legal move and check if the king is attacked"
  legal-move generator, kept in `tests/` (never in `src/`), used to
  differential-test `movegen.generate_legal_moves` (`set(fast(pos)) ==
  set(oracle(pos))`) against a large corpus of randomly-reached legal
  positions, not just the fixed perft positions. This is a defense-in-depth
  addition on top of exact perft counts, not a replacement for them.

## 14. Perft reference positions

Standard positions with published node counts (Chess Programming Wiki's
"Perft Results"), the fixture set for §13's perft gate. **These counts must
be re-verified against the canonical CPW page (or a trusted reference
engine's own `perft`/`divide` output) before being committed as hard test
assertions** — they are reproduced here from well-established reference
values to scope the test plan, not as a substitute for that verification
step.

| # | Position | FEN | Depth → expected nodes |
|---|---|---|---|
| 1 | Startpos | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 1:20, 2:400, 3:8902, 4:197281, 5:4865609, 6:119060324 |
| 2 | Kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1` | 1:48, 2:2039, 3:97862, 4:4085603, 5:193690690 |
| 3 | Position 3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | 1:14, 2:191, 3:2812, 4:43238, 5:674624, 6:11030083 |
| 4 | Position 4 | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | 1:6, 2:264, 3:9467, 4:422333, 5:15833292 |
| 5 | Position 5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 1:44, 2:1486, 3:62379, 4:2103487 |
| 6 | Position 6 | `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 1:46, 2:2079, 3:89890, 4:3894594 |

Between them, these positions exercise every legality rule the classical
generator has a dedicated code path for: promotions including
underpromotion (4, 5), en passant including the discovered-check edge case
(3), castling through/into/out of check and rights lost on rook capture
(2, 4), and pins/check evasion under a near-empty board (3) — exactly the
cases a naive move generator gets wrong first. Depths 1–4 are run on every
`pytest` invocation; depth 5–6 rows are marked `@pytest.mark.slow` (full
depth set run on a schedule or on demand) so the default suite stays fast.

## 15. Implementation roadmap

Each milestone's exit criterion is its test suite going green, not a
calendar date — the next milestone does not start until then, and in
particular **no search or evaluation code is written against a rules engine
that has not matched every §14 perft count exactly.**

### Milestone 1 — Board representation + move generation + perft correctness

**Build:** `constants.py`, `bitboard.py`, `attacks.py` (§5.2–5.4),
`zobrist.py`, `move.py`, `fen.py`, `board.py` (§3, §6), `movegen.py` (§5.5–
5.10), `perft.py`.

**Exit criteria:** `test_perft.py` matches every §14 position/depth exactly;
`test_attacks.py`, `test_board_invariants.py`,
`test_make_unmake_roundtrip.py`, `test_zobrist.py`, `test_fen.py` all green;
dedicated regression tests for en passant discovered check, castling
through/into/out of check, castling rights lost on rook capture, and
under-promotion. This is the single most important gate in the whole
project — a search bug later is very often a movegen bug wearing a
search-shaped disguise, so nothing downstream is trusted until this is
solid.

### Milestone 2 — Search + basic evaluation + can play a legal game

**Build:** `evaluate.py`'s `Evaluator` protocol, `material_pst_term`,
`CompositeEvaluator`, `default_evaluator()` (§10); `transposition.py`
(§9.2, wired but not yet exercised for speed); `search.py`'s `Search` class
with plain fixed-depth negamax + alpha-beta (§9.1), TT-backed move
ordering, no quiescence yet (`_negamax` bottoms out calling
`self.evaluator.evaluate(board)` directly at `depth == 0`, not
`_quiescence`), and a bare-bones iterative-deepening loop (§9.3) since it is
needed to pick a move under either a depth or node budget and costs almost
nothing beyond the depth-1 case.

**Exit criteria:** `test_evaluate.py` green in isolation (no search
involved); a differential test asserting `_negamax`'s result matches an
unpruned full-width minimax on small/shallow positions (catches
alpha-beta window bugs, which are otherwise invisible unless compared
against an unpruned search); a self-play smoke test plays a complete game
from the start position to checkmate/stalemate/draw with zero illegal
moves generated or accepted.

### Milestone 3 — UCI protocol + CLI

**Build:** `uci.py` (`UCIEngine`, `parse_go_limits`, §12), `cli.py`
(`main()`), console-script entry point already declared in `pyproject.toml`.

**Exit criteria:** `test_uci.py` green (protocol parsing and well-formed
`uciok`/`readyok`/`info`/`bestmove` sequencing over an in-memory stream); a
manual or scripted smoke test against a real GUI or `cutechess-cli` plays
complete games with zero protocol violations; `stop`/`quit` reliably
interrupt an in-flight search via `stop_event` without corrupting `Search`'s
state for the next `go`.

### Milestone 4 — Evaluation tuning polish, quiescence, TT, iterative deepening polish

**Build:** quiescence search wired into `_negamax`'s `depth == 0` case
(§9.5), `see_ge` static-exchange pruning, full MVV-LVA + killers + history
move ordering (§9.4) if not already complete from Milestone 2, iterative
deepening polish — PV extraction robustness (§9.3), the mate-distance TT
adjustment exercised end-to-end (§9.2), time-management refinement in
`parse_go_limits` — and the first evaluation-tuning pass: material/PST
weight adjustments, still via `default_evaluator()`'s existing
`CompositeEvaluator`/`Weights` machinery (no new terms yet — new terms are
Milestone 5).

**Exit criteria:** `test_transposition.py` (TT-on/TT-off agreement,
mate-distance correction, replacement-policy tests) green;
`test_search_quiescence.py` green (hand-built horizon-effect positions
resolved correctly); `test_search_tactics.py` (mate-in-N suite) solved
within budget; a node-count regression check confirming move ordering
does not *increase* node count at fixed depth versus Milestone 2's
baseline ordering.

### Milestone 5 — Opening book, endgame heuristics, performance

**Build (each item independently gated, lowest priority first):**
mobility/king-safety/pawn-structure evaluation terms added one at a time
via §10.2's extension path, each with its own isolated unit test and its
own `Weights` field; an incremental (make/unmake-updated) evaluation total
as a drop-in replacement for `material_pst_term`'s from-scratch recompute;
magic bitboards behind `attacks.py`'s existing signatures (§5.11), gated on
profiling actually showing sliding-attack generation as the bottleneck;
simple endgame heuristics (e.g. KPK-style rules, or driving a lone king to
the edge with mating material); an opening-book file format and lookup,
consulted only before the main search runs and never changing
`Search.search`'s contract; optional search extensions (null-move pruning,
late move reductions, aspiration windows, check extensions), each gated by
an A/B match at a fixed time control against the immediately prior version
that it must not lose measurable strength against.

**Exit criteria:** every Milestone 1–4 test suite remains green throughout
(this milestone is additive/optimization-only by construction — the test
suite is the proof); each new evaluation term passes its own unit test and
an engine-vs-engine match (self-play at fixed depth, or vs. a fixed
reference opponent) shows a non-negative trend before being kept, since a
term "correct" in isolation can still lose strength through interaction
effects; any performance change (magic bitboards, incremental eval) passes
a full differential-equality re-run of Milestone 1's perft suite and
Milestone 2/4's tactical suite with identical results, proving it changed
speed and not behavior.

## 16. Key trade-offs and where the proposals disagreed

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Sliding attacks | Classical ray-scanning | Magic bitboards (plain or fancy) | Correctness/testability-first priority; magics are a real but secondary speed win, deferred to Milestone 5 behind a stable seam (§5.1, §5.11) |
| Move representation | Packed 16-bit int | 26-bit int with piece/captured/promo embedded; frozen `Move` dataclass | Smallest scheme that still needs no board lookup to decode; a dataclass trades compactness/TT-friendliness for debuggability that a working `to_uci()` + tests already provide |
| Mailbox cache | Maintained alongside bitboards, single-writer in `make_move`/`unmake_move` | No mailbox (attack-table-only lookups) | O(1) "what's on this square" is used constantly (captures, `move_from_uci`, debugging); the single-writer discipline keeps it provably in sync, so the "no mailbox" proposal's speed argument didn't outweigh the testability/simplicity win |
| Legal move generation | Pseudo-legal + explicit pin/evasion masks, make/unmake fallback only for en passant | Always make/unmake every pseudo-legal move (simplest, slowest); two parallel implementations (slow oracle + fast masked generator) kept forever | The masked approach is already correctness-tractable (§5.5/5.6 are small, testable functions) without paying a make/unmake round trip on the common case; a permanent second production implementation was judged unnecessary complexity given §13's perft + optional differential-oracle testing already provide the same safety net |
| Search state ownership | A `Search` class owning TT/killers/history, reset via `.new_game()` | Free functions + a manually-threaded `SearchContext`; a nested `search/` subpackage with a `SearchEngine` facade | A single class is simple enough for a flat-module layout, gives UCI's `ucinewgame` one clear object to reset, and gives tests one object to construct fresh per case |
| Search interruption | Injectable `stop_event` (production) + `extra_stop` callable (tests) | Hardcoded `time.time()` checks only | Grafted from the hybrid proposal's dependency-inversion idea: makes search-interruption unit-testable with zero real wall-clock sleeps, while still supporting UCI's real `threading.Event` |
| Evaluation extensibility | `Evaluator` protocol + `CompositeEvaluator` over named term functions + `Weights` | A single monolithic `evaluate(board)` function extended in place | Directly satisfies the "add mobility/king-safety/pawn-structure later without breaking search" requirement; `search.py` only ever imports the one-method protocol |
| Module layout | Flat `src/chessengine/*.py` | Nested subpackages (`board/`, `moves/`, `eval/`, `search/`, `uci/`) with a strict `board → moves → search → uci → cli` import contract | Matches the project's existing flat package (`src/chessengine/__init__.py`, no subpackages yet); the same one-directional dependency discipline is achievable and enforced (§11) without the extra ceremony of subpackage `__init__.py`s at this project's size |
| TT correctness | Full-key verification per entry, depth-preferred replacement, explicit "TT must not change results" test | Bucketed/multi-way TT; always-replace | Simplicity: a single flat array with full-key check is correct and fast enough for a first version, and is the design all three input proposals converged on independently |
