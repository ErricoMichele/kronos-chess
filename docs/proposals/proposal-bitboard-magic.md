# Proposal: Bitboard + Magic-Number Chess Engine Architecture

- **Status:** Proposed
- **Target package:** `src/chessengine/`
- **Author:** francesco.errico@relatech.com (drafted with Claude Code)
- **Scope:** Board representation, move generation, search, evaluation, and
  the module layout needed to ship a UCI-compatible engine as described in
  `pyproject.toml` / `README.md`.

## 1. Goals and non-goals

**Goals**

- Pure-Python, dependency-free (`dependencies = []` in `pyproject.toml`)
  bitboard engine — no 0x88, no array-of-squares mailbox as the primary
  representation.
- Sliding-piece (bishop/rook/queen) attacks computed via **precomputed magic
  bitboard tables**, not ray-stepping loops, on the hot path.
- Compact, immutable **packed-integer move encoding** (no `Move` objects with
  many attributes on the hot path).
- **Make/unmake** via a small undo-info record pushed to a stack, so search
  never copies the board.
- **Negamax + alpha-beta**, **iterative deepening**, **quiescence search**,
  and a **transposition table keyed by Zobrist hashing**.
- An evaluation function with **material, piece-square tables (tapered),
  mobility, and basic king safety**.
- A test strategy anchored on **perft** node-count validation and **TT
  correctness** (search must not depend on TT for correctness, only speed).

**Non-goals (explicitly out of scope for this proposal)**

- SIMD/PEXT-based attack generation (Python has no portable intrinsic access
  to BMI2; magics are the right tool here).
- NNUE / neural evaluation, endgame tablebases, opening books — noted as
  future extensions, not designed here.
- Multi-threaded / lazy-SMP search (single-threaded negamax first; the
  module boundaries below are drawn so it *could* be added later without a
  rewrite).

## 2. Board representation

### 2.1 Square indexing

Little-Endian Rank-File (LERF) mapping, matching the convention used by most
open-source engines and mapping trivially to UCI square names:

```
square = rank * 8 + file          # rank, file in [0, 7]
bit 0  = a1                       bit 7  = h1
bit 56 = a8                       bit 63 = h8
```

A 64-bit bitboard is represented as a plain Python `int`, always masked back
into `[0, 2**64 - 1]` with a module-level constant:

```python
BB_ALL: int = 0xFFFF_FFFF_FFFF_FFFF

def mask64(x: int) -> int:
    return x & BB_ALL
```

Python ints are arbitrary precision, so every shift/multiply that is meant to
emulate 64-bit wraparound (this matters for the magic multiply, see §3) must
explicitly `& BB_ALL` afterwards. This is the one place 0x88-style bounds
tricks are *not* needed — LERF plus rank/file masks fully replace them for
edge detection (e.g. "does this knight jump wrap around the board" is answered
by a precomputed per-square knight-attack bitboard, not by range checks).

### 2.2 `Board` state

`Board` (in `board.py`) holds the minimum state needed to make/unmake moves
and to search, all as plain attributes (no nested objects) for fast access in
the hot loop:

```python
class Board:
    __slots__ = (
        "pieces",        # dict[int, int] or list[int]: 12 bitboards,
                          #   indexed by (color, piece_type) -> 0..11
        "occupancy",      # list[int]: [white, black, both] combined boards
        "side_to_move",   # int: 0 = white, 1 = black
        "castling_rights",# int: 4-bit mask (WK, WQ, BK, BQ)
        "ep_square",      # int: en-passant target square, or NO_SQUARE (-1/64)
        "halfmove_clock", # int: for 50-move rule
        "fullmove_number",# int
        "zobrist_key",    # int: incrementally maintained 64-bit hash (§7)
        "history",        # list[UndoInfo]: undo stack for unmake_move (§5)
    )
```

`pieces` is 12 bitboards (`WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK`),
each a plain `int`. `occupancy[WHITE]`, `occupancy[BLACK]`, and
`occupancy[BOTH]` are maintained incrementally alongside `pieces` on every
make/unmake so movegen never has to `reduce(or_, pieces)` on the hot path.

A **mailbox side table is deliberately not maintained** in the base design:
"what piece is on square X" is answered by scanning the 12 bitboards with a
tiny helper (`piece_at(board, square)`), which is only needed on the
capture/unmake path, not in the move-generation inner loop. If profiling later
shows this is hot, an 8-bit-per-square shadow array (`list[int]` of length 64)
can be added purely as a cache, updated in the same places `pieces` is
updated — this is a pure optimization, not a representation change, so it is
deferred.

### 2.3 Board constants module (`constants.py`)

- `Piece` int enum: `PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5`.
- `Color` int enum: `WHITE=0, BLACK=1`.
- File/rank masks: `FILE_A .. FILE_H`, `RANK_1 .. RANK_8`, plus
  `NOT_FILE_A`, `NOT_FILE_H`, `NOT_FILE_AB`, `NOT_FILE_GH` (needed to stop
  knight/king/pawn attacks wrapping around the board — the bitboard
  replacement for 0x88's off-board sentinel trick).
- Castling-rights bit flags `WK_CASTLE=1, WQ_CASTLE=2, BK_CASTLE=4, BQ_CASTLE=8`.
- `NO_SQUARE = 64` sentinel for "no en-passant square".

## 3. Magic bitboard generation strategy

### 3.1 Overview

For each of the 64 squares and for rooks and bishops separately we need a
function `attacks(square, occupancy) -> bitboard` that is O(1) and
branch-free. The classic magic bitboard trick:

1. Precompute, per square, a **relevant-occupancy mask** — the squares along
   the piece's rays *excluding the outer edge in that ray's direction*
   (an edge square always terminates a slide regardless of what's on it, so
   whether it's occupied doesn't change the attack set — dropping it from the
   mask shrinks the table without losing correctness).
2. Enumerate every subset of that mask (there are `2**popcount(mask)`
   subsets) using the **Carry-Rippler trick**:

   ```python
   def subsets(mask: int):
       subset = 0
       while True:
           yield subset
           subset = (subset - mask) & mask
           if subset == 0:
               break
   ```

3. For each subset (a hypothetical "occupied squares" pattern), compute the
   *true* attack bitboard with a slow, obviously-correct ray-tracer that
   walks each of the piece's directions one square at a time and stops at the
   first occupied square (inclusive, since sliders capture into that square):

   ```python
   def sliding_attacks_slow(square: int, occupied: int, deltas: list[tuple[int, int]]) -> int:
       attacks = 0
       r0, f0 = divmod(square, 8)
       for dr, df in deltas:
           r, f = r0 + dr, f0 + df
           while 0 <= r < 8 and 0 <= f < 8:
               sq = r * 8 + f
               attacks |= 1 << sq
               if occupied & (1 << sq):
                   break
               r += dr
               f += df
       return attacks
   ```

   This slow function is used **only at table-build time** (and in tests, as
   the oracle magics are checked against — see §10). It never runs during
   search.
4. Find a **magic number** `M` (a 64-bit int) such that the map

   ```
   index(occupied) = ((occupied & mask) * M) >> (64 - bits)
   ```

   is **collision-free** over all `2**bits` real subsets of `mask` (`bits =
   popcount(mask)`), i.e. any two subsets that map to the same index also
   produce the same true attack bitboard (constructive collisions, where two
   different occupancies happen to yield the same attack set, are fine and
   expected — only *destructive* collisions, where different attacks map to
   the same slot, are rejected).
5. Store `attack_table[square][index]` for every subset.

### 3.2 Fixed-shift ("plain") magics, not per-square fancy magics

We deliberately use **plain magic bitboards** with a single fixed index width
per piece type — 12 bits (4096 entries) for rooks, 9 bits (512 entries) for
bishops — rather than "fancy" magics with a per-square minimal shift and a
single packed flat array with per-square offsets.

Rationale: fancy magics save roughly 1–2 MB of table memory versus plain
magics; in a Python process that overhead is noise (a list of 4096 Python
ints is ~36 KB with `sys.getsizeof` bookkeeping, and there are only 64 of
them per piece type). The per-square-offset bookkeeping that fancy magics
need earns its complexity in C/C++ where cache-line locality dominates; here
it only adds code paths for tests to cover with no measurable runtime win, so
we take the simpler, easier-to-verify design. Fancy magics are noted as a
possible future micro-optimization if a `numpy`/C-extension rewrite ever
happens (see §11), not part of this proposal.

### 3.3 Magic search algorithm

A magic for a given `(square, mask)` is found by random search — this is the
standard, well-documented technique (see Chess Programming Wiki "Magic
Bitboards"), reproduced here for completeness since it runs inside this
repo's tooling rather than being copy-pasted from elsewhere:

```python
def find_magic(square: int, mask: int, bits: int, slow_attacks_for_all_subsets: dict[int, int],
                rng: random.Random) -> int:
    subset_list = list(subsets(mask))
    while True:
        magic = rng.getrandbits(64) & rng.getrandbits(64) & rng.getrandbits(64)  # sparse candidate
        if popcount((mask * magic) & BB_ALL >> 56) < 6:   # cheap high-bit heuristic, skip bad candidates early
            continue
        used = [None] * (1 << bits)
        if try_fill(subset_list, slow_attacks_for_all_subsets, magic, bits, used):
            return magic
```

`try_fill` computes `index = ((subset * magic) & BB_ALL) >> (64 - bits)` for
every subset, and fails fast the moment two different subsets with different
attack bitboards land on the same index. ANDing three random 64-bit numbers
biases the candidate towards having few set bits, which empirically finds
valid magics faster (fewer bits means fewer distinct high bits participate in
the multiply, which — counter-intuitively — makes it *easier* to hit a
low-collision multiplier for this specific use case; this is the standard
folklore heuristic used by every public magic-bitboard implementation).

This search is deterministic given a fixed PRNG seed (`random.Random(seed)`),
so magics are reproducible.

### 3.4 Where magics live: generated once, shipped as data

Two options were considered:

1. **Generate at import time.** Simple, no generated file to keep in sync,
   but the random search for all 64 rook squares can take on the order of a
   few hundred milliseconds to a couple of seconds depending on the machine
   and seed — paid on *every* process start, including every test run and
   every UCI engine launch from a GUI.
2. **Generate offline, ship as a checked-in data module.** Pay the search
   cost once, check the result in, and have `attacks.py` do only cheap,
   linear table-building work (`O(2**bits)` per square, no search) at import
   time.

**Decision: option 2.** A standalone script, `scripts/generate_magics.py`
(not part of the installed package), runs the search from §3.3 for all 64
squares × {rook, bishop} with a fixed seed and writes
`src/chessengine/data/magic_numbers.py`:

```python
# Auto-generated by scripts/generate_magics.py — do not hand-edit.
ROOK_MAGICS: tuple[int, ...] = (0x0080_0020_4000_1080, ...)   # 64 entries
BISHOP_MAGICS: tuple[int, ...] = (0x0021_0410_2044_0002, ...)  # 64 entries
```

At import time, `attacks.py`:

1. Recomputes the relevant-occupancy `mask` for each square (cheap, pure
   function of square + piece type — not worth storing as data).
2. Rebuilds each square's `2**bits`-entry attack table from `mask`,
   `ROOK_MAGICS[square]` / `BISHOP_MAGICS[square]`, and the slow ray-tracer,
   using the Carry-Rippler enumeration from §3.1 (cheap: at most 4096
   iterations × 64 squares × 2 piece types ≈ 500K simple operations, done
   once per process).
3. In debug/test builds only (guarded by an explicit `verify=True` argument
   used from the test suite, never from the hot import path), asserts there
   were no destructive collisions, so a corrupted or hand-edited data file
   fails loudly instead of silently mis-generating attacks.

If `data/magic_numbers.py` is ever missing or fails verification, `attacks.py`
falls back to calling the same search routine live (slow path), so the engine
is never *unable* to start — it just starts slower. This keeps "shipped data"
and "generated at import time" as one graceful-degradation design rather than
two competing ones.

### 3.5 Runtime attack lookup

```python
def rook_attacks(square: int, occupied: int) -> int:
    idx = (((occupied & ROOK_MASKS[square]) * ROOK_MAGICS[square]) & BB_ALL) >> (64 - ROOK_BITS)
    return ROOK_ATTACK_TABLE[square][idx]

def bishop_attacks(square: int, occupied: int) -> int:
    idx = (((occupied & BISHOP_MASKS[square]) * BISHOP_MAGICS[square]) & BB_ALL) >> (64 - BISHOP_BITS)
    return BISHOP_ATTACK_TABLE[square][idx]

def queen_attacks(square: int, occupied: int) -> int:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)
```

Knight, king, and (per-color) pawn-attack tables are simpler 64-entry
precomputed arrays with no occupancy dependence at all (`KNIGHT_ATTACKS[sq]`,
`KING_ATTACKS[sq]`, `PAWN_ATTACKS[color][sq]`) — generated directly at import
time in `attacks.py` from the same delta-list approach, no magic needed since
they aren't blockable sliders.

## 4. Move encoding

Moves are packed into a single Python `int` (fits comfortably in 24 bits, so
well within a machine word — no bignum overhead). No `Move` class is used on
the hot path; a thin set of free functions encode/decode fields. Field
layout (LSB first):

| Bits    | Width | Field                       | Values |
|---------|-------|------------------------------|--------|
| 0–5     | 6     | `from_square`                | 0–63 |
| 6–11    | 6     | `to_square`                  | 0–63 |
| 12–14   | 3     | `piece_type`                 | 0–5 (`Piece` enum) |
| 15–17   | 3     | `captured_type`               | 0–5, or `7` = none |
| 18–20   | 3     | `promotion_type`              | 0 = none, else N/B/R/Q |
| 21      | 1     | `flag_capture`                | 0/1 |
| 22      | 1     | `flag_double_push`            | 0/1 |
| 23      | 1     | `flag_en_passant`             | 0/1 |
| 24      | 1     | `flag_castle_kingside`        | 0/1 |
| 25      | 1     | `flag_castle_queenside`       | 0/1 |

```python
# moves.py
FROM_SHIFT, FROM_MASK = 0, 0x3F
TO_SHIFT,   TO_MASK   = 6, 0x3F
PIECE_SHIFT, PIECE_MASK = 12, 0x7
CAPTURED_SHIFT, CAPTURED_MASK = 15, 0x7
PROMO_SHIFT, PROMO_MASK = 18, 0x7
CAP_FLAG, DOUBLE_FLAG, EP_FLAG, OO_FLAG, OOO_FLAG = 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25

NO_PIECE = 7  # sentinel for "no captured piece"

def encode_move(frm: int, to: int, piece: int, *, captured: int = NO_PIECE,
                 promotion: int = 0, is_capture: bool = False, is_double_push: bool = False,
                 is_ep: bool = False, is_castle_k: bool = False, is_castle_q: bool = False) -> int:
    m = frm | (to << TO_SHIFT) | (piece << PIECE_SHIFT) | (captured << CAPTURED_SHIFT) | (promotion << PROMO_SHIFT)
    if is_capture:    m |= CAP_FLAG
    if is_double_push: m |= DOUBLE_FLAG
    if is_ep:         m |= EP_FLAG
    if is_castle_k:   m |= OO_FLAG
    if is_castle_q:   m |= OOO_FLAG
    return m

def move_from(m: int) -> int:      return m & FROM_MASK
def move_to(m: int) -> int:        return (m >> TO_SHIFT) & TO_MASK
def move_piece(m: int) -> int:     return (m >> PIECE_SHIFT) & PIECE_MASK
def move_captured(m: int) -> int:  return (m >> CAPTURED_SHIFT) & CAPTURED_MASK
def move_promotion(m: int) -> int: return (m >> PROMO_SHIFT) & PROMO_MASK
def is_capture(m: int) -> bool:    return bool(m & CAP_FLAG)
def is_en_passant(m: int) -> bool: return bool(m & EP_FLAG)
```

`side_to_move` and full en-passant/castling-rights context are **not**
encoded in the move — they live on `Board` and are recorded in the
`UndoInfo` (§5) at make-time, since they're properties of the position the
move was made *from*, not of the move itself. This keeps moves small,
comparable by plain integer equality (useful for TT best-move matching and
killer-move tables), and trivially hashable/storable in flat Python lists
(`MoveList = list[int]`).

A `to_uci(move: int) -> str` / `from_uci(board, s: str) -> int` pair in
`moves.py` handles the only place textual notation is needed (the UCI
front-end).

## 5. Make / unmake via undo-info struct

`Board.make_move(move: int)` mutates bitboards, occupancy, castling rights,
en-passant square, halfmove clock, side to move, and the Zobrist key in
place, and pushes one `UndoInfo` record. `Board.unmake_move()` pops the
record and reverses every mutation exactly, so no board copy is ever taken
during search — this is the standard technique that makes bitboard engines
fast, and it's why the undo record must capture *everything not otherwise
recoverable from the move itself*:

```python
# board.py
class UndoInfo:
    __slots__ = (
        "captured_type",      # Piece or NO_PIECE — needed since move_captured()
                               #   already encodes it, but keeping it here too
                               #   keeps unmake_move a pure function of UndoInfo
                               #   without re-decoding the move
        "castling_rights",    # castling rights *before* this move
        "ep_square",          # en-passant square *before* this move
        "halfmove_clock",     # halfmove clock *before* this move
        "zobrist_key",        # full key *before* this move (cheapest correct
                               #   way to undo an incrementally-updated hash;
                               #   see §7 for why we don't try to invert XORs)
    )
```

`make_move` sequence (pseudocode, elided error handling):

1. Snapshot `UndoInfo` from current board state, push to `board.history`.
2. Clear the moving piece's bit at `from`, set it at `to` in the relevant
   piece bitboard and in `occupancy[side_to_move]`.
3. If capturing (including en-passant, which removes a pawn on a *different*
   square than `to`): clear the captured piece's bit from its bitboard and
   from `occupancy[other_side]`.
4. If castling: also move the rook (second bitboard update).
5. If promoting: remove the pawn bit, add the promoted-piece bit instead of
   the pawn bit at `to`.
6. Update castling rights (moving a king or rook, or capturing on a rook's
   home square, clears the relevant bits).
7. Update `ep_square` (set only on a double pawn push, else `NO_SQUARE`).
8. Update `halfmove_clock` (reset on pawn move or capture, else +1).
9. Flip `side_to_move`, increment `fullmove_number` on black's move.
10. Incrementally update `zobrist_key` by XORing in/out exactly the pieces,
    rights, ep-file, and side-to-move keys that changed (§7) — cheaper than
    recomputing from scratch, and cross-checked against a from-scratch
    recompute in tests (§10.3).

`unmake_move` pops the `UndoInfo`, reverses steps 2–9 using the move's own
encoded fields (`move_from`, `move_to`, `move_captured`, `move_promotion`,
flags) plus the popped `UndoInfo` for the fields that aren't recoverable from
the move alone (rights/ep/halfmove/hash *before*), and restores
`zobrist_key` directly from `UndoInfo.zobrist_key` rather than re-deriving it
by inverse XOR — XOR is its own inverse so re-deriving would work too, but
storing the pre-move key is simpler, cannot drift, and costs one extra
64-bit int on the undo stack, which is negligible.

Because `occupancy[BOTH]` is derivable as `occupancy[WHITE] |
occupancy[BLACK]`, it is recomputed from the two color boards after each
mutation rather than patched separately, trading one cheap OR for one fewer
thing that can get out of sync.

## 6. Move generation

`movegen.py` generates **pseudo-legal moves first, then filters to legal**,
which is the right trade-off for a magic-bitboard engine (attack lookups are
O(1), so "generate all pseudo-legal moves, make each one, check if our king
is attacked, unmake if illegal" is cheap and much simpler than maintaining
pin/check bitboards incrementally):

1. **Pseudo-legal generation** (`generate_pseudo_legal(board) -> MoveList`):
   - Pawns: single/double pushes (blocked by `occupancy[BOTH]`), captures
     (`PAWN_ATTACKS[side][sq] & occupancy[other]`), en-passant (compare
     against `board.ep_square`), promotions (push/capture landing on rank
     1/8 emits one move per promotion piece).
   - Knights/King: `KNIGHT_ATTACKS[sq]` / `KING_ATTACKS[sq]` masked off
     same-color occupancy.
   - Bishops/Rooks/Queens: `bishop_attacks`/`rook_attacks`/`queen_attacks`
     from §3.5, masked off same-color occupancy.
   - Castling: rights bit set, squares between king/rook empty, and (checked
     here, cheaply, via the attack tables) king's start/pass-through/landing
     squares not attacked by the opponent.
2. **Legality filter**: for each pseudo-legal move, `make_move`; compute
   `is_square_attacked(board, king_square, by=opponent)` using the same
   attack tables in reverse (a square is attacked by an opponent slider if
   `rook_attacks(king_sq, occ) & opponent_rooks_or_queens`, etc. — this
   "attacks-from-king's-square" trick reuses the exact same magic lookups as
   move generation, so there is no second code path to keep correct); then
   `unmake_move`. Moves that leave the king in check are discarded.
3. `generate_legal_moves(board) -> MoveList` is the public entry point used
   by search's root and by perft; `generate_captures(board) -> MoveList`
   (a filtered subset, or a separate cheaper generator that skips quiet pawn
   pushes/knight-to-empty-square moves entirely) feeds quiescence search
   (§8.3).

This "make it and see if it's legal" approach is intentionally simpler than
a full pin/checkers-bitboard precomputation, at the cost of doing a
make/unmake per pseudo-legal move during generation. Given Python's per-call
overhead already dominates, this simplicity is worth it initially; a
"only generate king moves + evasions when in check, and mask pinned pieces'
moves along their pin ray" fast path is noted as a future optimization once
correctness is locked down by the perft suite (§10.1), not part of this
proposal's first cut.

## 7. Zobrist hashing scheme

`zobrist.py` builds, once at import time, fixed random 64-bit tables from a
seeded PRNG (so hashes are reproducible across runs/tests):

```python
_rng = random.Random(0xC0FFEE)  # fixed seed: reproducible hashes, not security-sensitive
PIECE_KEYS = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(12)]  # [piece_index][square]
SIDE_KEY = _rng.getrandbits(64)
CASTLING_KEYS = [_rng.getrandbits(64) for _ in range(16)]  # one per castling-rights bitmask value
EP_FILE_KEYS = [_rng.getrandbits(64) for _ in range(8)]    # keyed by file only, not full square
```

- **Piece keys**: XOR `PIECE_KEYS[piece_index][square]` in/out whenever a
  piece is placed/removed on a square (covers normal moves, captures,
  castling rook moves, and promotions, each as a "remove old piece" +
  "add new piece" pair).
- **Side-to-move key**: XOR `SIDE_KEY` on every move (it's a pure toggle).
- **Castling-rights key**: XOR out the key for the *old* rights value and
  XOR in the key for the *new* rights value whenever rights change (using a
  single 16-entry table indexed by the 4-bit rights mask, rather than 4
  independent per-flag keys, keeps "rights changed" a single lookup+XOR
  instead of four).
- **En-passant key**: XOR in `EP_FILE_KEYS[file]` when a double push creates
  an en-passant target, XOR it back out on the next move regardless of
  whether the capture happened (standard practice — the key represents
  "en-passant was *available*", which is a property of the position that
  affects legal moves, so two positions differing only in a long-expired ep
  square must still hash the same once that square is cleared). Only the
  file is keyed (not rank) since the rank is implied by side to move.

`compute_zobrist_from_scratch(board) -> int` (an independent, non-incremental
implementation that just iterates all 12 bitboards + rights + ep + side) is
kept in `zobrist.py` purely as a **test oracle** — §10.3 asserts it matches
the incrementally-maintained `board.zobrist_key` after every move in random
games. It is never called from the hot path.

The TT (§8.4) uses the low bits of `zobrist_key` as the table index and
stores the full 64-bit key in each entry to detect the (rare, expected)
collisions between different positions hashing to the same table slot.

## 8. Search architecture

### 8.1 Negamax with alpha-beta

`search.py` implements the standard negamax formulation (evaluation is
always from the side-to-move's perspective, so no separate min/max branches
are needed):

```python
def negamax(board: Board, depth: int, alpha: int, beta: int, ply: int, tt: TranspositionTable) -> int:
    alpha_orig = alpha
    tt_entry = tt.probe(board.zobrist_key)
    if tt_entry is not None and tt_entry.depth >= depth:
        if tt_entry.flag == EXACT:
            return tt_entry.score
        elif tt_entry.flag == LOWERBOUND:
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == UPPERBOUND:
            beta = min(beta, tt_entry.score)
        if alpha >= beta:
            return tt_entry.score

    if depth == 0:
        return quiescence(board, alpha, beta, ply)

    moves = order_moves(board, generate_legal_moves(board), tt_entry, ply)
    if not moves:
        return -MATE_SCORE + ply if in_check(board) else DRAW_SCORE

    best_score = -INF
    best_move = None
    for move in moves:
        undo = board.make_move(move)
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, tt)
        board.unmake_move(undo)
        if score > best_score:
            best_score, best_move = score, move
        alpha = max(alpha, score)
        if alpha >= beta:
            record_killer_and_history(move, depth, ply)  # §8.5
            break  # beta cutoff

    flag = EXACT if alpha_orig < best_score < beta else (LOWERBOUND if best_score >= beta else UPPERBOUND)
    tt.store(board.zobrist_key, depth, best_score, flag, best_move)
    return best_score
```

Checkmate/stalemate are detected exactly at the point moves are exhausted
(no legal moves): checkmate score is `-MATE_SCORE + ply` (so shorter mates
score higher in magnitude, which naturally makes the search prefer faster
mates and avoid slower ones without special-casing), stalemate is
`DRAW_SCORE` (0, or a small contempt term later).

### 8.2 Iterative deepening

The root driver in `search.py` calls `negamax` for `depth = 1, 2, 3, ...`
until a time/node budget is hit:

```python
def search_best_move(board: Board, tt: TranspositionTable, time_budget_s: float) -> int:
    deadline = time.monotonic() + time_budget_s
    best_move = None
    for depth in range(1, MAX_DEPTH + 1):
        score, best_move = root_search(board, depth, tt)
        report_info(depth, score, best_move)  # UCI `info depth ... score ... pv ...`
        if time.monotonic() >= deadline:
            break
    return best_move
```

This is not just a time-management convenience: each completed shallow pass
populates the TT with best moves that dramatically improve move ordering
(and thus alpha-beta cutoff rate) on the next, deeper pass — the standard
reason iterative deepening is *faster* than searching the target depth
directly, not merely safer under a clock. `root_search` is a thin wrapper
around the same `negamax` loop that additionally tracks the best move at the
root (negamax's return value alone doesn't identify which move produced it).

### 8.3 Quiescence search

`quiescence(board, alpha, beta, ply)` extends the leaf nodes of the main
search until the position is "quiet" (no immediately-winning captures
pending), which avoids the classic **horizon effect** (stopping mid-capture-
sequence and misjudging a hanging piece):

```python
def quiescence(board: Board, alpha: int, beta: int, ply: int) -> int:
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    for move in order_moves(board, generate_captures(board), None, ply):
        if not see_ge(board, move, 0):     # static-exchange pruning: skip clearly-losing captures
            continue
        undo = board.make_move(move)
        score = -quiescence(board, -beta, -alpha, ply + 1)
        board.unmake_move(undo)
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha
}
```

`stand_pat` (the static evaluation, as if the side to move could "stand pat"
and make no capture) both bounds the search and provides the fallback score
when no capture improves on it. `see_ge` (Static Exchange Evaluation) is a
small helper in `search.py` that estimates a capture sequence on one square
using the sorted attacker-value list from the magic attack tables, and is
used purely to prune obviously bad captures from quiescence — it is *not*
used to change search results, only to cut nodes, so a conservative/buggy
SEE only costs speed, never correctness, which is why it's acceptable to add
after the base quiescence search is validated.

### 8.4 Transposition table

`transposition.py` defines a fixed-size, power-of-two-sized array of entries
(no dynamic growth, no chaining — collisions are resolved by replacement, not
by storing multiple entries per slot, which keeps lookup O(1) with a single
array index):

```python
class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "best_move", "age")

class TranspositionTable:
    def __init__(self, size_mb: int = 64):
        num_entries = (size_mb * 1024 * 1024) // ENTRY_SIZE_BYTES
        self.size = 1 << (num_entries.bit_length() - 1)   # round down to a power of two
        self.mask = self.size - 1
        self.table: list[Optional[TTEntry]] = [None] * self.size
        self.generation = 0

    def probe(self, key: int) -> Optional[TTEntry]:
        entry = self.table[key & self.mask]
        return entry if entry is not None and entry.key == key else None

    def store(self, key: int, depth: int, score: int, flag: int, best_move: Optional[int]) -> None:
        idx = key & self.mask
        existing = self.table[idx]
        if existing is None or existing.age != self.generation or depth >= existing.depth:
            self.table[idx] = TTEntry(key, depth, score, flag, best_move, self.generation)
```

Replacement policy: **depth-preferred within the same search generation,
always-replace across generations** — a new `generation` starts at each
`search_best_move` call (each new root search / iterative-deepening run), so
stale entries from a previous, unrelated search get overwritten immediately,
while within one search a shallow entry doesn't evict a deeper, more
expensive-to-recompute one. `key` (the full 64-bit Zobrist hash, not just the
masked index) is always stored and checked on probe specifically so an
index collision (two different positions, same low bits) is detected instead
of silently returning the wrong entry's score — this is the "always verify
the full key" rule that makes a hash-collision *never* a correctness bug,
only, in the astronomically rare case of a full 64-bit collision, a
theoretical one shared by every engine that uses 64-bit Zobrist keys.

Mate scores stored in the TT are **distance-to-root corrected** on store and
uncorrected on retrieval (store `score - ply` if it's a mate score so it's
ply-independent in the table, add back `+ply` at the probing depth) — a
standard, easy-to-get-wrong detail called out explicitly here because §10.4
tests it directly.

### 8.5 Move ordering

`order_moves(board, moves, tt_entry, ply)` sorts (or partitions, e.g. via a
single pass that swaps a scored move to the front) the move list by:

1. TT move (`tt_entry.best_move`, if present) first — this is the single
   biggest driver of alpha-beta cutoff rate.
2. Captures sorted by MVV-LVA (Most Valuable Victim, Least Valuable
   Attacker: `captured_value * 16 - attacker_value`, computed straight from
   the packed move's `move_captured`/`move_piece` fields, §4 — no board
   lookup needed).
3. Killer moves: two per-ply slots (`killers[ply][0/1]`) recording quiet
   moves that caused a beta cutoff at that ply in a sibling branch.
4. History heuristic: a `[color][piece][to_square]` table incremented on
   every beta-cutoff quiet move, used to order the remaining quiet moves.

## 9. Evaluation function

`evaluate.py` exposes `evaluate(board) -> int`, always returned from the
**side-to-move's perspective** (negamax convention: positive means the side
to move is better), combining:

1. **Material**: standard centipawn values (`P=100, N=320, B=330, R=500,
   Q=900`), summed from bitboard popcounts (`popcount(board.pieces[WP]) *
   100`, etc.) — O(1) popcount per piece type via `int.bit_count()` (Python
   3.10+, matching the `requires-python = ">=3.10"` constraint already in
   `pyproject.toml`, so no manual popcount loop is needed).
2. **Piece-square tables (tapered)**: one 64-entry table per piece type for
   the middlegame and one for the endgame (`PST_MG[piece][square]`,
   `PST_EG[piece][square]`, black's tables mirrored from white's via
   `square ^ 56`). A **game-phase** value is computed from remaining
   non-pawn material (standard formula: sum of phase weights for
   N/B/R/Q still on the board, clamped to `[0, 24]`), and the final PST score
   linearly interpolates between the middlegame and endgame tables by that
   phase — this avoids the well-known problem of a single PST making bad
   king-safety/centralization trade-offs across the middlegame-to-endgame
   transition (e.g., king wants to stay back in the middlegame, forward in
   the endgame).
3. **Mobility**: a small bonus per legal-ish move available to each piece,
   computed cheaply straight from the attack tables already built for move
   generation (`popcount(bishop_attacks(sq, occ) & ~own_occupancy)` etc.,
   weighted per piece type) — deliberately *not* full legal-move mobility
   (which would require the make/unmake legality filter per piece); pseudo-
   legal attack-count mobility is the standard cheap approximation.
4. **King safety**: a small, explicitly-scoped first cut —
   - Pawn shield: bonus for own pawns on the three files in front of the
     king once castled.
   - Open/half-open file near the king: penalty if the king's file (or an
     adjacent file) has no own pawn, larger if the opponent also has a rook
     or queen on that file.
   - King-zone attacker count: a small penalty scaled by the number and
     value of enemy pieces attacking squares in the king's immediate
     3x3 zone (`KING_ATTACKS[king_sq]` reused directly), weighted so it
     matters far less than material/PST at this stage — tuning this term
     properly is an iterative, engine-specific process explicitly deferred
     past the first working version.
5. **Bishop pair**: a small flat bonus when a side holds both bishops (cheap
   `popcount(board.pieces[bishop]) >= 2` check), included alongside material
   since it is one of the best-understood, lowest-risk terms to add early.

All terms are simple weighted sums returning a single centipawn integer;
`evaluate.py` has no dependency on `search.py` (search depends on evaluate,
never the reverse), so evaluation can be unit-tested and tuned in isolation
(§10.5).

## 10. Module boundaries

```
src/chessengine/
├── __init__.py
├── constants.py      # Piece/Color enums, file/rank masks, castling flags — no deps
├── bitboard.py        # popcount/lsb/bit-iteration helpers, mask64 — no deps
├── attacks.py         # non-sliding attack tables + magic-based sliding attacks (§3)
│                       #   deps: constants, bitboard, data.magic_numbers
├── data/
│   └── magic_numbers.py  # generated data (§3.4) — no deps, no logic
├── zobrist.py         # Zobrist key tables + incremental-update helpers (§7)
│                       #   deps: constants
├── moves.py           # packed move encode/decode, UCI (de)serialization (§4)
│                       #   deps: constants
├── board.py           # Board + UndoInfo, make/unmake (§5)
│                       #   deps: constants, bitboard, attacks, zobrist, moves
├── movegen.py         # pseudo-legal + legal move generation (§6)
│                       #   deps: constants, bitboard, attacks, board, moves
├── evaluate.py         # material/PST/mobility/king-safety (§9)
│                       #   deps: constants, bitboard, attacks, board
├── transposition.py    # TTEntry, TranspositionTable (§8.4)
│                       #   deps: constants (score/flag enums only)
├── search.py           # negamax, quiescence, iterative deepening, ordering (§8)
│                       #   deps: board, movegen, evaluate, transposition, moves
├── perft.py            # perft(board, depth) / divide(board, depth) driver
│                       #   deps: board, movegen
├── uci.py              # UCI protocol loop (`position`, `go`, `isready`, ...)
│                       #   deps: board, movegen, search, moves
└── cli.py              # entry point wired to `chessengine` console script
                        #   deps: uci

scripts/
└── generate_magics.py  # offline magic-number search (§3.3–3.4), writes data/magic_numbers.py
                        #   deps: constants, bitboard (not imported by the package itself)
```

This is a strict DAG (no cycles): `constants`/`bitboard` are leaves;
`attacks` and `zobrist` depend only on those plus generated data;
`board`/`moves` sit in the middle; `movegen`/`evaluate` depend on `board`;
`search` is the only module that depends on `movegen` + `evaluate` +
`transposition` together; `uci`/`cli` are the outermost, user-facing layer.
This means `evaluate.py` and `movegen.py` can be independently unit-tested
against a bare `Board` with no `search.py` import at all, and `search.py`'s
correctness tests (§10.4) can swap in a trivial evaluation stub without
touching move generation.

## 11. Testing strategy

Tests live under `tests/` (already declared as `testpaths = ["tests"]` in
`pyproject.toml`) using `pytest` (already the declared `dev` dependency).

### 11.1 Perft validation (move generation correctness)

The core correctness net. `perft(board, depth)` counts leaf nodes of the
full legal-move tree; `divide(board, depth)` breaks the count down by root
move for debugging a mismatch down to the exact offending move.

Standard positions with known, published node counts (Chess Programming
Wiki's perft results) are the fixture set:

| Position | FEN (abbreviated) | Depth | Expected nodes |
|---|---|---|---|
| Startpos | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 1–6 | 20, 400, 8902, 197281, 4865609, 119060324 |
| Kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -` | 1–5 | 48, 2039, 97862, 4085603, 193690690 |
| Position 3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -` | 1–6 | 14, 191, 2812, 43238, 674624, 11030083 |
| Position 4 | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -` | 1–5 | 6, 264, 9467, 422333, 15833292 |
| Position 5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -` | 1–4 | 44, 1486, 62379, 2103487 |

Depths are chosen so the full suite runs in a reasonable CI time budget
(lower depths always included so a regression is caught fast; the deepest
row of each position is marked `@pytest.mark.slow` and run less frequently).
These specific positions are chosen because, between them, they exercise
every tricky legality rule at least once: promotions (all 4 pieces),
en-passant capture (including the discovered-check-via-en-passant edge
case), castling through/out of check, and pinned-piece restrictions —
exactly the cases a naive move generator gets wrong first.

A perft failure is diagnosed by recursively `divide`-ing both the engine and
a reference (starting from the documented per-move sub-counts in the same
CPW tables) until the first depth at which a single root move's count
diverges, which localizes the bug to one move instead of one whole-tree
count.

### 11.2 Magic table correctness

For every square and a large random sample of occupancy bitmasks (not just
the subsets used to build the table, to also catch mask errors — bits
outside the relevant mask must not affect the result, which is exactly what
a subset-only test would fail to catch):

```python
@pytest.mark.parametrize("square", range(64))
def test_rook_attacks_match_slow_ray_tracer(square):
    for _ in range(1000):
        occ = random.getrandbits(64)
        assert rook_attacks(square, occ) == sliding_attacks_slow(square, occ, ROOK_DELTAS)
```

Run for both rook and bishop, plus a dedicated test that every entry in
`data/magic_numbers.py` passes the no-destructive-collision check from §3.4
independent of the random-sampling test above (a magic could pass many
random samples yet still be technically wrong for an untested subset if it
were hand-edited incorrectly — the collision check is exhaustive over the
subset space, the random test is a defense-in-depth sanity check over the
*non*-subset space).

### 11.3 Zobrist / incremental-hash correctness

```python
def test_incremental_hash_matches_recompute_over_random_game():
    board = Board.starting_position()
    for _ in range(500):
        moves = generate_legal_moves(board)
        if not moves:
            break
        move = random.choice(moves)
        board.make_move(move)
        assert board.zobrist_key == compute_zobrist_from_scratch(board)
        board.unmake_move(...)  # also verify unmake restores the pre-move key exactly
```

Also: a **collision-rate sanity test** (not a correctness requirement, but a
smoke test that the seeded key tables aren't accidentally degenerate, e.g.
all zero) generating a large number of random legal positions and asserting
no two distinct positions in the sample collide.

### 11.4 Make/unmake round-trip

For every move in `generate_pseudo_legal(board)`, `make_move` then
`unmake_move` must restore the board to a state equal in **every** field
(all 12 piece bitboards, both occupancy boards, castling rights, ep square,
halfmove clock, side to move, zobrist key) to a snapshot taken before the
move — checked via `dataclasses`-style field-by-field comparison (or a
`__eq__`/`copy.deepcopy`-based snapshot helper used only in tests, never in
the engine itself, to avoid a persistent perf cost on `Board`).

### 11.5 TT correctness (must not change the *result*, only the speed)

The central invariant: **a search with the TT enabled must return the same
best move and score as the same search with the TT disabled**, at a fixed
depth, on a fixed position, modulo the well-known "same score, different
but equally-optimal move" ambiguity (handled by comparing scores exactly and
best-move only up to equal-score ties). This is tested by:

1. Running `negamax` at a small fixed depth (2–4, so brute-force comparison
   is fast) on a battery of tactical and quiet test positions, once with a
   `TranspositionTable` and once with a stub that always misses
   (`probe` always returns `None`, `store` a no-op), asserting equal scores.
2. A dedicated **mate-score distance correction** test (§8.4): store a mate
   score found at `ply=5`, probe it back as if reached at `ply=2`, and
   assert the corrected score reflects "mate in 3 *from here*", not the raw
   stored value — this is the single most common TT bug in hobby engines
   (returning a stale mate distance from a shallower or deeper part of the
   tree) and is worth a targeted test rather than relying on perft-style
   incidental coverage.
3. A **replacement-policy test**: fill a tiny (e.g. 16-entry) TT past
   capacity within one generation and assert shallower entries get evicted
   by deeper ones but not vice versa, then start a new "generation" and
   assert an old, deeper entry *is* now evictable by a new, shallower one
   (validates §8.4's two-tier replacement rule directly, rather than only
   observing its effect on search results).
4. A **key-collision-safety test**: manually craft two different `Board`
   states, force their Zobrist keys to collide by monkeypatching
   `zobrist_key` in a test-only helper, store one in the TT, and assert
   probing with the other's (colliding) index but different full key
   returns `None` rather than the wrong entry — this exercises the
   full-key-verification branch in `TranspositionTable.probe` directly,
   which the perft/search tests would otherwise only hit by astronomical
   chance.

### 11.6 Evaluation unit tests

- **Symmetry**: `evaluate(board)` on a position and `evaluate(board.mirrored())`
  (a horizontal color-flip helper used only in tests) must be equal — a
  broken PST or king-safety term that isn't properly mirrored for Black is a
  very common bug this catches immediately without needing search at all.
- Targeted term tests: e.g. a position with an isolated extra queen scores
  higher by roughly the queen's material value ± a small PST/mobility
  delta; a position with doubled/blocked pawns near the king scores lower on
  the king-safety term specifically (tested by calling the king-safety
  sub-function directly, not just the aggregate `evaluate`).

### 11.7 Search smoke tests (functional, not perft-scale)

A short list of well-known mate-in-N and "obvious best move" positions
(e.g. classic mate-in-2/mate-in-3 puzzles, a hanging-queen tactic) run
through `search_best_move` at a shallow-to-moderate depth/time budget,
asserting the returned move matches the known solution. These are slower
and less exhaustive than perft but validate the *search*, not just move
generation — perft alone says nothing about alpha-beta, quiescence, or move
ordering correctness.

## 12. Performance notes specific to pure-Python bitboards

- Every 64-bit-wraparound-sensitive operation (the magic multiply
  specifically) must mask with `& BB_ALL`; Python ints don't wrap, so a
  missing mask is a silent correctness bug, not a crash — this is called out
  explicitly in `attacks.py`'s module docstring and covered by §11.2's tests
  (a wrong mask would fail the random-occupancy comparison against the slow
  ray-tracer immediately).
- `int.bit_count()` (3.10+) is preferred over any hand-rolled popcount loop
  or lookup table — it's a C-level method call, faster than pure-Python
  alternatives and simpler than a de Bruijn/lookup-table popcount that would
  otherwise be "necessary" in a language without a native popcount.
- Bit iteration (`for sq in iterate_bits(bb): ...`) uses the standard
  "isolate lowest set bit, `bit_length() - 1` for its index, clear it"
  loop (`lsb = bb & -bb; sq = lsb.bit_length() - 1; bb ^= lsb`) rather than
  testing all 64 bits — this matters more in Python than in C, since each
  loop iteration has high fixed overhead, so touching only set bits (a
  sparse board has far fewer than 64) is a real win, not a micro-optimization.
- `Board`/`UndoInfo`/`TTEntry` all use `__slots__` to avoid per-instance
  `__dict__` overhead, since these are the highest-allocation-rate objects
  in the engine (one `UndoInfo` per ply of search, easily hundreds of
  thousands per move at reasonable depth/time budgets).
- No `numpy`/`ctypes`/C-extension dependency is introduced by this proposal
  (keeping `dependencies = []` intact); if profiling later shows the pure-
  Python interpreter loop itself (not algorithmic issues) to be the
  bottleneck, the module boundaries above are drawn so that `attacks.py` and
  the innermost `negamax`/`quiescence` loop could be replaced by a Cython or
  `ctypes`-backed implementation without changing `board.py`'s or
  `movegen.py`'s public interfaces — noted as a future option, not designed
  further here.

## 13. Key trade-offs summary

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Board rep | Bitboards (12 × `int`) | 0x88 / mailbox array | O(1) attack lookups via magics; 0x88 buys nothing once magics remove the need for edge-detection tricks |
| Magic layout | Plain (fixed shift/table size per piece type) | Fancy (per-square minimal shift, packed array) | Memory savings from fancy magics are noise in Python; plain magics are simpler to implement and test |
| Magic numbers | Generated offline, shipped as data, regenerated at import from masks+magics | Searched fresh at every import | Avoids paying random-search cost (up to ~seconds) on every process start, including every test run |
| Move generation | Pseudo-legal + make/unmake legality filter | Full pin/checkers bitboard tracking | Simpler, reuses existing attack tables with no new code path; optimize only if profiling demands it |
| Move representation | Packed `int` | `Move` dataclass/object | Cheaper to create/compare/store in TT and killer tables; no attribute-lookup overhead in hot loops |
| TT collision handling | Full-key verification per entry, no chaining | Bucketed/multi-way TT | Simplicity; a single flat array with full-key check is correct and fast enough for a first version |
| Undo mechanism | Explicit `UndoInfo` stack + in-place mutation | Copy-on-make (`board.copy()`) | Avoids copying 12 bitboards + metadata per node; standard technique in every serious bitboard engine |

## 14. Suggested implementation order (milestones)

1. `constants.py`, `bitboard.py` — foundational, fully covered by unit tests.
2. `attacks.py` non-sliding tables (knight/king/pawn) + `scripts/generate_magics.py`
   + `data/magic_numbers.py` + magic-based sliding attacks — gated by §11.2
   before anything else builds on it.
3. `zobrist.py` — independent of `board.py`, testable against its own
   from-scratch oracle immediately.
4. `moves.py` encode/decode — independent, unit-testable in isolation.
5. `board.py` (`Board`, `UndoInfo`, make/unmake) — gated by §11.4 round-trip
   tests before move generation is trusted to use it.
6. `movegen.py` — gated by the full §11.1 perft suite before search is
   trusted to use it (this is the single most important correctness gate in
   the whole project: a search bug is often actually a movegen bug wearing
   a search-shaped disguise).
7. `evaluate.py` — gated by §11.6, developed and tunable independently of
   search.
8. `transposition.py`, `search.py` (negamax → +quiescence → +iterative
   deepening → +TT → +move ordering, each as a separable step gated by
   §11.5/§11.7) — TT is added *after* a TT-free search already passes the
   mate-in-N smoke tests, specifically so §11.5's "TT must not change
   results" test has a trustworthy TT-free baseline to compare against.
9. `perft.py` CLI wiring, `uci.py`, `cli.py` — user-facing layer last, once
   the engine core is independently validated.
