# Proposal: Pragmatic Hybrid Architecture for `chessengine`

Status: Draft for review
Author: francesco.errico@relatech.com (proposal drafted by Claude)
Scope: `src/chessengine/`

## 1. Goals and non-goals

This proposal is optimized for **correctness-first, incremental delivery in
Python**, not for competing with C/C++ engines on raw nodes-per-second. Every
decision below is judged against four criteria, in this priority order:

1. **Correctness** — legal move generation is the foundation everything else
   sits on; a bug here silently corrupts search and evaluation results in ways
   that are hard to detect later. It must be provably correct via perft and
   regression suites before anything else is built on top of it.
2. **Testability** — every layer (bitboard helpers, attack generation, move
   generation, evaluation terms, search) must be unit-testable in isolation,
   with pure functions and dependency-inverted interfaces wherever the
   boundary matters.
3. **Clean layering** — rules engine, search engine, evaluation, and UCI
   adapter are four separate concerns with one-directional dependencies. No
   layer reaches upward or sideways into another's internals.
4. **A documented upgrade path for speed** — because Python bitwise ops on
   `int` will never match C, the architecture must expose narrow, stable
   seams (pure function signatures, protocol-based interfaces, plain-int
   state) where a faster implementation (Cython, Rust/PyO3, numpy, or just a
   smarter algorithm) can be swapped in later without touching callers.

Non-goals for the initial phases: multi-threaded search, NNUE-style
evaluation, tablebase probing, opening books, pondering. These are called out
as optional Phase 6 extensions and the architecture leaves room for them, but
none of the core interfaces below assume they exist.

## 2. Board representation: bitboards (with a mailbox cache) — and why

**Decision: bitboards are the canonical representation**, held as twelve
64-bit Python `int`s (one per `(color, piece_type)` pair) plus derived
occupancy unions, with a redundant 8x8 mailbox array (`piece_at: list[Piece |
None]` of length 64) maintained incrementally alongside them purely as an
O(1) "what's on this square" cache. This is itself the first "pragmatic
hybrid" decision in this document: the mailbox array is not the source of
truth and is never consulted by move generation — it exists only to make
`Position.piece_at(square)`, SAN generation, and debugging/printing O(1)
instead of requiring twelve `AND`-tests per query. It is updated in the same
`make_move`/`unmake_move` calls that update the bitboards, so it can never
drift out of sync if `Position` is the only mutator (enforced by keeping its
fields private and mutation confined to one module).

### Why bitboards over 0x88

In Python, neither representation has a *raw speed* advantage in the way it
would in C: 0x88's cheap off-board test (`sq & 0x88`) and bitboards' cheap
"is this square attacked" test (`AND` + truthiness) both ultimately run
through the CPython object model, and a big Python `int` used as a 64-bit
bitboard is not a machine word — it is a heap object with attendant
overhead. So this is **not** decided on a speed argument. It is decided on
three architectural arguments that matter more for this project:

- **Zobrist hashing is nearly free.** A transposition table is required for
  any search deeper than a few plies, and Zobrist hashing is naturally
  expressed as XOR-ing precomputed keys in and out as bitboards change. With
  a mailbox-only (0x88) representation you'd build the same key table anyway
  — bitboards just make the update sites (`make_move`) obvious and
  symmetric (set-bit / clear-bit / XOR-key, always in matching triples).
- **The future performance seam is bitboards, not mailboxes.** If/when this
  engine needs a faster move generator, the realistic paths are: magic
  bitboards / PEXT for sliding attacks, a Cython or Rust/PyO3 extension that
  operates on plain 64-bit integers, or vectorized batch operations. All of
  these consume and produce bitboards. Choosing bitboards now means the
  eventual native-extension boundary (`attacks.py`'s function signatures) is
  already the right shape; choosing 0x88 would mean re-deriving bitboards at
  that boundary anyway.
- **The literature and test corpora assume bitboards.** Perft reference
  positions, magic bitboard generators, and most public chess-programming
  references (Chess Programming Wiki, existing open-source engines used as
  cross-checks) are written in bitboard terms. Reusing that body of
  knowledge for correctness-checking is a direct, practical win for a
  correctness-first project.

0x88 remains a perfectly legitimate choice and would not be wrong — it is
simpler to hand-trace for a single square and avoids ever thinking about
64-bit integers. It loses out here specifically because of criterion 4
(swap-in path for speed) and because Zobrist/TT design is cleaner on top of
bitboards. If profiling ever shows the mailbox/mailbox-adjacent code paths
dominating, that's a localized, reversible decision — it does not touch the
rules engine's public API.

### `Position` sketch

```python
# chessengine/board/position.py
@dataclass
class Position:
    # Canonical state — bitboards, one per (color, piece_type), plus unions.
    pieces: dict[tuple[Color, PieceType], int]   # 12 bitboards, each a Python int
    occupied_by: dict[Color, int]                # 2 union bitboards
    occupied: int                                 # occupied_by[WHITE] | occupied_by[BLACK]

    # Cache, not source of truth — updated in lockstep by make_move/unmake_move only.
    mailbox: list[Piece | None]                   # length 64, O(1) piece_at()

    side_to_move: Color
    castling_rights: int          # 4-bit mask: WK, WQ, BK, BQ
    en_passant_square: int | None # target square, or None
    halfmove_clock: int
    fullmove_number: int
    zobrist_hash: int

    def make_move(self, move: "Move") -> "UndoInfo": ...
    def unmake_move(self, move: "Move", undo: "UndoInfo") -> None: ...
    def piece_at(self, square: int) -> "Piece | None": ...          # reads mailbox
    def is_square_attacked(self, square: int, by_color: Color) -> bool: ...
    def king_square(self, color: Color) -> int: ...
```

`make_move`/`unmake_move` are the *only* place bitboards, the mailbox, the
Zobrist hash, castling rights, and the en-passant square are mutated — this
single-writer discipline is what keeps the cache honest and is directly
testable (a property test can assert `mailbox` and `pieces` agree on every
square after any sequence of make/unmake calls).

## 3. Move encoding

Start with a small, explicit, immutable value type — not a packed integer —
because in a correctness-first phase, a move needs to be *readable in a
debugger and in test failure output*, not squeezed into 16 bits. Packing is
a pure optimization that can be introduced later behind the same public
surface without touching move generation, search logic, or the UCI adapter.

```python
# chessengine/moves/move.py
class MoveFlag(IntEnum):
    QUIET = 0
    DOUBLE_PAWN_PUSH = 1
    CASTLE_KINGSIDE = 2
    CASTLE_QUEENSIDE = 3
    CAPTURE = 4
    EN_PASSANT_CAPTURE = 5
    PROMOTION = 8          # combined with PROMOTION_PIECE, and | CAPTURE if applicable

@dataclass(frozen=True, slots=True)
class Move:
    from_square: int
    to_square: int
    flag: MoveFlag
    promotion: PieceType | None = None   # only set when flag has PROMOTION bit

    def is_capture(self) -> bool: ...
    def is_castle(self) -> bool: ...
    def to_uci(self) -> str: ...          # e.g. "e2e4", "e7e8q"

    @staticmethod
    def from_uci(text: str, position: "Position") -> "Move": ...  # needs position
                                                                    # to disambiguate
                                                                    # castling / en passant
```

`frozen=True, slots=True` gives cheap equality/hashing (useful in tests and
for killer-move/TT-move comparisons) without the overhead of a full class.
`to_uci`/`from_uci` are the *only* place move notation is understood —
that's the seam the UCI adapter uses, so nothing else in the codebase parses
or prints UCI move strings.

**Deferred, documented optimization:** once profiling in Phase 4 shows move
list construction or TT storage is a hot path, `Move` instances can be
additionally packed into a single 32-bit int (`from<<0 | to<<6 | flag<<12 |
promo<<16`) for internal search bookkeeping (killer tables, TT entries),
while `generate_legal_moves` keeps returning `Move` dataclasses at the
rules-engine boundary. This is explicitly *not* done in Phase 1–3: a packed
int is harder to unit-test and harder to read in a failing assertion, and at
that stage the priority is proving correctness, not shaving allocations.

## 4. Move generation strategy (the "rules engine")

The rules engine's job is exactly: **legal move generation, check
detection, pin detection, castling legality, en passant correctness** —
nothing about search or evaluation. It is exposed as pure functions over a
`Position`, never as stateful objects:

```python
# chessengine/moves/generator.py
def generate_pseudo_legal_moves(position: Position) -> list[Move]: ...
def generate_legal_moves(position: Position) -> list[Move]: ...
def is_in_check(position: Position, color: Color) -> bool: ...
```

### Two implementations, on purpose, at different phases

Correctness-first delivery means the *first* legal-move-generator should be
whichever version is easiest to prove correct, even if it is slower —
performance is a Phase 4 concern, not a Phase 1 concern.

- **Phase 1 (oracle): pseudo-legal + simulate.** Generate pseudo-legal moves
  per piece type using attack tables (`attacks.py`), then filter: for each
  pseudo-legal move, make it, check whether the mover's own king is attacked,
  unmake it, keep the move if not. This is slow (O(moves) make/unmake calls
  per position) but almost trivially correct to reason about and review,
  because "is this move legal" reduces to a single already-tested primitive
  (`is_square_attacked`).
- **Phase 4 (fast path): pin/checkers-mask legal generation.** Compute a
  `checkers` bitboard and, per pinned piece, a pin-ray mask, then generate
  moves that are legal by construction (a pinned piece may only move along
  its pin ray; if in check, non-king moves must land on a
  block-or-capture-the-checker mask). This avoids the make/unmake-per-move
  cost entirely.

Critically, **the Phase 1 oracle is not thrown away** — it stays in the
codebase as a permanent test oracle. Phase 4's fast generator is validated by
differential testing: for every position in the perft suite, and for a large
corpus of randomly-reached positions (random-legal-move self-play walks,
seeded for reproducibility), assert `set(generate_legal_moves_fast(pos)) ==
set(generate_legal_moves_oracle(pos))`. This "keep the slow, obviously
correct version around as a permanent oracle" pattern is the single most
important correctness technique in this proposal — it converts "we
optimized the move generator" from a leap of faith into a mechanically
checked equivalence.

### Special-move handling is explicit and separately tested

Each of the classic chess-engine bug sources gets its own named code path and
its own named regression test, rather than being folded into generic logic:

- **Castling**: requires (a) the mask of squares between king/rook empty,
  (b) king's start, transit, and end squares not attacked, (c) castling
  rights bit still set (cleared whenever the king moves, a rook moves off
  its home square, or a rook is captured on its home square — including when
  it's the *opponent's* move capturing that rook).
- **En passant**: the en-passant target square is state on `Position` (set
  on a double pawn push, cleared every other move); the classic bug — an en
  passant capture that exposes the king to a horizontal check because two
  pawns disappear from the same rank simultaneously — gets a dedicated test
  position (a known FEN where en passant is pseudo-legal but illegal for
  exactly this reason).
- **Promotion**: pseudo-legal generation emits four moves (Q/R/B/N) for every
  pawn push or capture that lands on the back rank; under-promotion is
  therefore free (no special-cased logic needed at the search layer beyond
  move ordering deprioritizing non-queen promotions).

### `perft` as the correctness harness

```python
# chessengine/moves/perft.py
def perft(position: Position, depth: int) -> int: ...
def perft_divide(position: Position, depth: int) -> dict[str, int]: ...
```

`perft`/`perft_divide` are used against the standard published-node-count
positions (initial position, "Kiwipete", and positions 3–6 from the
Chess Programming Wiki's perft results page) up to depth 5–6. This is the
non-negotiable gate before Phase 2 begins: **no search or evaluation code is
written against a rules engine that has not matched published perft counts
exactly.**

## 5. Search architecture

The search engine depends on exactly three things, all abstract: a
`Position` it can make/unmake moves on, `generate_legal_moves`, and an
`Evaluator` protocol (Section 6). It has **no knowledge of UCI** — no stdin,
no stdout, no text protocol — and no knowledge of which concrete evaluator
is plugged in.

```python
# chessengine/search/iterative.py
@dataclass
class SearchLimits:
    max_depth: int | None = None
    max_nodes: int | None = None
    movetime_ms: int | None = None
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int = 0
    binc_ms: int = 0

@dataclass
class SearchInfo:          # one per completed iterative-deepening depth
    depth: int
    score_cp: int
    nodes: int
    time_ms: int
    pv: list[Move]

@dataclass
class SearchResult:
    best_move: Move
    ponder_move: Move | None
    score_cp: int

class SearchEngine:
    def __init__(self, evaluator: "Evaluator", tt_size_mb: int = 64): ...

    def search(
        self,
        position: Position,
        limits: SearchLimits,
        on_info: Callable[[SearchInfo], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> SearchResult: ...
```

- `on_info` is a callback, not a print statement — the UCI adapter supplies
  a callback that formats and writes `info depth ... score cp ... pv ...` to
  stdout; a unit test supplies a callback that just appends to a list.
- `should_stop` is an injected predicate, not a hardcoded `time.time()`
  check — this is what makes search interruption testable without real
  wall-clock waits (a test can inject `should_stop=lambda: nodes_seen >= 100`
  or a fake clock).
- The search engine owns the transposition table and move-ordering state
  (killers, history) internally; these are never exposed to the UCI layer.

### Build order within search (each step gated by its own test before the next is added)

1. **Plain negamax alpha-beta**, fixed depth, no TT, no move ordering beyond
   "captures before quiets." Validated by a differential test against a
   brute-force full-width minimax on small/shallow positions — this exists
   specifically to catch alpha-beta pruning bugs (a mis-implemented
   alpha/beta window silently prunes a better move and is otherwise
   invisible unless you compare against unpruned search).
2. **Transposition table** (Zobrist-keyed, entries store depth, score, bound
   type `EXACT`/`LOWER_BOUND`/`UPPER_BOUND`, best move, age for replacement).
   Validated by running the same position with TT on vs TT off and asserting
   they agree on best move/score at a fixed depth (documented tolerance for
   cases where TT cutoffs legitimately change which equal-score move is
   returned).
3. **Iterative deepening + time management + `SearchInfo` reporting.**
   Validated by asserting each deeper iteration's PV is at least as good
   (score-wise, within the search's own numbers) and that time limits are
   respected using an injected fake clock.
4. **Move ordering**: TT move first, then MVV-LVA captures, then killer
   moves, then history heuristic. Validated by a node-count regression
   check (better ordering must not *increase* node count at fixed depth on
   the benchmark suite) and by the tactical suite in step 6 solving in fewer
   nodes than before.
5. **Quiescence search** (captures and check evasions only, at leaf nodes)
   to remove the horizon effect. Validated with hand-picked positions where
   naive fixed-depth search misjudges a hanging piece one ply past the
   horizon, and the quiescence-enabled search correctly does not.
6. **Tactical test suite**: a curated set of mate-in-2/3/4 and
   known-best-move positions (e.g., a small hand-picked set plus a subset of
   a public suite such as the Bratko-Kopec or WAC positions) solved within a
   fixed depth/time budget — this is the search engine's own version of the
   perft gate.

Null-move pruning, late move reductions, aspiration windows, and check
extensions are deliberately deferred to Phase 6 (Section 8) and are each
gated by an A/B match requirement (must not lose strength vs the prior
version at a fixed time control) rather than being added speculatively.

## 6. Evaluation interface design

```python
# chessengine/eval/interface.py
class Evaluator(Protocol):
    def evaluate(self, position: Position) -> int:
        """Return a score in centipawns from the side-to-move's perspective
        (negamax convention: positive is good for whoever is to move)."""
        ...
```

The search engine's only dependency is this `Protocol` — it never imports a
concrete evaluator. This is plain dependency inversion, and it buys two
concrete things for this project:

- **Phase 2 can validate the entire search+UCI pipeline against a trivial
  `MaterialOnlyEvaluator`** before any time is spent on positional
  evaluation, because `SearchEngine` doesn't care which `Evaluator` it was
  constructed with.
- **A future NNUE-style or otherwise opaque evaluator is a drop-in
  replacement** for `CompositeEvaluator` (below) with zero changes to
  search, because both satisfy the same one-method protocol.

### Composable, independently-tested terms

```python
# chessengine/eval/material.py
def material_term(position: Position) -> int: ...

# chessengine/eval/pst.py
def piece_square_term(position: Position) -> int: ...

# chessengine/eval/mobility.py
def mobility_term(position: Position) -> int: ...

# chessengine/eval/composite.py
@dataclass
class Weights:
    material: float = 1.0
    pst: float = 1.0
    mobility: float = 1.0
    king_safety: float = 1.0
    pawn_structure: float = 1.0

class CompositeEvaluator:
    def __init__(self, weights: Weights = Weights()):
        self._weights = weights

    def evaluate(self, position: Position) -> int:
        w = self._weights
        return int(
            w.material * material_term(position)
            + w.pst * piece_square_term(position)
            + w.mobility * mobility_term(position)
            + w.king_safety * king_safety_term(position)
            + w.pawn_structure * pawn_structure_term(position)
        )
```

Each `*_term` function is:

- **Pure** — takes a `Position`, returns an `int`, no hidden state, no
  dependency on search or on other terms.
- **Independently unit-testable** — e.g. "given a FEN with an extra queen
  for white, `material_term` returns approximately +900", "given a FEN with
  a centralized vs. a cornered knight, `piece_square_term` prefers the
  centralized one", entirely without invoking move generation or search.
- **Independently addable** — a new term (e.g. `passed_pawn_term`) is added
  as a new file + a new `Weights` field + one line in `CompositeEvaluator`,
  with no change to any existing term's code or tests.

Weights are **data, not code** — a plain dataclass — so that an offline
tuning process (Texel tuning / SPSA against a labeled game corpus, or simple
hand-tuning) can adjust them without touching term implementations, and so
that different `Weights` presets can be unit-tested independently of the
tuning process itself.

This satisfies the requirement directly: evaluation terms are added
independently, unit-tested in isolation, and the summation strategy
(`CompositeEvaluator`) is itself swappable behind the one-method
`Evaluator` protocol without the search engine ever noticing.

## 7. Module boundaries

```
src/chessengine/
  board/
    types.py        # Color, PieceType, Square/File/Rank enums, constants
    bitboard.py      # pure bit helpers: set_bit, pop_lsb, popcount, iterate_bits
    attacks.py       # knight/king/pawn tables + sliding attacks (ray-scan now,
                     #   magic-bitboard-ready signature for later)
    zobrist.py       # hash key tables + incremental update helpers
    position.py      # Position: bitboards + mailbox cache + make/unmake (single writer)
    fen.py           # FEN <-> Position, pure functions

  moves/
    move.py          # Move dataclass, MoveFlag, to_uci/from_uci
    generator.py      # generate_pseudo_legal_moves, generate_legal_moves,
                     #   is_in_check (Phase 1: simulate-based; Phase 4: pin/checker-mask)
    perft.py         # perft, perft_divide

  eval/
    interface.py     # Evaluator Protocol
    material.py, pst.py, mobility.py, king_safety.py, pawn_structure.py
    composite.py     # CompositeEvaluator + Weights

  search/
    transposition.py # TT entry + table, Zobrist-keyed
    ordering.py       # MVV-LVA, killers, history
    alphabeta.py      # negamax + quiescence
    iterative.py      # SearchLimits, SearchInfo, SearchResult, iterative-deepening loop
    engine.py         # SearchEngine facade — the only thing uci/ imports from search/

  uci/
    protocol.py       # parse UCI command lines -> internal command objects
    options.py        # declared UCI options (Hash, etc.) -> engine config
    adapter.py        # the stdin/stdout loop; ONLY module allowed to do UCI text I/O

  cli.py             # composition root: wires Position + SearchEngine + Evaluator
                     #   + UciAdapter together; console-script entry point

tests/
  board/, moves/ (incl. perft_data/ fixtures of known FEN + node counts),
  eval/ (one test module per term), search/ (tactical suite, TT/ordering
  regression, fake-clock time-management tests), uci/ (protocol parsing
  tests against a fake SearchEngine double)
```

**Enforced dependency direction** (the actual architectural contract):

```
board  <-  moves  <-  search  <-  uci  <-  cli
  ^         ^           ^ (via Evaluator
  |         |             protocol only)
  |         +-- eval  ----+
  |
  +-------- eval (read-only Position access)
```

- `board/` depends on nothing else in the package.
- `moves/` depends only on `board/`.
- `eval/` depends only on `board/` (read-only queries — a `piece_at`,
  `occupied`, never a mutator).
- `search/` depends on `board/` and `moves/` directly, and on `eval/` **only
  through the `Evaluator` protocol** — never importing a concrete evaluator
  module.
- `uci/` depends on `search/` **only through the `SearchEngine` facade**,
  plus `board/fen.py` (to parse the `position fen ...` command) and
  `moves/move.py` (to parse/print UCI move notation) — never on
  `search/alphabeta.py`, `search/transposition.py`, etc. directly, and never
  on `eval/` at all.
- `cli.py` is the single composition root: the only file permitted to import
  from every layer, because its entire job is dependency injection —
  constructing a `CompositeEvaluator`, a `SearchEngine`, and a `UciAdapter`
  and wiring them together.
- Nothing above `uci/` imports `uci/`; nothing in `board/` or `moves/`
  imports `eval/` or `search/`.

This is intentionally strict enough to be **mechanically checkable** — an
import-linter config (or a small AST-walking test in `tests/architecture/`
asserting no forbidden import edges exist) can enforce it in CI cheaply, and
is recommended as soon as Phase 1 is stable.

## 8. Phased implementation and testing roadmap

Each phase's exit criterion is a test suite, not a calendar date. A phase is
not "done" until its validation gate is green; the next phase does not start
until then.

### Phase 0 — Foundations (bitboard/attack/FEN primitives)

**Build**: `board/types.py`, `board/bitboard.py` (set_bit, pop_lsb, popcount,
iterate_bits), `board/attacks.py` (precomputed knight/king/pawn attack
tables; sliding attacks via simple ray-scan for now — magic bitboards are a
Phase 4 optimization, not a Phase 0 requirement), `board/fen.py`.

**Validate**: unit tests for every pure helper (edge/corner square cases for
knight and king attacks — a1, h1, a8, h8, edge files/ranks); FEN round-trip
tests (`parse(fen) -> serialize() == fen`) against the standard start
position, Kiwipete, and several hand-picked FENs covering castling rights,
en-passant target squares, and non-zero halfmove clocks.

### Phase 1 — Rules engine baseline (THE critical correctness gate)

**Build**: `board/position.py` (`make_move`/`unmake_move` for all move
types, incremental Zobrist hash), `board/zobrist.py`, `moves/move.py`,
`moves/generator.py` (pseudo-legal generation + simulate-based legal
filter), `moves/perft.py`.

**Validate**:
- `perft` matches published node counts *exactly* for: initial position
  (depth 1–6), Kiwipete (depth 1–5), and Chess Programming Wiki positions
  3–6 (depth 1–5).
- Dedicated regression tests for the classic bug positions: en-passant
  discovered check, castling through/into/out-of check, castling rights
  lost on rook capture, under-promotion.
- Property test: after any sequence of `make_move`/`unmake_move`, the
  mailbox cache agrees with the bitboards on every square, and the Zobrist
  hash returns to its exact prior value after every `unmake_move`.

Nothing in Phase 2+ begins until this phase's tests are fully green — this
is the single load-bearing correctness gate for the whole project.

### Phase 2 — Wire it end-to-end (naive search + trivial eval + minimal UCI)

**Build**: `eval/interface.py` + a `MaterialOnlyEvaluator`; plain
fixed-depth negamax alpha-beta in `search/alphabeta.py` (no TT, no
ordering); minimal `uci/adapter.py` supporting `uci`, `isready`,
`ucinewgame`, `position`, `go depth N`, `stop`, `quit`, `bestmove`; `cli.py`
composition root.

**Validate**: a self-play or GUI-driven smoke test (e.g., via
`python-chess` or a GUI/CLI harness such as `cutechess-cli`) plays complete
games with zero illegal-move UCI errors; a differential test asserts the
alpha-beta result matches an unpruned full-width minimax on small/shallow
positions (catches pruning-window bugs early, before they hide behind TT/
ordering complexity).

### Phase 3 — Search quality (TT, iterative deepening, ordering, quiescence)

**Build**: `search/transposition.py`, `search/iterative.py`
(`SearchLimits`/`SearchInfo`/time management via injected `should_stop`),
`search/ordering.py` (MVV-LVA, killers, history), quiescence search in
`search/alphabeta.py`.

**Validate**: TT on/off agreement test at fixed depth (documented tolerance
for cutoff-driven equal-score move differences); fake-clock time-management
tests (no real wall-clock sleeps in the test suite); node-count regression
benchmark (ordering must not increase node counts); a curated tactical
suite (mate-in-2/3/4 plus a subset of a public tactics suite) solved within
budget; quiescence-specific hand-built positions confirming the horizon
effect is resolved.

### Phase 4 — Faster legal move generation (the optimization seam)

**Build**: pin-mask/checkers-mask-based `generate_legal_moves` in
`moves/generator.py`, replacing per-move simulate-based filtering
(the Phase 1 simulate-based generator is *kept* as `generate_legal_moves_ref`
/ test oracle, not deleted); optionally, magic bitboards or PEXT for sliding
attacks behind the existing `attacks.py` signatures; optionally, a packed
move-int representation for TT/killer/history internals only.

**Validate**: differential equality test — `set(fast(pos)) ==
set(oracle(pos))` — across the full perft suite plus a large corpus of
randomly-reached legal positions (seeded random-legal-move self-play walks);
perft nodes/sec benchmark recorded to quantify the speedup; the entire
Phase 1–3 test suite remains green with zero behavior changes (this phase is
speed-only, and the test suite is the proof).

### Phase 5 — Evaluation depth and tuning infrastructure

**Build**: `pst.py`, `mobility.py`, `king_safety.py`, `pawn_structure.py`
added one at a time, each with its own isolated unit tests; `Weights`
extended one field at a time; an offline tuning script (Texel tuning/SPSA
against a labeled game corpus) kept outside the runtime engine path.

**Validate**: per-term unit tests (hand-built FENs with known expected
sign/magnitude); an engine-vs-engine match set (self-play at fixed depth, or
vs. a fixed reference opponent) tracked after each term is added, requiring
a non-negative score/Elo trend before the term is kept (a term that is
"correct" in isolation but loses strength due to interaction effects is
reverted or reweighted, not shipped); Phase 1–4 suites remain green
throughout.

### Phase 6 — Optional polish and extensions

**Build** (each item independently gated, lowest priority): null-move
pruning, late move reductions, aspiration windows, check extensions, refined
time management (soft/hard limits, sudden-death handling), the fuller UCI
option surface (`Hash`, `Threads` placeholder, `Ponder`), SAN generation/
parsing for PGN import/export, and — as external hooks only, not core
engine logic — an opening-book interface and a Syzygy-tablebase-probing
interface.

**Validate**: each pruning/extension/heuristic change is gated by an A/B
match at a fixed time control against the immediately prior version — it
must not lose measurable strength — before being kept; UCI conformance is
checked against a real GUI (e.g., a smoke test in Arena or CuteChess) in
addition to the unit-level protocol parsing tests.

## 9. Summary of key tradeoffs

| Decision | Chosen for | Given up | Mitigation |
|---|---|---|---|
| Bitboards as canonical state | Zobrist hashing, TT design, native-extension upgrade path, literature/perft corpora | Simplicity of 0x88's off-board test; no raw speed advantage in Python either way | Mailbox cache kept alongside for O(1) piece lookups without touching move-gen |
| Simulate-based legal move generation first | Obvious, reviewable correctness before any performance work | Slower move generation (extra make/unmake per candidate move) | Kept permanently as a test oracle; Phase 4's fast generator is validated against it, not trusted on its own |
| `Move` as a frozen dataclass, not a packed int | Debuggability, direct equality/testing, clear UCI boundary | Slightly higher per-move object overhead | Packed-int form deferred to Phase 4+, introduced only behind the same `to_uci`/equality surface, only where profiling justifies it |
| Search gated stepwise (alpha-beta -> TT -> ordering -> quiescence -> pruning) | Each feature is independently verifiable; no unverifiable complexity compounds | Slower path to peak playing strength | Each step has its own named test gate (differential minimax check, TT agreement check, node-count regression, tactical suite, A/B match) before the next step starts |
| Evaluation as summed independent terms behind an `Evaluator` protocol | Independent unit testing and tuning of each term; trivial to swap the whole evaluator (e.g., for an NNUE-style black box) later | Can't express nonlinear term interactions (e.g., bishop pair value depends on pawn structure) without an explicit interaction term | Acceptable for the correctness-first phases; the protocol boundary means a smarter (even nonlinear/learned) evaluator can replace `CompositeEvaluator` with zero changes to `search/` |
| Strict one-directional module boundaries (`board -> moves -> search -> uci -> cli`, `eval` parallel to `search`) | UCI is fully decoupled from search internals; rules engine has zero awareness of search or evaluation existing | Some upfront discipline/ceremony (protocols, facades) that a monolithic script wouldn't need | Cheap to enforce mechanically (import-linter or an AST-walking architecture test) once Phase 1 stabilizes |

## 10. What this buys, concretely

By the end of Phase 1, the project has a rules engine that is provably
correct against the standard perft corpus and stays that way, independent of
whatever happens to search or evaluation later. By the end of Phase 3, it has
a playable, UCI-compliant engine built entirely on interfaces (`Evaluator`
protocol, `SearchEngine` facade, `Move`/`Position` pure-data boundaries) that
can absorb a faster move generator (Phase 4), richer evaluation (Phase 5), or
deeper search techniques (Phase 6) as independent, separately-gated changes —
none of which require re-validating the layers below them, because each
layer's contract was tested in isolation from the start.
