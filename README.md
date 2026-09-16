# death-Token Chess Engine

A from-scratch chess engine written in Python: bitboard board representation,
legal move generation, alpha-beta search with iterative deepening, a tunable
evaluation function, and a UCI-compatible interface so it can be dropped into
any standard chess GUI.

## Status

Under active, incremental development. See `docs/` for design notes and
`tests/` for the perft/move-generation validation suite.

## Layout

- `src/chessengine/` — engine source (board, movegen, search, evaluate, uci, cli)
- `tests/` — unit tests and perft correctness tests
- `docs/` — architecture and design notes
