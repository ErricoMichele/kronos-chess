"""Console-script entry point for the UCI engine (architecture.md §12).

Per the module-boundary DAG (architecture.md §11), `cli.py` depends only on
`uci.py` — it is the outermost layer, wired as the `chessengine` console
script in `pyproject.toml` (`[project.scripts] chessengine =
"chessengine.cli:main"`).
"""

from __future__ import annotations

from .uci import UCIEngine


def main() -> None:
    """Run a UCI session over stdin/stdout until `quit` (or EOF)."""
    UCIEngine().run()


if __name__ == "__main__":
    main()
