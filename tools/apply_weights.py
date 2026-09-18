"""Apply tuned weights back to the codebase.

Reads a weights JSON file (produced by spsa_tune.py) and either prints the
new ``Weights(...)`` constructor call, or patches ``evaluate.py``'s
``default_evaluator()`` in-place.

Usage:
    # Print only (safe, no file changes):
    python tools/apply_weights.py tuned_weights.json

    # Apply directly to evaluate.py:
    python tools/apply_weights.py tuned_weights.json --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
EVALUATE_PY = _project_root / "src" / "chessengine" / "evaluate.py"

# The Weights fields in the order they appear in the dataclass.
WEIGHT_FIELDS = [
    "material_pst",
    "mobility",
    "king_safety",
    "pawn_structure",
    "endgame_mopup",
]


def load_weights(path: str | Path) -> dict[str, float]:
    """Load a tuned-weights JSON and return only the weight fields."""
    with open(path) as f:
        data = json.load(f)
    return {k: float(data[k]) for k in WEIGHT_FIELDS if k in data}


def format_weights_call(weights: dict[str, float]) -> str:
    """Format a ``Weights(...)`` constructor call with aligned kwargs."""
    lines = []
    for name in WEIGHT_FIELDS:
        if name in weights:
            lines.append(f"            {name}={weights[name]},")
    return "        weights=Weights(\n" + "\n".join(lines) + "\n        ),"


def print_weights(weights: dict[str, float]) -> None:
    """Print the new Weights constructor call to stdout."""
    print("New Weights constructor call:")
    print()
    print(format_weights_call(weights))
    print()


def apply_to_evaluate(weights: dict[str, float]) -> None:
    """Patch evaluate.py's default_evaluator() with the new weights."""
    source = EVALUATE_PY.read_text()

    # Match the existing Weights(...) block inside default_evaluator().
    pattern = r"(        weights=Weights\(\n)((?:            \w+=[\d.]+,\n)+)(        \),)"
    match = re.search(pattern, source)
    if not match:
        print("ERROR: Could not find Weights(...) block in evaluate.py", file=sys.stderr)
        sys.exit(1)

    # Build the replacement kwargs block.
    new_kwargs = ""
    for name in WEIGHT_FIELDS:
        if name in weights:
            new_kwargs += f"            {name}={weights[name]},\n"

    new_block = match.group(1) + new_kwargs + match.group(3)
    new_source = source[:match.start()] + new_block + source[match.end():]

    EVALUATE_PY.write_text(new_source)
    print(f"Updated {EVALUATE_PY}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply tuned weights to the chess engine"
    )
    parser.add_argument(
        "weights_file",
        help="Path to the tuned weights JSON file",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify evaluate.py (without this flag, just prints)",
    )
    args = parser.parse_args()

    weights = load_weights(args.weights_file)

    if not weights:
        print("ERROR: No weight fields found in the JSON file", file=sys.stderr)
        sys.exit(1)

    print_weights(weights)

    if args.apply:
        apply_to_evaluate(weights)
    else:
        print("(dry run -- pass --apply to modify evaluate.py)")


if __name__ == "__main__":
    main()
