"""SPSA tuner for evaluation weights.

Runs pairs of self-play matches: one with weights perturbed +delta,
one with -delta. The score difference estimates the gradient, and
weights are updated along it.

Usage:
    python tools/spsa_tune.py [--config tools/tune_config.json] [--output tuned_weights.json]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# Add project root to path so we can import both the engine and tests/.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "tests"))

from chessengine.evaluate import (
    CompositeEvaluator,
    Weights,
    default_evaluator,
    endgame_mopup_term,
    king_safety_term,
    material_pst_term,
    mobility_term,
    pawn_structure_term,
)
from chessengine.search import SearchLimits
from match_harness import DEFAULT_POSITIONS, play_match


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParamConfig:
    """Bounds and initial value for one tunable parameter."""
    name: str
    initial: float
    min_val: float
    max_val: float


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a tune_config.json file and return its contents."""
    with open(path) as f:
        return json.load(f)


def params_from_config(cfg: dict[str, Any]) -> list[ParamConfig]:
    """Extract the ordered list of tunable parameters from config."""
    params: list[ParamConfig] = []
    for name, spec in cfg["parameters"].items():
        params.append(ParamConfig(
            name=name,
            initial=spec["initial"],
            min_val=spec["min"],
            max_val=spec["max"],
        ))
    return params


# ---------------------------------------------------------------------------
# SPSA core
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def generate_perturbation(n: int, rng: random.Random) -> list[float]:
    """Generate a random perturbation vector of +/-1 values (Bernoulli)."""
    return [rng.choice([-1.0, 1.0]) for _ in range(n)]


def build_evaluator(params: list[ParamConfig], values: list[float]) -> CompositeEvaluator:
    """Construct a CompositeEvaluator from the given parameter values."""
    weight_fields = {f.name for f in fields(Weights)}
    kwargs: dict[str, float] = {}
    for pc, val in zip(params, values):
        if pc.name in weight_fields:
            kwargs[pc.name] = float(val)

    weights = Weights(**kwargs)
    return CompositeEvaluator(
        terms={
            "material_pst": material_pst_term,
            "mobility": mobility_term,
            "king_safety": king_safety_term,
            "pawn_structure": pawn_structure_term,
            "endgame_mopup": endgame_mopup_term,
        },
        weights=weights,
    )


def match_score(
    evaluator: CompositeEvaluator,
    baseline: CompositeEvaluator,
    positions: list[str],
    limits: SearchLimits,
) -> float:
    """Play a match and return a normalised score in [0, 1].

    1.0 = evaluator won every game, 0.0 = baseline won every game,
    0.5 = perfectly even.
    """
    result = play_match(evaluator, baseline, positions, limits)
    total_games = result.score_a + result.score_b
    if total_games == 0:
        return 0.5
    return result.score_a / total_games


def spsa_iteration(
    params: list[ParamConfig],
    theta: list[float],
    baseline: CompositeEvaluator,
    positions: list[str],
    limits: SearchLimits,
    a_k: float,
    c_k: float,
    rng: random.Random,
) -> list[float]:
    """Run one SPSA iteration and return the updated theta vector.

    Steps:
      1. Generate random perturbation delta (each component +/-1).
      2. Scale delta by c_k.
      3. Play match A: theta + c_k*delta  vs baseline.
      4. Play match B: theta - c_k*delta  vs baseline.
      5. Gradient estimate: g_i = (score_A - score_B) / (2 * c_k * delta_i)
      6. Update: theta += a_k * g
      7. Clamp to parameter bounds.
    """
    n = len(params)
    delta = generate_perturbation(n, rng)

    # Perturbed parameter vectors, clamped to bounds.
    theta_plus = [
        clamp(theta[i] + c_k * delta[i], params[i].min_val, params[i].max_val)
        for i in range(n)
    ]
    theta_minus = [
        clamp(theta[i] - c_k * delta[i], params[i].min_val, params[i].max_val)
        for i in range(n)
    ]

    eval_plus = build_evaluator(params, theta_plus)
    eval_minus = build_evaluator(params, theta_minus)

    score_plus = match_score(eval_plus, baseline, positions, limits)
    score_minus = match_score(eval_minus, baseline, positions, limits)

    # Gradient estimate per component.
    gradient = [0.0] * n
    for i in range(n):
        gradient[i] = (score_plus - score_minus) / (2.0 * c_k * delta[i])

    # Update and clamp to bounds.
    new_theta = [
        clamp(theta[i] + a_k * gradient[i], params[i].min_val, params[i].max_val)
        for i in range(n)
    ]

    return new_theta


# ---------------------------------------------------------------------------
# Decay schedule (standard Spall 1998 SPSA coefficients)
# ---------------------------------------------------------------------------

def ak_schedule(initial_a: float, k: int, decay: float, big_a: float = 10.0) -> float:
    """a_k = initial_a / (k + 1 + big_A)^decay"""
    return initial_a / ((k + 1 + big_a) ** decay)


def ck_schedule(initial_c: float, k: int, decay: float) -> float:
    """c_k = initial_c / (k + 1)^decay"""
    return initial_c / ((k + 1) ** decay)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_tuning(
    config_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    """Run the full SPSA tuning loop and return the best weights found."""
    cfg = load_config(config_path)
    params = params_from_config(cfg)

    iterations = cfg["iterations"]
    search_depth = cfg["search_depth"]
    initial_a = cfg["initial_a"]
    initial_c = cfg["initial_c"]
    a_decay = cfg["a_decay"]
    c_decay = cfg["c_decay"]

    # Subset of positions for speed (use first N based on games_per_match).
    games_per_match = cfg.get("games_per_match", 12)
    n_positions = max(1, games_per_match // 2)  # each position plays 2 games
    positions = DEFAULT_POSITIONS[:n_positions]

    limits = SearchLimits(max_depth=search_depth)
    baseline = default_evaluator()

    rng = random.Random(seed)

    # Initial theta from config.
    theta = [p.initial for p in params]

    best_theta = list(theta)
    best_score = 0.5  # neutral starting point

    if verbose:
        print(f"SPSA Tuning: {iterations} iterations, depth {search_depth}, "
              f"{len(positions)} positions x 2 colors = {len(positions) * 2} games/match")
        print(f"Parameters: {[p.name for p in params]}")
        print(f"Initial:    {theta}")
        print()

    for k in range(iterations):
        a_k = ak_schedule(initial_a, k, a_decay)
        c_k = ck_schedule(initial_c, k, c_decay)

        theta = spsa_iteration(params, theta, baseline, positions, limits, a_k, c_k, rng)

        # Evaluate current theta against baseline.
        current_eval = build_evaluator(params, theta)
        score = match_score(current_eval, baseline, positions, limits)

        if score > best_score:
            best_score = score
            best_theta = list(theta)

        if verbose:
            param_dict = {p.name: round(theta[i], 4) for i, p in enumerate(params)}
            print(f"Iter {k + 1:3d}/{iterations}: a_k={a_k:.4f} c_k={c_k:.4f} "
                  f"score={score:.3f} best={best_score:.3f}")
            print(f"  weights: {param_dict}")

        # Save checkpoint after each iteration.
        result: dict[str, Any] = {
            p.name: round(best_theta[i], 4) for i, p in enumerate(params)
        }
        result["_meta"] = {
            "iteration": k + 1,
            "best_score": round(best_score, 4),
            "total_iterations": iterations,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    final = {p.name: round(best_theta[i], 4) for i, p in enumerate(params)}

    if verbose:
        print(f"\nTuning complete. Best score: {best_score:.3f}")
        print(f"Best weights: {final}")
        print(f"Saved to: {output_path}")

    return final


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SPSA tuner for chess engine evaluation weights")
    parser.add_argument(
        "--config",
        default=str(_project_root / "tools" / "tune_config.json"),
        help="Path to tuning configuration JSON (default: tools/tune_config.json)",
    )
    parser.add_argument(
        "--output",
        default=str(_project_root / "tuned_weights.json"),
        help="Path to save best weights JSON (default: tuned_weights.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of iterations from config",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override search depth from config",
    )
    args = parser.parse_args()

    # Allow CLI overrides of config values.
    cfg = load_config(args.config)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    if args.depth is not None:
        cfg["search_depth"] = args.depth

    # Write back the (possibly overridden) config to a temp location so
    # run_tuning picks up the overrides.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp, indent=2)
        tmp_path = tmp.name

    try:
        run_tuning(tmp_path, args.output, seed=args.seed)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
