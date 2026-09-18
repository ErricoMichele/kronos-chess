"""Tests for the SPSA tuning framework (tools/spsa_tune.py).

Validates the core algorithmic building blocks -- perturbation generation,
weight clamping, gradient estimation shape, and a single-iteration smoke
test -- without running a full multi-iteration tuning loop (which is slow
due to self-play matches).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Make tools/ importable.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "tools"))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "tests"))

from spsa_tune import (
    ParamConfig,
    ak_schedule,
    build_evaluator,
    ck_schedule,
    clamp,
    generate_perturbation,
    load_config,
    match_score,
    params_from_config,
    spsa_iteration,
)
from chessengine.evaluate import CompositeEvaluator, Weights, default_evaluator
from chessengine.search import SearchLimits
from match_harness import DEFAULT_POSITIONS


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------

class TestPerturbation:
    def test_perturbation_shape(self) -> None:
        """Delta vector has the requested length."""
        rng = random.Random(42)
        delta = generate_perturbation(5, rng)
        assert len(delta) == 5

    def test_perturbation_values(self) -> None:
        """Every component of the perturbation is exactly +1 or -1."""
        rng = random.Random(123)
        delta = generate_perturbation(100, rng)
        assert set(delta).issubset({-1.0, 1.0})

    def test_perturbation_not_constant(self) -> None:
        """A large enough perturbation vector contains both +1 and -1."""
        rng = random.Random(99)
        delta = generate_perturbation(50, rng)
        assert -1.0 in delta and 1.0 in delta


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

class TestClamp:
    def test_clamp_within_bounds(self) -> None:
        assert clamp(1.5, 0.0, 3.0) == 1.5

    def test_clamp_below_min(self) -> None:
        assert clamp(-1.0, 0.0, 3.0) == 0.0

    def test_clamp_above_max(self) -> None:
        assert clamp(5.0, 0.0, 3.0) == 3.0

    def test_clamp_at_boundary(self) -> None:
        assert clamp(0.0, 0.0, 3.0) == 0.0
        assert clamp(3.0, 0.0, 3.0) == 3.0


# ---------------------------------------------------------------------------
# Decay schedules
# ---------------------------------------------------------------------------

class TestSchedules:
    def test_ak_decreases(self) -> None:
        """a_k should decrease as k increases."""
        vals = [ak_schedule(1.0, k, 0.602) for k in range(10)]
        for i in range(1, len(vals)):
            assert vals[i] < vals[i - 1]

    def test_ck_decreases(self) -> None:
        """c_k should decrease as k increases."""
        vals = [ck_schedule(5.0, k, 0.101) for k in range(10)]
        for i in range(1, len(vals)):
            assert vals[i] < vals[i - 1]

    def test_ak_positive(self) -> None:
        """a_k is always positive."""
        for k in range(100):
            assert ak_schedule(1.0, k, 0.602) > 0

    def test_ck_positive(self) -> None:
        """c_k is always positive."""
        for k in range(100):
            assert ck_schedule(5.0, k, 0.101) > 0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_default_config(self) -> None:
        cfg = load_config(_project_root / "tools" / "tune_config.json")
        assert "iterations" in cfg
        assert "parameters" in cfg
        assert cfg["iterations"] == 50

    def test_params_from_config(self) -> None:
        cfg = load_config(_project_root / "tools" / "tune_config.json")
        params = params_from_config(cfg)
        assert len(params) == 5
        names = [p.name for p in params]
        assert "material_pst" in names
        assert "endgame_mopup" in names


# ---------------------------------------------------------------------------
# Evaluator construction
# ---------------------------------------------------------------------------

class TestBuildEvaluator:
    def test_build_evaluator_returns_composite(self) -> None:
        params = [
            ParamConfig("material_pst", 1.0, 0.5, 2.0),
            ParamConfig("mobility", 1.0, 0.0, 3.0),
            ParamConfig("king_safety", 1.0, 0.0, 3.0),
            ParamConfig("pawn_structure", 1.0, 0.0, 3.0),
            ParamConfig("endgame_mopup", 3.0, 0.0, 10.0),
        ]
        values = [1.0, 1.0, 1.0, 1.0, 3.0]
        ev = build_evaluator(params, values)
        assert isinstance(ev, CompositeEvaluator)
        assert ev.weights.material_pst == 1.0
        assert ev.weights.endgame_mopup == 3.0

    def test_build_evaluator_respects_values(self) -> None:
        params = [
            ParamConfig("material_pst", 1.0, 0.5, 2.0),
            ParamConfig("mobility", 1.0, 0.0, 3.0),
            ParamConfig("king_safety", 1.0, 0.0, 3.0),
            ParamConfig("pawn_structure", 1.0, 0.0, 3.0),
            ParamConfig("endgame_mopup", 3.0, 0.0, 10.0),
        ]
        values = [1.5, 2.0, 0.5, 1.2, 5.0]
        ev = build_evaluator(params, values)
        assert ev.weights.material_pst == 1.5
        assert ev.weights.mobility == 2.0
        assert ev.weights.king_safety == 0.5
        assert ev.weights.pawn_structure == 1.2
        assert ev.weights.endgame_mopup == 5.0


# ---------------------------------------------------------------------------
# Gradient estimation shape
# ---------------------------------------------------------------------------

class TestGradientShape:
    """Verify that spsa_iteration returns a theta of the right shape and
    that values stay within bounds."""

    def test_iteration_returns_correct_shape(self) -> None:
        params = [
            ParamConfig("material_pst", 1.0, 0.5, 2.0),
            ParamConfig("mobility", 1.0, 0.0, 3.0),
            ParamConfig("king_safety", 1.0, 0.0, 3.0),
            ParamConfig("pawn_structure", 1.0, 0.0, 3.0),
            ParamConfig("endgame_mopup", 3.0, 0.0, 10.0),
        ]
        theta = [p.initial for p in params]
        baseline = default_evaluator()
        limits = SearchLimits(max_depth=1)
        rng = random.Random(42)

        # Use just 1 position for speed.
        new_theta = spsa_iteration(
            params, theta, baseline,
            [DEFAULT_POSITIONS[0]], limits,
            a_k=1.0, c_k=5.0, rng=rng,
        )
        assert len(new_theta) == len(theta)

    def test_iteration_respects_bounds(self) -> None:
        params = [
            ParamConfig("material_pst", 1.0, 0.5, 2.0),
            ParamConfig("mobility", 1.0, 0.0, 3.0),
            ParamConfig("king_safety", 1.0, 0.0, 3.0),
            ParamConfig("pawn_structure", 1.0, 0.0, 3.0),
            ParamConfig("endgame_mopup", 3.0, 0.0, 10.0),
        ]
        theta = [p.initial for p in params]
        baseline = default_evaluator()
        limits = SearchLimits(max_depth=1)
        rng = random.Random(7)

        new_theta = spsa_iteration(
            params, theta, baseline,
            [DEFAULT_POSITIONS[0]], limits,
            a_k=1.0, c_k=5.0, rng=rng,
        )
        for i, p in enumerate(params):
            assert p.min_val <= new_theta[i] <= p.max_val, (
                f"{p.name}: {new_theta[i]} not in [{p.min_val}, {p.max_val}]"
            )


# ---------------------------------------------------------------------------
# Single iteration smoke test
# ---------------------------------------------------------------------------

class TestSingleIteration:
    def test_single_iteration_no_error(self) -> None:
        """A single SPSA iteration at depth 1 completes without raising."""
        params = [
            ParamConfig("material_pst", 1.0, 0.5, 2.0),
            ParamConfig("mobility", 1.0, 0.0, 3.0),
            ParamConfig("king_safety", 1.0, 0.0, 3.0),
            ParamConfig("pawn_structure", 1.0, 0.0, 3.0),
            ParamConfig("endgame_mopup", 3.0, 0.0, 10.0),
        ]
        theta = [p.initial for p in params]
        baseline = default_evaluator()
        limits = SearchLimits(max_depth=1)
        rng = random.Random(0)

        new_theta = spsa_iteration(
            params, theta, baseline,
            [DEFAULT_POSITIONS[0]], limits,
            a_k=1.0, c_k=5.0, rng=rng,
        )
        # Just check it returned something valid.
        assert isinstance(new_theta, list)
        assert len(new_theta) == len(params)
