"""
tests/test_bo.py
================
Unit tests for engine/bo.py — MaterialBayesianOptimizer.

Covers:
  - Pareto front calculation (2D and 3D, mixed directions)
  - Edge cases: single point, all-dominated, all-Pareto
  - recommend_next_experiment returns valid dict with correct keys
  - Determinism via GLOBAL_RANDOM_STATE
"""

import numpy as np
import pandas as pd
import pytest

from contracts import GLOBAL_RANDOM_STATE
from engine.bo import MaterialBayesianOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def optimizer() -> MaterialBayesianOptimizer:
    return MaterialBayesianOptimizer()


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestCalculateParetoFront:
    def test_two_objectives_maximize(self, optimizer: MaterialBayesianOptimizer):
        """
        Simple 2-objective (both maximize) Pareto front.

        Points:  A(1, 5), B(2, 4), C(3, 3), D(4, 2), E(5, 1)
        All are Pareto-optimal — each is better in one objective.
        """
        metrics = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]], dtype=np.float64)
        mask = optimizer.calculate_pareto_front(metrics, ["maximize", "maximize"])
        assert np.all(mask)  # all are Pareto

    def test_two_objectives_mixed(self, optimizer: MaterialBayesianOptimizer):
        """
        Objective 0: maximize, Objective 1: minimize.

        Points:  A(1, 10), B(2, 5), C(3, 1), D(4, 8)
        After transform (negate minimize col):
          A(1, -10), B(2, -5), C(3, -1), D(4, -8)
        Pareto (larger-is-better for both):
          A dominated by B (2>=1, -5>=-10); A dominated by C (3>=1, -1>=-10)
          B dominated by C (3>=2, -1>=-5)
          C: best obj1 (maximized = minimized original)
          D(4,-8): 4>3, -8<-1 so C does not dominate D; 4>2, -8<-5 so B does not;
                   4>1, -8<-10? No. D is not dominated by anyone — D IS Pareto.
        Final Pareto: A(no), B(no), C(yes), D(yes)
        """
        metrics = np.array([[1, 10], [2, 5], [3, 1], [4, 8]], dtype=np.float64)
        mask = optimizer.calculate_pareto_front(metrics, ["maximize", "minimize"])
        expected = np.array([False, False, True, True])
        assert np.array_equal(mask, expected)

    def test_all_dominated(self, optimizer: MaterialBayesianOptimizer):
        """Single Pareto point dominates all others."""
        metrics = np.array([[5, 5], [1, 1], [2, 2], [3, 3]], dtype=np.float64)
        mask = optimizer.calculate_pareto_front(metrics, ["maximize", "maximize"])
        expected = np.array([True, False, False, False])
        assert np.array_equal(mask, expected)

    def test_single_point(self, optimizer: MaterialBayesianOptimizer):
        """Single point is trivially Pareto-optimal."""
        metrics = np.array([[3.14, 2.72]], dtype=np.float64)
        mask = optimizer.calculate_pareto_front(metrics, ["maximize", "minimize"])
        assert mask[0]

    def test_three_objectives(self, optimizer: MaterialBayesianOptimizer):
        """3-objective Pareto front."""
        metrics = np.array(
            [
                [1, 1, 1],  # dominated by A(2,2,2)
                [2, 2, 2],  # Pareto
                [3, 1, 1],  # Pareto (best in obj0)
                [1, 3, 1],  # Pareto (best in obj1)
                [1, 1, 3],  # Pareto (best in obj2)
                [2, 2, 1],  # dominated by A(2,2,2)
            ],
            dtype=np.float64,
        )
        mask = optimizer.calculate_pareto_front(
            metrics, ["maximize", "maximize", "maximize"]
        )
        expected = np.array([False, True, True, True, True, False])
        assert np.array_equal(mask, expected)

    def test_invalid_direction(self, optimizer: MaterialBayesianOptimizer):
        with pytest.raises(ValueError, match="Unsupported direction"):
            optimizer.calculate_pareto_front(
                np.array([[1, 2]]), ["maximize", "invalid"]
            )

    def test_direction_count_mismatch(self, optimizer: MaterialBayesianOptimizer):
        with pytest.raises(ValueError, match="target_directions length"):
            optimizer.calculate_pareto_front(
                np.array([[1, 2, 3]]), ["maximize", "maximize"]
            )


# ---------------------------------------------------------------------------
# recommend_next_experiment
# ---------------------------------------------------------------------------

class TestRecommendNextExperiment:
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "A": [0.2, 0.3, 0.4],
            "B": [0.3, 0.4, 0.5],
            "C": [0.5, 0.3, 0.1],
            "temp": [100, 200, 150],
            "yield": [80, 85, 90],
            "cost": [15, 12, 10],
        })

    def test_returns_dict_with_correct_keys(
        self, optimizer: MaterialBayesianOptimizer, sample_data: pd.DataFrame
    ):
        result = optimizer.recommend_next_experiment(
            historical_df=sample_data,
            comp_cols=["A", "B", "C"],
            param_cols=["temp"],
            target_cols=["yield", "cost"],
            comp_bounds={"A": (0.0, 1.0), "B": (0.0, 1.0), "C": (0.0, 1.0)},
            process_bounds={"temp": (50, 300)},
            target_directions=["maximize", "minimize"],
        )
        expected_keys = {"A", "B", "C", "temp"}
        assert set(result.keys()) == expected_keys
        for k, v in result.items():
            assert isinstance(v, float)

    def test_missing_bound_raises(
        self, optimizer: MaterialBayesianOptimizer, sample_data: pd.DataFrame
    ):
        with pytest.raises(ValueError, match="Missing bound"):
            optimizer.recommend_next_experiment(
                historical_df=sample_data,
                comp_cols=["A", "B", "C"],
                param_cols=["temp"],
                target_cols=["yield", "cost"],
                comp_bounds={"A": (0.0, 1.0), "B": (0.0, 1.0)},
                process_bounds={"temp": (50, 300)},
                target_directions=["maximize", "minimize"],
            )

    def test_deterministic(
        self, optimizer: MaterialBayesianOptimizer, sample_data: pd.DataFrame
    ):
        """Same inputs → same recommendation."""
        kwargs = dict(
            historical_df=sample_data,
            comp_cols=["A", "B", "C"],
            param_cols=["temp"],
            target_cols=["yield", "cost"],
            comp_bounds={"A": (0.0, 1.0), "B": (0.0, 1.0), "C": (0.0, 1.0)},
            process_bounds={"temp": (50, 300)},
            target_directions=["maximize", "minimize"],
        )
        r1 = optimizer.recommend_next_experiment(**kwargs)
        r2 = optimizer.recommend_next_experiment(**kwargs)
        for k in r1:
            assert r1[k] == pytest.approx(r2[k], abs=1e-12)