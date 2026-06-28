# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`mat_matrix_analytics` implements a **protocol-driven analytics track** for a materials-informatics platform. The codebase is part of a multi-repo monorepo (`mat_matrix`, `mat_matrix_analytics`, `mat_matrix_math`, `mat_matrix_gui`, `mat_matrix_vcs`), each owning a separate domain with no cross-imports — only `contracts.py` is shared by symlink/copy.

The core engine (`core/analytics.py`) provides three public services:

1. **Robust anomaly detection** — `robust_anomaly_detection()`: ingests compositional (simplex) + numeric columns → ilr transform → scikit-learn `MinCovDet` (FAST-MCD) → robust Mahalanobis distance → χ² threshold.
2. **Distribution drift detection** — `calculate_distribution_drift()`: two-sample KS test per column with Bonferroni correction; both groups must have ≥ 30 rows.
3. **Anomaly explainer** — `anomaly_explain()`: feature-level contribution to Mahalanobis distance via ablation in ilr/numeric space.
4. **Sliding window drift monitor** — `sliding_window_drift_monitor()`: batch-level sequential drift detection with automatic window rolling.

All randomness is seeded from `contracts.GLOBAL_RANDOM_STATE = 42`.

## Code architecture

```
mat_matrix_analytics/
├── contracts.py              # Protocol definitions (shared contract layer)
├── core/
│   └── analytics.py          # RobustAnalyticsEngine (only implementation)
├── tests/
│   ├── contract_sanity.py    # Protocol structural sanity check
│   ├── test_robust_analytics.py    # 25 tests: anomaly, drift, explainer
│   └── test_sliding_window_drift.py # 8 tests: sliding window monitor
└── requirements.txt
```

The protocol hierarchy (`contracts.py`) defines 5 tracks: `CompositionalMathProtocol`, `BayesianOptimizationProtocol`, `StorageEngineProtocol`, `VersionControlSystemProtocol`, `AdvancedAnalyticsProtocol`, `SpectralFeatureProtocol`. Each track is implemented in a separate package — this repo only owns the analytics track.

## Key design constraints

- **No third-party stats libraries beyond scikit-learn + scipy.** FAST-MCD must come from `sklearn.covariance.MinCovDet`.
- **Compositional data** is always converted to ilr space (isometric log-ratio) via a built-in SBP matrix before MCD fitting.
- **n ≥ 30 guard** on drift detection — `ValueError` if either group is smaller.
- **Bonferroni correction** applied as `drift_alpha / len(metrics_cols)`.
- **Random seed** is hard-bound to `contracts.GLOBAL_RANDOM_STATE`.
- **mypy --strict** compliance required on `core/analytics.py`.

## Common commands

```bash
# Run all tests
python -X utf8 -m pytest tests/ -v --tb=short

# Run a single test file
python -X utf8 -m pytest tests/test_robust_analytics.py -v

# Run a single test class
python -X utf8 -m pytest tests/test_robust_analytics.py::TestAnomalyExplainer -v

# Run a single test
python -X utf8 -m pytest tests/test_robust_analytics.py::TestAnomalyExplainer::test_explain_anomalous_samples -v

# Type check (mypy --strict)
python -X utf8 -m mypy --strict core/analytics.py

# Contract sanity check
python -X utf8 -c "exec(open('tests/contract_sanity.py').read()); verify_all_protocols()"

# Run all tests and type check in sequence
python -X utf8 -m pytest tests/ -q --tb=short && python -X utf8 -m mypy --strict core/analytics.py
```

The `Python -X utf8` flag is required on Windows to avoid `gbk` encoding errors with Chinese docstrings.