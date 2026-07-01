"""
engine/bo.py
============
Bayesian optimisation engine for material formulation problems.

Concrete implementation of :class:`contracts.BayesianOptimizationProtocol`.
Class name MUST be ``MaterialBayesianOptimizer`` per project convention.

All random sampling reads ``contracts.GLOBAL_RANDOM_STATE`` to guarantee
deterministic reproducibility.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from contracts import GLOBAL_RANDOM_STATE, BayesianOptimizationProtocol


class MaterialBayesianOptimizer(BayesianOptimizationProtocol):
    """
    Multi-objective Bayesian optimisation engine for material formulation
    under simplex (composition) and process-parameter box constraints.

    Uses a simplified Pareto-front approach for multi-objective trade-off
    recommendation.  In a full production system this would be backed by
    TPE / GPyTorch GPR; here we provide a deterministic, numerically
    verified scaffold that respects the protocol contract.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_pareto_front(
        self, metrics_matrix: np.ndarray, target_directions: List[str]
    ) -> np.ndarray:
        """
        Compute the Pareto-front mask for a multi-objective matrix.

        Parameters
        ----------
        metrics_matrix : np.ndarray
            Shape ``(n_experiments, n_objectives)``.  Each column is one
            objective metric.
        target_directions : List[str]
            Length ``n_objectives``.  Each entry is ``"maximize"`` or
            ``"minimize"``, declaring the optimisation direction for the
            corresponding column.

        Returns
        -------
        np.ndarray
            Boolean array of length ``n_experiments``; ``True`` means the
            experiment lies on the Pareto front.
        """
        metrics = np.asarray(metrics_matrix, dtype=np.float64)
        n, m = metrics.shape

        if len(target_directions) != m:
            raise ValueError(
                f"target_directions length ({len(target_directions)}) must "
                f"match metrics_matrix columns ({m})"
            )

        # Convert every objective to "larger is better"
        transformed = metrics.copy()
        for j, direction in enumerate(target_directions):
            d = direction.strip().lower()
            if d == "maximize":
                pass  # already larger-is-better
            elif d == "minimize":
                transformed[:, j] = -transformed[:, j]
            else:
                raise ValueError(
                    f"Unsupported direction '{direction}'; expected "
                    f"'maximize' or 'minimize'"
                )

        # Standard Pareto-dominance check
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            # i dominates j iff transformed[i] >= transformed[j] in all
            # objectives AND strictly > in at least one.
            for j in range(n):
                if i == j:
                    continue
                if np.all(transformed[i] >= transformed[j]) and np.any(
                    transformed[i] > transformed[j]
                ):
                    is_pareto[j] = False

        return is_pareto

    def recommend_next_experiment(
        self,
        historical_df: pd.DataFrame,
        comp_cols: List[str],
        param_cols: List[str],
        target_cols: List[str],
        comp_bounds: Dict[str, Tuple[float, float]],
        process_bounds: Dict[str, Tuple[float, float]],
        target_directions: List[str],
    ) -> Dict[str, float]:
        """
        Recommend the next experiment's physical-space parameters.
 
        Strategy
        --------
        1. **Composition sampling** — Dirichlet(α=1) gives a perfectly uniform
           distribution over the standard simplex; rejection sampling then
           enforces per-component box constraints while preserving that
           uniformity within the feasible sub-simplex.
        2. **Process-parameter sampling** — independent uniform sampling over
           each parameter's box (mathematically correct for a hyper-rectangle).
        3. **Target prediction** — nearest-neighbour lookup in the joint
           variable space; each candidate inherits the targets of its closest
           historical point.
        4. **Pareto filter** — among all 5 000 candidates, retain those whose
           predicted targets are not dominated by any *other* candidate.
        5. **Diversity selection** — among Pareto-optimal candidates, pick the
           one farthest from any historical experiment (exploration heuristic).
           Reuses the distance matrix already computed in step 3 at zero cost.
 
        Parameters
        ----------
        historical_df : pd.DataFrame
            Past experimental data (must be non-empty).
        comp_cols : List[str]
            Column names for composition variables (constrained to sum = 1).
        param_cols : List[str]
            Column names for process-parameter variables.
        target_cols : List[str]
            Column names for objective metrics.
        comp_bounds : Dict[str, Tuple[float, float]]
            Per-component lower/upper bounds.
        process_bounds : Dict[str, Tuple[float, float]]
            Lower/upper bounds for each process parameter.
        target_directions : List[str]
            ``"maximize"`` or ``"minimize"`` for each target column.
 
        Returns
        -------
        Dict[str, float]
            Recommended next experiment as ``{column: value}``.
 
        Raises
        ------
        ValueError
            On missing bounds, infeasible constraints, empty inputs, or
            rejection-sampling failure.
        """
        # -------------------------------------------------------------------
        # 0. Setup & validation
        # -------------------------------------------------------------------
        rng = np.random.default_rng(GLOBAL_RANDOM_STATE)
 
        comp_cols_list  = list(comp_cols)
        param_cols_list = list(param_cols)
        all_var_cols    = comp_cols_list + param_cols_list
        d_comp          = len(comp_cols_list)
 
        # Guard: basic structural checks
        if not all_var_cols:
            raise ValueError(
                "At least one of comp_cols or param_cols must be non-empty."
            )
        if not target_cols:
            raise ValueError("target_cols must not be empty.")
        if len(target_directions) != len(target_cols):
            raise ValueError(
                f"target_directions length ({len(target_directions)}) must equal "
                f"target_cols length ({len(target_cols)})."
            )
        if historical_df.empty:
            raise ValueError(
                "historical_df must contain at least one experiment."
            )
 
        # Guard: every declared column must have a bound
        for col in comp_cols_list:
            if col not in comp_bounds:
                raise ValueError(
                    f"Missing bound for composition variable '{col}'. "
                    f"Provide a (lo, hi) tuple in comp_bounds."
                )
        for col in param_cols_list:
            if col not in process_bounds:
                raise ValueError(
                    f"Missing bound for process parameter '{col}'. "
                    f"Provide a (lo, hi) tuple in process_bounds."
                )
 
        n_candidates = 5_000
 
        # -------------------------------------------------------------------
        # 1. Process-parameter candidates — uniform over each box dimension
        # -------------------------------------------------------------------
        if param_cols_list:
            process_candidates: Dict[str, np.ndarray] = {}
            for col in param_cols_list:
                lo, hi = process_bounds[col]
                process_candidates[col] = rng.uniform(lo, hi, size=n_candidates)
            candidate_df_params = pd.DataFrame(
                process_candidates, index=range(n_candidates)
            )
        else:
            candidate_df_params = pd.DataFrame(index=range(n_candidates))
 
        # -------------------------------------------------------------------
        # 2. Composition candidates — Dirichlet(1) + rejection sampling
        #
        #    Dirichlet([1, …, 1]) is the uniform distribution on the standard
        #    simplex.  Accepting only samples that satisfy every per-component
        #    bound preserves uniformity on the feasible sub-simplex (the
        #    conditional of a uniform distribution is still uniform).
        #
        #    Safety fuse: if the feasible region has near-zero measure the
        #    loop raises ValueError after MAX_ATTEMPTS doublings of chunk_size,
        #    preventing an infinite hang.
        # -------------------------------------------------------------------
        if d_comp > 0:
            # ── Pre-flight feasibility check (catches the obvious impossible
            #    cases before touching the RNG at all) ──────────────────────
            lower_sum = sum(comp_bounds[c][0] for c in comp_cols_list)
            upper_sum = sum(comp_bounds[c][1] for c in comp_cols_list)
 
            if lower_sum > 1.0 + 1e-9:
                raise ValueError(
                    f"Infeasible comp_bounds: lower bounds sum to {lower_sum:.6f} > 1.0. "
                    f"The simplex constraint (∑components = 1) cannot be satisfied."
                )
            if upper_sum < 1.0 - 1e-9:
                raise ValueError(
                    f"Infeasible comp_bounds: upper bounds sum to {upper_sum:.6f} < 1.0. "
                    f"The simplex constraint (∑components = 1) cannot be satisfied."
                )
 
            MAX_ATTEMPTS  = 30
            CHUNK_CAP     = 200_000
            valid_comps: list[np.ndarray] = []
            chunk_size    = n_candidates * 5   # generous initial batch
            attempt       = 0
            total_sampled = 0
 
            while len(valid_comps) < n_candidates:
                attempt += 1
                if attempt > MAX_ATTEMPTS:
                    pass_rate = len(valid_comps) / max(total_sampled, 1)
                    raise ValueError(
                        f"Dirichlet rejection sampling failed after {MAX_ATTEMPTS} attempts "
                        f"(estimated pass-rate ≈ {pass_rate:.4%}). "
                        f"comp_bounds likely define a near-empty feasible sub-simplex.\n"
                        f"  lower bounds sum = {lower_sum:.4f}\n"
                        f"  upper bounds sum = {upper_sum:.4f}"
                    )
 
                sampled_comps  = rng.dirichlet(np.ones(d_comp), size=chunk_size)
                total_sampled += chunk_size
 
                # Vectorised per-component feasibility check
                is_valid = np.ones(chunk_size, dtype=bool)
                for idx, col in enumerate(comp_cols_list):
                    lo, hi  = comp_bounds[col]
                    is_valid &= (sampled_comps[:, idx] >= lo) & (sampled_comps[:, idx] <= hi)
 
                valid_comps.extend(sampled_comps[is_valid])
 
                # Adaptively double chunk (capped) to maintain throughput when
                # the feasible sub-simplex is small
                if len(valid_comps) < n_candidates:
                    chunk_size = min(chunk_size * 2, CHUNK_CAP)
 
            valid_comps_arr   = np.array(valid_comps[:n_candidates])   # exact capacity
            candidate_df_comp = pd.DataFrame(valid_comps_arr, columns=comp_cols_list)
 
        else:
            candidate_df_comp = pd.DataFrame(index=range(n_candidates))
 
        # Assemble unified candidate pool, preserving all_var_cols column order
        candidate_df = pd.concat(
            [candidate_df_comp, candidate_df_params], axis=1
        )[all_var_cols]
 
        # -------------------------------------------------------------------
        # 3. Nearest-neighbour target prediction (fully vectorised)
        #
        # Memory analysis — peak holds TWO (N, H, d) float64 tensors:
        #   • the diff tensor  (cand - hist)
        #   • the squared diff tensor  diff²
        #
        #   Peak bytes = 2 × N × H × d × 8
        #
        #   N=5000, H=100,  d=10  →   ~80 MB  ✅ fast path
        #   N=5000, H=1000, d=20  →  ~1.6 GB  ⚠ chunked path activates
        #   N=5000, H=5000, d=20  →  ~8.0 GB  ❌ chunked path activates
        # -------------------------------------------------------------------
        hist_targets = historical_df[target_cols].values    # (H, M)
        hist_vars    = historical_df[all_var_cols].values   # (H, d)
        cand_vars    = candidate_df.values                  # (N, d)
 
        H = hist_vars.shape[0]
        d = hist_vars.shape[1]
        N = cand_vars.shape[0]
 
        BYTES_FLOAT64      = 8
        MEMORY_LIMIT_BYTES = 400 * 1024 * 1024   # 400 MB
        # Factor of 2: diff + diff² both reside in memory simultaneously
        peak_bytes = 2 * N * H * d * BYTES_FLOAT64
 
        if peak_bytes <= MEMORY_LIMIT_BYTES:
            # ── Fast path: single vectorised kernel ────────────────────────
            dists = np.sqrt(
                np.sum(
                    (cand_vars[:, np.newaxis, :] - hist_vars[np.newaxis, :, :]) ** 2,
                    axis=2,
                )
            )   # (N, H)
        else:
            # ── Chunked path: slice candidates to stay under memory limit ──
            # Divide limit equally between diff and diff²
            safe_per_tensor = MEMORY_LIMIT_BYTES // 2
            cand_chunk      = max(1, safe_per_tensor // (H * d * BYTES_FLOAT64))
            dists           = np.empty((N, H), dtype=np.float64)
 
            for start in range(0, N, cand_chunk):
                end   = min(start + cand_chunk, N)
                chunk = cand_vars[start:end]            # (chunk, d)
                dists[start:end] = np.sqrt(
                    np.sum(
                        (chunk[:, np.newaxis, :] - hist_vars[np.newaxis, :, :]) ** 2,
                        axis=2,
                    )
                )
 
        nn_indices   = np.argmin(dists, axis=1)    # (N,) — closest history row per candidate
        pred_targets = hist_targets[nn_indices]    # (N, M) — inherited predicted targets
 
        # -------------------------------------------------------------------
        # 4. Pareto filter: retain candidates not dominated by any other candidate
        # -------------------------------------------------------------------
        pareto_mask    = self.calculate_pareto_front(pred_targets, target_directions)
        pareto_indices = np.where(pareto_mask)[0]
 
        if len(pareto_indices) == 0:
            # Fallback: optimise primary objective only
            direction = target_directions[0].strip().lower()
            best_idx  = (
                int(np.argmax(pred_targets[:, 0]))
                if direction == "maximize"
                else int(np.argmin(pred_targets[:, 0]))
            )
            return {col: float(candidate_df[col].iloc[best_idx]) for col in all_var_cols}
 
        # -------------------------------------------------------------------
        # 5. Diversity selection
        #
        # Reuses the dists matrix from step 3 — no extra distance call needed.
        # min_dists_to_hist[i] = distance from candidate i to its nearest
        # historical experiment.  We pick the Pareto candidate that maximises
        # this, steering exploration toward under-sampled regions.
        # -------------------------------------------------------------------
        min_dists_to_hist  = np.min(dists, axis=1)             # (N,)
        pareto_min_dists   = min_dists_to_hist[pareto_indices]  # (|Pareto|,)
        best_candidate_idx = pareto_indices[np.argmax(pareto_min_dists)]
 
        return {
            col: float(candidate_df[col].iloc[best_candidate_idx])
            for col in all_var_cols
        }
 