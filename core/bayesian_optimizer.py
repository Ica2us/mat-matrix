"""
Mock TPEBasedOptimizer — implements BayesianOptimizationProtocol.

Pareto 前沿使用简化的支配度评分；下一实验推荐返回边界内的随机点。
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from contracts import GLOBAL_RANDOM_STATE


class TPEBasedOptimizer:
    """Mock TPE 贝叶斯优化器。"""

    def __init__(self) -> None:
        self._rng = random.Random(GLOBAL_RANDOM_STATE)
        self._np_rng = np.random.default_rng(GLOBAL_RANDOM_STATE)

    def calculate_pareto_front(
        self,
        metrics_matrix: np.ndarray,
        target_directions: List[str],
    ) -> np.ndarray:
        """简化 Pareto: 取各目标归一化后按方向加权评分，取前 20%。"""
        n = metrics_matrix.shape[0]
        if n == 0:
            return np.array([], dtype=bool)

        normalized = (metrics_matrix - metrics_matrix.min(axis=0)) / (
            metrics_matrix.max(axis=0) - metrics_matrix.min(axis=0) + 1e-12
        )
        signs = np.array([-1 if d == "maximize" else 1 for d in target_directions])
        scores = normalized @ signs
        threshold = np.percentile(scores, 80)
        return scores >= threshold

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
        """返回边界内随机推荐。"""
        recommendation: Dict[str, float] = {}
        for col, (lo, hi) in comp_bounds.items():
            recommendation[col] = round(self._rng.uniform(lo, hi), 4)
        for col, (lo, hi) in process_bounds.items():
            recommendation[col] = round(self._rng.uniform(lo, hi), 2)
        return recommendation