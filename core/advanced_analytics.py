"""
Mock RobustAnalyticsEngine — implements AdvancedAnalyticsProtocol.

异常检测使用 Z-Score 替代 MCD；分布偏移使用真实 scipy KS 检验。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class RobustAnalyticsEngine:
    """Mock 高级分析引擎。"""

    def robust_anomaly_detection(
        self,
        df: pd.DataFrame,
        comp_cols: List[str],
        numeric_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Z-Score 异常检测。返回 (布尔 Mask, 最大 Z-Score 数组)。"""
        numeric_data = df[numeric_cols].values
        means = np.nanmean(numeric_data, axis=0)
        stds = np.nanstd(numeric_data, axis=0)
        stds = np.where(stds == 0, 1e-12, stds)
        z_scores = np.abs((numeric_data - means) / stds)
        max_z = np.max(z_scores, axis=1)
        mask = max_z > 3.0
        return mask, max_z

    def calculate_distribution_drift(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        metrics_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        if len(df_a) < 30 or len(df_b) < 30:
            raise ValueError("两组样本量均必须 >= 30")
        result: Dict[str, Dict[str, Any]] = {}
        for col in metrics_cols:
            if col not in df_a.columns or col not in df_b.columns:
                continue
            stat, pval = ks_2samp(df_a[col].dropna(), df_b[col].dropna())
            result[col] = {"ks_stat": round(stat, 4), "p_value": round(pval, 4)}
        return result