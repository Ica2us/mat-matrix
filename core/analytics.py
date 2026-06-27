"""
core/analytics.py — RobustAnalyticsEngine
============================================

Implements AdvancedAnalyticsProtocol from contracts.py.

Outlier detection (robust_anomaly_detection)
  - Converts compositional (simplex) columns to ilr space via SBP matrix.
  - Merges with numeric columns → high-dimensional real space.
  - Uses scikit-learn's MinCovDet (FAST-MCD) for robust location / scatter.
  - Flags samples whose robust Mahalanobis distance > χ²_{dof, 0.99}.

Distribution drift (calculate_distribution_drift)
  - Rigorous guard: both groups must have ≥ 30 rows.
  - Per-column two-sample KS test with Bonferroni correction.
  - Returns per-column statistic, p-value, corrected threshold, and verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest
from sklearn.covariance import MinCovDet

from contracts import GLOBAL_RANDOM_STATE, AdvancedAnalyticsProtocol


# =====================================================================
# 内部工具：SBP 矩阵构建与 ilr 变换（纯 numpy，无外部依赖）
# =====================================================================

def _build_sbp_matrix(d: int) -> np.ndarray:
    """构建 d × (d-1) 序贯二分对等（SBP）矩阵。

    对成分列 j = 0 .. d-2：
      - 前 j+1 个位置设为 +1
      - 接下来 1 个位置设为 - (j+1)
      - 其余位置设为 0

    Args:
        d: 成分数量（单纯形维度）

    Returns:
        shape (d, d-1) 的 SBP 矩阵 Ψ
    """
    psi = np.zeros((d, d - 1))
    for j in range(d - 1):
        psi[: j + 1, j] = +1.0
        psi[j + 1, j] = -float(j + 1)
    return psi


def _ilr_transform(X: np.ndarray) -> np.ndarray:
    """等距对数比变换。

    Args:
        X: shape (n, d), 严格正值的单纯形配方矩阵（每行和为 1）

    Returns:
        shape (n, d-1) 的无约束实数矩阵
    """
    d = X.shape[1]
    psi = _build_sbp_matrix(d)
    norms = np.sqrt(np.sum(psi ** 2, axis=0))  # shape (d-1,)
    logX = np.log(X)  # shape (n, d)
    # Z = logX @ (psi / norms)   — each column of psi normalized
    psi_normed = psi / norms[np.newaxis, :]  # (d, d-1)
    Z = logX @ psi_normed  # (n, d-1)
    return Z


# =====================================================================
# RobustAnalyticsEngine
# =====================================================================

class RobustAnalyticsEngine(AdvancedAnalyticsProtocol):
    """AdvancedAnalyticsProtocol 的稳健实现。

    异常检测:
      1. 提取 comp_cols → 确定性零值替代 → ilr 变换
      2. 拼接 numeric_cols（标准化）
      3. FAST-MCD (MinCovDet) 拟合 → 稳健马氏距离
      4. χ² 阈值判定（α = 0.01）

    分布偏移:
      1. 断言两组样本量 ≥ 30
      2. 每列独立 KS 检验
      3. Bonferroni 校正 α' = 0.05 / len(metrics_cols)
    """

    def __init__(
        self,
        anomaly_alpha: float = 0.01,
        drift_alpha: float = 0.05,
        zero_eps: float = 1e-6,
        random_state: int = GLOBAL_RANDOM_STATE,
    ) -> None:
        """初始化引擎。

        Args:
            anomaly_alpha: 异常检测 χ² 阈值显著性水平（默认 0.01）
            drift_alpha:    漂移检测总体显著性水平（默认 0.05）
            zero_eps:       零值替代扰动值
            random_state:   随机种子（强制从 contracts 读取）
        """
        self.anomaly_alpha = anomaly_alpha
        self.drift_alpha = drift_alpha
        self.zero_eps = zero_eps
        self.random_state = random_state

    # -----------------------------------------------------------------
    # 零值替代
    # -----------------------------------------------------------------

    @staticmethod
    def _multiplicative_zero_replacement(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """确定性零值替代 + 行和归一化。

        对每行：将 0 替换为 eps，将非零元素等比缩放以保证行和为 1。
        """
        Xr = X.copy().astype(np.float64)
        for i in range(Xr.shape[0]):
            row = Xr[i]
            zero_mask = row <= 0.0
            if np.any(zero_mask):
                nz = zero_mask.sum()
                # 替换零为 eps
                row[zero_mask] = eps
                # 等比缩减非零元素
                row[~zero_mask] *= (1.0 - nz * eps) / row[~zero_mask].sum()
            Xr[i] = row
        return Xr

    # -----------------------------------------------------------------
    # robust_anomaly_detection
    # -----------------------------------------------------------------

    def robust_anomaly_detection(
        self,
        df: pd.DataFrame,
        comp_cols: List[str],
        numeric_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """稳健异常检测。

        流程:
          - comp_cols → 零值替代 → ilr 变换 → (n, d_comp-1)
          - numeric_cols → z-score 标准化 → (n, d_num)
          - 拼接 → (n, p)
          - FAST-MCD → 稳健马氏距离
          - χ²(p, 1-α) 阈值判定异常

        Args:
            df:          输入 DataFrame
            comp_cols:   成分（配方）列名列表
            numeric_cols: 数值指标列名列表

        Returns:
            (anomaly_mask, mahalanobis_distances)
              anomaly_mask:         shape (n,), True 表示异常
              mahalanobis_distances: shape (n,), 稳健马氏距离
        """
        n = len(df)
        if n == 0:
            raise ValueError("DataFrame 为空，无法执行异常检测")

        # --- 1. 成分数据处理：零值替代 → ilr ---
        if comp_cols:
            comp_data = df[comp_cols].values.astype(np.float64)
            comp_data = self._multiplicative_zero_replacement(comp_data, self.zero_eps)
            ilr_data = _ilr_transform(comp_data)  # (n, d_comp-1)
        else:
            ilr_data = np.empty((n, 0))

        # --- 2. 数值列 z-score 标准化 ---
        if numeric_cols:
            num_data = df[numeric_cols].values.astype(np.float64)
            mean_ = np.nanmean(num_data, axis=0)
            std_ = np.nanstd(num_data, axis=0)
            std_[std_ == 0] = 1.0  # 防除零
            num_data = (num_data - mean_) / std_
        else:
            num_data = np.empty((n, 0))

        # --- 3. 拼接 ---
        X = np.column_stack([ilr_data, num_data])  # (n, p)
        p = X.shape[1]
        if p == 0:
            raise ValueError("comp_cols 和 numeric_cols 至少应提供一个非空列表")

        if n <= p:
            raise ValueError(
                f"样本量 ({n}) 必须大于特征维度 ({p})，否则 MCD 协方差矩阵奇异"
            )

        # --- 4. FAST-MCD 稳健拟合 ---
        mcd = MinCovDet(random_state=self.random_state)
        mcd.fit(X)

        # --- 5. 稳健马氏距离 ---
        mahal = mcd.mahalanobis(X)  # shape (n,)

        # --- 6. χ² 阈值判定 ---
        threshold = chi2.ppf(1.0 - self.anomaly_alpha, df=p)
        anomaly_mask = mahal > threshold

        return anomaly_mask, mahal

    # -----------------------------------------------------------------
    # calculate_distribution_drift
    # -----------------------------------------------------------------

    def calculate_distribution_drift(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        metrics_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """数据集分布偏移校验。

        刚性防御:
          - 两组样本量均 ≥ 30，否则抛出 ValueError
          - 每列独立 KS 检验 + Bonferroni 校正

        Args:
            df_a:         参考组
            df_b:         目标组
            metrics_cols: 待校验的指标列

        Returns:
            {
              "<column>": {
                "ks_statistic": float,
                "p_value":       float,
                "bonferroni_alpha": float,
                "is_drifted":    bool,   # p_value < bonferroni_alpha
              },
              ...
            }
        """
        n_a, n_b = len(df_a), len(df_b)

        # --- 刚性断言 ---
        if n_a < 30 or n_b < 30:
            raise ValueError(
                f"分布漂移检测要求两组样本量均 ≥ 30，当前 df_a={n_a}, df_b={n_b}"
            )

        if not metrics_cols:
            return {}

        # Bonferroni 校正
        m = len(metrics_cols)
        bonf_alpha = self.drift_alpha / m

        result: Dict[str, Dict[str, Any]] = {}
        for col in metrics_cols:
            series_a = df_a[col].dropna().values.astype(np.float64)
            series_b = df_b[col].dropna().values.astype(np.float64)

            if len(series_a) == 0 or len(series_b) == 0:
                result[col] = {
                    "ks_statistic": np.nan,
                    "p_value": np.nan,
                    "bonferroni_alpha": bonf_alpha,
                    "is_drifted": True,  # 数据缺失视为漂移
                    "note": "列中存在全部 NaN，无法执行 KS 检验",
                }
                continue

            stat, pv = kstest(series_a, series_b)
            result[col] = {
                "ks_statistic": float(stat),
                "p_value": float(pv),
                "bonferroni_alpha": bonf_alpha,
                "is_drifted": bool(pv < bonf_alpha),
            }

        return result