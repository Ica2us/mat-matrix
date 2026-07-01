"""
core/analytics.py — RobustAnalyticsEngine
============================================

Implements AdvancedAnalyticsProtocol from contracts.py.

Outlier detection (robust_anomaly_detection)
  - Converts compositional (simplex) columns to ilr space via SBP matrix.
  - Merges with numeric columns → high-dimensional real space.
  - Uses scikit-learn's MinCovDet (FAST-MCD) for robust location / scatter.
  - Flags samples whose robust squared Mahalanobis distance > χ²_{dof, 0.99}.

Distribution drift (calculate_distribution_drift)
  - Rigorous guard: both groups must have ≥ 30 rows.
  - Per-column two-sample KS test with Bonferroni correction.
  - Returns per-column statistic, p-value, corrected threshold, and verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import chi2, ks_2samp
from sklearn.covariance import MinCovDet  # type: ignore[import-untyped]

from contracts import AdvancedAnalyticsProtocol
from core.math_space import CompositionalMathTransformer, _helmert_contrast_matrix


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
        random_state: int = 42,
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
        self._math = CompositionalMathTransformer()

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
          - FAST-MCD → 稳健（平方）马氏距离
          - χ²(p, 1-α) 阈值判定异常

        Args:
            df:          输入 DataFrame
            comp_cols:   成分（配方）列名列表
            numeric_cols: 数值指标列名列表

        Returns:
            (anomaly_mask, squared_mahalanobis)
              anomaly_mask:       shape (n,), True 表示异常
              squared_mahalanobis: shape (n,), 稳健平方马氏距离 (MD²)
        """
        n = len(df)
        if n == 0:
            raise ValueError("DataFrame 为空，无法执行异常检测")

        # --- 1. 成分数据处理：零值替代 → ilr ---
        if comp_cols:
            comp_data = df[comp_cols].values.astype(np.float64)
            comp_data = self._math.multiplicative_zero_replacement(comp_data, self.zero_eps)
            ilr_data = self._math.ilr_transform(comp_data)  # (n, d_comp-1)
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

        # --- 5. 稳健平方马氏距离 (MinCovDet.mahalanobis 返回 MD²) ---
        mahal_sq = mcd.mahalanobis(X)  # shape (n,)

        # --- 6. χ² 阈值判定（χ² 分布的正是平方距离）---
        threshold = chi2.ppf(1.0 - self.anomaly_alpha, df=p)
        anomaly_mask = mahal_sq > threshold

        return anomaly_mask, mahal_sq

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

            stat, pv = ks_2samp(series_a.tolist(), series_b.tolist())
            stat_f = float(stat)  # type: ignore[arg-type]
            pv_f = float(pv)  # type: ignore[arg-type]
            result[col] = {
                "ks_statistic": stat_f,
                "p_value": pv_f,
                "bonferroni_alpha": bonf_alpha,
                "is_drifted": bool(pv_f < bonf_alpha),
            }

        return result

    # -----------------------------------------------------------------
    # anomaly_explain
    # -----------------------------------------------------------------

    def anomaly_explain(
        self,
        df: pd.DataFrame,
        comp_cols: List[str],
        numeric_cols: List[str],
        top_k: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """对 robust_anomaly_detection 标记的异常样本，反向解释各特征贡献度。

        核心思想：
          马氏距离 MD² = Z Sigma^-1 Z^T（Z 为标准化后的 MCD 空间向量）。
          将 MD² 分解为各原始特征的边际贡献：
            - 对每个特征 j，计算将该特征固定（置为 MCD 中心）后的 MD²_ablated(j)
            - 贡献度 Delta(j) = MD² - MD²_ablated(j)，归一化为百分比

        注：
          - comp_cols 和 numeric_cols 输入必须与 robust_anomaly_detection 一致。
          - 当异常样本过多时，解释计算会逐样本消融，效率 O(K * p * p')。
          - 返回字典仅包含 anomaly_mask 为 True 的行索引。
          返回原始特征名（comp_cols 按原始成分名，numeric_cols 原名）。

        Args:
            df:          输入 DataFrame
            comp_cols:   成分（配方）列名列表
            numeric_cols: 数值指标列名列表
            top_k:       可选，只返回贡献度最高的前 K 个特征（默认全部）

        Returns:
            {
              <row_index>: {
                "squared_mahalanobis": float,               # 该样本的平方马氏距离 (MD²)
                "is_anomaly":   bool,               # 是否异常
                "contributions": {
                  <feature_name>: float,             # 归一化贡献度百分比 (0-100)
                },
                "top_features": [str, ...],          # 按贡献度降序排列
              },
              ...  # 仅包含 anomaly_mask 为 True 的行
            }
        """
        n = len(df)
        if n == 0:
            raise ValueError("DataFrame 为空，无法执行异常解释")

        # --- 1. 复现 MCD 拟合管线（提取内部状态） ---
        if comp_cols:
            comp_data = df[comp_cols].values.astype(np.float64)
            comp_data = self._math.multiplicative_zero_replacement(comp_data, self.zero_eps)
            ilr_data = self._math.ilr_transform(comp_data)
        else:
            ilr_data = np.empty((n, 0))

        if numeric_cols:
            num_data = df[numeric_cols].values.astype(np.float64)
            num_mean = np.nanmean(num_data, axis=0)
            num_std = np.nanstd(num_data, axis=0)
            num_std[num_std == 0] = 1.0
            num_data_z = (num_data - num_mean) / num_std
        else:
            num_data_z = np.empty((n, 0))

        X = np.column_stack([ilr_data, num_data_z])
        p = X.shape[1]
        if p == 0:
            raise ValueError("comp_cols 和 numeric_cols 至少应提供一个非空列表")
        if n <= p:
            raise ValueError(f"样本量 ({n}) 必须大于特征维度 ({p})")

        mcd = MinCovDet(random_state=self.random_state)
        mcd.fit(X)

        mahal_sq = mcd.mahalanobis(X)
        threshold = chi2.ppf(1.0 - self.anomaly_alpha, df=p)
        anomaly_mask = mahal_sq > threshold

        result: Dict[int, Dict[str, Any]] = {}
        anomalous_indices = np.where(anomaly_mask)[0]
        if len(anomalous_indices) == 0:
            return result

        # --- 2. 提取 MCD 参数用于逐样本消融 ---
        center = mcd.location_  # shape (p,)
        cov_inv = np.linalg.inv(mcd.covariance_)  # shape (p, p)

        # 构建 ilr 到 comp 的权重映射（通过 Helmert 对比矩阵）：
        #   每个 ilr_j 是所有成分的加权和
        #   我们反过来计算每个原始成分对 ilr 坐标的敏感度
        if comp_cols:
            d = len(comp_cols)
            H = _helmert_contrast_matrix(d)
            contrast = H[1:, :]  # (d-1, d)
            # contrast 的行是归一化正交基，行范数为 1
            ilr_weight: Optional[np.ndarray] = contrast.T  # (d, d-1)，每列是成分到该 ilr 坐标的权重
        else:
            ilr_weight = None

        for idx in anomalous_indices:
            x = X[idx]  # shape (p,)

            # MD² = (x-μ) Σ⁻¹ (x-μ)ᵀ
            delta = x - center
            md_sq = float(delta @ cov_inv @ delta)

            # 逐特征消融：将该特征固定在 MCD 中心，重新计算 MD²
            contributions: Dict[str, float] = {}

            # --- numeric 特征：直接逐列消融 ---
            num_offset = ilr_data.shape[1]  # ilr 占据前 p_ilr 列
            for j, name in enumerate(numeric_cols):
                col_idx = num_offset + j
                delta_ablated = delta.copy()
                delta_ablated[col_idx] = 0.0  # 置为该特征在 MCD 中心 → 零偏移
                md_sq_ablated = float(delta_ablated @ cov_inv @ delta_ablated)
                contrib = md_sq - md_sq_ablated
                contributions[name] = max(contrib, 0.0)

            # --- comp 特征：通过 ilr 权重间接贡献 ---
            if comp_cols and ilr_weight is not None:
                # 每个成分 i 的变化通过 ilr_weight[i, :] 传播到所有 ilr 坐标
                # 消融成分 i = 将 delta 在 ilr 子空间沿方向 ilr_weight[i,:] 置零
                for i, name in enumerate(comp_cols):
                    w = ilr_weight[i, :]  # shape (d-1,)
                    w_norm_sq = float(w @ w)
                    if w_norm_sq < 1e-15:
                        contributions[name] = 0.0
                        continue
                    # 计算 delta_ilr 在 w 方向上的投影量
                    delta_ilr = delta[: num_offset]  # ilr 部分的偏移
                    proj = float(delta_ilr @ w) / w_norm_sq
                    # 消融：从 delta_ilr 中减去 w 方向分量
                    delta_ablated = delta.copy()
                    delta_ablated[:num_offset] = delta_ilr - proj * w
                    md_sq_ablated = float(delta_ablated @ cov_inv @ delta_ablated)
                    contrib = md_sq - md_sq_ablated
                    contributions[name] = max(contrib, 0.0)

            # --- 归一化到百分比 ---
            total = sum(contributions.values())
            if total > 0:
                contributions = {k: (v / total) * 100.0 for k, v in contributions.items()}
            else:
                contributions = {k: 0.0 for k in contributions}

            # 按贡献度降序排列
            sorted_features = sorted(contributions, key=contributions.__getitem__, reverse=True)
            if top_k is not None:
                sorted_features = sorted_features[:top_k]

            result[int(idx)] = {
                "squared_mahalanobis": float(mahal_sq[idx]),  # MD²
                "is_anomaly": True,
                "contributions": contributions,
                "top_features": sorted_features,
            }

        return result

    # -----------------------------------------------------------------
    # sliding_window_drift_monitor
    # -----------------------------------------------------------------

    def sliding_window_drift_monitor(
        self,
        reference_df: pd.DataFrame,
        target_batches: List[pd.DataFrame],
        metrics_cols: List[str],
        window_size: int = 50,
        drift_axis: int = 0,
        return_details: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """滑动窗口分布漂移监控。

        在时序/批次场景下，持续监控新到达批次相对于滑动窗口参考分布是否发生漂移。

        核心设计:
          1. 初始窗口 = reference_df（必须 >= 30 行）。
          2. 每到达一个 target_batch，若 len(target_batch) >= 30，
             用 calculate_distribution_drift 检测该批次与当前窗口的分布差异。
          3. Bonferroni 校正应用于 metrics_cols。
          4. 漂移标记：若任何一列 is_drifted = True，该批次视为漂移批次。
          5. 若漂移发生则将窗口滚动到当前批次；否则并入窗口，
             超过 window_size 时丢弃最旧行。

        Args:
            reference_df:   参考窗口 DataFrame（必须 >= 30 行）
            target_batches: 一个或多个待检测的目标批次
            metrics_cols:   待监控的指标列
            window_size:    滑动窗口最大行数（默认 50）
            drift_axis:     0 返回聚合摘要 Dict；1 返回逐批次 List
            return_details: 是否在逐批次结果中保留完整的列级漂移明细

        Returns:
            若 drift_axis=0:
              { "total_batches": int, "drifted_batches": int,
                "drifted_columns": Dict[str,int], "cumulative_alarm": bool,
                "batch_summaries": [...] }
            若 drift_axis=1:
              [ { "batch_index": int, "any_drifted": bool,
                  "drifted_cols": [str], "details": {...} }, ... ]
        """
        if len(reference_df) < 30:
            raise ValueError(
                f"参考窗口至少需要 30 行，当前 {len(reference_df)} 行"
            )

        if not isinstance(target_batches, list):
            target_batches = [target_batches]

        window = reference_df.copy()
        results_axis1: List[Dict[str, Any]] = []
        drifted_batches = 0
        drifted_columns_counter: Dict[str, int] = {}

        for batch_idx, batch in enumerate(target_batches):
            if len(batch) < 30:
                summary: Dict[str, Any] = {
                    "batch_index": batch_idx,
                    "any_drifted": None,
                    "drifted_cols": [],
                    "note": f"批次 {batch_idx} 样本量 {len(batch)} < 30，跳过检测",
                }
                if return_details:
                    summary["details"] = {}
                if drift_axis == 1:
                    results_axis1.append(summary)
                continue

            drift_result = self.calculate_distribution_drift(
                window, batch, metrics_cols
            )

            drifted_cols = [
                col for col, v in drift_result.items()
                if v.get("is_drifted", False)
            ]
            any_drifted = len(drifted_cols) > 0

            if any_drifted:
                drifted_batches += 1
                for col in drifted_cols:
                    drifted_columns_counter[col] = (
                        drifted_columns_counter.get(col, 0) + 1
                    )
                window = batch.copy()
            else:
                window = pd.concat([window, batch], ignore_index=True)
                if len(window) > window_size:
                    window = window.iloc[-window_size:].reset_index(drop=True)

            summary = {
                "batch_index": batch_idx,
                "any_drifted": any_drifted,
                "drifted_cols": drifted_cols,
            }
            if return_details:
                summary["details"] = drift_result

            results_axis1.append(summary)

        if drift_axis == 0:
            return {
                "total_batches": len(
                    [b for b in target_batches if len(b) >= 30]
                ),
                "drifted_batches": drifted_batches,
                "drifted_columns": drifted_columns_counter,
                "cumulative_alarm": drifted_batches > 0,
                "batch_summaries": results_axis1,
            }
        else:
            return results_axis1