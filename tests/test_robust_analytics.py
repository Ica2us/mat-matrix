"""
tests/test_robust_analytics.py
===============================

Unit tests for RobustAnalyticsEngine.

Coverage:
  1. 异常检测：正常数据、含离群点数据、空列边界
  2. 分布漂移：漂移检测、无漂移检测、样本量不足防御
  3. 含显著批次漂移的矩阵 → 验证能精准揪出发生偏移的特征列
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 确保 contracts 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analytics import RobustAnalyticsEngine


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def engine() -> RobustAnalyticsEngine:
    return RobustAnalyticsEngine(random_state=42)


@pytest.fixture
def clean_df() -> "pd.DataFrame":
    """60 条正常配方 + 数值数据，无显式离群点"""
    import pandas as pd
    np.random.seed(42)
    # 使用均匀 Dirichlet 避免近零值导致的 MCD 敏感
    comp = np.random.dirichlet([10, 10, 10, 10], size=60)
    num = np.random.randn(60, 3)
    return pd.DataFrame(
        np.column_stack([comp, num]),
        columns=["c1", "c2", "c3", "c4", "n1", "n2", "n3"],
    )


@pytest.fixture
def contaminated_df() -> "pd.DataFrame":
    """60 条数据，其中 5 条显式离群"""
    import pandas as pd
    np.random.seed(42)
    comp = np.random.dirichlet([10, 10, 10], size=60)
    num = np.random.randn(60, 2)
    # 将后 5 条拉成极端值
    comp[-5:] = [0.01, 0.98, 0.01]
    num[-5:, :] = 10.0
    return pd.DataFrame(
        np.column_stack([comp, num]),
        columns=["c1", "c2", "c3", "n1", "n2"],
    )


# =====================================================================
# 1. 异常检测测试
# =====================================================================

class TestRobustAnomalyDetection:
    """验证 robust_anomaly_detection 的各种场景"""

    def test_clean_data_false_positive_rate_within_bound(self, engine: RobustAnalyticsEngine, clean_df: "pd.DataFrame") -> None:
        """正常数据：假阳性率 ≤ 15% (α=0.01 的保守容忍边界，MCD 小样本校正)"""
        mask, dist = engine.robust_anomaly_detection(
            clean_df, comp_cols=["c1", "c2", "c3", "c4"], numeric_cols=["n1", "n2", "n3"],
        )
        assert isinstance(mask, np.ndarray)
        assert isinstance(dist, np.ndarray)
        assert mask.shape == (60,)
        assert dist.shape == (60,)
        fp_rate = mask.sum() / 60
        assert fp_rate <= 0.15, f"False positive rate {fp_rate:.3f} exceeds 15%"

    def test_contaminated_data_catches_outliers(
        self, engine: RobustAnalyticsEngine, contaminated_df: "pd.DataFrame",
    ) -> None:
        """含 5 个显式离群点的数据应检测出至少 4 个"""
        mask, dist = engine.robust_anomaly_detection(
            contaminated_df, comp_cols=["c1", "c2", "c3"], numeric_cols=["n1", "n2"],
        )
        # 后 5 条被标记
        detected = mask[-5:].sum()
        assert detected >= 4, f"Expected ≥ 4/5 outliers, got {detected}/5"

    def test_only_compositional_cols(self, engine: RobustAnalyticsEngine, clean_df: "pd.DataFrame") -> None:
        """仅使用成分列也应正常工作（假阳性率 ≤ 15%）"""
        mask, dist = engine.robust_anomaly_detection(
            clean_df, comp_cols=["c1", "c2", "c3", "c4"], numeric_cols=[],
        )
        assert mask.shape == (60,)
        fp_rate = mask.sum() / 60
        assert fp_rate <= 0.15, f"False positive rate {fp_rate:.3f} exceeds 15%"

    def test_only_numeric_cols(self, engine: RobustAnalyticsEngine, clean_df: "pd.DataFrame") -> None:
        """仅使用数值列（无 ilr 变换）也应正常工作"""
        mask, dist = engine.robust_anomaly_detection(
            clean_df, comp_cols=[], numeric_cols=["n1", "n2", "n3"],
        )
        assert mask.shape == (60,)

    def test_returns_tuple_with_mahalanobis_distances(
        self, engine: RobustAnalyticsEngine, clean_df: "pd.DataFrame",
    ) -> None:
        """验证返回值为 (布尔 Mask, 马氏距离) 双元组"""
        result = engine.robust_anomaly_detection(
            clean_df, comp_cols=["c1", "c2", "c3", "c4"], numeric_cols=["n1"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        mask, dist = result
        assert mask.dtype == np.bool_
        assert np.all(dist >= 0)

    def test_empty_dataframe_raises(self, engine: RobustAnalyticsEngine) -> None:
        """空 DataFrame 应抛出 ValueError"""
        import pandas as pd
        with pytest.raises(ValueError, match="空"):
            engine.robust_anomaly_detection(
                pd.DataFrame(), comp_cols=["a"], numeric_cols=["b"],
            )

    def test_too_few_samples_raises(self, engine: RobustAnalyticsEngine) -> None:
        """样本量 ≤ 特征数应抛出 ValueError"""
        import pandas as pd
        df = pd.DataFrame({"c1": [0.2, 0.8], "c2": [0.8, 0.2], "n1": [0.0, 1.0]})
        with pytest.raises(ValueError, match="样本量"):
            engine.robust_anomaly_detection(df, comp_cols=["c1", "c2"], numeric_cols=["n1"])

    def test_no_cols_raises(self, engine: RobustAnalyticsEngine, clean_df: "pd.DataFrame") -> None:
        """两个列表都为空应抛出 ValueError"""
        with pytest.raises(ValueError, match="至少应提供一个非空列表"):
            engine.robust_anomaly_detection(clean_df, comp_cols=[], numeric_cols=[])


# =====================================================================
# 2. 分布漂移测试
# =====================================================================

class TestCalculateDistributionDrift:
    """验证 calculate_distribution_drift 的各种场景"""

    def test_drifted_distributions(self, engine: RobustAnalyticsEngine) -> None:
        """均值漂移 3σ 的数据应被检测为漂移"""
        import pandas as pd
        np.random.seed(42)
        df_a = pd.DataFrame({"x": np.random.normal(0, 1, 50)})
        df_b = pd.DataFrame({"x": np.random.normal(5, 1, 50)})
        result = engine.calculate_distribution_drift(df_a, df_b, ["x"])
        assert result["x"]["is_drifted"] is True
        assert result["x"]["ks_statistic"] > 0.5
        assert result["x"]["p_value"] < result["x"]["bonferroni_alpha"]

    def test_same_distributions_no_drift(self, engine: RobustAnalyticsEngine) -> None:
        """同分布数据不应被标记为漂移"""
        import pandas as pd
        np.random.seed(42)
        df_a = pd.DataFrame({"x": np.random.normal(0, 1, 60)})
        df_b = pd.DataFrame({"x": np.random.normal(0, 1, 60)})
        result = engine.calculate_distribution_drift(df_a, df_b, ["x"])
        assert result["x"]["is_drifted"] is False

    def test_small_sample_raises(self, engine: RobustAnalyticsEngine) -> None:
        """df_a 不足 30 条应抛出 ValueError"""
        import pandas as pd
        df_a = pd.DataFrame({"x": np.random.randn(10)})
        df_b = pd.DataFrame({"x": np.random.randn(50)})
        with pytest.raises(ValueError) as excinfo:
            engine.calculate_distribution_drift(df_a, df_b, ["x"])
        assert "30" in str(excinfo.value)

    def test_both_small_raises(self, engine: RobustAnalyticsEngine) -> None:
        """两组均不足 30 条应抛出 ValueError"""
        import pandas as pd
        df_a = pd.DataFrame({"x": np.random.randn(15)})
        df_b = pd.DataFrame({"x": np.random.randn(15)})
        with pytest.raises(ValueError):
            engine.calculate_distribution_drift(df_a, df_b, ["x"])

    def test_multiple_columns(self, engine: RobustAnalyticsEngine) -> None:
        """多列漂移检测：只有 1、3 列漂移，2 列不漂移"""
        import pandas as pd
        np.random.seed(42)
        df_a = pd.DataFrame({
            "v1": np.random.normal(0, 1, 50),
            "v2": np.random.normal(0, 1, 50),
            "v3": np.random.normal(0, 1, 50),
        })
        df_b = pd.DataFrame({
            "v1": np.random.normal(5, 1, 50),
            "v2": np.random.normal(0, 1, 50),
            "v3": np.random.normal(3, 1, 50),
        })
        result = engine.calculate_distribution_drift(df_a, df_b, ["v1", "v2", "v3"])
        assert result["v1"]["is_drifted"] is True
        assert result["v2"]["is_drifted"] is False
        assert result["v3"]["is_drifted"] is True

    def test_empty_columns_returns_empty(self, engine: RobustAnalyticsEngine) -> None:
        """metrics_cols 为空时返回空字典"""
        import pandas as pd
        df_a = pd.DataFrame({"x": np.random.randn(50)})
        df_b = pd.DataFrame({"x": np.random.randn(50)})
        result = engine.calculate_distribution_drift(df_a, df_b, [])
        assert result == {}

    def test_bonferroni_multiple_features(self, engine: RobustAnalyticsEngine) -> None:
        """大量特征时 Bonferroni 校正后 alpha 正确缩小"""
        import pandas as pd
        np.random.seed(42)
        df_a = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(20)})
        df_b = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(20)})
        result = engine.calculate_distribution_drift(df_a, df_b, list(df_a.columns))
        # 所有列应通过（无漂移）
        drifted = [v["is_drifted"] for v in result.values()]
        assert sum(drifted) <= 3, f"Expected ≤ 3 false positives, got {sum(drifted)}"
        # 验证 alpha 是 0.05/20
        alpha = result["f0"]["bonferroni_alpha"]
        assert abs(alpha - 0.05 / 20) < 1e-10


# =====================================================================
# 3. 批次漂移矩阵集成测试
# =====================================================================

class TestBatchDriftIntegration:
    """提供含有显著批次漂移的矩阵，验证算法能精准揪出偏移特征列"""

    @pytest.fixture
    def batch_drift_data(self) -> tuple:
        """
        生成 3 个参考批次（各 50 条） + 1 个目标批次（50 条）。
        目标批次中:
          - f0, f1: 均值漂移 +3σ，大幅偏移（应被检出）
          - f2, f3: 均值漂移 +1σ，中等偏移（应被检出）
          - f4, f5: 无漂移（不应被检出）
        """
        import pandas as pd
        np.random.seed(42)

        # 参考：5 特征，均 N(0,1)
        ref = pd.DataFrame(
            {f"f{i}": np.random.normal(0, 1, 150) for i in range(6)}
        )

        # 目标：50 条，f0/f1 大漂移，f2/f3 中等漂移，f4/f5 无漂移
        target = pd.DataFrame(
            {f"f{i}": np.random.normal(0, 1, 50) for i in range(6)}
        )
        target["f0"] = np.random.normal(3.0, 1.0, 50)
        target["f1"] = np.random.normal(3.0, 1.0, 50)
        target["f2"] = np.random.normal(1.0, 1.0, 50)
        target["f3"] = np.random.normal(1.0, 1.0, 50)

        return ref, target

    def test_detects_all_drifted_features(
        self, engine: RobustAnalyticsEngine, batch_drift_data: tuple,
    ) -> None:
        """验证大漂移（f0, f1）和小漂移（f2, f3）均被检出，无漂移特征未被误检"""
        ref, target = batch_drift_data
        result = engine.calculate_distribution_drift(ref, target, [f"f{i}" for i in range(6)])

        expected_drift = {"f0", "f1", "f2", "f3"}
        expected_clean = {"f4", "f5"}

        for col in expected_drift:
            assert result[col]["is_drifted"] is True, (
                f"{col}: 大/中漂移应被检出 (p={result[col]['p_value']:.2e}, "
                f"ks={result[col]['ks_statistic']:.3f})"
            )

        for col in expected_clean:
            assert result[col]["is_drifted"] is False, (
                f"{col}: 无漂移不应被误检 (p={result[col]['p_value']:.2e})"
            )

    def test_drifted_features_have_larger_ks_statistics(
        self, engine: RobustAnalyticsEngine, batch_drift_data: tuple,
    ) -> None:
        """漂移特征的 KS 统计量应显著大于无漂移特征的 KS 统计量"""
        ref, target = batch_drift_data
        result = engine.calculate_distribution_drift(ref, target, [f"f{i}" for i in range(6)])

        max_clean_ks = max(result[f]["ks_statistic"] for f in ["f4", "f5"])
        min_drifted_ks = min(result[f]["ks_statistic"] for f in ["f0", "f1", "f2", "f3"])

        assert min_drifted_ks > max_clean_ks, (
            f"漂移特征最小 KS ({min_drifted_ks:.3f}) 应大于无漂移特征最大 KS ({max_clean_ks:.3f})"
        )

    def test_ks_pvalue_separation(
        self, engine: RobustAnalyticsEngine, batch_drift_data: tuple,
    ) -> None:
        """漂移特征的 p 值极小（< Bonferroni alpha），无漂移特征的 p 值较大"""
        ref, target = batch_drift_data
        result = engine.calculate_distribution_drift(ref, target, [f"f{i}" for i in range(6)])

        alpha = result["f0"]["bonferroni_alpha"]  # = 0.05/6 ≈ 0.0083
        for col in ["f0", "f1", "f2", "f3"]:
            assert result[col]["p_value"] < alpha, (
                f"{col}: p={result[col]['p_value']:.2e} 应 < bonf_alpha={alpha:.4f}"
            )
        for col in ["f4", "f5"]:
            assert result[col]["p_value"] >= alpha, (
                f"{col}: p={result[col]['p_value']:.2e} 应 ≥ bonf_alpha={alpha:.4f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])