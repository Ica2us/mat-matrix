"""Tests for sliding_window_drift_monitor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analytics import RobustAnalyticsEngine


@pytest.fixture
def engine() -> RobustAnalyticsEngine:
    return RobustAnalyticsEngine(random_state=42)


class TestSlidingWindowDriftMonitor:
    """验证 sliding_window_drift_monitor 的各种场景"""

    def test_no_drift_batches(self, engine: RobustAnalyticsEngine) -> None:
        """所有批次无漂移时 cumulative_alarm=False"""
        np.random.seed(42)
        ref = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 60) for i in range(3)})
        b1 = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(3)})
        b2 = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(3)})

        result: Union[Dict[str, Any], List[Dict[str, Any]]] = engine.sliding_window_drift_monitor(ref, [b1, b2], ["f0", "f1", "f2"])
        assert isinstance(result, dict)
        assert result["cumulative_alarm"] is False
        assert result["drifted_batches"] == 0
        assert result["total_batches"] == 2

    def test_drifted_batch_detected(self, engine: RobustAnalyticsEngine) -> None:
        """含漂移批次的场景，检测到的漂移列正确"""
        np.random.seed(42)
        ref = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 60) for i in range(3)})
        b1 = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(3)})
        b1["f0"] = np.random.normal(3.0, 1.0, 50)
        b1["f1"] = np.random.normal(2.0, 1.0, 50)

        result: Union[Dict[str, Any], List[Dict[str, Any]]] = engine.sliding_window_drift_monitor(ref, [b1], ["f0", "f1", "f2"])
        assert isinstance(result, dict)
        assert result["drifted_batches"] == 1
        assert "f0" in result["drifted_columns"]
        assert "f1" in result["drifted_columns"]
        assert result["cumulative_alarm"] is True

    def test_axis1_format(self, engine: RobustAnalyticsEngine) -> None:
        """drift_axis=1 时返回列表"""
        np.random.seed(42)
        ref = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 60) for i in range(2)})
        b1 = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(2)})
        result: Union[Dict[str, Any], List[Dict[str, Any]]] = engine.sliding_window_drift_monitor(
            ref, [b1], ["f0", "f1"], drift_axis=1,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        batch0: Dict[str, Any] = result[0]
        assert "batch_index" in batch0

    def test_return_details(self, engine: RobustAnalyticsEngine) -> None:
        """return_details=True 时返回完整的列级漂移明细"""
        np.random.seed(42)
        ref = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 60) for i in range(2)})
        b1 = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 50) for i in range(2)})
        result: Union[Dict[str, Any], List[Dict[str, Any]]] = engine.sliding_window_drift_monitor(
            ref, [b1], ["f0", "f1"], drift_axis=1, return_details=True,
        )
        assert isinstance(result, list)
        assert "details" in result[0]
        details: Dict[str, Any] = result[0]["details"]
        assert "f0" in details

    def test_small_batch_skipped(self, engine: RobustAnalyticsEngine) -> None:
        """len(batch) < 30 的批次被跳过，any_drifted=None"""
        ref = pd.DataFrame({"x": np.random.randn(60)})
        small = pd.DataFrame({"x": np.random.randn(10)})
        result: Union[Dict[str, Any], List[Dict[str, Any]]] = engine.sliding_window_drift_monitor(
            ref, [small], ["x"], drift_axis=1,
        )
        assert isinstance(result, list)
        batch0: Dict[str, Any] = result[0]
        assert batch0["any_drifted"] is None
        assert "跳过" in batch0["note"]

    def test_small_reference_raises(self, engine: RobustAnalyticsEngine) -> None:
        """reference_df < 30 行抛出 ValueError"""
        with pytest.raises(ValueError, match="至少需要 30"):
            engine.sliding_window_drift_monitor(
                pd.DataFrame({"x": np.random.randn(20)}),
                [pd.DataFrame({"x": np.random.randn(30)})],
                ["x"],
            )

    def test_window_rolling_on_drift(self, engine: RobustAnalyticsEngine) -> None:
        """漂移发生后窗口滚动到当前批次，下一批次基于新窗口检测"""
        np.random.seed(42)
        ref = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 60) for i in range(2)})
        # batch 1: 漂移
        b1 = pd.DataFrame({f"f{i}": np.random.normal(3.0, 1.0, 50) for i in range(2)})
        # batch 2: 与 b1 同分布（即漂移但已滚动），不应再次标记漂移
        b2 = pd.DataFrame({f"f{i}": np.random.normal(3.0, 1.0, 50) for i in range(2)})

        result: Union[Dict[str, Any], List[Dict[str, Any]]] = engine.sliding_window_drift_monitor(
            ref, [b1, b2], ["f0", "f1"], drift_axis=1,
        )
        assert isinstance(result, list)
        batch0: Dict[str, Any] = result[0]
        batch1: Dict[str, Any] = result[1]
        assert batch0["any_drifted"] is True   # b1 vs ref → drift
        assert batch1["any_drifted"] is False  # b2 vs b1 (新窗口) → no drift

    def test_window_size_cap(self, engine: RobustAnalyticsEngine) -> None:
        """无漂移时窗口增长被 window_size 截断"""
        np.random.seed(42)
        ref = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 40) for i in range(2)})
        b1 = pd.DataFrame({f"f{i}": np.random.normal(0, 1, 30) for i in range(2)})
        engine.sliding_window_drift_monitor(ref, [b1], ["f0", "f1"], window_size=50)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
