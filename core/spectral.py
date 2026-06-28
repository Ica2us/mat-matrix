"""
CachedSpectralAligner — SpectralFeatureProtocol 的具体实现（光谱缓存对齐器）。

职责：
1. align_peak_positions: 将样品峰位刚性对齐到参考系（基于 np.interp 插值）。
2. get_or_compute_aligned_profile: 持久化缓存调度层，避免重复 O(n²) 计算。

缓存键 = SHA256(raw_file_hash + SHA256(x_ref))，缓存文件为 .npy 格式。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from core.storage_engine import DuckDBStorageEngine

logger = logging.getLogger(__name__)


class CachedSpectralAligner:
    """光谱特征峰位对齐器（带持久化缓存）。"""

    def __init__(self, cache_dir: str = "cache/spectral") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._storage = DuckDBStorageEngine(cache_dir)

    # -----------------------------------------------------------------
    # 原子级无状态计算
    # -----------------------------------------------------------------

    def align_peak_positions(
        self,
        x_ref: np.ndarray,
        x_sample: np.ndarray,
        intensity_sample: np.ndarray,
    ) -> np.ndarray:
        """将样品峰位刚性对齐到 x_ref 参考系。

        使用 np.interp 完成线性插值对齐；x_ref 范围之外的区域填零。
        """
        x_ref = np.asarray(x_ref, dtype=float).ravel()
        x_sample = np.asarray(x_sample, dtype=float).ravel()
        intensity_sample = np.asarray(intensity_sample, dtype=float).ravel()

        if x_ref.size == 0 or x_sample.size == 0 or intensity_sample.size == 0:
            logger.warning("align_peak_positions: 输入数组为空，返回零向量")
            return np.zeros_like(x_ref)

        if intensity_sample.size != x_sample.size:
            raise ValueError(
                f"intensity_sample 长度 ({intensity_sample.size}) "
                f"必须与 x_sample 长度 ({x_sample.size}) 一致"
            )

        # 计算重叠区间
        lo = max(x_ref.min(), x_sample.min())
        hi = min(x_ref.max(), x_sample.max())

        if lo >= hi:
            logger.warning("align_peak_positions: x_ref 与 x_sample 无重叠区间，返回零向量")
            return np.zeros_like(x_ref)

        # 在重叠区间内插值对齐
        aligned = np.interp(x_ref, x_sample, intensity_sample, left=0.0, right=0.0)
        return aligned

    # -----------------------------------------------------------------
    # 持久化缓存调度层
    # -----------------------------------------------------------------

    def get_or_compute_aligned_profile(
        self,
        raw_file_hash: str,
        x_ref: np.ndarray,
        x_sample: np.ndarray,
        intensity_sample: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        """持久化缓存调度。

        若缓存存在则零拷贝加载；否则计算并持久化。
        返回 (对齐后的强度向量, 缓存文件哈希值 value_hash)。
        """
        x_ref_bytes = np.asarray(x_ref).tobytes()
        x_ref_hash = hashlib.sha256(x_ref_bytes).hexdigest()

        cache_raw = (raw_file_hash + x_ref_hash).encode("utf-8")
        cache_key = hashlib.sha256(cache_raw).hexdigest()

        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            logger.debug("Spectral cache HIT: %s", cache_key)
            aligned = np.load(str(cache_path), allow_pickle=False)
        else:
            logger.debug("Spectral cache MISS: %s", cache_key)
            aligned = self.align_peak_positions(x_ref, x_sample, intensity_sample)
            np.save(str(cache_path), aligned)

        value_hash = hashlib.sha256(aligned.tobytes()).hexdigest()
        return aligned, value_hash