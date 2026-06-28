"""
Mock DuckDBStorageEngine — implements StorageEngineProtocol.

在真正的 DuckDB 引擎就绪前，使用 pandas + Parquet 文件系统作为存根。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CACHE_ROOT = Path("./cache")


class DuckDBStorageEngine:
    """基于 Parquet 文件的 Mock 存储引擎。"""

    def __init__(self, cache_dir: str | Path = "cache") -> None:
        self._root = Path(cache_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame) -> str:
        """将 DataFrame 持久化为 SHA-256 命名的 Parquet 文件，返回哈希值。"""
        hash_input = pd.util.hash_pandas_object(df).values.tobytes()
        file_hash = hashlib.sha256(hash_input).hexdigest()
        path = self._root / f"{file_hash}.parquet"
        if not path.exists():
            df.to_parquet(path, index=False)
        return file_hash

    def load_dataframe(self, data_file_hash: str) -> pd.DataFrame:
        """通过哈希值读取对应的 Parquet 文件。"""
        path = self._root / f"{data_file_hash}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Storage: 未找到哈希 {data_file_hash} 对应的缓存文件")
        return pd.read_parquet(path)