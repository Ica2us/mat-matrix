"""
DuckDB-backed storage engine — columnar persistence via Parquet + SHA-256 content addressing.

Implements ``StorageEngineProtocol`` defined in :mod:`contracts`.
"""

import hashlib
import os
from pathlib import Path

import duckdb
import pandas as pd

from contracts import StorageEngineProtocol


class DuckDBStorageEngine(StorageEngineProtocol):
    """Persist DataFrames as SHA-256-named Parquet files, load via DuckDB zero-copy."""

    def __init__(self, data_dir: str | os.PathLike[str] = "v/data") -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def save_dataframe(self, df: pd.DataFrame) -> str:
        """Persist *df* as a Parquet file named ``<sha256>.parquet``.

        Returns the hex digest (hash) — the caller's content-address.
        """
        raw = df.to_parquet(index=False)
        digest = hashlib.sha256(raw).hexdigest()
        path = self._data_dir / f"{digest}.parquet"
        if not path.exists():
            path.write_bytes(raw)
        return digest

    def load_dataframe(self, data_file_hash: str) -> pd.DataFrame:
        """Zero-copy read of the Parquet file identified by *data_file_hash*."""
        path = self._data_dir / f"{data_file_hash}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet data file not found for hash: {data_file_hash}"
            )
        return duckdb.read_parquet(str(path)).df()
