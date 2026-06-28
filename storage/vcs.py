"""
Git-like Merkle DAG content-addressable version control engine.

Implements ``VersionControlSystemProtocol`` from :mod:`contracts`.
- Version index in SQLite.
- Manifest files as JSON blobs keyed by SHA-256.
- Multi-parent merge commits for a real Merkle DAG.
- LCS-based SOP diff and RFC 6902 version patch generation.
"""

import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from contracts import VersionControlSystemProtocol


class MerkleDAGVersionControl(VersionControlSystemProtocol):
    """Content-addressed version control using a Merkle DAG topology."""

    def __init__(self, index_dir: str | os.PathLike[str] = "v") -> None:
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_dir = self._index_dir / "manifests"
        self._manifest_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._index_dir / "vcs_index.sqlite"
        self._conn = sqlite3.connect(
            str(self._db_path), timeout=30.0, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout=30000;")
        self._init_schema()

    # -----------------------------------------------------------------
    # public API (VersionControlSystemProtocol)
    # -----------------------------------------------------------------

    def commit_version(
        self,
        parent_ids: Optional[List[str]],
        sop_sequence: List[str],
        parameters: Dict[str, Any],
        data_file_hash: str,
    ) -> str:
        """Build a manifest, store it as ``<sha256>.json``, index in SQLite.

        *parent_ids* may be ``None`` (root commit), a single-element list
        (linear commit), or multiple (merge commit — real DAG branch).
        Returns the *commit_id* (SHA-256 of the manifest body).
        """
        parents = sorted(parent_ids) if parent_ids else []

        manifest: Dict[str, Any] = {
            "parents": parents,
            "sop_sequence": sop_sequence,
            "parameters": parameters,
            "data_file_hash": data_file_hash,
        }

        body = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        commit_id = hashlib.sha256(body).hexdigest()

        manifest_path = self._manifest_dir / f"{commit_id}.json"
        if not manifest_path.exists():
            manifest_path.write_bytes(body)

        parent_json = json.dumps(parents)
        sop_json = json.dumps(sop_sequence)
        params_json = json.dumps(parameters, sort_keys=True)

        self._conn.execute(
            """INSERT OR IGNORE INTO versions
               (commit_id, parents, sop_sequence, parameters, data_file_hash)
               VALUES (?, ?, ?, ?, ?)""",
            (commit_id, parent_json, sop_json, params_json, data_file_hash),
        )
        self._conn.commit()

        return commit_id

    def calculate_sop_diff(
        self, seq_a: List[str], seq_b: List[str]
    ) -> Dict[str, List[str]]:
        """Deterministic LCS diff between two SOP sequences.

        Returns ``{"only_in_a": [...], "only_in_b": [...], "common": [...]}``.
        """
        lcs = self._longest_common_subsequence(seq_a, seq_b)
        lcs_set = set(lcs)

        # preserve order for each side
        only_a: List[str] = []
        seen_a = set()
        for s in seq_a:
            if s not in lcs_set and s not in seen_a:
                only_a.append(s)
                seen_a.add(s)

        only_b: List[str] = []
        seen_b = set()
        for s in seq_b:
            if s not in lcs_set and s not in seen_b:
                only_b.append(s)
                seen_b.add(s)

        return {
            "only_in_a": only_a,
            "only_in_b": only_b,
            "common": lcs,
        }

    def generate_version_patch(
        self, v1_id: str, v2_id: str
    ) -> Dict[str, Any]:
        """Compare two commits and produce an RFC 6902–style JSON Patch dict.

        Fields compared: *sop_sequence*, *parameters*, *data_file_hash*.
        """
        m1 = self._load_manifest(v1_id)
        m2 = self._load_manifest(v2_id)

        patch: Dict[str, Any] = {}

        # --- data file hash ---
        if m1["data_file_hash"] != m2["data_file_hash"]:
            patch["data_file_hash"] = {
                "op": "replace",
                "path": "/data_file_hash",
                "from": m1["data_file_hash"],
                "value": m2["data_file_hash"],
            }

        # --- parameters RFC 6902 diff (shallow) ---
        params_patch = self._json_patch_dict(m1["parameters"], m2["parameters"])
        if params_patch:
            patch["parameters"] = params_patch

        # --- sop_sequence LCS diff ---
        sop_diff = self.calculate_sop_diff(m1["sop_sequence"], m2["sop_sequence"])
        if sop_diff["only_in_a"] or sop_diff["only_in_b"]:
            patch["sop_sequence"] = sop_diff

        return patch

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS versions (
                commit_id      TEXT PRIMARY KEY,
                parents        TEXT NOT NULL,
                sop_sequence   TEXT NOT NULL,
                parameters     TEXT NOT NULL,
                data_file_hash TEXT NOT NULL
            )"""
        )

    def _load_manifest(self, commit_id: str) -> Dict[str, Any]:
        path = self._manifest_dir / f"{commit_id}.json"
        if not path.exists():
            raise ValueError(f"Commit not found: {commit_id}")
        return json.loads(path.read_bytes())

    @staticmethod
    def _longest_common_subsequence(a: List[str], b: List[str]) -> List[str]:
        """Classic DP LCS — purely deterministic, O(mn)."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # backtrack
        i, j = m, n
        result: List[str] = []
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                result.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        result.reverse()
        return result

    @staticmethod
    def _json_patch_dict(
        d1: Dict[str, Any], d2: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Shallow dict diff → list of RFC 6902 patch operations.

        Float values are compared with ``math.isclose`` (rtol=1e-5, atol=1e-8)
        to suppress false diffs from binary floating-point drift.
        """
        ops: List[Dict[str, Any]] = []
        keys = sorted(set(d1) | set(d2))
        for k in keys:
            path = f"/{k}"
            if k not in d1:
                ops.append({"op": "add", "path": path, "value": d2[k]})
            elif k not in d2:
                ops.append({"op": "remove", "path": path})
            elif _values_differ(d1[k], d2[k]):
                ops.append({"op": "replace", "path": path, "value": d2[k]})
        return ops


def _values_differ(a: Any, b: Any) -> bool:
    """Check whether two values differ, using tolerant float comparison."""
    if isinstance(a, float) and isinstance(b, float):
        return not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8)
    if isinstance(a, int) and isinstance(b, float):
        return _values_differ(float(a), b)
    if isinstance(a, float) and isinstance(b, int):
        return _values_differ(a, float(b))
    try:
        return bool(a != b)
    except Exception:
        return True