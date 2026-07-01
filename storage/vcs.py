"""
Git-like Merkle DAG content-addressable version control engine.

Implements ``VersionControlSystemProtocol`` from :mod:`contracts`.
- Version index in SQLite.
- Manifest files as JSON blobs keyed by SHA-256.
- Multi-parent merge commits for a real Merkle DAG.
- LCS-based SOP diff and RFC 6902 version patch generation.
- Commit metadata: author, message, equipment, branch.
- Commit log browsing via ``list_commits`` / ``get_commit``.
"""

import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, cast

from contracts import VersionControlSystemProtocol


class _StorageProtocol(Protocol):
    """Minimal duck-typing for the storage engine used by checkout."""
    def load_dataframe(self, data_file_hash: str) -> "pd.DataFrame": ...


class MerkleDAGVersionControl(VersionControlSystemProtocol):
    """Content-addressed version control using a Merkle DAG topology.

    Parameters
    ----------
    index_dir : str or os.PathLike
        Directory for the SQLite index and manifest JSON blobs.
    storage_engine : optional
        An object with ``load_dataframe(hash) -> pd.DataFrame``.
        Required by ``checkout_dataframe``; if omitted that method will raise.
    """

    def __init__(
        self,
        index_dir: str | os.PathLike[str] = "v",
        storage_engine: Optional[_StorageProtocol] = None,
    ) -> None:
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

        self._storage = storage_engine

    # -----------------------------------------------------------------
    # public API (VersionControlSystemProtocol)
    # -----------------------------------------------------------------

    def commit_version(
        self,
        parent_ids: Optional[List[str]],
        sop_sequence: List[str],
        parameters: Dict[str, Any],
        data_file_hash: str,
        *,
        author: str = "",
        message: str = "",
        equipment: str = "",
        branch: str = "main",
    ) -> str:
        """Build a manifest, store it as ``<sha256>.json``, index in SQLite.

        *parent_ids* may be ``None`` (root commit), a single-element list
        (linear commit), or multiple (merge commit — real DAG branch).

        Keyword-only metadata (*author*, *message*, *equipment*, *branch*)
        are stored in both the manifest JSON and the SQLite index.

        Returns the *commit_id* (SHA-256 of the manifest body).
        """
        parents = sorted(parent_ids) if parent_ids else []

        manifest: Dict[str, Any] = {
            "parents": parents,
            "sop_sequence": sop_sequence,
            "parameters": parameters,
            "data_file_hash": data_file_hash,
            "author": author,
            "message": message,
            "equipment": equipment,
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
               (commit_id, parents, sop_sequence, parameters, data_file_hash,
                author, message, equipment, branch)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (commit_id, parent_json, sop_json, params_json, data_file_hash,
             author, message, equipment, branch),
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

        Fields compared: *sop_sequence*, *parameters*, *data_file_hash*,
        *author*, *message*, *equipment*.
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

        # --- metadata diff (author, message, equipment) ---
        meta_diff = {}
        if m1.get("author", "") != m2.get("author", ""):
            meta_diff["author"] = m1.get("author", "")
        if m1.get("message", "") != m2.get("message", ""):
            meta_diff["message"] = m1.get("message", "")
        if m1.get("equipment", "") != m2.get("equipment", ""):
            meta_diff["equipment"] = m1.get("equipment", "")
        if meta_diff:
            patch["metadata"] = meta_diff

        return patch

    # -----------------------------------------------------------------
    # Commit log / history browsing
    # -----------------------------------------------------------------

    def list_commits(
        self,
        branch: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        author: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return commit metadata dicts sorted by created_at DESC.

        Each dict contains: commit_id, parents, author, message,
        created_at, branch, equipment, sop_sequence, parameters,
        data_file_hash.

        Parameters
        ----------
        branch : str or None
            Filter to a specific branch.  ``None`` means all branches.
        limit : int
            Maximum number of rows (default 50).
        offset : int
            Pagination offset (default 0).
        author : str or None
            Filter by exact author match.
        since : str or None
            ISO-8601 datetime string; only commits with
            ``created_at >= since`` are returned.
        """
        query = "SELECT * FROM versions WHERE 1=1"
        params: List[Any] = []

        if branch is not None:
            query += " AND branch = ?"
            params.append(branch)
        if author is not None:
            query += " AND author = ?"
            params.append(author)
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since)

        query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return []

        columns = [desc[0] for desc in self._conn.execute(query, params).description]
        # Re-fetch with cursor to get description
        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(zip(columns, row))
            # Deserialize JSON fields
            d["parents"] = json.loads(d["parents"])
            d["sop_sequence"] = json.loads(d["sop_sequence"])
            d["parameters"] = json.loads(d["parameters"])
            result.append(d)
        return result

    def get_commit(self, commit_id: str) -> Dict[str, Any]:
        """Return full metadata for a single commit.

        Raises ``ValueError`` if the commit does not exist.
        """
        cursor = self._conn.execute(
            "SELECT * FROM versions WHERE commit_id = ?", (commit_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Commit not found: {commit_id}")

        columns = [desc[0] for desc in cursor.description]
        d = dict(zip(columns, row))
        # Deserialize JSON fields
        d["parents"] = json.loads(d["parents"])
        d["sop_sequence"] = json.loads(d["sop_sequence"])
        d["parameters"] = json.loads(d["parameters"])
        return d

    def checkout_dataframe(self, commit_id: str) -> "pd.DataFrame":
        """Load the DataFrame associated with *commit_id* from the storage engine.

        Requires that a *storage_engine* was provided at construction time.

        Raises
        ------
        ValueError
            If the commit is unknown, the data file is missing, or no storage
            engine was injected.
        """
        import pandas as pd  # deferred to avoid top-level circular import

        if self._storage is None:
            raise ValueError(
                "checkout_dataframe requires a storage_engine; none was provided "
                "at construction."
            )
        manifest = self._load_manifest(commit_id)
        data_hash = manifest["data_file_hash"]
        return self._storage.load_dataframe(data_hash)

    # -----------------------------------------------------------------
    # DAG traversal
    # -----------------------------------------------------------------

    def find_merge_base(self, commit_a: str, commit_b: str) -> str:
        """Find the lowest common ancestor (merge base) of two commits.

        Walks the parent chain of *commit_a* to construct an ancestor set,
        then BFS-walks *commit_b*'s chain, returning the first ancestor
        that appears in the set.  The root commit (no parents) is always
        a fallback.

        Raises ``ValueError`` if either commit is unknown.
        """
        self.get_commit(commit_a)  # validate existence
        self.get_commit(commit_b)

        # Build full ancestor set for commit_a
        ancestors: set[str] = set()

        def _collect(cid: str) -> None:
            if cid in ancestors:
                return
            ancestors.add(cid)
            try:
                c = self.get_commit(cid)
            except ValueError:
                return
            for pid in c.get("parents", []):
                _collect(pid)

        _collect(commit_a)

        # BFS from commit_b — first match is the LCA
        from collections import deque
        queue: deque[str] = deque([commit_b])
        visited: set[str] = set()
        while queue:
            cid = queue.popleft()
            if cid in ancestors:
                return cid
            if cid in visited:
                continue
            visited.add(cid)
            try:
                c = self.get_commit(cid)
            except ValueError:
                continue
            for pid in c.get("parents", []):
                queue.append(pid)

        raise ValueError(
            f"No merge base found between {commit_a[:16]}... and {commit_b[:16]}..."
        )

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
        # Migrate: add new columns if they don't yet exist
        for col_def in [
            "author TEXT DEFAULT ''",
            "message TEXT DEFAULT ''",
            "created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))",
            "branch TEXT DEFAULT 'main'",
            "equipment TEXT DEFAULT ''",
        ]:
            try:
                self._conn.execute(f"ALTER TABLE versions ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    def _load_manifest(self, commit_id: str) -> Dict[str, Any]:
        path = self._manifest_dir / f"{commit_id}.json"
        if not path.exists():
            raise ValueError(f"Commit not found: {commit_id}")
        return cast(Dict[str, Any], json.loads(path.read_bytes()))

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