"""
Mock GitLikeVCS — implements VersionControlSystemProtocol.

使用内存字典模拟 Merkle DAG，LCS 对比为真实实现。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


class GitLikeVCS:
    """Mock 版本控制系统。"""

    def __init__(self) -> None:
        self._commits: dict[str, dict[str, Any]] = {}
        self._counter: int = 0

    def commit_version(
        self,
        parent_ids: Optional[List[str]],
        sop_sequence: List[str],
        parameters: Dict[str, Any],
        data_file_hash: str,
    ) -> str:
        self._counter += 1
        commit_id = f"mock-commit-{self._counter}"
        self._commits[commit_id] = {
            "parent_ids": parent_ids or [],
            "sop_sequence": list(sop_sequence),
            "parameters": dict(parameters),
            "data_file_hash": data_file_hash,
        }
        return commit_id

    def calculate_sop_diff(self, seq_a: List[str], seq_b: List[str]) -> Dict[str, List[str]]:
        """LCS 动态规划，返回 {added, removed, common}。"""
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 回溯
        added, removed, common = [], [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq_a[i - 1] == seq_b[j - 1]:
                common.append(seq_a[i - 1])
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
                added.append(seq_b[j - 1])
                j -= 1
            else:
                removed.append(seq_a[i - 1])
                i -= 1

        added.reverse()
        removed.reverse()
        common.reverse()
        return {"added": added, "removed": removed, "common": common}

    def generate_version_patch(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        if v1_id not in self._commits or v2_id not in self._commits:
            raise ValueError(f"未找到版本: {v1_id} 或 {v2_id}")
        c1 = self._commits[v1_id]
        c2 = self._commits[v2_id]
        patch = {"changed_keys": [], "sop_diff": None}
        for key in c1:
            if key not in c2:
                patch["changed_keys"].append({"op": "remove", "path": f"/{key}"})
            elif c1[key] != c2[key]:
                patch["changed_keys"].append({"op": "replace", "path": f"/{key}"})
        for key in c2:
            if key not in c1:
                patch["changed_keys"].append({"op": "add", "path": f"/{key}"})
        patch["sop_diff"] = self.calculate_sop_diff(
            c1.get("sop_sequence", []), c2.get("sop_sequence", [])
        )
        return patch