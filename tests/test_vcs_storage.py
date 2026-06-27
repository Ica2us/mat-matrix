"""Unit tests for DuckDBStorageEngine & MerkleDAGVersionControl.

Simulates a 3-version Merkle DAG with branching and merging,
validates hash addressing, LCS diffs, and RFC 6902 patch generation.
"""

import gc
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from storage.db_engine import DuckDBStorageEngine
from storage.vcs import MerkleDAGVersionControl


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def tmp_data_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    # Close any lingering SQLite connections so Windows can delete files
    for m in gc.get_objects():
        if isinstance(m, sqlite3.Connection):
            try:
                m.close()
            except Exception:
                pass
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def engine(tmp_data_dir):
    data_dir = tmp_data_dir / "data"
    return DuckDBStorageEngine(data_dir)


@pytest.fixture
def vcs(tmp_data_dir):
    index_dir = tmp_data_dir / "vcs_index"
    return MerkleDAGVersionControl(index_dir)


# ======================================================================
# Test DuckDBStorageEngine
# ======================================================================

class TestDuckDBStorageEngine:
    def test_save_and_load_dataframe(self, engine: DuckDBStorageEngine):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        h = engine.save_dataframe(df)
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex
        loaded = engine.load_dataframe(h)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_returns_same_hash_for_identical_data(
        self, engine: DuckDBStorageEngine
    ):
        df = pd.DataFrame({"x": [10, 20]})
        h1 = engine.save_dataframe(df)
        h2 = engine.save_dataframe(df)
        assert h1 == h2  # content-addressable

    def test_load_missing_raises(self, engine: DuckDBStorageEngine):
        with pytest.raises(FileNotFoundError):
            engine.load_dataframe("0" * 64)


# ======================================================================
# Test MerkleDAGVersionControl — LCS
# ======================================================================

class TestLCSSopDiff:
    def test_identical_sequences(self, vcs: MerkleDAGVersionControl):
        seq = ["A", "B", "C"]
        diff = vcs.calculate_sop_diff(seq, seq)
        assert diff["common"] == seq
        assert diff["only_in_a"] == []
        assert diff["only_in_b"] == []

    def test_completely_different(self, vcs: MerkleDAGVersionControl):
        diff = vcs.calculate_sop_diff(["A", "B"], ["C", "D"])
        assert diff["common"] == []
        assert diff["only_in_a"] == ["A", "B"]
        assert diff["only_in_b"] == ["C", "D"]

    def test_partial_overlap(self, vcs: MerkleDAGVersionControl):
        diff = vcs.calculate_sop_diff(
            ["Mix", "Grind", "Heat", "Cool"],
            ["Mix", "Heat", "Quench", "Cool"],
        )
        assert diff["common"] == ["Mix", "Heat", "Cool"]
        assert diff["only_in_a"] == ["Grind"]
        assert diff["only_in_b"] == ["Quench"]

    def test_lcs_deterministic(self, vcs: MerkleDAGVersionControl):
        r1 = vcs.calculate_sop_diff(["a", "b", "c"], ["b", "a", "d"])
        r2 = vcs.calculate_sop_diff(["a", "b", "c"], ["b", "a", "d"])
        assert r1 == r2


# ======================================================================
# Test MerkleDAGVersionControl — commit + DAG (multi-parent)
# ======================================================================

class TestMerkleDAGVersionControl:
    def test_root_commit_no_parent(self, vcs: MerkleDAGVersionControl, engine):
        df = pd.DataFrame({"formula": ["A", "B"]})
        data_hash = engine.save_dataframe(df)
        cid = vcs.commit_version(
            parent_ids=None,
            sop_sequence=["Mix", "Press"],
            parameters={"T": 150.0},
            data_file_hash=data_hash,
        )
        assert isinstance(cid, str) and len(cid) == 64

    def test_linear_chain(self, vcs: MerkleDAGVersionControl, engine):
        df = pd.DataFrame({"v": [1.0]})
        h1 = engine.save_dataframe(df)
        c1 = vcs.commit_version(None, ["A"], {"p": 1}, h1)

        h2 = engine.save_dataframe(pd.DataFrame({"v": [2.0]}))
        c2 = vcs.commit_version([c1], ["A", "B"], {"p": 2}, h2)

        h3 = engine.save_dataframe(pd.DataFrame({"v": [3.0]}))
        c3 = vcs.commit_version([c2], ["A", "B", "C"], {"p": 3}, h3)

        # verify chain
        m1 = vcs._load_manifest(c1)
        m2 = vcs._load_manifest(c2)
        m3 = vcs._load_manifest(c3)
        assert m1["parents"] == []
        assert m2["parents"] == [c1]
        assert m3["parents"] == [c2]

    def test_merge_commit_multi_parent(self, vcs, engine):
        """Build a real DAG branch and merge — true Merkle DAG with 2 parents."""
        # root
        h_root = engine.save_dataframe(pd.DataFrame({"x": [0.0]}))
        root = vcs.commit_version(None, ["Init"], {"a": 0}, h_root)

        # branch-A
        h_a = engine.save_dataframe(pd.DataFrame({"x": [1.0]}))
        c_a = vcs.commit_version([root], ["Init", "StepA"], {"a": 1}, h_a)

        # branch-B (from root)
        h_b = engine.save_dataframe(pd.DataFrame({"x": [2.0]}))
        c_b = vcs.commit_version([root], ["Init", "StepB"], {"a": 2}, h_b)

        # merge commit — two parents
        h_m = engine.save_dataframe(pd.DataFrame({"x": [3.0]}))
        merge = vcs.commit_version(
            [c_a, c_b],
            ["Init", "StepA", "StepB"],
            {"a": 3, "merged": True},
            h_m,
        )

        manifest = vcs._load_manifest(merge)
        assert sorted(manifest["parents"]) == sorted([c_a, c_b])

    def test_commit_id_deterministic(self, vcs, engine):
        df = pd.DataFrame({"x": [1]})
        h = engine.save_dataframe(df)
        c1 = vcs.commit_version([], ["A"], {"t": 1}, h)
        c2 = vcs.commit_version([], ["A"], {"t": 1}, h)
        assert c1 == c2


# ======================================================================
# Test generate_version_patch
# ======================================================================

class TestGenerateVersionPatch:
    def test_patch_between_two_versions(self, vcs, engine):
        h1 = engine.save_dataframe(pd.DataFrame({"y": [10.0]}))
        h2 = engine.save_dataframe(pd.DataFrame({"y": [20.0]}))

        c1 = vcs.commit_version(None, ["Mix", "Heat"], {"T": 100.0, "P": 1.0}, h1)
        c2 = vcs.commit_version(
            [c1], ["Mix", "Heat", "Cool"], {"T": 150.0, "P": 1.0, "added": True}, h2
        )

        patch = vcs.generate_version_patch(c1, c2)

        # data file changed
        assert patch["data_file_hash"]["op"] == "replace"

        # params: T changed, "added" added
        param_ops = {op["op"]: op for op in patch["parameters"]}
        assert "replace" in [op["op"] for op in patch["parameters"]]
        # "added" is new
        add_ops = [op for op in patch["parameters"] if op["op"] == "add"]
        assert any(op["path"] == "/added" for op in add_ops)

        # sop: LCS-based
        assert patch["sop_sequence"]["common"] == ["Mix", "Heat"]
        assert patch["sop_sequence"]["only_in_b"] == ["Cool"]

    def test_patch_identical_versions(self, vcs, engine):
        h = engine.save_dataframe(pd.DataFrame({"z": [99.0]}))
        c = vcs.commit_version(None, ["A"], {"k": "v"}, h)
        patch = vcs.generate_version_patch(c, c)
        assert patch == {}  # empty patch

    def test_float_drift_not_false_patch(self, vcs, engine):
        """二进制浮点漂移不应污染版本链——math.isclose 容差过滤"""
        df = pd.DataFrame({"z": [99.0]})
        h = engine.save_dataframe(df)

        # 0.1 + 0.2 在二进制下 ≠ 0.3 —— 经典浮点陷阱
        temp_value = 0.1 + 0.2

        c1 = vcs.commit_version(None, ["A"], {"temp": 0.3}, h)
        c2 = vcs.commit_version(None, ["A"], {"temp": temp_value}, h)
        patch = vcs.generate_version_patch(c1, c2)
        assert patch == {}  # 应视为相等，不出 Patch

    def test_patch_merge_scenario(self, vcs, engine):
        """Generate patch between a branch commit and the merge result."""
        h_root = engine.save_dataframe(pd.DataFrame({"d": [0.0]}))
        root = vcs.commit_version(None, ["R1"], {"x": 1}, h_root)

        h_a = engine.save_dataframe(pd.DataFrame({"d": [1.0]}))
        c_a = vcs.commit_version([root], ["R1", "A"], {"x": 10}, h_a)

        h_b = engine.save_dataframe(pd.DataFrame({"d": [2.0]}))
        c_b = vcs.commit_version([root], ["R1", "B"], {"x": 20}, h_b)

        h_m = engine.save_dataframe(pd.DataFrame({"d": [3.0]}))
        merge = vcs.commit_version(
            [c_a, c_b], ["R1", "A", "B"], {"x": 99, "merged": True}, h_m
        )

        patch = vcs.generate_version_patch(c_a, merge)
        assert "sop_sequence" in patch
        assert patch["sop_sequence"]["only_in_b"] == ["B"]


# ======================================================================
# Integration: full workflow — 3 versions + merge
# ======================================================================

def test_full_merkle_dag_workflow(engine, vcs):
    """Simulate 3 version evolutions plus a merge; verify every invariant."""
    # ---- Version 1: root ----
    v1_df = pd.DataFrame({"batch": [1], "result": [0.5]})
    v1_hash = engine.save_dataframe(v1_df)
    v1_id = vcs.commit_version(
        parent_ids=None,
        sop_sequence=["Weigh", "Mix", "Press"],
        parameters={"temp": 100.0, "pressure": 10.0},
        data_file_hash=v1_hash,
    )

    # ---- Version 2: linear child ----
    v2_df = pd.DataFrame({"batch": [2], "result": [0.7]})
    v2_hash = engine.save_dataframe(v2_df)
    v2_id = vcs.commit_version(
        parent_ids=[v1_id],
        sop_sequence=["Weigh", "Mix", "Press", "Heat"],
        parameters={"temp": 150.0, "pressure": 10.0},
        data_file_hash=v2_hash,
    )

    # ---- Version 3: branch from v1 ----
    v3_df = pd.DataFrame({"batch": [3], "result": [0.9]})
    v3_hash = engine.save_dataframe(v3_df)
    v3_id = vcs.commit_version(
        parent_ids=[v1_id],
        sop_sequence=["Weigh", "Mix", "Sinter", "Cool"],
        parameters={"temp": 200.0, "pressure": 15.0},
        data_file_hash=v3_hash,
    )

    # ---- Merge v2 + v3 ----
    merge_df = pd.DataFrame({"batch": [4], "result": [0.85]})
    merge_hash = engine.save_dataframe(merge_df)
    merge_id = vcs.commit_version(
        parent_ids=[v2_id, v3_id],
        sop_sequence=["Weigh", "Mix", "Press", "Heat", "Sinter", "Cool"],
        parameters={"temp": 180.0, "pressure": 12.0, "merged": True},
        data_file_hash=merge_hash,
    )

    # ========== Assertions ==========

    # 1. Hash addressing — load back data
    pd.testing.assert_frame_equal(engine.load_dataframe(v1_hash), v1_df)
    pd.testing.assert_frame_equal(engine.load_dataframe(v2_hash), v2_df)
    pd.testing.assert_frame_equal(engine.load_dataframe(v3_hash), v3_df)

    # 2. Merge commit has two parents
    merge_manifest = vcs._load_manifest(merge_id)
    assert len(merge_manifest["parents"]) == 2
    assert v2_id in merge_manifest["parents"]
    assert v3_id in merge_manifest["parents"]

    # 3. LCS diff between v2 and v3
    diff = vcs.calculate_sop_diff(
        ["Weigh", "Mix", "Press", "Heat"],
        ["Weigh", "Mix", "Sinter", "Cool"],
    )
    assert diff["common"] == ["Weigh", "Mix"]  # LCS
    assert "Press" in diff["only_in_a"]
    assert "Heat" in diff["only_in_a"]
    assert "Sinter" in diff["only_in_b"]
    assert "Cool" in diff["only_in_b"]

    # 4. Patch between v1 and merge
    patch = vcs.generate_version_patch(v1_id, merge_id)
    assert "data_file_hash" in patch
    assert "parameters" in patch
    assert "sop_sequence" in patch

    # 5. Deterministic commit ids
    again = vcs.commit_version(
        parent_ids=[v2_id, v3_id],
        sop_sequence=["Weigh", "Mix", "Press", "Heat", "Sinter", "Cool"],
        parameters={"temp": 180.0, "pressure": 12.0, "merged": True},
        data_file_hash=merge_hash,
    )
    assert again == merge_id

    # 6. Patch between root and self → empty
    assert vcs.generate_version_patch(v1_id, v1_id) == {}