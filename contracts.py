from typing import List, Dict, Any, Optional, Tuple, Protocol
import numpy as np
import pandas as pd

# =====================================================================
# 全局可复现性配置 (Reproducibility Contract)
# =====================================================================
GLOBAL_RANDOM_STATE: int = 42  # 所有涉及随机性/近似迭代的算法（MCD, TPE, t-SNE）必须强制读取此种子


# =====================================================================
# TRACK 1: 数学与优化矩阵 (Math & BO Track)
# =====================================================================

class CompositionalMathProtocol(Protocol):
    """
    负责配方空间（单纯形 S^d）与实数空间 (R^(d-1)) 的双向确定性几何转换。
    注意：零值替代属于确定性正则化（Deterministic Regularization），并非无偏操作。
    """
    
    def multiplicative_zero_replacement(self, X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        确定性零值替代：检测输入矩阵中的 0 值，用 eps 替换，并等比例缩放同行非零元素，
        严格保证行和为 1.0，最大程度保留子组成之间的原始相对比例。
        """
        ...

    def ilr_transform(self, X: np.ndarray) -> np.ndarray:
        """等距对数比变换：将严格正值的单纯形配方矩阵投影到无约束的实数空间。"""
        ...

    def inverse_ilr_transform(self, Z: np.ndarray) -> np.ndarray:
        """逆等距对数比变换：将实数空间矩阵解码回单纯形配方空间。"""
        ...


class BayesianOptimizationProtocol(Protocol):
    """
    在物理单纯形与工艺边界双重约束下，利用 TPE 与内部 ILR-GPR 代理模型，
    严格沿用户定义的方向（最大化/最小化）进行多目标 Pareto 推荐。
    """
    
    def calculate_pareto_front(self, metrics_matrix: np.ndarray, target_directions: List[str]) -> np.ndarray:
        """计算多目标 Pareto 前沿面。返回长度为 n 的布尔 Mask。"""
        ...

    def recommend_next_experiment(
        self, 
        historical_df: pd.DataFrame, 
        comp_cols: List[str], 
        param_cols: List[str], 
        target_cols: List[str],
        comp_bounds: Dict[str, Tuple[float, float]],  
        process_bounds: Dict[str, Tuple[float, float]], 
        target_directions: List[str]  
    ) -> Dict[str, float]:
        """输入历史实验数据集，推荐下一个实验的最佳物理空间参数字典（Dict[str, float]）。"""
        ...


# =====================================================================
# TRACK 2: 存储与版本控制矩阵 (VCS & Storage Track)
# =====================================================================

class StorageEngineProtocol(Protocol):
    """基于 DuckDB 与文件系统的列式及非结构化大文件高效寻址底座"""
    
    def save_dataframe(self, df: pd.DataFrame) -> str:
        """将高通量 DataFrame 持久化为以其 SHA-256 哈希命名的本地 Parquet 文件，返回该 data_file_hash"""
        ...

    def load_dataframe(self, data_file_hash: str) -> pd.DataFrame:
        """通过 DuckDB 零拷贝技术，高效读取对应的 Parquet 文件并返回 Pandas DataFrame"""
        ...


class VersionControlSystemProtocol(Protocol):
    """
    Git-like Merkle DAG 内容寻址控制引擎。
    支持分支合并提交（Merge Commit），操作流对比退化为线性过程序列的 LCS。
    """

    def commit_version(
        self, 
        parent_ids: Optional[List[str]],  
        sop_sequence: List[str], 
        parameters: Dict[str, Any], 
        data_file_hash: str
    ) -> str:
        """生成并持久化 Manifest JSON 文件，并将索引同步写入 SQLite。返回唯一的 commit_id"""
        ...

    def calculate_sop_diff(self, seq_a: List[str], seq_b: List[str]) -> Dict[str, List[str]]:
        """利用动态规划（LCS）对比两条线性工艺过程序列。"""
        ...

    def generate_version_patch(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        """沿 Merkle DAG 回溯，对比两个版本的元数据差异（返回 RFC 6902 JSON Patch）。"""
        ...


# =====================================================================
# TRACK 3: 确定性稳健统计与分析矩阵 (Analytics Track)
# =====================================================================

class AdvancedAnalyticsProtocol(Protocol):
    """负责多维空间数据清洗与小样本限制下的统计分布偏移校验"""

    def robust_anomaly_detection(
        self, 
        df: pd.DataFrame, 
        comp_cols: List[str], 
        numeric_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """稳健异常检测。返回元组: (异常布尔 Mask, 稳健马氏距离数值数组)"""
        ...

    def calculate_distribution_drift(
        self, 
        df_a: pd.DataFrame, 
        df_b: pd.DataFrame, 
        metrics_cols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """数据集偏移校验。强制两组样本量均 >= 30，否则抛出 ValueError。"""
        ...


# =====================================================================
# TRACK 4: 原始物理信号特征工程 (Spectral Track - 零成本缓存加固版)
# =====================================================================

class SpectralFeatureProtocol(Protocol):
    """
    专门承接 XRD 衍射谱图、DSC 吸放热曲线等高维非结构化非表格信号的特征对齐与状态存储，
    根治 $O(n^2)$ 级 DTW 算法在多线程/并发调用下的算力冗余与 UI 卡死隐患。
    """

    def align_peak_positions(
        self, 
        x_ref: np.ndarray, 
        x_sample: np.ndarray, 
        intensity_sample: np.ndarray
    ) -> np.ndarray:
        """
        仪器特征峰位对齐（原子级无状态计算）：
        将样品的坐标系刚性对齐到标准参考坐标系 x_ref 下，返回对齐后的特征强度向量。
        """
        ...

    def get_or_compute_aligned_profile(
        self, 
        raw_file_hash: str, 
        x_ref: np.ndarray, 
        x_sample: np.ndarray, 
        intensity_sample: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        持久化缓存调度层（工程刚性防御）：
        1. 检查是否存在以 raw_file_hash + SHA256(x_ref) 为唯一键的持久化缓存文件。
        2. 若存在，通过 StorageEngine 零拷贝直接加载返回；
        3. 若不存在，调用 align_peak_positions 计算，并以新哈希命名持久化到缓存目录。
        返回元组: (对齐后的强度向量, 缓存文件哈希 value_hash)
        """
        ...