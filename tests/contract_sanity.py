import inspect
from typing import List, Dict, Optional, Tuple
from contracts import (
    CompositionalMathProtocol, 
    BayesianOptimizationProtocol, 
    VersionControlSystemProtocol, 
    AdvancedAnalyticsProtocol,
    SpectralFeatureProtocol
)

def verify_all_protocols() -> None:
    # -----------------------------------------------------------------
    # 1. 验证 VCS 支持多父节点（Merkle DAG 核心逻辑）
    # -----------------------------------------------------------------
    commit_sig = inspect.signature(VersionControlSystemProtocol.commit_version)
    if 'parent_ids' not in commit_sig.parameters:
        raise TypeError("VCS Blocker: 必须支持 parent_ids 列表！")
        
    expected_vcs_annotations = ('Optional[List[str]]', Optional[List[str]])
    if commit_sig.parameters['parent_ids'].annotation not in expected_vcs_annotations:
        raise TypeError("VCS Blocker: parent_ids 必须是 Optional[List[str]] 类型！")

    # -----------------------------------------------------------------
    # 2. 验证 BO 空间彻底分离且方向明确
    # -----------------------------------------------------------------
    bo_sig = inspect.signature(BayesianOptimizationProtocol.recommend_next_experiment)
    for param in ['comp_bounds', 'process_bounds', 'target_directions']:
        if param not in bo_sig.parameters:
            raise TypeError(f"BO Blocker: 缺少必要参数 {param}！")
            
    expected_bo_returns = ('Dict[str, float]', Dict[str, float])
    if bo_sig.return_annotation not in expected_bo_returns:
        raise TypeError("BO Blocker: 返回类型必须收紧为 Dict[str, float]！")

    # -----------------------------------------------------------------
    # 3. 验证 Analytics 输出包含马氏距离数值
    # -----------------------------------------------------------------
    mcd_sig = inspect.signature(AdvancedAnalyticsProtocol.robust_anomaly_detection)
    import numpy as np
    expected_mcd_returns = ('Tuple[np.ndarray, np.ndarray]', Tuple[np.ndarray, np.ndarray])
    if mcd_sig.return_annotation not in expected_mcd_returns:
        raise TypeError("Analytics Major: 必须返回 (Mask, 马氏距离) 双元组！")

    # -----------------------------------------------------------------
    # 4. 验证 Spectral Track 没有被架空（防止非结构化数据崩溃）
    # -----------------------------------------------------------------
    spectral_sig = inspect.signature(SpectralFeatureProtocol.align_peak_positions)
    if 'x_ref' not in spectral_sig.parameters or 'intensity_sample' not in spectral_sig.parameters:
        raise TypeError("Spectral Blocker: 必须包含 x_ref 参考系与强度输入！")
    
    cache_sig = inspect.signature(SpectralFeatureProtocol.get_or_compute_aligned_profile)
    if 'raw_file_hash' not in cache_sig.parameters:
        raise TypeError("Spectral Blocker: 缓存方法必须包含 raw_file_hash！")
        
    expected_cache_returns = ('Tuple[np.ndarray, str]', Tuple[np.ndarray, str])
    if cache_sig.return_annotation not in expected_cache_returns:
        raise TypeError("Spectral Blocker: 缓存方法返回值必须为 (强度向量, 缓存哈希) 双元组！")

    # -----------------------------------------------------------------
    # 5. 验证 CompositionalMathProtocol 刚性（成分数据几何底座）
    # -----------------------------------------------------------------
    math_sig = inspect.signature(CompositionalMathProtocol.multiplicative_zero_replacement)
    if 'X' not in math_sig.parameters or 'eps' not in math_sig.parameters:
        raise TypeError("Math Blocker: 零值替代必须包含输入矩阵 X 与扰动项 eps！")
    
    ilr_sig = inspect.signature(CompositionalMathProtocol.ilr_transform)
    expected_ilr_returns = ('np.ndarray', np.ndarray)
    if ilr_sig.return_annotation not in expected_ilr_returns:
        raise TypeError("Math Blocker: ILR 变换返回类型必须为 np.ndarray！")

    # 只有跨越了上面重重的 raise 刀山火海，才有资格执行到这里
    print("================================================================")
    print("✅ 契约自检刚性完整性静态检查 100% 闭环通过！")
    print("   [Math-CoDA]: 通过   | [VCS-DAG]: 通过        | [BO-双解耦]: 通过 ")
    print("   [Analytics-MCD]: 通过| [Spectral-缓存桩]: 通过")
    print("================================================================")

if __name__ == "__main__":
    verify_all_protocols()