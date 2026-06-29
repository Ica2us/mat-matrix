# 材料研发实验数据归档与溯源系统 — 功能缺口分析

**Context:** 基于当前代码库（mat_matrix），从实验员视角审视一个面向生产环境的材料研发数据管理、溯源与版本对比系统还缺少什么。

---

## 一、版本控制与历史追溯（VCS 层面）

### 现状
- Merkle DAG 结构正确（支持多父节点 merge commit）
- `commit_version` / `calculate_sop_diff` / `generate_version_patch` 已实现
- SQLite 索引存放版本信息

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **提交日志浏览 (commit log)** | 🔴 HIGH | 无法「查看最近提交记录」；`list_commits()` / `get_log()` 不存在 |
| **按条件搜索历史** | 🔴 HIGH | 无法回答「哪些实验的压力 > 200？」、「谁在上周提交的？」 |
| **还原/回滚 (revert/rollback)** | 🔴 HIGH | 无法「回到实验 #5 的配方」 |
| **检出历史数据 (checkout)** | 🔴 HIGH | 无法加载某个历史 commit 的 DataFrame 到当前视图 |
| **分支管理** | 🟡 MEDIUM | 无 `create_branch()`、`list_branches()`、`switch_branch()` |
| **标签 (tag)** | 🟡 MEDIUM | 无法标记「v1.0 - 基准配方」供快速引用 |
| **提交日志/消息** | 🟡 MEDIUM | commit 不含 human-readable 的消息/备注 |
| **DAG 可视化** | 🟡 MEDIUM | 无 commit 血缘关系图（类似 git log --graph） |
| **作者/时间戳** | 🟡 MEDIUM | 无 `author`、`created_at` 字段 |
| **数据血缘 (lineage)** | 🟡 MEDIUM | 无法追踪「这个模型推荐是从哪些历史实验推导出来的」 |
| **与旧版本做 diff** | 🟢 LOW | `generate_version_patch` 已实现，但 `_load_manifest` 不是公开方法 |

---

## 二、实验元数据与附属文件

### 现状
- Commit manifest 只存 `parents`, `sop_sequence`, `parameters`, `data_file_hash`
- `RightDetailPanel` 显示了「操作人、时间、设备」但从未写入 VCS

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **非表格文件附件** | 🔴 HIGH | 无法将 SEM 图像、XRD 谱图、日志文件关联到实验 commit |
| **实验人员/设备/注释元数据** | 🔴 HIGH | commit 不含操作人、设备编号、实验日期、实验条件（温湿度等） |
| **Manifest 附加信息字段** | 🟡 MEDIUM | manifest 缺少 `author`, `message`, `attachments`, `tags` |
| **逐行实验备注** | 🟡 MEDIUM | `RightDetailPanel` 的 Markdown 注释只在内存，不保存到 VCS |
| **关键帧标注与 manifest 关联** | 🟡 MEDIUM | `in_situ_widget` 的注释代码提及 `manifest.annotations` 但不存在 |

---

## 三、数据分析与质量监控

### 现状
- MCD 异常检测、分布漂移检测、滑动窗口漂移监控
- `anomaly_explain` 分解出各特征贡献度
- BO 推荐引擎（nearest-neighbor 占位）

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **真正的贝叶斯代理模型** | 🔴 HIGH | 目前是 nearest-neighbor 占位，需要 GP/TPE |
| **谱图特征处理 (XRD/DSC)** | 🔴 HIGH | `SpectralFeatureProtocol` 定义在 contracts 但未实现 |
| **多目标优化 > 2 维可视化** | 🟡 MEDIUM | UI 只支持 2 目标 Pareto，无平行坐标图 |
| **不确定性量化 (UQ)** | 🟡 MEDIUM | 推荐只有点估计，无置信区间 |
| **实验设计 (DoE)** | 🟡 MEDIUM | 无析因设计、空间填充设计、D-最优设计 |
| **批次效应校正** | 🟡 MEDIUM | 漂移检测有，校正无（ComBat 等） |
| **缺失值/删失数据处理** | 🟡 MEDIUM | 隐式 drop NaN，无多重插补 |
| **主动学习批量推荐** | 🟢 LOW | 每次只推荐 1 个实验，不能批量 |

---

## 四、数据导入/导出与文件管理

### 现状
- 单一 CSV 导入，`ImportTypeInferenceEngine` 自动推理列角色

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **数据导出** | 🔴 HIGH | 无 CSV/Excel/Parquet 导出按钮 |
| **多文件导入与批对比** | 🔴 HIGH | 不能同时加载两组实验数据并对比 |
| **Excel 导入** | 🟡 MEDIUM | 只支持 CSV |
| **大型文件异步导入** | 🟡 MEDIUM | CSV 读取在 UI 线程，大文件会卡死 |
| **存储目录管理** | 🟢 LOW | `v/data` 硬编码，不能「打开其他仓库」 |

---

## 五、可视化与交互

### 现状
- 条件格式表格、2D 散点图、2D Pareto 前沿、折线趋势图、异常贡献度条形图

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **直方图/分布图** | 🟡 MEDIUM | 无各列分布概览 |
| **相关矩阵热力图** | 🟡 MEDIUM | 无 pairwise 相关性可视化 |
| **箱线图/小提琴图** | 🟡 MEDIUM | 无批次间分布对比 |
| **三元相图 (ternary plot)** | 🟡 MEDIUM | 3 组分配方空间无专用图 |
| **平行坐标图** | 🟢 LOW | N 目标权衡 |
| **交互式约束编辑** | 🟡 MEDIUM | comp_bounds/process_bounds 由代码硬算，无 UI 编辑 |

---

## 六、用户与工作流管理

### 现状
- 单用户、单窗口、无项目概念

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **项目/工作区管理** | 🟡 MEDIUM | 不能保存/打开项目（UI 状态、列配置、标签页等） |
| **多用户/权限** | 🟢 LOW | 无登录，无角色 |
| **撤销 (undo)** | 🟡 MEDIUM | 表格编辑无撤销栈 |
| **未保存变更提示** | 🟡 MEDIUM | 修改后关闭无提醒 |
| **报告生成** | 🟡 MEDIUM | 无 PDF/HTML 报告输出 |

---

## 七、测试与基础设施

### 现状
- 8 个测试文件，核心算法测试覆盖良好
- `requirements.txt` 有 `pytest`, `mypy`

### 缺口

| 功能 | 严重度 | 说明 |
|------|--------|------|
| **GUI 组件测试** | 🔴 HIGH | staging_panel, canvas, table_view, in_situ_widget 均无测试 |
| **端到端管线测试** | 🔴 HIGH | import → analytics → BO → VCS → display 无自动化测试 |
| **CI/CD 流水线** | 🔴 HIGH | 无 GitHub Actions / 其他 CI |
| **覆盖率工具** | 🟡 MEDIUM | 无 pytest-cov，无 .coveragerc |
| **pytest 配置** | 🟡 MEDIUM | 无 conftest.py，无 markers，无 pyproject.toml 配置 |
| **基于属性的测试 (hypothesis)** | 🟡 MEDIUM | 对 ILR 往返、Pareto 正确性、VCS 确定性等适用 |
| **性能/规模测试** | 🟡 MEDIUM | 大数据量（1000+ commit、100+ 特征）无验证 |
| **mypy 配置** | 🟢 LOW | 有 mypy 依赖但无配置文件 |
| **ImportTypeInferenceEngine 测试** | 🟢 LOW | 列角色推理逻辑完全未测试 |

---

## 优先级总结

| 优先级 | 内容 |
|--------|------|
| **P0 — 必须补** | commit log 浏览、历史 checkout、非表格文件附件、实验元数据入 VCS、数据导出、谱图处理、GUI 组件测试、端到端测试 |
| **P1 — 尽快补** | 分支/标签、搜索历史、回滚、真正的贝叶斯模型、直方图/箱线图/相关矩阵、多文件导入对比、CI/CD |
| **P2 — 锦上添花** | 三元相图、平行坐标、DoE、批次校正、UQ、报告生成、undo、项目工作区、hypothesis 测试、性能测试 |
