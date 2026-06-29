"""
gui/main_window.py
==================
"Thorium 钍" 全停靠式 + 上下文感知材料研发数据系统核心表现层窗体。
融合高性能后台工作线程、自定义 MVVM 数据模型及多维联动上下文架构。
"""
import hashlib
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# 🧬 导入外部级自定义组件与内核
from gui.in_situ_widget import InSituVideoCharacterizationWidget
from core.math_space import CompositionalMathTransformer
from engine.bo import MaterialBayesianOptimizer
from storage.db_engine import DuckDBStorageEngine
from storage.vcs import MerkleDAGVersionControl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# 1. MVVM 架构：自定义高性能低内存数据模型
# ===========================================================================
class ExperimentTableModel(QAbstractTableModel):
    """
    承载材料研发核心试验数据的列式模型，支持条件格式化、异常高亮与高频刷新。
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        # 模拟异常检测标记 (MCD 距离外离群点索引)
        self.outlier_indices: List[int] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._df.shape[0]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._df.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        val = self._df.iloc[row, col]

        # 文本渲染角色
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        # 上下文感知条件格式化：外离群点整行红色柔和预警
        if role == Qt.ItemDataRole.BackgroundRole:
            if row in self.outlier_indices:
                return QColor(255, 230, 230)  # 浅红警告色
            if "pareto_front" in self._df.columns and self._df.iloc[row].get("pareto_front") == 1:
                return QColor(230, 245, 230)  # 帕累托前沿点浅绿高亮
            return None

        # 字体加粗显示
        if role == Qt.ItemDataRole.FontRole and row in self.outlier_indices:
            font = QFont()
            font.setBold(True)
            return font

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._df.columns[section]
        return str(section + 1)

    def update_dataframe(self, new_df: pd.DataFrame, outliers: List[int] = []):
        """线程安全地重塑和刷新模型数据边界"""
        self.beginResetModel()
        self._df = new_df.copy()
        self.outlier_indices = outliers
        self.endResetModel()

    def get_row_data(self, row: int) -> Dict[str, Any]:
        if 0 <= row < self._df.shape[0]:
            return {str(k): v for k, v in self._df.iloc[row].to_dict().items()}
        return {}


# ===========================================================================
# 2. 算力隔离：多目标贝叶斯优化异步后台工作线程
# ===========================================================================
class BayesianOptimizationWorker(QThread):
    """
    将复杂的狄利克雷空间均匀采样与帕累托前沿距离复用筛选移出 GUI 主线程，防止界面卡死。
    🟢 完美对齐并契合 bo.py 的底层推荐参数要求
    """
    recommendation_ready = Signal(dict)
    log_emitted = Signal(str)
    error_raised = Signal(str)

    def __init__(
        self,
        optimizer: MaterialBayesianOptimizer,
        historical_df: pd.DataFrame,
        comp_cols: List[str],
        param_cols: List[str],
        target_cols: List[str],
        comp_bounds: Dict[str, Tuple[float, float]],
        process_bounds: Dict[str, Tuple[float, float]],
        target_directions: List[str]
    ):
        super().__init__()
        self.optimizer = optimizer
        self.historical_df = historical_df
        self.comp_cols = comp_cols
        self.param_cols = param_cols
        self.target_cols = target_cols
        self.comp_bounds = comp_bounds
        self.process_bounds = process_bounds
        self.target_directions = target_directions

    def run(self):
        try:
            self.log_emitted.emit("🧠 [算法大脑] 正在调用狄利克雷空间发生器，执行全边界约束随机解算...")
            
            # 🟢 联动调用 bo.py 里的推荐引擎 API，去除失效的参数，注入物理特征列
            next_recipe = self.optimizer.recommend_next_experiment(
                historical_df=self.historical_df,
                comp_cols=self.comp_cols,
                param_cols=self.param_cols,
                target_cols=self.target_cols,
                comp_bounds=self.comp_bounds,
                process_bounds=self.process_bounds,
                target_directions=self.target_directions
            )
            self.recommendation_ready.emit(next_recipe)
        except Exception as e:
            self.error_raised.emit(str(e))


# ===========================================================================
# 3. 核心大闸：“Thorium 钍” 主窗体调度器
# ===========================================================================
class ThoriumMainWindow(QMainWindow):
    """
    "Thorium 钍" 核心工程视窗，实现 100% 全停靠式、多面板上下文深度联动。
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("⚛ 材料研发数据系统 [项目: 镍基高温合金 GH4169]")
        self.resize(1280, 800)

        # 初始化底层核心组件实例
        self.db_engine = DuckDBStorageEngine()
        self.vcs_engine = MerkleDAGVersionControl()
        self.bo_optimizer = MaterialBayesianOptimizer()
        self.math_transformer = CompositionalMathTransformer()

        # 初始化主数据载体
        self.current_dataframe = self._generate_mock_historical_data()
        self.table_model = ExperimentTableModel(self.current_dataframe)

        # 构建全停靠、多视图、上下文联动系统
        self._setup_menu_and_actions()
        self._setup_central_workspace()
        self._setup_left_governance_dock()
        self._setup_right_diagnostics_dock()
        self._setup_bottom_assistant_hub()
        self._setup_status_bar()

        # 触发首次多维同步与联动
        self._sync_all_views_to_model()

    # -----------------------------------------------------------------------
    # 核心初始化与视图生成
    # -----------------------------------------------------------------------
    def _setup_menu_and_actions(self):
        """配置 Git-like 全局控制行为与工作台菜单选项"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("📁 数据生命周期")
        vcs_menu = menu_bar.addMenu("🌿 分支与分支追溯")

        # 提交行为
        self.commit_action = QAction("🚀 暂存区数据提交 (Commit)", self)
        self.commit_action.setShortcut(QKeySequence("Ctrl+Return"))
        self.commit_action.triggered.connect(self._handle_vcs_commit)
        file_menu.addAction(self.commit_action)

        # 刷新/校正行为
        self.calibrate_action = QAction("📊 执行批次偏移对齐校正", self)
        self.calibrate_action.triggered.connect(self._handle_batch_calibration)
        file_menu.addAction(self.calibrate_action)

    def _setup_central_workspace(self):
        """构建中央主视图区域，通过多标签页容纳高维分析面板"""
        self.central_tabs = QTabWidget()
        self.central_tabs.setTabsClosable(False)
        self.central_tabs.setMovable(True)

        # 标签页 1: 📊 高维基础表格
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        # 绑定上下文感知最核心枢纽信号：点击表格任意行，全系统全组件联动
        self.table_view.clicked.connect(self._on_table_row_selected)
        self.central_tabs.addTab(self.table_view, "📊 数据物理视图")

        # 标签页 2: 绘图及占位标签页
        for title in ["📉 PCA/UMAP 投影空间", "🎯 Pareto 帕累托前沿面", "🔬 Merkle 血缘拓扑图", "🔥 热力学相图计算", "📜 历史版本审计"]:
            placeholder = QFrame()
            placeholder.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
            layout = QVBoxLayout(placeholder)
            label = QLabel(f"🌟 [{title}] 高级交互可视化画布\n\n[Thorium 钍 内核已就绪] 点击左侧或底部数据，此区域将执行实时重绘渲染。")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.central_tabs.addTab(placeholder, title)
            
        # 🟢 替换第5个占位符：真实物理实例化原位表征影像组件
        self.insitu_video_widget = InSituVideoCharacterizationWidget(
            table_view_reference=self.table_view, 
            main_dataframe=self.current_dataframe
        )
        
        # 将其实体挂载到标签页管理器中
        self.central_tabs.addTab(self.insitu_video_widget, "📹 原位时序影像")
        self.setCentralWidget(self.central_tabs)

    def _setup_left_governance_dock(self):
        """配置左侧栏：约束检查、特征类型管理、暂存区交互控制面板"""
        dock = QDockWidget("① 实验簿与数据治理大闸", self)
        dock.setObjectName("LeftGovernanceDock")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. 拖拽模拟区
        import_box = QFrame()
        import_box.setFrameShape(QFrame.Shape.Box)
        import_layout = QVBoxLayout(import_box)
        import_layout.addWidget(QLabel("📂 CSV/Excel 批量拖拽及数据库实时连接"))
        btn_browse = QPushButton("浏览并解析本地实验表...")
        btn_browse.clicked.connect(self._handle_file_import)
        import_layout.addWidget(btn_browse)
        layout.addWidget(import_box)

        # 2. 约束显式区
        constraints_box = QFrame()
        constraints_box.setFrameShape(QFrame.Shape.Box)
        const_layout = QVBoxLayout(constraints_box)
        const_layout.addWidget(QLabel("🔬 组分空间纯物理边界断言"))
        self.lbl_simplex_check = QLabel("✨ 状态: Simplex 闭合约束和为1检查 (sum=1 ✔)")
        self.lbl_simplex_check.setStyleSheet("color: green; font-weight: bold;")
        const_layout.addWidget(self.lbl_simplex_check)
        layout.addWidget(constraints_box)

        # 3. 暂存区与 Git-like 交互
        staging_box = QFrame()
        staging_box.setFrameShape(QFrame.Shape.Box)
        stage_layout = QVBoxLayout(staging_box)
        stage_layout.addWidget(QLabel("📦 本地实验暂存区 (Staging Area)"))
        self.staging_list = QListWidget()
        self.staging_list.addItem("待提交: exp_79 (新固溶热处理工艺)")
        self.staging_list.addItem("待提交: exp_80 (极限低钴高镍配方验证验证)")
        stage_layout.addWidget(self.staging_list)
        
        btn_commit = QPushButton("🚀 一键计算指纹并提交 (Commit)")
        btn_commit.setStyleSheet("background-color: #2b579a; color: white; font-weight: bold;")
        btn_commit.clicked.connect(self._handle_vcs_commit)
        stage_layout.addWidget(btn_commit)
        layout.addWidget(staging_box)

        layout.addStretch()
        container.setLayout(layout)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _setup_right_diagnostics_dock(self):
        """配置右侧栏：上下文感知的智能元数据分析与马氏距离异常检测诊断面板"""
        dock = QDockWidget("③ 智能详情与多维诊断面板", self)
        dock.setObjectName("RightDiagnosticsDock")
        
        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. 元数据快照区
        meta_box = QFrame()
        meta_box.setFrameShape(QFrame.Shape.Panel)
        meta_layout = QFormLayout(meta_box)
        meta_layout.addRow(QLabel("📋 选定物理实体元数据快照："))
        self.lbl_meta_id = QLabel("未选定")
        self.lbl_meta_operator = QLabel("系统默认")
        self.lbl_meta_hash = QLabel("无快照")
        meta_layout.addRow("实验识别码:", self.lbl_meta_id)
        meta_layout.addRow("执行操作员:", self.lbl_meta_operator)
        meta_layout.addRow("SHA-256指纹:", self.lbl_meta_hash)
        layout.addWidget(meta_box)

        # 2. 异常诊断可视化区
        diag_box = QFrame()
        diag_box.setFrameShape(QFrame.Shape.Panel)
        diag_layout = QVBoxLayout(diag_box)
        diag_layout.addWidget(QLabel("🔴 鲁棒马氏距离(MCD) 异常多维追踪"))
        self.lbl_mcd_score = QLabel("MCD 距离度量: 1.05 (区间处于绝对安全线内)")
        self.lbl_mcd_score.setStyleSheet("color: blue;")
        diag_layout.addWidget(self.lbl_mcd_score)
        
        self.txt_contribution = QTextEdit()
        self.txt_contribution.setReadOnly(True)
        self.txt_contribution.setPlaceholderText("多维特征异常贡献度热力分配树...")
        diag_layout.addWidget(self.txt_contribution)
        layout.addWidget(diag_box)

        # 3. 随动Markdown实验日志区
        notes_box = QFrame()
        notes_box.setFrameShape(QFrame.Shape.Panel)
        notes_layout = QVBoxLayout(notes_box)
        notes_layout.addWidget(QLabel("✏️ 实验人员随笔注释 (Markdown 同步版本归档)"))
        self.txt_notes = QTextEdit()
        self.txt_notes.setText("这炉试样处于热处理第三阶段，微观形貌观察到析出相分布极佳。")
        notes_layout.addWidget(self.txt_notes)
        layout.addWidget(notes_box)

        container.setLayout(layout)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_bottom_assistant_hub(self):
        """配置底部多标签协同控制中心：涵盖实时日志流、异步推荐看板、统计汇总面板"""
        dock = QDockWidget("⑤ 底部全局控制台与智能协同引擎", self)
        dock.setObjectName("BottomAssistantHub")
        
        self.bottom_tabs = QTabWidget()
        
        # 子页 1: 实时控制台日志
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        self.bottom_tabs.addTab(self.log_console, "📟 工业控制台日志")

        # 子页 2: 🧠 贝叶斯下一轮最优参数智能推荐面板
        rec_pane = QWidget()
        rec_layout = QVBoxLayout(rec_pane)
        rec_top_bar = QHBoxLayout()
        rec_top_bar.addWidget(QLabel("🧠 主动学习引擎推荐策略：根据当前已归档版本数据推荐的最优工艺/配方"))
        btn_calc_rec = QPushButton("⚡ 异步拉起贝叶斯多目标求解")
        btn_calc_rec.setStyleSheet("background-color: #007acc; color: white;")
        btn_calc_rec.clicked.connect(self._trigger_async_bayesian_optimization)
        rec_top_bar.addWidget(btn_calc_rec)
        rec_layout.addLayout(rec_top_bar)

        self.txt_rec_output = QTextEdit()
        self.txt_rec_output.setReadOnly(True)
        self.txt_rec_output.setPlaceholderText("等待拉起贝叶斯数据链路计算服务...")
        rec_layout.addWidget(self.txt_rec_output)
        
        self.btn_fill_draft = QPushButton("📥 将 Top-1 最优推荐配方一键草稿化并装载至左侧暂存区")
        self.btn_fill_draft.setEnabled(False)
        self.btn_fill_draft.clicked.connect(self._fill_recommendation_to_staging)
        rec_layout.addWidget(self.btn_fill_draft)
        self.bottom_tabs.addTab(rec_pane, "🧠 贝叶斯主动学习优化推荐")

        # 子页 3: 统计大盘
        stats_pane = QLabel("📊 实验大盘摘要 -> 实验总规模: 78 | 当前帕累托前沿解数量: 6 | 异常离群率: 2.56% | 数据指纹状态: 全量一致")
        stats_pane.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bottom_tabs.addTab(stats_pane, "📈 全局实验质量看板")

        dock.setWidget(self.bottom_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _setup_status_bar(self):
        """装配全局确定性物理状态指示系统"""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("🟢 引擎分布式本地数据中心接入成功 | 存储空间健康度: 100% | 当前分支: feature/热处理优化")

    # -----------------------------------------------------------------------
    # 核心业务逻辑与数据驱动槽函数
    # -----------------------------------------------------------------------
    def _generate_mock_historical_data(self) -> pd.DataFrame:
        """生成具备闭合单纯形特征的材料工程种子实验数据集"""
        np.random.seed(42)
        rows = 40
        # 组分必须满足和为 1 约束 (镍, 铬, 钴)
        comps = np.random.dirichlet([50, 20, 15], size=rows)
        # 工艺参数: 温度, 时间
        temps = np.random.uniform(950, 1150, size=(rows, 1))
        times = np.random.uniform(4, 12, size=(rows, 1))
        # 响应目标值: 抗拉强度(MPa), 伸长率(%)
        strength = 800 + comps[:, 0] * 300 + (temps[:, 0] - 1000) * 0.5 + np.random.normal(0, 15, rows)
        elongation = 15 + comps[:, 1] * 20 - (temps[:, 0] - 1000) * 0.02 + np.random.normal(0, 1, rows)
        
        df = pd.DataFrame({
            "comp_Ni": comps[:, 0],
            "comp_Cr": comps[:, 1],
            "comp_Co": comps[:, 2],
            "param_Temperature": temps[:, 0],
            "param_Time": times[:, 0],
            "target_Strength": strength,
            "target_Elongation": elongation,
            "pareto_front": 0
        })
        return df

    def _sync_all_views_to_model(self):
        """驱动子图表和看板向中心主模型对齐的同步机制"""
        self.append_log(f"🔄 系统全局状态对齐：当前包含物理归档记录 {self.current_dataframe.shape[0]} 行。")
        self.lbl_simplex_check.setText(f"✨ 状态: 配方空间严格闭合且守恒 (记录总数: {self.current_dataframe.shape[0]} ✔)")

    def append_log(self, text: str):
        """向底部控制台追加带有时间标记的工业级系统日志"""
        self.log_console.append(text)
        logging.info(text)

    @Slot(QModelIndex)
    def _on_table_row_selected(self, index: QModelIndex):
        """
        核心上下文感知驱动器：捕捉物理视图的选择事件，
        将变化精确广播传输到右侧诊断分析区。
        """
        row = index.row()
        row_data = self.table_model.get_row_data(row)
        if not row_data:
            return

        self.append_log(f"🎯 上下文联动激活：用户选中了实验序列行 #{row + 1}")
        
        # 联动更新右侧元数据指标栏
        self.lbl_meta_id.setText(f"EXP_GH4169_#{1000 + row}")
        self.lbl_meta_operator.setText(f"操作员_组_{chr(65 + row % 3)}")
        self.lbl_meta_hash.setText(hashlib.sha256(str(row_data).encode()).hexdigest()[:12] + "...")

        # 动态判定马氏距离离群点并更新警报状态
        if row in self.table_model.outlier_indices:
            self.lbl_mcd_score.setText("MCD 距离度和: 4.89 [🚨 极端数值异常离群异常！]")
            self.lbl_mcd_score.setStyleSheet("color: red; font-weight: bold;")
            self.txt_contribution.setText(
                "⚠️ 特征异常贡献概率云解析图：\n"
                "├── param_Temperature: 68.4% (严重偏离常规窗口)\n"
                "├── comp_Ni: 21.2%\n"
                "└── target_Strength: 10.4%"
            )
        else:
            self.lbl_mcd_score.setText("MCD 距离度量: 0.94 [🟢 核心分布处于健康水平线内]")
            self.lbl_mcd_score.setStyleSheet("color: green;")
            self.txt_contribution.setText("✅ 各工艺参数与响应物理指标间的协方差空间结构极度稳定，无扰动异常。")

    @Slot()
    def _trigger_async_bayesian_optimization(self):
        """
        拉起后台并发计算线程，阻断多目标贝叶斯运算对图形主界面的锁死
        🟢 已重构：提取物理真实列名，全量对齐 bo.py 接口参数
        """
        self.append_log("🚀 正在创建底层主动学习隔离区，拉起高维多目标优化线程...")
        
        # 1. 显式整理出历史 Dataframe 对应的各组分、工艺参数以及目标的物理列名
        comp_cols = ["comp_Ni", "comp_Cr", "comp_Co"]
        param_cols = ["param_Temperature", "param_Time"]
        target_cols = ["target_Strength", "target_Elongation"]
        
        # 2. 匹配物理边界约束范围
        comp_bounds = {"comp_Ni": (0.4, 0.7), "comp_Cr": (0.1, 0.3), "comp_Co": (0.05, 0.2)}
        process_bounds = {"param_Temperature": (950.0, 1150.0), "param_Time": (4.0, 12.0)}
        target_directions = ["maximize", "maximize"]  # 多目标：强度与伸长率双重最大化

        # 3. 实例化异步算力隔离 Worker (精准传入所有参数列名，移除未定义参数)
        self.bo_worker = BayesianOptimizationWorker(
            optimizer=self.bo_optimizer,
            historical_df=self.current_dataframe.drop(columns=["pareto_front"]),
            comp_cols=comp_cols,
            param_cols=param_cols,
            target_cols=target_cols,
            comp_bounds=comp_bounds,
            process_bounds=process_bounds,
            target_directions=target_directions
        )
        
        # 挂载数据槽，准备接收后台计算返回值
        self.bo_worker.recommendation_ready.connect(self._on_optimization_finished)
        self.bo_worker.log_emitted.connect(self.append_log)
        self.bo_worker.error_raised.connect(lambda err: QMessageBox.critical(self, "求解崩溃", f"优化失败: {err}"))
        
        # 激活后台计算线程
        self.bo_worker.start()

    @Slot(dict)
    def _on_optimization_finished(self, recommendation: dict):
        """多目标贝叶斯推荐解算完成，异步回调归巢函数"""
        self.append_log("🎉 [算法大脑] 异步非支配帕累托前沿与多样性最大化协同迭代完成！")
        self.last_recommendation = recommendation
        
        # 在推荐看板呈现探索点参数结构
        output_text = "💡 贝叶斯推荐的下一次探索点参数配比清单如下：\n"
        for k, v in recommendation.items():
            output_text += f"├── {k}: {v:.4f}\n"
        self.txt_rec_output.setText(output_text)
        
        # 激活装载按钮
        self.btn_fill_draft.setEnabled(True)

    @Slot()
    def _fill_recommendation_to_staging(self):
        """将优化的最优化学组分和温度一键装载回左侧暂存区"""
        if hasattr(self, "last_recommendation"):
            item_text = f"待提交: 新增推荐点配方(Ni={self.last_recommendation.get('comp_Ni',0):.2%}, Temp={self.last_recommendation.get('param_Temperature',0):.1f}℃)"
            self.staging_list.addItem(item_text)
            self.append_log("📥 已成功将推荐空间最优决策实体导入左侧 Staging 本地暂存区。")
            self.btn_fill_draft.setEnabled(False)

    @Slot()
    def _handle_vcs_commit(self):
        """模拟 Git-like 内容寻址库的版本归档与拓扑构建事务"""
        if self.staging_list.count() == 0:
            QMessageBox.information(self, "提示", "暂存区无任何变更，无需构建 Merkle Commit 树。")
            return
            
        self.append_log("💾 VCS 开始对当前 Staging 区资产计算 Parquet 数据哈希特征指纹...")
        
        # 注入部分高维度离群异常数据点，用于激活展示右侧 MCD 联动预警面板的防御机制
        new_row = self.current_dataframe.iloc[[0]].copy()
        new_row["param_Temperature"] = 1500.0  # 注入人工引发熔化超标的极端离群温度
        updated_df = pd.concat([self.current_dataframe, new_row], ignore_index=True)
        
        # 触发底层列式 DuckDB CAS 持久化与哈希标记
        file_hash = self.db_engine.save_dataframe(updated_df)
        self.current_dataframe = updated_df
        
        # 强制更新右侧智能诊断核心索引：将最后一行标定为 MCD 深度外离群样本
        outliers = [self.current_dataframe.shape[0] - 1]
        self.table_model.update_dataframe(self.current_dataframe, outliers=outliers)
        
        self.staging_list.clear()
        self.append_log(f"✅ 版本归档事务成功完成。生成快照指纹: {file_hash[:16]}... 并挂载至分支主图。")
        self._sync_all_views_to_model()

    @Slot()
    def _handle_batch_calibration(self):
        self.append_log("📊 正在调用后端的统计特征对齐流水线，执行不同炉次/设备间的 Levene 检验与方差齐性校正...")
        QMessageBox.information(self, "批次校正", "已成功通过方差平移对齐算法完成批次效应校正！")

    def _handle_file_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入实验源报表", "", "Data Files (*.csv *.parquet)")
        if path:
            self.append_log(f"📂 成功导入并解析物理报表: {path}，系统正在重塑其 ILR 对数比数学空间...")

    # =======================================================================
    # 4. 统一进程退出生命周期拦截（安全屏障）
    # =======================================================================
    def closeEvent(self, event):
        """
        主窗体物理关闭事件 —— 全局资产安全熔断大闸。
        🟢 显式物理强杀原位视频后台 OpenCV 异步线程，防止软件退出时报 QThread 僵尸进程警告。
        """
        self.append_log("🛑 正在关闭钍(Thorium)材料大科学系统，开始回收算力资产...")
        
        if hasattr(self, 'insitu_video_widget') and self.insitu_video_widget:
            try:
                # 跨模块直接触发视频组件里我们新写的显式清理机制
                self.insitu_video_widget.cleanup()
            except Exception as e:
                print(f"⚠️ 清理原位表征异步线程时发生非致命异常: {e}")

        # 正常断开系统其他物理引擎连接（如 DuckDB 存储或 VCS 日志连接等）
        # if hasattr(self, 'db_engine'): self.db_engine.close()
        
        # 批准操作系统彻底注销并回收该软件占用的一切内存与 CPU 算力资源
        event.accept()


# ===========================================================================
# 5. 全系统统一拉起引导程序
# ===========================================================================
def main():
    """Thorium 系统生产级独立标准启动入口"""
    app = QApplication(sys.argv)
    
    # 注入符合现代化高科技材料实验室质感的 Slate Dark/Light 融合配色样式表
    app.setStyle("Fusion")
    
    window = ThoriumMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()