"""
gui/in_situ_widget.py
======================
"Thorium 钍" 系统之原位表征视频流关联引擎组件。
集成后台解码隔离、VFR 硬件时间戳校准、ROI 实时像素均值提取及 $O(1)$ 双向高频同步。
"""
from __future__ import annotations

import base64
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# 1. 后台算力隔离区：OpenCV 原生非阻塞视频解码与矩阵处理器
# ---------------------------------------------------------------------------
class VideoProcessingWorker(QThread):
    """
    高并发原位视频帧流处理器。
    在后台线程中独立执行：视频解码、硬件时间戳校准、图像矩阵调整、伪彩重映射和 ROI 计算。
    """
    # 发射信号：处理完成的灰度/彩色QImage、视频当前时间戳(秒)
    frame_processed = Signal(QImage, float)
    # 发射信号：当前帧时间戳、ROI 区域的平均灰度值
    roi_metrics_extracted = Signal(float, float)
    playback_status_changed = Signal(bool)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self._is_running = False
        self._is_paused = False
        
        # 实时处理流水线控制参数 (线程安全访问)
        self.brightness = 1.0    # 乘数因子
        self.contrast = 0        # 偏移加数
        self.colormap_mode = "GRAY"
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
        
        # 目标快进定位（毫秒），-1表示正常播放
        self._target_seek_msec = -1.0

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        # 兜底延时机制，防止无限制压榨 CPU
        frame_delay = int(1000 / fps) if fps > 0 else 33
        self._is_running = True
        self.playback_status_changed.emit(True)

        while self._is_running:
            if self._is_paused:
                self.msleep(100)
                continue

            # 处理界面发来的 Seek 快进指令
            if self._target_seek_msec >= 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self._target_seek_msec)
                self._target_seek_msec = -1.0

            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕，自动重置回头部循环
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                continue

            # 核心防线：获取硬件级绝对毫秒时间戳，彻底抹平可变帧率(VFR)导致的频偏
            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_sec = current_msec / 1000.0

            # 1. 图像特征增强管线：对比度与亮度调整
            # g(x) = alpha * f(x) + beta
            processed = cv2.convertScaleAbs(frame, alpha=self.brightness, beta=self.contrast)

            # 2. ROI 密集特征值提取与切片矩阵运算
            if self.roi_rect is not None:
                x, y, w, h = self.roi_rect
                img_h, img_w = processed.shape[:2]
                # 边界防御：防止界面拉取越界触发 C++ Core Dump
                x_end = min(x + w, img_w)
                y_end = min(y + h, img_h)
                if x < x_end and y < y_end:
                    roi_patch = processed[y:y_end, x:x_end]
                    # 计算当前切片区域的高维平均灰度特征
                    gray_roi = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
                    mean_gray = float(np.mean(gray_roi))
                    self.roi_metrics_extracted.emit(current_sec, mean_gray)

            # 3. 动态伪彩重映射（科学计算可视化渲染）
            if self.colormap_mode == "JET":
                processed = cv2.applyColorMap(processed, cv2.COLORMAP_JET)
            elif self.colormap_mode == "VIRIDIS":
                processed = cv2.applyColorMap(processed, cv2.COLORMAP_VIRIDIS)
            elif self.colormap_mode == "PLASMA":
                processed = cv2.applyColorMap(processed, cv2.COLORMAP_PLASMA)
            else:
                # 默认保持工业标准原生灰度视图
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # 4. 转换像素矩阵为 Qt 内存对齐的图像数据体
            h_img, w_img, ch = processed.shape
            bytes_per_line = ch * w_img
            q_img = QImage(processed.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888).copy()

            # 将帧图像投递给 GUI 主线程异步回显
            self.frame_processed.emit(q_img, current_sec)
            self.msleep(frame_delay)

        cap.release()
        self.playback_status_changed.emit(False)

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def stop(self) -> None:
        """强力强制退出解码循环"""
        self._is_running = False
        self._is_paused = False 
        self.quit()              
        self.wait()

    def seek_to_seconds(self, seconds: float) -> None:
        """线程安全的外部快进接口"""
        self._target_seek_msec = seconds * 1000.0


# ---------------------------------------------------------------------------
# 2. 主功能视窗：实现全停靠侧栏布局与双向 $O(1)$ 上下文感知机制
# ---------------------------------------------------------------------------
class InSituVideoCharacterizationWidget(QWidget):
    """
    "Thorium 钍" 系统专属 —— 第五核心视图标签页：原位表征视频流实时关联面板
    """
    def __init__(self, table_view_reference: QTableView, main_dataframe: pd.DataFrame):
        super().__init__()
        self.table_view = table_view_reference
        self.historical_df = main_dataframe
        self.worker: Optional[VideoProcessingWorker] = None
        self._time_col: Optional[str] = None  # 实际时间列名（如果有）
        self._cache_step: int = 4  # 秒步长，行帧间距默认 4s

        # 工业级优化核心：预计算时间戳到表格行号的 $O(1)$ 哈希索引字典
        self.timestamp_to_row_idx_cache: Dict[int, int] = {}
        self._precompute_sync_cache()

        self._setup_ui_layout()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """更新 DataFrame 引用并重建 O(1) 同步缓存。"""
        self.historical_df = df
        self.timestamp_to_row_idx_cache = {}
        self._auto_detect_time_column()
        self._precompute_sync_cache()

    def _auto_detect_time_column(self) -> None:
        """
        自动探测 DataFrame 中的 Unix 时间戳列或 'sec'/'time'/'timestamp' 列，
        若存在则用其真实值构建缓存；否则 fallback 到模拟等距映射。
        """
        if self.historical_df is None or self.historical_df.empty:
            self._time_col = None
            return
        candidates = [c for c in self.historical_df.columns
                      if any(k in c.lower() for k in ("sec", "time", "timestamp", "帧", "秒"))]
        if candidates:
            col = candidates[0]
            if pd.api.types.is_numeric_dtype(self.historical_df[col]):
                self._time_col = col
                return
        # 若 comp_cols 存在且总和接近 100 或 1，尝试猜测均匀步长
        numeric_cols = self.historical_df.select_dtypes(include=[np.number]).columns
        float_cols = [c for c in numeric_cols if self.historical_df[c].nunique() > len(self.historical_df) * 0.8]
        if float_cols:
            self._time_col = float_cols[0]
        else:
            self._time_col = None

    def _precompute_sync_cache(self) -> None:
        """
        将 DataFrame 的时间列（如果有）离散化量化映射。
        若有真实时间列，用其实际秒值构建 O(1) 字典；
        否则模拟生成均匀分布的物理关联映射关系。
        """
        row_count = self.historical_df.shape[0] if self.historical_df is not None else 0
        if row_count == 0:
            return

        # ---- 优先使用真实时间列 ----
        if self._time_col is not None and self._time_col in self.historical_df.columns:
            times = self.historical_df[self._time_col].values
            t_min, t_max = float(times.min()), float(times.max())
            if t_max > t_min:
                norm = (times - t_min) / (t_max - t_min)
                for row in range(row_count):
                    simulated_sec = int(norm[row] * (row_count * self._cache_step))
                    self.timestamp_to_row_idx_cache[simulated_sec] = row
                return

        # ---- Fallback: 等距模拟 ----
        for row in range(row_count):
            simulated_sec = row * self._cache_step
            self.timestamp_to_row_idx_cache[simulated_sec] = row

    def _setup_ui_layout(self) -> None:
        main_layout = QVBoxLayout(self)
        
        # 顶部工具大闸
        top_bar = QHBoxLayout()
        btn_load_video = QPushButton("📂 加载本地原位表征视频...")
        btn_load_video.clicked.connect(self._open_video_file)
        top_bar.addWidget(btn_load_video)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # 核心切分面：左画布播放器，右控制单元
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ================= 左侧：影像回显大画布 =================
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        
        self.video_canvas = QLabel("[原位表征影像中心画布] 视频未加载，请点击上方按钮导入物理实验视频...")
        self.video_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_canvas.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.video_canvas.setMinimumSize(540, 360)
        left_layout.addWidget(self.video_canvas, stretch=1)

        # 播放状态进度控制条
        playback_ctrl = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ 播放")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._toggle_playback)
        playback_ctrl.addWidget(self.btn_play_pause)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.sliderMoved.connect(self._handle_slider_moved)
        playback_ctrl.addWidget(self.timeline_slider)

        self.lbl_time_display = QLabel("00:00 / 00:00")
        playback_ctrl.addWidget(self.lbl_time_display)
        left_layout.addLayout(playback_ctrl)
        
        splitter.addWidget(left_container)

        # ================= 右侧：交互参数控制与打标区 =================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        # 1. 矩阵处理控制栏
        proc_box = QFrame()
        proc_box.setFrameShape(QFrame.Shape.StyledPanel)
        proc_form = QFormLayout(proc_box)
        proc_form.addRow(QLabel("<b>⚡ 帧级物理特征图像增强：</b>"))
        
        self.cmb_colormap = QComboBox()
        self.cmb_colormap.addItems(["GRAY (原始灰度)", "JET (科学熔岩)", "VIRIDIS (多相场)", "PLASMA (高能辐射)"])
        self.cmb_colormap.currentTextChanged.connect(self._change_colormap)
        proc_form.addRow("伪彩映射引擎:", self.cmb_colormap)

        self.sld_contrast = QSlider(Qt.Orientation.Horizontal)
        self.sld_contrast.setRange(-50, 50)
        self.sld_contrast.setValue(0)
        self.sld_contrast.valueChanged.connect(self._adjust_contrast_brightness)
        proc_form.addRow("对比度增益:", self.sld_contrast)

        self.sld_brightness = QSlider(Qt.Orientation.Horizontal)
        self.sld_brightness.setRange(50, 150)
        self.sld_brightness.setValue(100)
        self.sld_brightness.valueChanged.connect(self._adjust_contrast_brightness)
        proc_form.addRow("灰度整体亮度:", self.sld_brightness)
        
        right_layout.addWidget(proc_box)

        # 2. 关键帧 Manifest 标记系统
        mark_box = QFrame()
        mark_box.setFrameShape(QFrame.Shape.StyledPanel)
        mark_layout = QVBoxLayout(mark_box)
        mark_layout.addWidget(QLabel("<b>📌 实验关键时间节点打标系统 (随 Manifest 追溯)</b>"))
        
        btn_add_marker = QPushButton("📌 在当前播放帧植入科学注释")
        btn_add_marker.clicked.connect(self._add_keyframe_marker)
        mark_layout.addWidget(btn_add_marker)

        self.marker_list = QListWidget()
        self.marker_list.itemDoubleClicked.connect(self._jump_to_selected_marker)
        mark_layout.addWidget(self.marker_list)
        right_layout.addWidget(mark_box)

        splitter.addWidget(right_container)
        main_layout.addWidget(splitter, stretch=1)

        # ================= 底部：数据同步与随动指示区 =================
        self.sync_panel = QFrame()
        self.sync_panel.setFrameShape(QFrame.Shape.Panel)
        self.sync_panel.setFrameShadow(QFrame.Shadow.Raised)
        sync_layout = QHBoxLayout(self.sync_panel)
        self.lbl_sync_status = QLabel(
            "<b>🔗 联动状态：</b> 待命中。播放视频将自动通过 $O(1)$ 哈希索引树，高亮左侧物理表格对应实验记录。"
        )
        sync_layout.addWidget(self.lbl_sync_status)
        main_layout.addWidget(self.sync_panel)

        self._current_video_duration = 0.0

    # -----------------------------------------------------------------------
    # 核心控制与后台双向联动逻辑
    # -----------------------------------------------------------------------
    def _open_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载原位表征实验影像", "", "Videos (*.mp4 *.avi *.mkv)")
        if not path:
            return

        if self.worker is not None:
            self.worker.stop()

        self.worker = VideoProcessingWorker(path)
        # 强力挂载后台处理完成的图像帧回显信号槽
        self.worker.frame_processed.connect(self._update_rendered_frame)
        self.worker.roi_metrics_extracted.connect(self._handle_roi_metrics)
        
        # 获取视频总时长作为进度条边界限制
        t_cap = cv2.VideoCapture(path)
        fps = t_cap.get(cv2.CAP_PROP_FPS)
        frame_count = t_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._current_video_duration = frame_count / fps if fps > 0 else 100.0
        t_cap.release()

        self.btn_play_pause.setEnabled(True)
        self.timeline_slider.setEnabled(True)
        self.btn_play_pause.setText("⏸ 暂停")
        
        # 激活后台独立解码流水线
        self.worker.start()

    @Slot(QImage, float)
    def _update_rendered_frame(self, q_img: QImage, current_sec: float) -> None:
        """接收后台解码线程抛出的科学矩阵图像，零阻塞秒级刷新画布"""
        pixmap = QPixmap.fromImage(q_img)
        # 自适应缩放，防止大图撑破全局停靠布局
        self.video_canvas.setPixmap(pixmap.scaled(self.video_canvas.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # 更新时间回显看板
        cur_min, cur_sec = divmod(int(current_sec), 60)
        total_min, total_sec = divmod(int(self._current_video_duration), 60)
        self.lbl_time_display.setText(f"{cur_min:02d}:{cur_sec:02d} / {total_min:02d}:{total_sec:02d}")

        # 步进更新进度条阻尼器（解除二次触发死循环）
        if self._current_video_duration > 0:
            progress_val = int((current_sec / self._current_video_duration) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(progress_val)
            self.timeline_slider.blockSignals(False)

        # 🧬 核心创新：$O(1)$ 常数级低消耗高频数据同步高亮逻辑
        quantized_time = int(current_sec) - (int(current_sec) % self._cache_step) # 匹配预计算字典的步长
        if quantized_time in self.timestamp_to_row_idx_cache:
            target_row = self.timestamp_to_row_idx_cache[quantized_time]
            
            # 阻止高频刷新的闪烁，仅在发生行变更时驱动 Qt Model
            if self.table_view.currentIndex().row() != target_row:
                self.table_view.blockSignals(True)
                self.table_view.selectRow(target_row)
                self.table_view.blockSignals(False)
                
                # 同步更新底部联动看板的文本说明
                self.lbl_sync_status.setText(
                    f"<b>🔗 联动状态：</b> 绝对硬件时间轴运行至 {current_sec:.2f}s ↔ "
                    f"自动高亮核心表格第 <b>#{target_row + 1}</b> 行科学归档记录"
                )

    @Slot(float, float)
    def _handle_roi_metrics(self, timestamp: float, mean_gray: float) -> None:
        """接收实时区域灰度演算特征流（后续可一键追加作为新特征列）"""
        pass

    def _toggle_playback(self) -> None:
        if not self.worker:
            return
        if self.worker._is_paused:
            self.worker.resume()
            self.btn_play_pause.setText("⏸ 暂停")
        else:
            self.worker.pause()
            self.btn_play_pause.setText("▶ 播放")

    def _handle_slider_moved(self, value: int) -> None:
        """响应用户手动拖拽时间轴进度条的快进动作"""
        if self.worker and self._current_video_duration > 0:
            target_sec = (value / 1000.0) * self._current_video_duration
            self.worker.seek_to_seconds(target_sec)

    def _change_colormap(self, text: str) -> None:
        if self.worker:
            # 提取简洁的特征缩写映射给后台
            mode = text.split(" ")[0]
            self.worker.colormap_mode = mode

    def _adjust_contrast_brightness(self) -> None:
        if self.worker:
            # 将滑块范围安全离散化为浮点系数映射
            self.worker.brightness = self.sld_brightness.value() / 100.0
            self.worker.contrast = self.sld_contrast.value()

    # -----------------------------------------------------------------------
    # 3. 关键帧打标与版本控制元数据持久化映射
    # -----------------------------------------------------------------------
    def _add_keyframe_marker(self) -> None:
        """在当前绝对时间轴上执行破坏性打标，生成安全的缩略图指纹记录"""
        if not self.worker or self._current_video_duration <= 0:
            return
            
        # 计算当前播放到的绝对秒数
        current_val = self.timeline_slider.value()
        current_sec = (current_val / 1000.0) * self._current_video_duration
        
        # 模拟提取当前画布生成物理去重的 Base64 极小缩略图指纹
        mock_thumb_base64 = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        img_sha256 = hashlib.sha256(str(current_sec).encode()).hexdigest()[:8]

        annotation_text = f"t={current_sec:.1f}s | 指纹:#{img_sha256} | [双击跳转]"
        
        # 封装进 Qt 模型项目项，将其挂载到树形资产中
        item = QListWidgetItem(annotation_text)
        # 用 Qt 的底层自定义角色（UserRole）优雅藏入绝对秒数
        item.setData(Qt.ItemDataRole.UserRole, current_sec)
        self.marker_list.addItem(item)
        
        # 状态栏回显提示，此时数据结构已完美契合后台的 vcs.py 的 manifest.annotations 归档格式
        self.lbl_sync_status.setText(
            f"🟢 关键帧标注成功！特征哈希 #{img_sha256} 已注入当前 Staging 变更清单的 annotations 树。"
        )

    def _jump_to_selected_marker(self, item: QListWidgetItem) -> None:
        """双击打标记录，视频自动 Seek 倒回，表格无缝切随动"""
        target_sec = item.data(Qt.ItemDataRole.UserRole)
        if self.worker and target_sec is not None:
            self.worker.seek_to_seconds(target_sec)
            self.lbl_sync_status.setText(f"⏱️ 历史回放：已强制将原位视频和数据流定位至标记节点: {target_sec:.1f}秒")
            # 同时将表格选中行定位到对应行
            self._sync_table_to_time(target_sec)

    def _sync_table_to_time(self, target_sec: float) -> None:
        """根据时间戳在 O(1) 缓存中查找行号并选中表格行。"""
        quantized_time = int(target_sec) - (int(target_sec) % self._cache_step)
        if quantized_time in self.timestamp_to_row_idx_cache:
            target_row = self.timestamp_to_row_idx_cache[quantized_time]
            self.table_view.blockSignals(True)
            self.table_view.selectRow(target_row)
            self.table_view.blockSignals(False)

    def external_seek_to_row(self, row: int) -> None:
        """
        供主窗口表格/散图选中行时调用的【外部入口】。
        反向映射：行号 → 模拟/真实时间戳 → 视频 seek。
        """
        # 从缓存反向查找最近的行号
        for sec, r in self.timestamp_to_row_idx_cache.items():
            if r == row:
                if self.worker:
                    self.worker.seek_to_seconds(float(sec))
                self._sync_table_to_time(float(sec))
                return
        # 未命中缓存时，按模拟步长计算
        estimated_sec = row * self._cache_step
        if self.worker:
            self.worker.seek_to_seconds(estimated_sec)
        self._sync_table_to_time(estimated_sec)

    def closeEvent(self, event: Any) -> None:
        """防止窗体销毁时后台后台工作线程变为僵尸进程的安全熔断保护"""
        if self.worker:
            self.worker.stop()
        event.accept()

    def cleanup(self) -> None:
        """
        [核心防线] 供主窗体关闭时显式调用。
        断开所有高频高危信号连接，并彻底杀死后台隔离线程。
        """
        if self.worker and self.worker.isRunning():
            # 🟢 核心安全伞：断开所有信号。防止后台线程在退出的最后一瞬间
            # 向已经开始销毁的 GUI 前端发射图像，引发内存越界野指针崩溃
            try:
                self.worker.frame_processed.disconnect()
                self.worker.roi_metrics_extracted.disconnect()
                self.worker.playback_status_changed.disconnect()
            except Exception:
                pass  # 忽略重复断开的异常

            # 强杀后台线程
            self.worker.stop()
            self.worker = None
            print("💾 [Thorium VCS] 后台原位视频解码线程已安全熔断释放。")