#!/usr/bin/env python
"""
mat_matrix_gui — 配方优化系统

用法:
    python main.py          # 直接启动 GUI
    python main.py --help   # 查看选项
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path


def _bootstrap_pythonpath() -> None:
    """将项目根目录加入 sys.path，确保所有 import 路径可解析。"""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> None:
    _bootstrap_pythonpath()

    parser = argparse.ArgumentParser(description="mat_matrix_gui — 配方优化系统")
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="以离屏模式运行（无显示设备时使用，CI 测试用）",
    )
    args = parser.parse_args()

    if args.offscreen:
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import os
    os.environ.setdefault("QT_API", "pyside6")  # 强制 matplotlib 用 PySide6 而非 PyQt6

    from gui.main_window import MainWindow
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()