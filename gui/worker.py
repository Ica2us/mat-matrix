"""
异步工作池 — 基于 QThreadPool + QRunnable 的通用异步调度器。

Worker 包装任意可调用对象，通过 WorkerSignals 向主线程安全传递结果。
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional, Tuple

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class WorkerSignals(QObject):
    """QRunnable 的信号容器（QRunnable 不能直接继承 QObject）。"""

    started = Signal()
    finished = Signal()
    result = Signal(object)  # 任意 Python 对象 (np.ndarray, dict, str, ...)
    error = Signal(tuple)    # (type, message, traceback_text)
    progress = Signal(int)   # 0–100


class Worker(QRunnable):
    """通用异步任务包装器。

    用法:
        worker = Worker(fn, arg1, arg2, kwarg1=val1)
        worker.signals.result.connect(self.on_result)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._is_cancelled = False

    @Slot()
    def run(self) -> None:
        if self._is_cancelled:
            return
        self.signals.started.emit()
        try:
            # 默认注入 progress_callback；若目标函数不接受则回退
            import inspect
            sig = inspect.signature(self.fn)
            kwargs = dict(self.kwargs)
            if "progress_callback" in sig.parameters:
                kwargs["progress_callback"] = self.signals.progress.emit
            result = self.fn(*self.args, **kwargs)
            if not self._is_cancelled:
                self.signals.result.emit(result)
        except Exception:
            tb = traceback.format_exc()
            exc_type, exc_val, _ = __import__("sys").exc_info()
            exc_info = (exc_type, str(exc_val), tb)
            self.signals.error.emit(exc_info)
        finally:
            self.signals.finished.emit()

    def cancel(self) -> None:
        """请求取消（运行中的任务需自行检查 _is_cancelled）。"""
        self._is_cancelled = True


class WorkerPool:
    """便捷提交器 — 链式 API。"""

    @staticmethod
    def submit(
        fn: Callable[..., Any],
        on_result: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Tuple[Any, ...]], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Worker:
        worker = Worker(fn, *args, **kwargs)
        if on_result is not None:
            worker.signals.result.connect(on_result)
        if on_error is not None:
            worker.signals.error.connect(on_error)
        if on_progress is not None:
            worker.signals.progress.connect(on_progress)
        if on_finished is not None:
            worker.signals.finished.connect(on_finished)
        QThreadPool.globalInstance().start(worker)
        return worker