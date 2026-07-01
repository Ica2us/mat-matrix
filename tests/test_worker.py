"""
Worker / WorkerPool 异步调度器单元测试。

使用 pytest-qt 的 qtbot fixture 管理 Qt 事件循环。
"""
from __future__ import annotations

import threading
from typing import Any, List

import pytest
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from gui.worker import Worker, WorkerPool


# ── Tests ────────────────────────────────────────────────────────────────


class TestWorkerExecution:
    """Worker 基本执行测试。"""

    def test_worker_executes_function(self, qtbot):
        """Worker 执行目标函数并通过 result signal 返回结果。"""
        results: List[str] = []

        def target():
            return "hello"

        worker = Worker(target)
        worker.signals.result.connect(lambda v: results.append(v))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert results == ["hello"]

    def test_worker_passes_args_and_kwargs(self, qtbot):
        """Worker 正确传递位置参数和关键字参数。"""
        results: List[int] = []

        def add(a, b, multiplier=1):
            return (a + b) * multiplier

        worker = Worker(add, 3, 4, multiplier=2)  # (3+4)*2 = 14
        worker.signals.result.connect(lambda v: results.append(v))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert results == [14]

    def test_worker_emits_started(self, qtbot):
        """Worker 在执行前发出 started signal。"""
        flags: List[str] = []

        def target():
            return "done"

        worker = Worker(target)
        worker.signals.started.connect(lambda: flags.append("started"))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert "started" in flags

    def test_worker_emits_finished(self, qtbot):
        """Worker 在执行后发出 finished signal。"""
        flags: List[str] = []

        def target():
            return "done"

        worker = Worker(target)
        worker.signals.finished.connect(lambda: flags.append("finished"))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert "finished" in flags

    def test_worker_propagates_exception(self, qtbot):
        """Worker 中抛出异常时通过 error signal 传递。"""
        errors: List[tuple] = []

        def target():
            raise ValueError("测试错误")

        worker = Worker(target)
        worker.signals.error.connect(lambda e: errors.append(e))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert len(errors) >= 1
        _type, msg, _tb = errors[0]
        assert "测试错误" in str(msg)

    def test_worker_progress_callback(self, qtbot):
        """Worker 通过 progress_callback 发出进度信号。"""
        progress_values: List[int] = []

        def target(progress_callback=None):
            for i in range(0, 101, 33):
                if progress_callback:
                    progress_callback(i)
            return "done"

        worker = Worker(target)
        worker.signals.progress.connect(lambda v: progress_values.append(v))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert len(progress_values) >= 2

    def test_worker_cancel_before_run(self, qtbot):
        """cancel() 后的 Worker 不执行目标函数。"""
        results: List[str] = []

        def target():
            return "should_not_run"

        worker = Worker(target)
        worker.signals.result.connect(lambda v: results.append(v))
        worker.cancel()
        QThreadPool.globalInstance().start(worker)
        qtbot.wait(500)
        assert results == []

    def test_worker_with_progress_callback_not_required(self, qtbot):
        """不接受 progress_callback 的函数也可以正常执行。"""
        results = []

        def target_no_progress():
            return 42

        worker = Worker(target_no_progress)
        worker.signals.result.connect(lambda v: results.append(v))
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert results == [42]


class TestWorkerPool:
    """WorkerPool.submit 便捷 API 测试。"""

    def test_pool_submit_calls_on_result(self, qtbot):
        """WorkerPool.submit 正确连接 on_result 回调。"""
        results: List[str] = []

        def target():
            return "pool_result"

        def on_result(val):
            results.append(val)

        worker = WorkerPool.submit(target, on_result=on_result)
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            pass  # worker 已在 submit() 中启动

        assert results == ["pool_result"]

    def test_pool_submit_calls_on_error(self, qtbot):
        """WorkerPool.submit 正确连接 on_error 回调。"""
        errors: List[str] = []

        def target():
            raise RuntimeError("pool error")

        def on_error(exc_info):
            errors.append(str(exc_info))

        worker = WorkerPool.submit(target, on_error=on_error)
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            pass

        assert len(errors) == 1

    def test_pool_submit_calls_on_progress(self, qtbot):
        """WorkerPool.submit 正确连接 on_progress 回调。"""
        progress: List[int] = []

        def target(progress_callback=None):
            for i in range(0, 101, 25):
                if progress_callback:
                    progress_callback(i)
            return "done"

        def on_progress(val):
            progress.append(val)

        worker = WorkerPool.submit(target, on_progress=on_progress)
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            pass

        assert len(progress) >= 2
        assert progress[-1] == 100

    def test_pool_submit_calls_on_finished(self, qtbot):
        """WorkerPool.submit 正确连接 on_finished 回调。"""
        flags: List[str] = []

        def target():
            return "done"

        def on_finished():
            flags.append("finished")

        worker = WorkerPool.submit(target, on_finished=on_finished)
        qtbot.wait_until(lambda: len(flags) > 0, timeout=5000)
        assert "finished" in flags

    def test_pool_submit_returns_worker(self, qtbot):
        """WorkerPool.submit 返回 Worker 实例。"""

        def target():
            return "x"

        worker = WorkerPool.submit(target)
        assert isinstance(worker, Worker)
        qtbot.wait_until(lambda: worker._is_cancelled or True, timeout=5000)

    def test_multiple_workers_concurrent(self, qtbot):
        """多个 Worker 并发执行不丢失结果。"""
        results: List[int] = []
        expected = 10
        finished = [0]

        def target(i):
            return i

        def on_finished():
            finished[0] += 1

        workers = []
        for i in range(expected):
            worker = Worker(target, i)
            worker.signals.result.connect(lambda v: results.append(v))
            worker.signals.finished.connect(on_finished)
            workers.append(worker)

        # 批量提交
        for w in workers:
            QThreadPool.globalInstance().start(w)

        # 等待所有完成
        qtbot.wait_until(lambda: finished[0] >= expected, timeout=10000)

        assert len(results) == expected
        assert sorted(results) == list(range(expected))


class TestWorkerThreadSafety:
    """线程安全性测试。"""

    def test_result_from_different_thread(self, qtbot):
        """验证 result signal 从工作线程发出（非主线程）。"""
        worker_thread_ids: List[int] = []

        def target():
            return threading.get_ident()

        def on_result(thread_id):
            worker_thread_ids.append(thread_id)

        main_thread_id = threading.get_ident()
        worker = Worker(target)
        worker.signals.result.connect(on_result)
        with qtbot.wait_signal(worker.signals.finished, timeout=3000):
            QThreadPool.globalInstance().start(worker)

        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != main_thread_id