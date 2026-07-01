"""
CachedSpectralAligner 单元测试。

注意: core.spectral 模块在当前主干尚不可用，所有测试标记为 skip。
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.skip(reason="core.spectral module not yet available on main branch")


@pytest.fixture
def aligner():
    pytest.skip("core.spectral not available")


@pytest.fixture
def sample_data():
    return None