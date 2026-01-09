"""
pytest 共享配置和 fixtures

提供测试所需的通用 fixtures，包括：
- 模拟K线数据
- 威科夫事件样本
- 数据快照加载器
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from wyckoff_ai.schemas import WyckoffEvent


# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "data"
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """测试数据目录"""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def snapshots_dir() -> Path:
    """快照数据目录"""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOTS_DIR


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    生成模拟K线数据
    
    包含100根K线，带有典型的价格走势模式
    """
    np.random.seed(42)
    n = 100
    
    # 生成基础时间序列
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    
    # 生成价格走势（带有趋势和波动）
    base_price = 40000
    trend = np.cumsum(np.random.randn(n) * 50)  # 随机漫步
    noise = np.random.randn(n) * 100
    close_prices = base_price + trend + noise
    
    # 确保价格为正
    close_prices = np.maximum(close_prices, 1000)
    
    # 生成 OHLC
    high_prices = close_prices + np.abs(np.random.randn(n) * 200)
    low_prices = close_prices - np.abs(np.random.randn(n) * 200)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # 生成成交量
    base_volume = 1000
    volume = base_volume + np.abs(np.random.randn(n) * 500)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })
    
    return df


@pytest.fixture
def sample_ohlcv_with_features(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """带有特征的K线数据"""
    from wyckoff_ai.features import compute_features
    return compute_features(sample_ohlcv_df)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """
    上涨趋势的K线数据
    
    用于测试看涨事件检测
    """
    np.random.seed(123)
    n = 100
    
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    
    # 明显的上涨趋势
    base_price = 40000
    trend = np.linspace(0, 5000, n)  # 稳定上涨
    noise = np.random.randn(n) * 50
    close_prices = base_price + trend + noise
    
    high_prices = close_prices + np.abs(np.random.randn(n) * 100)
    low_prices = close_prices - np.abs(np.random.randn(n) * 100)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    volume = 1000 + np.abs(np.random.randn(n) * 300)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """
    下跌趋势的K线数据
    
    用于测试看跌事件检测
    """
    np.random.seed(456)
    n = 100
    
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    
    # 明显的下跌趋势
    base_price = 45000
    trend = np.linspace(0, -5000, n)  # 稳定下跌
    noise = np.random.randn(n) * 50
    close_prices = base_price + trend + noise
    
    high_prices = close_prices + np.abs(np.random.randn(n) * 100)
    low_prices = close_prices - np.abs(np.random.randn(n) * 100)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    volume = 1000 + np.abs(np.random.randn(n) * 300)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })


@pytest.fixture
def ranging_df() -> pd.DataFrame:
    """
    横盘震荡的K线数据
    
    用于测试区间识别
    """
    np.random.seed(789)
    n = 100
    
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    
    # 在区间内震荡
    base_price = 42000
    oscillation = np.sin(np.linspace(0, 8 * np.pi, n)) * 500
    noise = np.random.randn(n) * 50
    close_prices = base_price + oscillation + noise
    
    high_prices = close_prices + np.abs(np.random.randn(n) * 80)
    low_prices = close_prices - np.abs(np.random.randn(n) * 80)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    volume = 1000 + np.abs(np.random.randn(n) * 200)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })


@pytest.fixture
def sample_events() -> list[WyckoffEvent]:
    """示例威科夫事件列表"""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    return [
        WyckoffEvent(
            type="SC",
            ts=(base_time + timedelta(hours=10)).isoformat(),
            price=39000.0,
            confidence=0.75,
            evidence=["大阴线", "放量下跌", "长下影线"],
        ),
        WyckoffEvent(
            type="AR",
            ts=(base_time + timedelta(hours=15)).isoformat(),
            price=40500.0,
            confidence=0.70,
            evidence=["快速反弹", "成交量减少"],
        ),
        WyckoffEvent(
            type="ST",
            ts=(base_time + timedelta(hours=25)).isoformat(),
            price=39200.0,
            confidence=0.65,
            evidence=["缩量回测", "未创新低"],
        ),
        WyckoffEvent(
            type="SOS",
            ts=(base_time + timedelta(hours=40)).isoformat(),
            price=41000.0,
            confidence=0.80,
            evidence=["突破前高", "放量上涨"],
        ),
    ]


@pytest.fixture
def climax_pattern_df() -> pd.DataFrame:
    """
    包含卖出高潮（SC）模式的K线数据
    
    特征：
    - 前面有下跌趋势
    - 出现放量大阴线
    - 带有长下影线（表示买盘介入）
    """
    np.random.seed(111)
    n = 50
    
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]
    
    # 构建下跌后的卖出高潮
    close_prices = []
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    price = 45000
    for i in range(n):
        if i < 30:
            # 下跌阶段
            price -= np.random.uniform(50, 150)
            vol = 800 + np.random.uniform(0, 200)
        elif i == 30:
            # SC: 大阴线，放量，长下影
            open_p = price
            low_p = price - 1500  # 大幅下探
            close_p = price - 500  # 收盘价远离最低点
            high_p = price + 100
            vol = 3000  # 放量
            close_prices.append(close_p)
            high_prices.append(high_p)
            low_prices.append(low_p)
            open_prices.append(open_p)
            volumes.append(vol)
            price = close_p
            continue
        else:
            # SC后反弹
            price += np.random.uniform(20, 100)
            vol = 1000 + np.random.uniform(0, 300)
        
        noise = np.random.uniform(-50, 50)
        close_p = price + noise
        high_p = close_p + np.abs(np.random.randn() * 100)
        low_p = close_p - np.abs(np.random.randn() * 100)
        open_p = close_prices[-1] if close_prices else price
        
        close_prices.append(close_p)
        high_prices.append(high_p)
        low_prices.append(low_p)
        open_prices.append(open_p)
        volumes.append(vol)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


# ============== 快照工具 ==============

class SnapshotManager:
    """快照管理器，用于回归测试"""
    
    def __init__(self, snapshots_dir: Path):
        self.snapshots_dir = snapshots_dir
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(self, name: str, data: dict) -> None:
        """保存快照"""
        path = self.snapshots_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    def load_snapshot(self, name: str) -> dict | None:
        """加载快照"""
        path = self.snapshots_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def compare_snapshot(self, name: str, current: dict, update: bool = False) -> tuple[bool, str]:
        """
        比较当前数据与快照
        
        Args:
            name: 快照名称
            current: 当前数据
            update: 是否更新快照
        
        Returns:
            (是否匹配, 差异描述)
        """
        saved = self.load_snapshot(name)
        
        if saved is None:
            if update:
                self.save_snapshot(name, current)
                return True, "快照已创建"
            return False, "快照不存在，运行 pytest --update-snapshots 创建"
        
        # 比较关键字段
        differences = []
        for key in set(list(saved.keys()) + list(current.keys())):
            if key not in saved:
                differences.append(f"新增字段: {key}")
            elif key not in current:
                differences.append(f"缺失字段: {key}")
            elif saved[key] != current[key]:
                differences.append(f"字段 {key} 变化: {saved[key]} -> {current[key]}")
        
        if differences:
            if update:
                self.save_snapshot(name, current)
                return True, f"快照已更新: {'; '.join(differences)}"
            return False, "\n".join(differences)
        
        return True, "匹配"


@pytest.fixture
def snapshot_manager(snapshots_dir: Path) -> SnapshotManager:
    """快照管理器fixture"""
    return SnapshotManager(snapshots_dir)


# ============== pytest 钩子 ==============

def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="更新回归测试快照",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="运行慢速测试",
    )


@pytest.fixture
def update_snapshots(request) -> bool:
    """是否更新快照"""
    return request.config.getoption("--update-snapshots")


def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="需要 --slow 选项来运行")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

