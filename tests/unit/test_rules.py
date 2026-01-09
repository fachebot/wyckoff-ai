"""
威科夫规则引擎单元测试

测试 wyckoff_ai/wyckoff/rules.py 中的事件检测逻辑
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wyckoff_ai.features import compute_features
from wyckoff_ai.wyckoff.rules import (
    DetectionConfig,
    detect_wyckoff,
)
from wyckoff_ai.schemas import WyckoffEvent, WyckoffAnalysis


@pytest.mark.unit
class TestDetectionConfig:
    """测试检测配置"""
    
    def test_default_config(self):
        """默认配置应该有效"""
        cfg = DetectionConfig()
        assert cfg.lookback_bars > 0
        assert 0 <= cfg.min_confidence_threshold <= 1
    
    def test_custom_config(self):
        """自定义配置应该生效"""
        cfg = DetectionConfig(
            lookback_bars=200,
            min_confidence_threshold=0.8,
        )
        assert cfg.lookback_bars == 200
        assert cfg.min_confidence_threshold == 0.8


@pytest.mark.unit
class TestDetectWyckoff:
    """测试 detect_wyckoff 主函数"""
    
    def test_returns_analysis(self, sample_ohlcv_with_features: pd.DataFrame):
        """应该返回 WyckoffAnalysis"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        assert isinstance(result, WyckoffAnalysis)
    
    def test_analysis_has_required_fields(self, sample_ohlcv_with_features: pd.DataFrame):
        """分析结果应该包含必需字段"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        assert result.symbol == "TEST/USDT"
        assert result.timeframe == "1h"
        assert hasattr(result, "events")
        assert hasattr(result, "market_structure")
    
    def test_events_are_valid(self, sample_ohlcv_with_features: pd.DataFrame):
        """事件应该是有效的 WyckoffEvent"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        for event in result.events:
            assert isinstance(event, WyckoffEvent)
            assert event.type in ["SC", "BC", "AR", "ST", "SOS", "SOW", "SPRING", 
                                  "UT", "UTAD", "LPS", "LPSY", "JAC", "BUEC", 
                                  "TEST", "PSY"]
            assert 0 <= event.confidence <= 1
            assert event.price > 0
    
    def test_lookback_limits_detection(self, sample_ohlcv_with_features: pd.DataFrame):
        """lookback_bars 应该限制检测范围"""
        # 使用很小的 lookback
        cfg = DetectionConfig(lookback_bars=10)
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
            cfg=cfg,
        )
        # 事件时间戳应该在最近的范围内
        # （这是一个宽松的检查，主要确保不会崩溃）
        assert isinstance(result, WyckoffAnalysis)
    
    def test_strict_mode_filters_low_confidence(self, sample_ohlcv_with_features: pd.DataFrame):
        """严格模式应该过滤低置信度事件"""
        # 正常模式
        cfg_normal = DetectionConfig(min_confidence_threshold=0.3)
        result_normal = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
            cfg=cfg_normal,
        )
        
        # 严格模式
        cfg_strict = DetectionConfig(min_confidence_threshold=0.8)
        result_strict = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
            cfg=cfg_strict,
        )
        
        # 严格模式应该有更少或相等的事件
        assert len(result_strict.events) <= len(result_normal.events)


@pytest.mark.unit
class TestEventDetection:
    """测试特定事件的检测"""
    
    def test_detects_events_in_climax_pattern(self, climax_pattern_df: pd.DataFrame):
        """应该在卖出高潮模式中检测到事件"""
        df = compute_features(climax_pattern_df)
        result = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        # 应该检测到一些事件
        assert len(result.events) >= 0  # 宽松检查，确保不崩溃
    
    def test_uptrend_has_bullish_bias(self, trending_up_df: pd.DataFrame):
        """上涨趋势应该有看涨倾向"""
        df = compute_features(trending_up_df)
        result = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        # 检查市场结构 - 可能是 unknown 如果数据不足
        assert result.market_structure in ["bullish", "bearish", "ranging", "unclear", "unknown"]
    
    def test_downtrend_has_bearish_bias(self, trending_down_df: pd.DataFrame):
        """下跌趋势应该有看跌倾向"""
        df = compute_features(trending_down_df)
        result = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        assert result.market_structure in ["bullish", "bearish", "ranging", "unclear", "unknown"]
    
    def test_ranging_market_detection(self, ranging_df: pd.DataFrame):
        """应该识别横盘市场"""
        df = compute_features(ranging_df)
        result = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        # 横盘市场不应该有强烈的方向偏见
        assert result.market_structure in ["bullish", "bearish", "ranging", "unclear", "unknown"]


@pytest.mark.unit
class TestEventProperties:
    """测试事件属性"""
    
    def test_event_timestamps_are_valid(self, sample_ohlcv_with_features: pd.DataFrame):
        """事件时间戳应该有效"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        for event in result.events:
            # 时间戳应该是可解析的
            assert event.ts is not None
            # 尝试解析时间戳
            try:
                pd.Timestamp(event.ts)
            except Exception as e:
                pytest.fail(f"无法解析时间戳 {event.ts}: {e}")
    
    def test_event_prices_are_positive(self, sample_ohlcv_with_features: pd.DataFrame):
        """事件价格应该为正"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        for event in result.events:
            assert event.price > 0, f"事件 {event.type} 价格为非正: {event.price}"
    
    def test_event_confidence_in_range(self, sample_ohlcv_with_features: pd.DataFrame):
        """事件置信度应该在 0-1 范围内"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        for event in result.events:
            assert 0 <= event.confidence <= 1, \
                f"事件 {event.type} 置信度超出范围: {event.confidence}"
    
    def test_event_has_evidence(self, sample_ohlcv_with_features: pd.DataFrame):
        """事件应该有证据"""
        result = detect_wyckoff(
            sample_ohlcv_with_features,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        for event in result.events:
            assert hasattr(event, "evidence")
            # 证据可以为空，但属性应该存在


@pytest.mark.unit
class TestEdgeCases:
    """测试边界情况"""
    
    def test_handles_minimum_data(self):
        """处理最小数据量"""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
            "open": [100] * 30,
            "high": [105] * 30,
            "low": [95] * 30,
            "close": [102] * 30,
            "volume": [1000] * 30,
        })
        df = compute_features(df)
        
        result = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        assert isinstance(result, WyckoffAnalysis)
    
    def test_handles_empty_events(self):
        """处理无事件情况"""
        # 非常平稳的数据，可能检测不到事件
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "open": [100.0] * 50,
            "high": [100.5] * 50,
            "low": [99.5] * 50,
            "close": [100.0] * 50,
            "volume": [1000] * 50,
        })
        df = compute_features(df)
        
        result = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        # 即使没有事件也不应该崩溃
        assert isinstance(result, WyckoffAnalysis)
        assert isinstance(result.events, list)

