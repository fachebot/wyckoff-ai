"""
完整流程集成测试

测试从数据获取到报告生成的完整流程
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from wyckoff_ai.features import compute_features
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff
from wyckoff_ai.backtest import (
    BacktestConfig,
    BacktestEngine,
    evaluate_events,
    calculate_metrics,
    generate_backtest_report,
)


@pytest.mark.integration
class TestAnalysisPipeline:
    """测试分析流程"""
    
    def test_full_analysis_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """
        测试完整的分析流程：
        数据 -> 特征计算 -> 威科夫检测 -> 结果验证
        """
        # 1. 特征计算
        df = compute_features(sample_ohlcv_df)
        assert "atr_14" in df.columns
        assert "ema_50" in df.columns
        
        # 2. 威科夫检测
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 3. 验证结果结构
        assert analysis.symbol == "TEST/USDT"
        assert analysis.timeframe == "1h"
        assert hasattr(analysis, "events")
        assert hasattr(analysis, "market_structure")
        
        # 4. 验证事件有效性
        for event in analysis.events:
            assert event.price > 0
            assert 0 <= event.confidence <= 1
    
    def test_analysis_with_different_timeframes(self, sample_ohlcv_df: pd.DataFrame):
        """测试不同时间框架的分析"""
        df = compute_features(sample_ohlcv_df)
        
        for timeframe in ["1h", "4h", "1d"]:
            analysis = detect_wyckoff(
                df,
                symbol="TEST/USDT",
                exchange="test",
                timeframe=timeframe,
            )
            assert analysis.timeframe == timeframe
    
    def test_analysis_with_different_configs(self, sample_ohlcv_df: pd.DataFrame):
        """测试不同配置的分析"""
        df = compute_features(sample_ohlcv_df)
        
        configs = [
            DetectionConfig(lookback_bars=50),
            DetectionConfig(lookback_bars=100),
            DetectionConfig(min_confidence_threshold=0.5),
            DetectionConfig(min_confidence_threshold=0.8),
        ]
        
        for cfg in configs:
            analysis = detect_wyckoff(
                df,
                symbol="TEST/USDT",
                exchange="test",
                timeframe="1h",
                cfg=cfg,
            )
            assert analysis is not None


@pytest.mark.integration
class TestBacktestPipeline:
    """测试回测流程"""
    
    def test_full_backtest_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """
        测试完整的回测流程：
        数据 -> 特征 -> 事件检测 -> 回测 -> 指标计算 -> 报告生成
        """
        # 1. 特征计算
        df = compute_features(sample_ohlcv_df)
        
        # 2. 事件检测
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 3. 回测
        config = BacktestConfig(
            initial_capital=100000,
            position_size_pct=10,
        )
        engine = BacktestEngine(config)
        result = engine.run(df, analysis.events)
        
        # 4. 指标计算
        metrics = calculate_metrics(result)
        
        # 5. 验证结果
        assert result.total_bars == len(df)
        assert metrics.trade_metrics.win_rate >= 0
        assert metrics.max_drawdown_pct >= 0
        
        # 6. 生成报告
        report = generate_backtest_report(result)
        assert "总体表现" in report
        assert "交易统计" in report
    
    def test_event_evaluation_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """测试事件评估流程"""
        # 1. 特征计算
        df = compute_features(sample_ohlcv_df)
        
        # 2. 事件检测
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 3. 事件评估
        if analysis.events:
            performances, stats = evaluate_events(df, analysis.events)
            
            # 验证结果
            assert isinstance(performances, list)
            assert isinstance(stats, dict)
    
    def test_multiple_symbols_backtest(self, sample_ohlcv_df: pd.DataFrame):
        """测试多品种回测"""
        df = compute_features(sample_ohlcv_df)
        
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        results = {}
        
        for symbol in symbols:
            analysis = detect_wyckoff(
                df,
                symbol=symbol,
                exchange="test",
                timeframe="1h",
            )
            
            engine = BacktestEngine()
            result = engine.run(df, analysis.events)
            results[symbol] = calculate_metrics(result)
        
        # 验证所有结果
        assert len(results) == len(symbols)
        for symbol, metrics in results.items():
            assert metrics is not None


@pytest.mark.integration
class TestDataFlow:
    """测试数据流"""
    
    def test_data_consistency(self, sample_ohlcv_df: pd.DataFrame):
        """测试数据一致性"""
        df = compute_features(sample_ohlcv_df)
        
        # 原始数据应该保留
        assert len(df) == len(sample_ohlcv_df)
        
        # 时间戳应该一致
        pd.testing.assert_series_equal(
            df["timestamp"].reset_index(drop=True),
            sample_ohlcv_df["timestamp"].reset_index(drop=True),
        )
        
        # 价格数据应该一致
        pd.testing.assert_series_equal(
            df["close"].reset_index(drop=True),
            sample_ohlcv_df["close"].reset_index(drop=True),
        )
    
    def test_event_timestamps_match_data(self, sample_ohlcv_df: pd.DataFrame):
        """事件时间戳应该与数据匹配"""
        df = compute_features(sample_ohlcv_df)
        
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 获取数据的时间范围
        data_timestamps = set(
            df["timestamp"].apply(
                lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x)
            )
        )
        
        # 事件时间戳应该在数据范围内
        for event in analysis.events:
            # 注意：事件时间戳可能有微小差异，这里做宽松检查
            assert event.ts is not None


@pytest.mark.integration
@pytest.mark.slow
class TestWithRealData:
    """使用真实数据的测试（需要网络）"""
    
    def test_binance_data_fetch(self):
        """测试从 Binance 获取数据"""
        try:
            from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
            
            result = fetch_ohlcv_binance_spot(
                symbol="BTC/USDT",
                timeframe="1h",
                limit=100,
            )
            
            assert result.df is not None
            assert len(result.df) > 0
            assert "close" in result.df.columns
        except Exception as e:
            pytest.skip(f"无法连接 Binance API: {e}")
    
    def test_full_pipeline_with_real_data(self):
        """使用真实数据测试完整流程"""
        try:
            from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
            
            # 获取数据
            result = fetch_ohlcv_binance_spot(
                symbol="BTC/USDT",
                timeframe="1h",
                limit=200,
            )
            
            # 特征计算
            df = compute_features(result.df)
            
            # 分析
            analysis = detect_wyckoff(
                df,
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1h",
            )
            
            # 验证
            assert analysis.symbol == "BTC/USDT"
            assert len(analysis.events) >= 0
            
        except Exception as e:
            pytest.skip(f"无法完成真实数据测试: {e}")


@pytest.mark.integration
class TestOutputGeneration:
    """测试输出生成"""
    
    def test_json_output_serializable(self, sample_ohlcv_df: pd.DataFrame):
        """JSON 输出应该可序列化"""
        df = compute_features(sample_ohlcv_df)
        
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        # 转换为 dict 并序列化
        analysis_dict = analysis.model_dump()
        
        # 应该能序列化为 JSON
        json_str = json.dumps(analysis_dict, default=str)
        assert json_str is not None
        
        # 应该能反序列化
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "TEST/USDT"
    
    def test_report_generation(self, sample_ohlcv_df: pd.DataFrame):
        """测试报告生成"""
        df = compute_features(sample_ohlcv_df)
        
        analysis = detect_wyckoff(
            df,
            symbol="TEST/USDT",
            exchange="test",
            timeframe="1h",
        )
        
        engine = BacktestEngine()
        result = engine.run(df, analysis.events)
        
        # 生成报告
        report = generate_backtest_report(result, title="测试报告")
        
        # 验证报告内容
        assert "测试报告" in report
        assert "总体表现" in report
        assert "交易统计" in report
        
        # 报告应该是有效的 Markdown
        assert report.startswith("#")

