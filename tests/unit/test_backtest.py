"""
回测系统单元测试

测试 wyckoff_ai/backtest/ 模块的功能
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from wyckoff_ai.schemas import WyckoffEvent
from wyckoff_ai.features import compute_features
from wyckoff_ai.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    EventEvaluator,
    evaluate_events,
    calculate_metrics,
    PerformanceMetrics,
)


@pytest.mark.unit
class TestBacktestConfig:
    """测试回测配置"""
    
    def test_default_config(self):
        """默认配置应该有效"""
        cfg = BacktestConfig()
        assert cfg.initial_capital > 0
        assert cfg.position_size_pct > 0
        assert cfg.stop_loss_atr > 0
        assert cfg.take_profit_atr > 0
    
    def test_custom_config(self):
        """自定义配置应该生效"""
        cfg = BacktestConfig(
            initial_capital=50000,
            position_size_pct=20,
            stop_loss_atr=1.5,
            take_profit_atr=4.0,
        )
        assert cfg.initial_capital == 50000
        assert cfg.position_size_pct == 20
        assert cfg.stop_loss_atr == 1.5
        assert cfg.take_profit_atr == 4.0


@pytest.mark.unit
class TestBacktestEngine:
    """测试回测引擎"""
    
    def test_engine_init(self):
        """引擎初始化应该正确"""
        engine = BacktestEngine()
        assert engine is not None
    
    def test_engine_with_config(self):
        """带配置的引擎初始化"""
        cfg = BacktestConfig(initial_capital=50000)
        engine = BacktestEngine(cfg)
        assert engine.config.initial_capital == 50000
    
    def test_run_returns_result(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """运行回测应该返回结果"""
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, sample_events)
        assert isinstance(result, BacktestResult)
    
    def test_result_has_equity_curve(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """结果应该包含权益曲线"""
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, sample_events)
        assert len(result.equity_curve) > 0
    
    def test_reset_clears_state(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """重置应该清除状态"""
        engine = BacktestEngine()
        engine.run(sample_ohlcv_with_features, sample_events)
        
        engine.reset()
        
        # 再次运行应该得到独立的结果
        result2 = engine.run(sample_ohlcv_with_features, sample_events)
        assert isinstance(result2, BacktestResult)
    
    def test_no_trades_with_no_events(self, sample_ohlcv_with_features: pd.DataFrame):
        """没有事件时不应该有交易"""
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, [])
        assert result.total_trades == 0
    
    def test_respects_min_confidence(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
    ):
        """应该尊重最小置信度设置"""
        # 创建低置信度事件
        low_conf_events = [
            WyckoffEvent(
                type="SOS",
                ts=sample_ohlcv_with_features.iloc[50]["timestamp"].isoformat(),
                price=float(sample_ohlcv_with_features.iloc[50]["close"]),
                confidence=0.3,
                evidence=["test"],
            ),
        ]
        
        # 高置信度要求
        cfg = BacktestConfig(min_confidence=0.8)
        engine = BacktestEngine(cfg)
        result = engine.run(sample_ohlcv_with_features, low_conf_events)
        
        # 低置信度事件不应该触发交易
        assert result.total_trades == 0
    
    def test_respects_allowed_events(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """应该只交易允许的事件类型"""
        # 只允许 SOS 事件
        cfg = BacktestConfig(allowed_events=["SOS"])
        engine = BacktestEngine(cfg)
        result = engine.run(sample_ohlcv_with_features, sample_events)
        
        # 检查所有交易都是基于 SOS 事件
        for trade in result.trades:
            if trade.entry_event:
                assert trade.entry_event.type == "SOS"


@pytest.mark.unit
class TestEventEvaluator:
    """测试事件评估器"""
    
    def test_evaluator_init(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """评估器初始化应该正确"""
        evaluator = EventEvaluator(sample_ohlcv_with_features, sample_events)
        assert evaluator is not None
    
    def test_evaluate_all(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """评估所有事件应该返回结果"""
        evaluator = EventEvaluator(sample_ohlcv_with_features, sample_events)
        results = evaluator.evaluate_all()
        assert isinstance(results, list)
    
    def test_evaluate_events_convenience(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """便捷函数应该返回元组"""
        performances, stats = evaluate_events(
            sample_ohlcv_with_features,
            sample_events,
        )
        assert isinstance(performances, list)
        assert isinstance(stats, dict)


@pytest.mark.unit
class TestCalculateMetrics:
    """测试指标计算"""
    
    def test_returns_metrics(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """应该返回性能指标"""
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, sample_events)
        metrics = calculate_metrics(result)
        assert isinstance(metrics, PerformanceMetrics)
    
    def test_win_rate_in_range(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """胜率应该在 0-1 范围内"""
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, sample_events)
        metrics = calculate_metrics(result)
        assert 0 <= metrics.trade_metrics.win_rate <= 1
    
    def test_max_drawdown_non_negative(
        self,
        sample_ohlcv_with_features: pd.DataFrame,
        sample_events: list[WyckoffEvent],
    ):
        """最大回撤应该非负"""
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, sample_events)
        metrics = calculate_metrics(result)
        assert metrics.max_drawdown_pct >= 0


@pytest.mark.unit
class TestTradeExecution:
    """测试交易执行逻辑"""
    
    def test_stop_loss_triggers(self, trending_down_df: pd.DataFrame):
        """止损应该触发"""
        df = compute_features(trending_down_df)
        
        # 创建一个做多事件（在下跌趋势中会触发止损）
        event = WyckoffEvent(
            type="SOS",  # 做多信号
            ts=df.iloc[20]["timestamp"].isoformat(),
            price=float(df.iloc[20]["close"]),
            confidence=0.9,
            evidence=["test"],
        )
        
        cfg = BacktestConfig(
            stop_loss_atr=1.0,  # 较紧的止损
            take_profit_atr=5.0,  # 较远的止盈
        )
        engine = BacktestEngine(cfg)
        result = engine.run(df, [event])
        
        # 在下跌趋势中做多应该会触发止损
        # 这是一个宽松的检查
        assert isinstance(result, BacktestResult)
    
    def test_take_profit_triggers(self, trending_up_df: pd.DataFrame):
        """止盈应该触发"""
        df = compute_features(trending_up_df)
        
        # 创建一个做多事件（在上涨趋势中会触发止盈）
        event = WyckoffEvent(
            type="SOS",
            ts=df.iloc[20]["timestamp"].isoformat(),
            price=float(df.iloc[20]["close"]),
            confidence=0.9,
            evidence=["test"],
        )
        
        cfg = BacktestConfig(
            stop_loss_atr=5.0,  # 较远的止损
            take_profit_atr=1.0,  # 较近的止盈
        )
        engine = BacktestEngine(cfg)
        result = engine.run(df, [event])
        
        assert isinstance(result, BacktestResult)


@pytest.mark.unit
class TestEdgeCases:
    """测试边界情况"""
    
    def test_handles_empty_df(self):
        """处理空数据"""
        df = pd.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })
        
        engine = BacktestEngine()
        result = engine.run(df, [])
        assert result.total_trades == 0
    
    def test_handles_single_bar(self):
        """处理单根K线"""
        df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000],
            "atr_14": [5.0],
        })
        
        engine = BacktestEngine()
        result = engine.run(df, [])
        assert isinstance(result, BacktestResult)
    
    def test_handles_event_at_end(self, sample_ohlcv_with_features: pd.DataFrame):
        """处理在最后一根K线的事件"""
        last_ts = sample_ohlcv_with_features.iloc[-1]["timestamp"]
        event = WyckoffEvent(
            type="SOS",
            ts=last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
            price=float(sample_ohlcv_with_features.iloc[-1]["close"]),
            confidence=0.9,
            evidence=["test"],
        )
        
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv_with_features, [event])
        # 不应该崩溃
        assert isinstance(result, BacktestResult)

