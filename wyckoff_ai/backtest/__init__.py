"""
回测系统模块

提供威科夫事件和交易策略的回测功能，包括：
- 事件后验评估（MFE/MAE统计）
- 交易模拟引擎
- 性能指标统计
- 回测报告生成
"""

from wyckoff_ai.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from wyckoff_ai.backtest.event_eval import (
    EventEvaluator,
    EventPerformance,
    evaluate_events,
)
from wyckoff_ai.backtest.metrics import (
    calculate_metrics,
    PerformanceMetrics,
    TradeMetrics,
)
from wyckoff_ai.backtest.report import generate_backtest_report

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "EventEvaluator",
    "EventPerformance",
    "evaluate_events",
    "calculate_metrics",
    "PerformanceMetrics",
    "TradeMetrics",
    "generate_backtest_report",
]

