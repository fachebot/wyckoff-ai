"""
威科夫分析可视化模块

提供 K 线图表、事件标注、状态机转换图等可视化功能
"""
from wyckoff_ai.visualization.candlestick import (
    create_candlestick_chart,
    create_analysis_chart,
    save_chart,
    add_events_to_chart,
    add_support_resistance,
    add_range_highlight,
    add_ema_lines,
)
from wyckoff_ai.visualization.state_diagram import (
    create_state_diagram,
    create_phase_progress_chart,
    create_timeline_chart,
    create_combined_state_view,
)
from wyckoff_ai.visualization.report_charts import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_trade_distribution_chart,
    create_event_performance_chart,
    create_backtest_summary_chart,
    generate_html_report_with_charts,
)

__all__ = [
    # K线图表
    "create_candlestick_chart",
    "create_analysis_chart",
    "save_chart",
    "add_events_to_chart",
    "add_support_resistance",
    "add_range_highlight",
    "add_ema_lines",
    # 状态机图表
    "create_state_diagram",
    "create_phase_progress_chart",
    "create_timeline_chart",
    "create_combined_state_view",
    # 回测报告图表
    "create_equity_curve_chart",
    "create_drawdown_chart",
    "create_trade_distribution_chart",
    "create_event_performance_chart",
    "create_backtest_summary_chart",
    "generate_html_report_with_charts",
]

