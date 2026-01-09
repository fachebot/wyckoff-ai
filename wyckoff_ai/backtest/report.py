"""
回测报告生成模块

生成回测结果的Markdown报告，包括：
- 总体性能摘要
- 交易统计
- 事件表现分析
- 资金曲线图（文本版）
"""
from __future__ import annotations

from datetime import datetime

from wyckoff_ai.backtest.engine import BacktestResult, Trade, TradeDirection, TradeStatus
from wyckoff_ai.backtest.event_eval import EventTypeStats
from wyckoff_ai.backtest.metrics import PerformanceMetrics, calculate_metrics


def _format_pct(value: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{value:.{decimals}f}%"


def _format_number(value: float, decimals: int = 2) -> str:
    """格式化数字"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def _format_ratio(value: float, decimals: int = 2) -> str:
    """格式化比率"""
    if value == float("inf"):
        return "∞"
    return f"{value:.{decimals}f}"


def _progress_bar(value: float, max_value: float = 1.0, width: int = 20) -> str:
    """生成文本进度条"""
    if max_value <= 0:
        return "-" * width
    
    ratio = min(max(value / max_value, 0), 1)
    filled = int(ratio * width)
    return "=" * filled + "-" * (width - filled)


def generate_backtest_report(
    result: BacktestResult,
    title: str = "威科夫策略回测报告",
    include_trades: bool = True,
    max_trades_shown: int = 20,
) -> str:
    """
    生成回测报告
    
    Args:
        result: 回测结果
        title: 报告标题
        include_trades: 是否包含交易明细
        max_trades_shown: 显示的最大交易数量
    
    Returns:
        Markdown格式的报告
    """
    metrics = calculate_metrics(result)
    
    lines = []
    
    # 标题
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **回测区间**：{result.start_time[:19]} ~ {result.end_time[:19]}")
    lines.append(f"- **总K线数**：{result.total_bars}")
    lines.append("")
    
    # 总体表现
    lines.append("## 📊 总体表现")
    lines.append("")
    
    # 收益指标
    return_icon = "📈" if metrics.total_return >= 0 else "📉"
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| {return_icon} 总收益 | {_format_number(metrics.total_return)} ({_format_pct(metrics.total_return_pct)}) |")
    lines.append(f"| 💰 初始资金 | {_format_number(result.config.initial_capital)} |")
    lines.append(f"| 💵 最终权益 | {_format_number(result.equity_curve[-1] if result.equity_curve else result.config.initial_capital)} |")
    lines.append(f"| 📊 年化收益 | {_format_pct(metrics.annualized_return)} |")
    lines.append(f"| 📉 最大回撤 | {_format_pct(metrics.max_drawdown_pct)} |")
    lines.append(f"| ⚡ Sharpe比率 | {_format_ratio(metrics.sharpe_ratio)} |")
    lines.append(f"| 📈 Sortino比率 | {_format_ratio(metrics.sortino_ratio)} |")
    lines.append(f"| 🎯 Calmar比率 | {_format_ratio(metrics.calmar_ratio)} |")
    lines.append("")
    
    # 交易统计
    lines.append("## 📈 交易统计")
    lines.append("")
    
    tm = metrics.trade_metrics
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 🔢 总交易次数 | {tm.total_trades} |")
    lines.append(f"| ✅ 盈利次数 | {tm.winning_trades} |")
    lines.append(f"| ❌ 亏损次数 | {tm.losing_trades} |")
    lines.append(f"| 🎯 胜率 | [{_progress_bar(tm.win_rate)}] {_format_pct(tm.win_rate * 100)} |")
    lines.append(f"| 💰 总盈利 | {_format_number(tm.total_profit)} |")
    lines.append(f"| 💸 总亏损 | {_format_number(tm.total_loss)} |")
    lines.append(f"| 📊 盈亏比 | {_format_ratio(tm.profit_factor)} |")
    lines.append(f"| 📈 平均盈利 | {_format_number(tm.avg_profit)} |")
    lines.append(f"| 📉 平均亏损 | {_format_number(tm.avg_loss)} |")
    lines.append(f"| 💡 期望值 | {_format_number(tm.expectancy)} |")
    lines.append(f"| ⏱️ 平均持仓 | {tm.avg_bars_in_trade:.1f} 根K线 |")
    lines.append(f"| 🏆 最大连胜 | {tm.max_consecutive_wins} |")
    lines.append(f"| 💔 最大连亏 | {tm.max_consecutive_losses} |")
    lines.append("")
    
    # MFE/MAE 分析
    lines.append("## 📐 MFE/MAE 分析")
    lines.append("")
    lines.append("> MFE (Maximum Favorable Excursion): 最大有利偏移")
    lines.append("> MAE (Maximum Adverse Excursion): 最大不利偏移")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 📈 平均MFE | {_format_pct(tm.avg_mfe)} |")
    lines.append(f"| 📉 平均MAE | {_format_pct(tm.avg_mae)} |")
    lines.append(f"| 📊 MFE/MAE比 | {_format_ratio(tm.mfe_mae_ratio)} |")
    lines.append("")
    
    # 按方向统计
    if metrics.stats_by_direction:
        lines.append("## 🔀 按方向统计")
        lines.append("")
        lines.append(f"| 方向 | 交易数 | 胜率 | 盈亏比 | 平均收益 | 期望值 |")
        lines.append(f"|------|--------|------|--------|----------|--------|")
        
        for direction, stats in metrics.stats_by_direction.items():
            dir_name = "做多 📈" if direction == "long" else "做空 📉"
            lines.append(
                f"| {dir_name} | {stats.total_trades} | "
                f"{_format_pct(stats.win_rate * 100)} | "
                f"{_format_ratio(stats.profit_factor)} | "
                f"{_format_number(stats.avg_trade)} | "
                f"{_format_number(stats.expectancy)} |"
            )
        lines.append("")
    
    # 按事件类型统计
    if metrics.stats_by_event:
        lines.append("## 📋 按事件类型统计")
        lines.append("")
        lines.append(f"| 事件 | 交易数 | 胜率 | 盈亏比 | 平均MFE | 平均MAE | 期望值 |")
        lines.append(f"|------|--------|------|--------|---------|---------|--------|")
        
        # 按期望值排序
        sorted_events = sorted(
            metrics.stats_by_event.items(),
            key=lambda x: x[1].expectancy,
            reverse=True
        )
        
        for event_type, stats in sorted_events:
            exp_icon = "✅" if stats.expectancy > 0 else "❌"
            lines.append(
                f"| {event_type} | {stats.total_trades} | "
                f"{_format_pct(stats.win_rate * 100)} | "
                f"{_format_ratio(stats.profit_factor)} | "
                f"{_format_pct(stats.avg_mfe)} | "
                f"{_format_pct(stats.avg_mae)} | "
                f"{exp_icon} {_format_number(stats.expectancy)} |"
            )
        lines.append("")
    
    # 配置信息
    lines.append("## ⚙️ 回测配置")
    lines.append("")
    lines.append(f"| 参数 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 初始资金 | {_format_number(result.config.initial_capital)} |")
    lines.append(f"| 单笔仓位 | {result.config.position_size_pct}% |")
    lines.append(f"| 最大持仓 | {result.config.max_positions} |")
    lines.append(f"| 止损距离 | {result.config.stop_loss_atr} ATR |")
    lines.append(f"| 止盈距离 | {result.config.take_profit_atr} ATR |")
    lines.append(f"| 移动止损 | {'是' if result.config.use_trailing_stop else '否'} |")
    lines.append(f"| 最小置信度 | {_format_pct(result.config.min_confidence * 100)} |")
    lines.append(f"| 最大持仓时间 | {result.config.max_bars_in_trade} 根 |")
    lines.append(f"| 手续费 | {result.config.commission_pct}% |")
    lines.append(f"| 滑点 | {result.config.slippage_pct}% |")
    lines.append("")
    
    # 交易明细
    if include_trades and result.trades:
        lines.append("## 📝 交易明细")
        lines.append("")
        
        # 只显示最近的交易
        trades_to_show = result.trades[-max_trades_shown:]
        if len(result.trades) > max_trades_shown:
            lines.append(f"> 仅显示最近 {max_trades_shown} 笔交易（共 {len(result.trades)} 笔）")
            lines.append("")
        
        lines.append(f"| # | 方向 | 入场时间 | 入场价 | 出场价 | 收益% | MFE | MAE | 状态 | 原因 |")
        lines.append(f"|---|------|----------|--------|--------|-------|-----|-----|------|------|")
        
        for trade in trades_to_show:
            dir_icon = "📈" if trade.direction == TradeDirection.LONG else "📉"
            pnl_icon = "✅" if trade.pnl > 0 else "❌" if trade.pnl < 0 else "➖"
            
            status_map = {
                TradeStatus.CLOSED: "平仓",
                TradeStatus.STOPPED: "止损",
                TradeStatus.TARGET_HIT: "止盈",
                TradeStatus.OPEN: "持仓中",
            }
            
            lines.append(
                f"| {trade.trade_id} | {dir_icon} | "
                f"{trade.entry_time[:16]} | "
                f"{_format_number(trade.entry_price, 2)} | "
                f"{_format_number(trade.exit_price or 0, 2)} | "
                f"{pnl_icon} {_format_pct(trade.pnl_pct)} | "
                f"{_format_pct(trade.mfe)} | "
                f"{_format_pct(trade.mae)} | "
                f"{status_map.get(trade.status, '未知')} | "
                f"{trade.entry_reason[:20]}... |"
            )
        lines.append("")
    
    # 资金曲线（简化文本版）
    if result.equity_curve and len(result.equity_curve) > 10:
        lines.append("## 📈 资金曲线")
        lines.append("")
        lines.append("```")
        
        # 采样显示
        n_points = min(50, len(result.equity_curve))
        step = len(result.equity_curve) // n_points
        
        min_eq = min(result.equity_curve)
        max_eq = max(result.equity_curve)
        range_eq = max_eq - min_eq if max_eq > min_eq else 1
        
        width = 40
        
        for i in range(0, len(result.equity_curve), step):
            eq = result.equity_curve[i]
            pos = int((eq - min_eq) / range_eq * width)
            bar = " " * pos + "█"
            lines.append(f"{_format_number(eq, 0):>10} |{bar}")
        
        lines.append("```")
        lines.append("")
    
    # 风险提示
    lines.append("## ⚠️ 风险提示")
    lines.append("")
    lines.append("1. 历史回测结果不代表未来表现")
    lines.append("2. 回测假设所有订单都能按预期成交，实际交易可能存在滑点和流动性问题")
    lines.append("3. 未考虑资金管理和心理因素对实际交易的影响")
    lines.append("4. 建议在模拟盘验证后再进行实盘交易")
    lines.append("")
    
    return "\n".join(lines)


def generate_event_stats_report(
    stats: dict[str, EventTypeStats],
    title: str = "威科夫事件后验分析报告",
) -> str:
    """
    生成事件统计报告
    
    Args:
        stats: 事件类型 -> EventTypeStats 的字典
        title: 报告标题
    
    Returns:
        Markdown格式的报告
    """
    lines = []
    
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 总览表格
    lines.append("## 📊 事件表现总览")
    lines.append("")
    lines.append(f"| 事件 | 样本数 | 24根胜率 | 24根中位收益 | 平均MFE | 平均MAE | 方向准确率 |")
    lines.append(f"|------|--------|----------|--------------|---------|---------|------------|")
    
    # 按24根胜率排序
    sorted_stats = sorted(
        stats.items(),
        key=lambda x: x[1].win_rate_24,
        reverse=True
    )
    
    for event_type, s in sorted_stats:
        win_icon = "✅" if s.win_rate_24 > 0.5 else "❌" if s.win_rate_24 < 0.5 else "➖"
        lines.append(
            f"| **{event_type}** | {s.count} | "
            f"{win_icon} {_format_pct(s.win_rate_24 * 100)} | "
            f"{_format_pct(s.median_return_24)} | "
            f"{_format_pct(s.avg_mfe)} | "
            f"{_format_pct(s.avg_mae)} | "
            f"{_format_pct(s.direction_accuracy * 100)} |"
        )
    lines.append("")
    
    # 详细分析
    lines.append("## 📈 详细收益分析")
    lines.append("")
    lines.append(f"| 事件 | 6根平均 | 12根平均 | 24根平均 | 48根平均 | 6根中位 | 12根中位 | 24根中位 | 48根中位 |")
    lines.append(f"|------|---------|----------|----------|----------|---------|----------|----------|----------|")
    
    for event_type, s in sorted_stats:
        lines.append(
            f"| {event_type} | "
            f"{_format_pct(s.avg_return_6)} | "
            f"{_format_pct(s.avg_return_12)} | "
            f"{_format_pct(s.avg_return_24)} | "
            f"{_format_pct(s.avg_return_48)} | "
            f"{_format_pct(s.median_return_6)} | "
            f"{_format_pct(s.median_return_12)} | "
            f"{_format_pct(s.median_return_24)} | "
            f"{_format_pct(s.median_return_48)} |"
        )
    lines.append("")
    
    # 胜率分析
    lines.append("## 🎯 胜率分析")
    lines.append("")
    lines.append(f"| 事件 | 6根胜率 | 12根胜率 | 24根胜率 | 48根胜率 |")
    lines.append(f"|------|---------|----------|----------|----------|")
    
    for event_type, s in sorted_stats:
        lines.append(
            f"| {event_type} | "
            f"{_format_pct(s.win_rate_6 * 100)} | "
            f"{_format_pct(s.win_rate_12 * 100)} | "
            f"{_format_pct(s.win_rate_24 * 100)} | "
            f"{_format_pct(s.win_rate_48 * 100)} |"
        )
    lines.append("")
    
    # 盈亏分析
    lines.append("## 💰 盈亏分析")
    lines.append("")
    lines.append(f"| 事件 | 盈亏比 | 平均盈利 | 平均亏损 | 置信度相关性 |")
    lines.append(f"|------|--------|----------|----------|--------------|")
    
    for event_type, s in sorted_stats:
        corr_icon = "📈" if s.confidence_correlation > 0.3 else "📉" if s.confidence_correlation < -0.3 else "➖"
        lines.append(
            f"| {event_type} | "
            f"{_format_ratio(s.profit_factor)} | "
            f"{_format_pct(s.avg_win)} | "
            f"{_format_pct(s.avg_loss)} | "
            f"{corr_icon} {s.confidence_correlation:.2f} |"
        )
    lines.append("")
    
    # 建议
    lines.append("## 💡 交易建议")
    lines.append("")
    
    # 找出表现最好的事件
    best_events = [
        (et, s) for et, s in sorted_stats
        if s.win_rate_24 > 0.55 and s.count >= 3
    ]
    
    if best_events:
        lines.append("### 推荐交易的事件")
        lines.append("")
        for et, s in best_events[:3]:
            lines.append(f"- **{et}**: 胜率 {_format_pct(s.win_rate_24 * 100)}，样本数 {s.count}")
        lines.append("")
    
    # 找出表现最差的事件
    worst_events = [
        (et, s) for et, s in sorted_stats
        if s.win_rate_24 < 0.45 and s.count >= 3
    ]
    
    if worst_events:
        lines.append("### 谨慎交易的事件")
        lines.append("")
        for et, s in worst_events[:3]:
            lines.append(f"- **{et}**: 胜率 {_format_pct(s.win_rate_24 * 100)}，样本数 {s.count}")
        lines.append("")
    
    return "\n".join(lines)

