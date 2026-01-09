"""
性能指标计算模块

计算各种交易策略性能指标，包括：
- 收益指标
- 风险指标
- 风险调整收益指标
- 交易统计指标
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from wyckoff_ai.backtest.engine import BacktestResult, Trade, TradeDirection


@dataclass
class TradeMetrics:
    """交易统计指标"""
    # 基本统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    # 胜率
    win_rate: float = 0.0
    
    # 收益
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    
    # 平均收益
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    
    # 最大单笔
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # 连续统计
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # 盈亏比
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0  # avg_profit / avg_loss
    expectancy: float = 0.0  # 期望值
    
    # 持仓时间
    avg_bars_in_winning_trade: float = 0.0
    avg_bars_in_losing_trade: float = 0.0
    avg_bars_in_trade: float = 0.0
    
    # MFE/MAE
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_mae_ratio: float = 0.0


@dataclass
class PerformanceMetrics:
    """综合性能指标"""
    # 收益指标
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # 风险指标
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # K线数
    
    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sterling_ratio: float = 0.0
    
    # 其他
    recovery_factor: float = 0.0  # net_profit / max_drawdown
    profit_to_drawdown: float = 0.0
    
    # 交易统计
    trade_metrics: TradeMetrics = field(default_factory=TradeMetrics)
    
    # 时间信息
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0
    
    # 条件统计
    stats_by_direction: dict[str, TradeMetrics] = field(default_factory=dict)
    stats_by_event: dict[str, TradeMetrics] = field(default_factory=dict)


def calculate_trade_metrics(trades: list[Trade]) -> TradeMetrics:
    """计算交易统计指标"""
    metrics = TradeMetrics()
    
    if not trades:
        return metrics
    
    metrics.total_trades = len(trades)
    
    # 分类
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    breakeven = [t for t in trades if t.pnl == 0]
    
    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.breakeven_trades = len(breakeven)
    
    # 胜率
    metrics.win_rate = len(wins) / len(trades)
    
    # 收益
    profits = [t.pnl for t in wins]
    losses_vals = [abs(t.pnl) for t in losses]
    
    metrics.total_profit = sum(profits) if profits else 0.0
    metrics.total_loss = sum(losses_vals) if losses_vals else 0.0
    metrics.net_profit = metrics.total_profit - metrics.total_loss
    
    # 平均收益
    metrics.avg_profit = np.mean(profits) if profits else 0.0
    metrics.avg_loss = np.mean(losses_vals) if losses_vals else 0.0
    metrics.avg_trade = np.mean([t.pnl for t in trades])
    
    # 最大单笔
    all_pnls = [t.pnl for t in trades]
    metrics.max_profit = max(all_pnls) if all_pnls else 0.0
    metrics.max_loss = min(all_pnls) if all_pnls else 0.0
    
    # 连续统计
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for t in trades:
        if t.pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif t.pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0
    
    metrics.max_consecutive_wins = max_wins
    metrics.max_consecutive_losses = max_losses
    
    # 盈亏比
    if metrics.total_loss > 0:
        metrics.profit_factor = metrics.total_profit / metrics.total_loss
    elif metrics.total_profit > 0:
        metrics.profit_factor = float("inf")
    
    if metrics.avg_loss > 0:
        metrics.payoff_ratio = metrics.avg_profit / metrics.avg_loss
    elif metrics.avg_profit > 0:
        metrics.payoff_ratio = float("inf")
    
    # 期望值
    metrics.expectancy = (
        metrics.win_rate * metrics.avg_profit
        - (1 - metrics.win_rate) * metrics.avg_loss
    )
    
    # 持仓时间
    if wins:
        metrics.avg_bars_in_winning_trade = np.mean([t.bars_held for t in wins])
    if losses:
        metrics.avg_bars_in_losing_trade = np.mean([t.bars_held for t in losses])
    metrics.avg_bars_in_trade = np.mean([t.bars_held for t in trades])
    
    # MFE/MAE
    mfes = [t.mfe for t in trades]
    maes = [t.mae for t in trades]
    
    metrics.avg_mfe = np.mean(mfes) if mfes else 0.0
    metrics.avg_mae = np.mean(maes) if maes else 0.0
    
    if metrics.avg_mae > 0:
        metrics.mfe_mae_ratio = metrics.avg_mfe / metrics.avg_mae
    
    return metrics


def calculate_equity_metrics(
    equity_curve: list[float],
    initial_capital: float,
    bars_per_year: int = 252 * 24,  # 默认1小时K线
) -> dict:
    """计算权益曲线相关指标"""
    if len(equity_curve) < 2:
        return {}
    
    equity = np.array(equity_curve)
    
    # 收益
    total_return = equity[-1] - initial_capital
    total_return_pct = (equity[-1] / initial_capital - 1) * 100
    
    # 收益率序列
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 波动率
    volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
    annualized_volatility = volatility * np.sqrt(bars_per_year)
    
    # 年化收益
    n_bars = len(equity_curve)
    if n_bars > 0 and equity[0] > 0:
        total_mult = equity[-1] / equity[0]
        years = n_bars / bars_per_year
        if years > 0 and total_mult > 0:
            annualized_return = (total_mult ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0
    else:
        annualized_return = 0.0
    
    # 回撤计算
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    
    max_drawdown_pct = float(np.max(drawdown))
    max_drawdown = float(np.max(peak - equity))
    avg_drawdown = float(np.mean(drawdown))
    
    # 最大回撤持续时间
    max_dd_duration = 0
    current_dd_duration = 0
    
    for i in range(len(drawdown)):
        if drawdown[i] > 0:
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        else:
            current_dd_duration = 0
    
    # Sharpe ratio
    avg_return = float(np.mean(returns)) if len(returns) > 0 else 0.0
    if volatility > 0:
        sharpe = avg_return / volatility * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0
    
    # Sortino ratio
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = float(np.std(negative_returns))
        if downside_std > 0:
            sortino = avg_return / downside_std * np.sqrt(bars_per_year)
        else:
            sortino = 0.0
    else:
        sortino = 0.0
    
    # Calmar ratio
    if max_drawdown_pct > 0:
        calmar = annualized_return / max_drawdown_pct
    else:
        calmar = 0.0
    
    # Sterling ratio (使用平均回撤)
    if avg_drawdown > 0:
        sterling = annualized_return / avg_drawdown
    else:
        sterling = 0.0
    
    # Recovery factor
    if max_drawdown > 0:
        recovery_factor = total_return / max_drawdown
    else:
        recovery_factor = 0.0
    
    return {
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_drawdown": avg_drawdown,
        "max_drawdown_duration": max_dd_duration,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "sterling_ratio": sterling,
        "recovery_factor": recovery_factor,
    }


def calculate_metrics(
    result: BacktestResult,
    bars_per_year: int = 252 * 24,
) -> PerformanceMetrics:
    """
    计算完整的性能指标
    
    Args:
        result: 回测结果
        bars_per_year: 每年的K线数量（用于年化计算）
    
    Returns:
        PerformanceMetrics: 综合性能指标
    """
    metrics = PerformanceMetrics()
    
    # 交易统计
    metrics.trade_metrics = calculate_trade_metrics(result.trades)
    
    # 权益曲线指标
    if result.equity_curve:
        equity_metrics = calculate_equity_metrics(
            result.equity_curve,
            result.config.initial_capital,
            bars_per_year,
        )
        
        metrics.total_return = equity_metrics.get("total_return", 0.0)
        metrics.total_return_pct = equity_metrics.get("total_return_pct", 0.0)
        metrics.annualized_return = equity_metrics.get("annualized_return", 0.0)
        metrics.volatility = equity_metrics.get("volatility", 0.0)
        metrics.annualized_volatility = equity_metrics.get("annualized_volatility", 0.0)
        metrics.max_drawdown = equity_metrics.get("max_drawdown", 0.0)
        metrics.max_drawdown_pct = equity_metrics.get("max_drawdown_pct", 0.0)
        metrics.avg_drawdown = equity_metrics.get("avg_drawdown", 0.0)
        metrics.max_drawdown_duration = equity_metrics.get("max_drawdown_duration", 0)
        metrics.sharpe_ratio = equity_metrics.get("sharpe_ratio", 0.0)
        metrics.sortino_ratio = equity_metrics.get("sortino_ratio", 0.0)
        metrics.calmar_ratio = equity_metrics.get("calmar_ratio", 0.0)
        metrics.sterling_ratio = equity_metrics.get("sterling_ratio", 0.0)
        metrics.recovery_factor = equity_metrics.get("recovery_factor", 0.0)
    
    # 时间信息
    metrics.start_date = result.start_time
    metrics.end_date = result.end_time
    metrics.trading_days = result.total_bars
    
    # 按方向统计
    long_trades = [t for t in result.trades if t.direction == TradeDirection.LONG]
    short_trades = [t for t in result.trades if t.direction == TradeDirection.SHORT]
    
    if long_trades:
        metrics.stats_by_direction["long"] = calculate_trade_metrics(long_trades)
    if short_trades:
        metrics.stats_by_direction["short"] = calculate_trade_metrics(short_trades)
    
    # 按事件类型统计
    by_event: dict[str, list[Trade]] = {}
    for trade in result.trades:
        if trade.entry_event:
            et = trade.entry_event.type
            if et not in by_event:
                by_event[et] = []
            by_event[et].append(trade)
    
    for event_type, trades in by_event.items():
        metrics.stats_by_event[event_type] = calculate_trade_metrics(trades)
    
    return metrics


def compare_results(
    results: list[BacktestResult],
    names: list[str] | None = None,
) -> dict:
    """
    比较多个回测结果
    
    Args:
        results: 回测结果列表
        names: 结果名称列表
    
    Returns:
        比较表格数据
    """
    if names is None:
        names = [f"策略{i+1}" for i in range(len(results))]
    
    comparison = {
        "names": names,
        "total_return": [],
        "win_rate": [],
        "profit_factor": [],
        "sharpe_ratio": [],
        "max_drawdown_pct": [],
        "total_trades": [],
    }
    
    for result in results:
        metrics = calculate_metrics(result)
        
        comparison["total_return"].append(metrics.total_return_pct)
        comparison["win_rate"].append(metrics.trade_metrics.win_rate * 100)
        comparison["profit_factor"].append(metrics.trade_metrics.profit_factor)
        comparison["sharpe_ratio"].append(metrics.sharpe_ratio)
        comparison["max_drawdown_pct"].append(metrics.max_drawdown_pct)
        comparison["total_trades"].append(metrics.trade_metrics.total_trades)
    
    return comparison

