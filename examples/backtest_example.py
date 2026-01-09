"""
回测系统使用示例

演示如何使用回测系统验证威科夫事件策略的有效性。

运行方式:
    python examples/backtest_example.py
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.features import compute_features
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff
from wyckoff_ai.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    EventEvaluator,
    evaluate_events,
    calculate_metrics,
    generate_backtest_report,
)
from wyckoff_ai.backtest.report import generate_event_stats_report

console = Console()


def example_event_evaluation():
    """
    示例1: 事件后验评估
    
    评估每种威科夫事件发生后的价格表现，
    计算MFE/MAE、胜率、收益率等指标。
    """
    console.print(Panel("[bold cyan]示例1: 事件后验评估[/bold cyan]"))
    
    # 获取数据
    console.print("[dim]获取 BTC/USDT 1h 数据...[/dim]")
    fr = fetch_ohlcv_binance_spot(symbol="BTC/USDT", timeframe="1h", limit=500)
    df = compute_features(fr.df)
    console.print(f"[dim]获取到 {len(df)} 根K线[/dim]")
    
    # 检测事件
    console.print("[dim]检测威科夫事件...[/dim]")
    cfg = DetectionConfig(lookback_bars=400)
    analysis = detect_wyckoff(
        df,
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        cfg=cfg,
    )
    console.print(f"[dim]检测到 {len(analysis.events)} 个事件[/dim]")
    
    if not analysis.events:
        console.print("[yellow]未检测到事件[/yellow]")
        return
    
    # 评估事件
    performances, stats = evaluate_events(df, analysis.events)
    
    # 显示结果
    table = Table(title="事件后验表现")
    table.add_column("事件", style="cyan")
    table.add_column("样本数", justify="right")
    table.add_column("24根胜率", justify="right")
    table.add_column("中位收益", justify="right")
    table.add_column("平均MFE", justify="right")
    table.add_column("平均MAE", justify="right")
    table.add_column("方向准确率", justify="right")
    
    for event_type, s in sorted(stats.items(), key=lambda x: x[1].win_rate_24, reverse=True):
        icon = "✅" if s.win_rate_24 > 0.5 else "❌" if s.win_rate_24 < 0.5 else "➖"
        table.add_row(
            event_type,
            str(s.count),
            f"{icon} {s.win_rate_24*100:.1f}%",
            f"{s.median_return_24:.2f}%",
            f"{s.avg_mfe:.2f}%",
            f"{s.avg_mae:.2f}%",
            f"{s.direction_accuracy*100:.1f}%",
        )
    
    console.print(table)
    
    # 生成报告
    report = generate_event_stats_report(stats, title="BTC/USDT 威科夫事件后验分析")
    with open("out/event_eval_example.md", "w", encoding="utf-8") as f:
        f.write(report)
    console.print("[green]报告已保存到 out/event_eval_example.md[/green]")


def example_simple_backtest():
    """
    示例2: 简单回测
    
    使用默认配置进行回测，验证威科夫事件策略的整体表现。
    """
    console.print(Panel("[bold cyan]示例2: 简单回测[/bold cyan]"))
    
    # 获取数据
    console.print("[dim]获取 BTC/USDT 1h 数据...[/dim]")
    fr = fetch_ohlcv_binance_spot(symbol="BTC/USDT", timeframe="1h", limit=500)
    df = compute_features(fr.df)
    
    # 检测事件
    cfg = DetectionConfig(lookback_bars=400)
    analysis = detect_wyckoff(
        df,
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        cfg=cfg,
    )
    
    if not analysis.events:
        console.print("[yellow]未检测到事件[/yellow]")
        return
    
    # 配置回测
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=10,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        min_confidence=0.6,
        max_bars_in_trade=48,
    )
    
    # 运行回测
    engine = BacktestEngine(config)
    result = engine.run(df, analysis.events)
    
    # 计算指标
    metrics = calculate_metrics(result)
    
    # 显示结果
    console.print("")
    console.print("[bold]回测结果:[/bold]")
    console.print(f"  总交易数: {result.total_trades}")
    console.print(f"  胜率: {metrics.trade_metrics.win_rate*100:.1f}%")
    console.print(f"  总收益: {metrics.total_return_pct:.2f}%")
    console.print(f"  最大回撤: {metrics.max_drawdown_pct:.2f}%")
    console.print(f"  Sharpe比率: {metrics.sharpe_ratio:.2f}")
    console.print(f"  盈亏比: {metrics.trade_metrics.profit_factor:.2f}")
    console.print(f"  平均MFE: {metrics.trade_metrics.avg_mfe:.2f}%")
    console.print(f"  平均MAE: {metrics.trade_metrics.avg_mae:.2f}%")
    
    # 生成报告
    report = generate_backtest_report(result, title="BTC/USDT 威科夫策略回测")
    with open("out/backtest_example.md", "w", encoding="utf-8") as f:
        f.write(report)
    console.print("[green]报告已保存到 out/backtest_example.md[/green]")


def example_filtered_backtest():
    """
    示例3: 过滤事件回测
    
    只交易特定的高胜率事件，验证选择性交易的效果。
    """
    console.print(Panel("[bold cyan]示例3: 过滤事件回测[/bold cyan]"))
    
    # 获取数据
    fr = fetch_ohlcv_binance_spot(symbol="BTC/USDT", timeframe="1h", limit=500)
    df = compute_features(fr.df)
    
    # 检测事件
    cfg = DetectionConfig(lookback_bars=400)
    analysis = detect_wyckoff(
        df,
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        cfg=cfg,
    )
    
    if not analysis.events:
        console.print("[yellow]未检测到事件[/yellow]")
        return
    
    # 只交易看涨事件（SOS, SPRING, LPS, JAC）
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=10,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
        min_confidence=0.7,  # 更高置信度
        max_bars_in_trade=48,
        allowed_events=["SOS", "SPRING", "LPS", "JAC", "TEST"],  # 只交易看涨事件
    )
    
    engine = BacktestEngine(config)
    result = engine.run(df, analysis.events)
    metrics = calculate_metrics(result)
    
    console.print("")
    console.print("[bold]只交易看涨事件的结果:[/bold]")
    console.print(f"  总交易数: {result.total_trades}")
    console.print(f"  胜率: {metrics.trade_metrics.win_rate*100:.1f}%")
    console.print(f"  总收益: {metrics.total_return_pct:.2f}%")
    console.print(f"  盈亏比: {metrics.trade_metrics.profit_factor:.2f}")


def example_compare_configs():
    """
    示例4: 对比不同配置
    
    对比不同止损止盈配置的表现，寻找最优参数。
    """
    console.print(Panel("[bold cyan]示例4: 对比不同配置[/bold cyan]"))
    
    # 获取数据
    fr = fetch_ohlcv_binance_spot(symbol="BTC/USDT", timeframe="1h", limit=500)
    df = compute_features(fr.df)
    
    # 检测事件
    cfg = DetectionConfig(lookback_bars=400)
    analysis = detect_wyckoff(
        df,
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        cfg=cfg,
    )
    
    if not analysis.events:
        console.print("[yellow]未检测到事件[/yellow]")
        return
    
    # 测试不同配置
    configs = [
        ("保守 (1.5/2 ATR)", BacktestConfig(stop_loss_atr=1.5, take_profit_atr=2.0)),
        ("标准 (2/3 ATR)", BacktestConfig(stop_loss_atr=2.0, take_profit_atr=3.0)),
        ("激进 (2.5/4 ATR)", BacktestConfig(stop_loss_atr=2.5, take_profit_atr=4.0)),
        ("宽松 (3/5 ATR)", BacktestConfig(stop_loss_atr=3.0, take_profit_atr=5.0)),
    ]
    
    table = Table(title="不同配置对比")
    table.add_column("配置", style="cyan")
    table.add_column("交易数", justify="right")
    table.add_column("胜率", justify="right")
    table.add_column("收益%", justify="right")
    table.add_column("回撤%", justify="right")
    table.add_column("盈亏比", justify="right")
    
    for name, config in configs:
        engine = BacktestEngine(config)
        result = engine.run(df, analysis.events)
        metrics = calculate_metrics(result)
        
        table.add_row(
            name,
            str(result.total_trades),
            f"{metrics.trade_metrics.win_rate*100:.1f}%",
            f"{metrics.total_return_pct:.2f}%",
            f"{metrics.max_drawdown_pct:.2f}%",
            f"{metrics.trade_metrics.profit_factor:.2f}",
        )
    
    console.print(table)


def example_walk_forward():
    """
    示例5: Walk-Forward 测试
    
    使用滚动窗口进行回测，验证策略的稳健性。
    """
    console.print(Panel("[bold cyan]示例5: Walk-Forward 测试[/bold cyan]"))
    
    # 获取更多数据
    fr = fetch_ohlcv_binance_spot(symbol="BTC/USDT", timeframe="1h", limit=1000)
    df = compute_features(fr.df)
    
    # 检测事件
    cfg = DetectionConfig(lookback_bars=500)
    analysis = detect_wyckoff(
        df,
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1h",
        cfg=cfg,
    )
    
    if not analysis.events:
        console.print("[yellow]未检测到事件[/yellow]")
        return
    
    # Walk-forward 回测
    config = BacktestConfig()
    engine = BacktestEngine(config)
    results = engine.run_walk_forward(df, analysis.events, n_splits=5)
    
    table = Table(title="Walk-Forward 结果")
    table.add_column("周期", style="cyan")
    table.add_column("交易数", justify="right")
    table.add_column("胜率", justify="right")
    table.add_column("收益%", justify="right")
    
    for i, result in enumerate(results):
        metrics = calculate_metrics(result)
        table.add_row(
            f"周期 {i+1}",
            str(result.total_trades),
            f"{metrics.trade_metrics.win_rate*100:.1f}%",
            f"{metrics.total_return_pct:.2f}%",
        )
    
    console.print(table)
    
    # 计算整体统计
    total_trades = sum(r.total_trades for r in results)
    avg_win_rate = sum(
        calculate_metrics(r).trade_metrics.win_rate for r in results if r.total_trades > 0
    ) / len([r for r in results if r.total_trades > 0]) if results else 0
    
    console.print("")
    console.print(f"[bold]整体统计:[/bold]")
    console.print(f"  总交易数: {total_trades}")
    console.print(f"  平均胜率: {avg_win_rate*100:.1f}%")


def main():
    """运行所有示例"""
    import os
    os.makedirs("out", exist_ok=True)
    
    console.print(Panel("[bold magenta]威科夫回测系统示例[/bold magenta]", expand=False))
    console.print("")
    
    example_event_evaluation()
    console.print("")
    
    example_simple_backtest()
    console.print("")
    
    example_filtered_backtest()
    console.print("")
    
    example_compare_configs()
    console.print("")
    
    example_walk_forward()
    console.print("")
    
    console.print("[bold green]所有示例完成！[/bold green]")


if __name__ == "__main__":
    main()

