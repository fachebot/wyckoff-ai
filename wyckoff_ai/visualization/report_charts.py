"""
报告图表生成模块

生成可嵌入报告的可视化图表
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wyckoff_ai.backtest.engine import BacktestResult, TradeDirection
from wyckoff_ai.backtest.metrics import calculate_metrics
from wyckoff_ai.logging import get_logger

logger = get_logger("visualization.report_charts")


def create_equity_curve_chart(
    result: BacktestResult,
    title: str = "资金曲线",
    height: int = 400,
) -> go.Figure:
    """
    创建资金曲线图
    
    Args:
        result: 回测结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    if not result.equity_curve:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="无资金曲线数据", showarrow=False, font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # 资金曲线
    fig.add_trace(
        go.Scatter(
            y=result.equity_curve,
            mode="lines",
            name="资金",
            line=dict(color="#00BCD4", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 188, 212, 0.1)",
        )
    )
    
    # 初始资金线
    fig.add_hline(
        y=result.config.initial_capital,
        line=dict(color="gray", width=1, dash="dash"),
        annotation_text="初始资金",
    )
    
    # 最高点
    max_eq = max(result.equity_curve)
    max_idx = result.equity_curve.index(max_eq)
    fig.add_annotation(
        x=max_idx, y=max_eq,
        text=f"最高: {max_eq:,.0f}",
        showarrow=True, arrowhead=2,
        font=dict(color="#4CAF50"),
    )
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        height=height,
        xaxis=dict(title="K线序号"),
        yaxis=dict(title="资金"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    
    return fig


def create_drawdown_chart(
    result: BacktestResult,
    title: str = "回撤曲线",
    height: int = 300,
) -> go.Figure:
    """
    创建回撤曲线图
    
    Args:
        result: 回测结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    if not result.equity_curve:
        fig = go.Figure()
        return fig
    
    # 计算回撤
    equity = result.equity_curve
    peak = equity[0]
    drawdowns = []
    
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100 if peak > 0 else 0
        drawdowns.append(dd)
    
    fig = go.Figure()
    
    # 回撤曲线
    fig.add_trace(
        go.Scatter(
            y=drawdowns,
            mode="lines",
            name="回撤",
            line=dict(color="#F44336", width=2),
            fill="tozeroy",
            fillcolor="rgba(244, 67, 54, 0.2)",
        )
    )
    
    # 最大回撤标记
    min_dd = min(drawdowns)
    min_idx = drawdowns.index(min_dd)
    fig.add_annotation(
        x=min_idx, y=min_dd,
        text=f"最大回撤: {min_dd:.1f}%",
        showarrow=True, arrowhead=2,
        font=dict(color="#F44336"),
    )
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        height=height,
        xaxis=dict(title="K线序号"),
        yaxis=dict(title="回撤 (%)", range=[min(drawdowns) * 1.1, 5]),
        showlegend=False,
    )
    
    return fig


def create_trade_distribution_chart(
    result: BacktestResult,
    title: str = "交易收益分布",
    height: int = 400,
) -> go.Figure:
    """
    创建交易收益分布图
    
    Args:
        result: 回测结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    if not result.trades:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="无交易数据", showarrow=False, font=dict(size=16)
        )
        return fig
    
    pnl_pcts = [t.pnl_pct for t in result.trades]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("收益分布", "累计收益"),
        column_widths=[0.5, 0.5],
    )
    
    # 1. 直方图
    wins = [p for p in pnl_pcts if p >= 0]
    losses = [p for p in pnl_pcts if p < 0]
    
    fig.add_trace(
        go.Histogram(x=wins, name="盈利", marker_color="#4CAF50", opacity=0.7, nbinsx=20),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=losses, name="亏损", marker_color="#F44336", opacity=0.7, nbinsx=20),
        row=1, col=1,
    )
    
    # 2. 累计收益曲线
    cumulative = []
    total = 0
    for p in pnl_pcts:
        total += p
        cumulative.append(total)
    
    colors = ["#4CAF50" if c >= 0 else "#F44336" for c in cumulative]
    
    fig.add_trace(
        go.Scatter(
            y=cumulative,
            mode="lines+markers",
            name="累计收益",
            line=dict(color="#00BCD4", width=2),
            marker=dict(size=6, color=colors),
        ),
        row=1, col=2,
    )
    
    fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=2)
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        barmode="overlay",
    )
    
    fig.update_xaxes(title_text="收益 (%)", row=1, col=1)
    fig.update_xaxes(title_text="交易序号", row=1, col=2)
    fig.update_yaxes(title_text="频次", row=1, col=1)
    fig.update_yaxes(title_text="累计收益 (%)", row=1, col=2)
    
    return fig


def create_event_performance_chart(
    result: BacktestResult,
    title: str = "按事件类型表现",
    height: int = 400,
) -> go.Figure:
    """
    创建事件类型表现图
    
    Args:
        result: 回测结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    if not result.stats_by_event:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="无事件统计数据", showarrow=False, font=dict(size=16)
        )
        return fig
    
    # 准备数据
    events = list(result.stats_by_event.keys())
    win_rates = [result.stats_by_event[e].get("win_rate", 0) * 100 for e in events]
    total_pnls = [result.stats_by_event[e].get("total_pnl", 0) for e in events]
    counts = [result.stats_by_event[e].get("count", 0) for e in events]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("胜率", "总盈亏"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )
    
    # 胜率
    colors_wr = ["#4CAF50" if wr >= 50 else "#F44336" for wr in win_rates]
    fig.add_trace(
        go.Bar(
            x=events, y=win_rates,
            name="胜率",
            marker_color=colors_wr,
            text=[f"{wr:.1f}%" for wr in win_rates],
            textposition="outside",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=50, line=dict(color="gray", dash="dash"), row=1, col=1)
    
    # 总盈亏
    colors_pnl = ["#4CAF50" if p >= 0 else "#F44336" for p in total_pnls]
    fig.add_trace(
        go.Bar(
            x=events, y=total_pnls,
            name="总盈亏",
            marker_color=colors_pnl,
            text=[f"{p:,.0f}" for p in total_pnls],
            textposition="outside",
        ),
        row=1, col=2,
    )
    fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=2)
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        height=height,
        showlegend=False,
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="胜率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="盈亏", row=1, col=2)
    
    return fig


def create_monthly_returns_heatmap(
    result: BacktestResult,
    title: str = "月度收益热力图",
    height: int = 400,
) -> go.Figure:
    """
    创建月度收益热力图（如果数据跨越多月）
    
    Args:
        result: 回测结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    if not result.trades:
        fig = go.Figure()
        return fig
    
    # 按月聚合收益
    monthly_returns = {}
    
    for trade in result.trades:
        if trade.exit_time:
            # 解析日期
            try:
                month_key = trade.exit_time[:7]  # YYYY-MM
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0
                monthly_returns[month_key] += trade.pnl_pct
            except (IndexError, ValueError):
                continue
    
    if not monthly_returns:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="数据不足以生成月度图表", showarrow=False, font=dict(size=14)
        )
        return fig
    
    months = sorted(monthly_returns.keys())
    returns = [monthly_returns[m] for m in months]
    
    # 颜色映射
    colors = ["#F44336" if r < 0 else "#4CAF50" for r in returns]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=months,
            y=returns,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in returns],
            textposition="outside",
        )
    )
    
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        height=height,
        xaxis=dict(title="月份", tickangle=45),
        yaxis=dict(title="收益 (%)"),
        showlegend=False,
    )
    
    return fig


def create_backtest_summary_chart(
    result: BacktestResult,
    title: str = "回测结果汇总",
    height: int = 900,
) -> go.Figure:
    """
    创建回测汇总图（多子图）
    
    Args:
        result: 回测结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(
            "资金曲线", "回撤曲线",
            "交易收益分布", "累计收益",
            "按事件类型胜率", "按方向统计"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )
    
    # 1. 资金曲线
    if result.equity_curve:
        fig.add_trace(
            go.Scatter(
                y=result.equity_curve,
                mode="lines",
                name="资金",
                line=dict(color="#00BCD4", width=2),
            ),
            row=1, col=1,
        )
    
    # 2. 回撤曲线
    if result.equity_curve:
        equity = result.equity_curve
        peak = equity[0]
        drawdowns = []
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        
        fig.add_trace(
            go.Scatter(
                y=drawdowns,
                mode="lines",
                name="回撤",
                line=dict(color="#F44336", width=2),
                fill="tozeroy",
                fillcolor="rgba(244, 67, 54, 0.2)",
            ),
            row=1, col=2,
        )
    
    # 3. 交易收益分布
    if result.trades:
        pnl_pcts = [t.pnl_pct for t in result.trades]
        wins = [p for p in pnl_pcts if p >= 0]
        losses = [p for p in pnl_pcts if p < 0]
        
        fig.add_trace(
            go.Histogram(x=wins, name="盈利", marker_color="#4CAF50", opacity=0.7),
            row=2, col=1,
        )
        fig.add_trace(
            go.Histogram(x=losses, name="亏损", marker_color="#F44336", opacity=0.7),
            row=2, col=1,
        )
        
        # 4. 累计收益
        cumulative = []
        total = 0
        for p in pnl_pcts:
            total += p
            cumulative.append(total)
        
        fig.add_trace(
            go.Scatter(
                y=cumulative,
                mode="lines",
                name="累计收益",
                line=dict(color="#00BCD4", width=2),
            ),
            row=2, col=2,
        )
    
    # 5. 按事件类型胜率
    if result.stats_by_event:
        events = list(result.stats_by_event.keys())
        win_rates = [result.stats_by_event[e].get("win_rate", 0) * 100 for e in events]
        colors_wr = ["#4CAF50" if wr >= 50 else "#F44336" for wr in win_rates]
        
        fig.add_trace(
            go.Bar(x=events, y=win_rates, marker_color=colors_wr, showlegend=False),
            row=3, col=1,
        )
    
    # 6. 按方向统计
    if result.trades:
        long_wins = sum(1 for t in result.trades if t.direction == TradeDirection.LONG and t.pnl > 0)
        long_losses = sum(1 for t in result.trades if t.direction == TradeDirection.LONG and t.pnl <= 0)
        short_wins = sum(1 for t in result.trades if t.direction == TradeDirection.SHORT and t.pnl > 0)
        short_losses = sum(1 for t in result.trades if t.direction == TradeDirection.SHORT and t.pnl <= 0)
        
        fig.add_trace(
            go.Bar(
                x=["多头盈", "多头亏", "空头盈", "空头亏"],
                y=[long_wins, long_losses, short_wins, short_losses],
                marker_color=["#4CAF50", "#F44336", "#4CAF50", "#F44336"],
                showlegend=False,
            ),
            row=3, col=2,
        )
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        template="plotly_dark",
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    
    return fig


def generate_html_report_with_charts(
    result: BacktestResult,
    title: str = "威科夫策略回测报告",
) -> str:
    """
    生成带交互式图表的 HTML 报告
    
    Args:
        result: 回测结果
        title: 报告标题
        
    Returns:
        完整 HTML 内容
    """
    metrics = calculate_metrics(result)
    
    # 生成图表
    equity_chart = create_equity_curve_chart(result)
    drawdown_chart = create_drawdown_chart(result)
    distribution_chart = create_trade_distribution_chart(result)
    event_chart = create_event_performance_chart(result)
    
    # 转换为 HTML div
    equity_html = equity_chart.to_html(full_html=False, include_plotlyjs=False)
    drawdown_html = drawdown_chart.to_html(full_html=False, include_plotlyjs=False)
    distribution_html = distribution_chart.to_html(full_html=False, include_plotlyjs=False)
    event_html = event_chart.to_html(full_html=False, include_plotlyjs=False)
    
    # 构建 HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00bcd4;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 188, 212, 0.3);
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 188, 212, 0.2);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value.positive {{
            color: #4CAF50;
        }}
        .metric-value.negative {{
            color: #F44336;
        }}
        .metric-value.neutral {{
            color: #00bcd4;
        }}
        .metric-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .chart-section {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .chart-section h2 {{
            color: #00bcd4;
            margin-top: 0;
            font-size: 1.3em;
            border-bottom: 1px solid rgba(0, 188, 212, 0.3);
            padding-bottom: 10px;
        }}
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {title}</h1>
        <p class="subtitle">回测区间: {result.start_time[:19]} ~ {result.end_time[:19]} | 总K线: {result.total_bars}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.total_return >= 0 else 'negative'}">{metrics.total_return_pct:.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.trade_metrics.win_rate >= 0.5 else 'negative'}">{metrics.trade_metrics.win_rate * 100:.1f}%</div>
                <div class="metric-label">胜率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.sharpe_ratio >= 1 else 'neutral' if metrics.sharpe_ratio >= 0 else 'negative'}">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe 比率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{metrics.max_drawdown_pct:.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.trade_metrics.profit_factor >= 1.5 else 'neutral' if metrics.trade_metrics.profit_factor >= 1 else 'negative'}">{metrics.trade_metrics.profit_factor:.2f}</div>
                <div class="metric-label">盈亏比</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{result.total_trades}</div>
                <div class="metric-label">总交易</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2>💰 资金曲线</h2>
            {equity_html}
        </div>
        
        <div class="chart-section">
            <h2>📉 回撤分析</h2>
            {drawdown_html}
        </div>
        
        <div class="chart-section">
            <h2>📈 交易分析</h2>
            {distribution_html}
        </div>
        
        <div class="chart-section">
            <h2>📋 事件表现</h2>
            {event_html}
        </div>
        
        <div class="chart-section" style="text-align: center; color: #666;">
            <p>⚠️ 风险提示：历史回测结果不代表未来表现。建议在模拟盘验证后再进行实盘交易。</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

