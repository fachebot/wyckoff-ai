"""
K 线图表可视化模块

使用 Plotly 创建交互式 K 线图，支持：
- 威科夫事件标注
- 关键价位线
- 区间高亮
- 成交量子图
- 技术指标叠加
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wyckoff_ai.logging import get_logger
from wyckoff_ai.schemas import WyckoffAnalysis, WyckoffEvent

logger = get_logger("visualization.candlestick")

# 事件颜色映射
EVENT_COLORS = {
    # 看涨事件 - 绿色系
    "SC": "#00C853",      # 卖出高潮 - 亮绿
    "AR": "#69F0AE",      # 自动反弹 - 浅绿
    "ST": "#00E676",      # 二次测试 - 绿
    "SOS": "#1B5E20",     # 强势信号 - 深绿
    "LPS": "#4CAF50",     # 最后支撑点 - 中绿
    "SPRING": "#76FF03",  # 弹簧 - 黄绿
    "JAC": "#00BFA5",     # 跳过小溪 - 青绿
    "BUEC": "#64FFDA",    # 回测 - 浅青
    "TEST": "#A5D6A7",    # 测试 - 淡绿
    
    # 看跌事件 - 红色系
    "BC": "#FF1744",      # 买入高潮 - 亮红
    "SOW": "#D50000",     # 弱势信号 - 深红
    "LPSY": "#F44336",    # 最后供应点 - 红
    "UT": "#FF5252",      # 上冲 - 浅红
    "UTAD": "#FF8A80",    # 派发后上冲 - 粉红
    
    # 中性事件 - 蓝色/灰色系
    "PSY": "#2196F3",     # 初步供应/支撑 - 蓝
    "TR": "#9E9E9E",      # 交易区间 - 灰
}

# 事件方向
EVENT_DIRECTION = {
    "SC": "bullish", "AR": "bullish", "ST": "bullish", "SOS": "bullish",
    "LPS": "bullish", "SPRING": "bullish", "JAC": "bullish", "BUEC": "bullish",
    "TEST": "bullish",
    "BC": "bearish", "SOW": "bearish", "LPSY": "bearish", "UT": "bearish",
    "UTAD": "bearish",
    "PSY": "neutral", "TR": "neutral",
}


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "K线图",
    show_volume: bool = True,
    height: int = 800,
) -> go.Figure:
    """
    创建基础 K 线图
    
    Args:
        df: OHLCV 数据，需要 timestamp, open, high, low, close, volume 列
        title: 图表标题
        show_volume: 是否显示成交量
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    logger.debug(f"创建K线图: {len(df)} 根K线")
    
    # 确保时间戳格式正确
    if "timestamp" in df.columns:
        x_data = pd.to_datetime(df["timestamp"])
    else:
        x_data = df.index
    
    # 创建子图
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("", "成交量"),
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # K 线图
    fig.add_trace(
        go.Candlestick(
            x=x_data,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
            decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
        ),
        row=1, col=1,
    )
    
    # 成交量
    if show_volume and "volume" in df.columns:
        colors = ["#26A69A" if c >= o else "#EF5350" 
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(
                x=x_data,
                y=df["volume"],
                name="成交量",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1,
        )
    
    # 样式设置
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        height=height,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=40),
    )
    
    # 隐藏周末/非交易时间的空白
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # 隐藏周末
        ]
    )
    
    return fig


def add_events_to_chart(
    fig: go.Figure,
    events: list[WyckoffEvent],
    df: pd.DataFrame,
    show_labels: bool = True,
    min_confidence: float = 0.5,
) -> go.Figure:
    """
    在图表上添加威科夫事件标注
    
    Args:
        fig: Plotly Figure 对象
        events: 威科夫事件列表
        df: OHLCV 数据
        show_labels: 是否显示事件标签
        min_confidence: 最小置信度阈值
        
    Returns:
        更新后的 Figure
    """
    logger.debug(f"添加 {len(events)} 个事件标注")
    
    # 过滤低置信度事件
    filtered_events = [e for e in events if e.confidence >= min_confidence]
    
    # 按类型分组
    bullish_events = []
    bearish_events = []
    neutral_events = []
    
    for event in filtered_events:
        direction = EVENT_DIRECTION.get(event.type, "neutral")
        if direction == "bullish":
            bullish_events.append(event)
        elif direction == "bearish":
            bearish_events.append(event)
        else:
            neutral_events.append(event)
    
    # 添加事件标记
    def add_event_markers(events: list[WyckoffEvent], symbol: str, name: str):
        if not events:
            return
            
        x_vals = [pd.to_datetime(e.ts) for e in events]
        y_vals = [e.price for e in events]
        colors = [EVENT_COLORS.get(e.type, "#FFFFFF") for e in events]
        texts = [f"{e.type}<br>置信度: {e.confidence:.0%}" for e in events]
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    symbol=symbol,
                    size=15,
                    color=colors,
                    line=dict(width=2, color="white"),
                ),
                text=[e.type for e in events] if show_labels else None,
                textposition="top center",
                textfont=dict(size=10, color="white"),
                hovertemplate="<b>%{text}</b><br>价格: %{y:.2f}<br>时间: %{x}<extra></extra>",
                customdata=texts,
                name=name,
            ),
            row=1, col=1,
        )
    
    add_event_markers(bullish_events, "triangle-up", "看涨事件")
    add_event_markers(bearish_events, "triangle-down", "看跌事件")
    add_event_markers(neutral_events, "diamond", "中性事件")
    
    return fig


def add_support_resistance(
    fig: go.Figure,
    support_levels: list[float],
    resistance_levels: list[float],
    df: pd.DataFrame,
) -> go.Figure:
    """
    添加支撑/阻力位线
    
    Args:
        fig: Plotly Figure 对象
        support_levels: 支撑位列表
        resistance_levels: 阻力位列表
        df: OHLCV 数据（用于确定线的范围）
        
    Returns:
        更新后的 Figure
    """
    if "timestamp" in df.columns:
        x_start = pd.to_datetime(df["timestamp"].iloc[0])
        x_end = pd.to_datetime(df["timestamp"].iloc[-1])
    else:
        x_start = df.index[0]
        x_end = df.index[-1]
    
    # 支撑位 - 绿色虚线
    for i, level in enumerate(support_levels):
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[level, level],
                mode="lines",
                line=dict(color="#4CAF50", width=1, dash="dash"),
                name=f"支撑 {level:.2f}" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="y",
            ),
            row=1, col=1,
        )
    
    # 阻力位 - 红色虚线
    for i, level in enumerate(resistance_levels):
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[level, level],
                mode="lines",
                line=dict(color="#F44336", width=1, dash="dash"),
                name=f"阻力 {level:.2f}" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="y",
            ),
            row=1, col=1,
        )
    
    return fig


def add_range_highlight(
    fig: go.Figure,
    range_low: float,
    range_high: float,
    df: pd.DataFrame,
    label: str = "交易区间",
) -> go.Figure:
    """
    添加区间高亮
    
    Args:
        fig: Plotly Figure 对象
        range_low: 区间下限
        range_high: 区间上限
        df: OHLCV 数据
        label: 区间标签
        
    Returns:
        更新后的 Figure
    """
    if "timestamp" in df.columns:
        x_start = pd.to_datetime(df["timestamp"].iloc[0])
        x_end = pd.to_datetime(df["timestamp"].iloc[-1])
    else:
        x_start = df.index[0]
        x_end = df.index[-1]
    
    # 添加矩形区域
    fig.add_shape(
        type="rect",
        x0=x_start, x1=x_end,
        y0=range_low, y1=range_high,
        fillcolor="rgba(128, 128, 128, 0.2)",
        line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
        row=1, col=1,
    )
    
    # 添加中线
    range_mid = (range_low + range_high) / 2
    fig.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[range_mid, range_mid],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dot"),
            name=f"{label} 中线",
            hoverinfo="y",
        ),
        row=1, col=1,
    )
    
    return fig


def add_ema_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    periods: list[int] = [20, 50, 200],
) -> go.Figure:
    """
    添加 EMA 均线
    
    Args:
        fig: Plotly Figure 对象
        df: 带有 close 列的数据
        periods: EMA 周期列表
        
    Returns:
        更新后的 Figure
    """
    if "timestamp" in df.columns:
        x_data = pd.to_datetime(df["timestamp"])
    else:
        x_data = df.index
    
    colors = ["#FFD700", "#00BFFF", "#FF69B4"]  # 金色、蓝色、粉色
    
    for i, period in enumerate(periods):
        col_name = f"ema_{period}"
        if col_name in df.columns:
            ema_data = df[col_name]
        else:
            ema_data = df["close"].ewm(span=period, adjust=False).mean()
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=ema_data,
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1),
                name=f"EMA{period}",
                opacity=0.7,
            ),
            row=1, col=1,
        )
    
    return fig


def create_analysis_chart(
    df: pd.DataFrame,
    analysis: WyckoffAnalysis,
    title: str | None = None,
    show_volume: bool = True,
    show_ema: bool = True,
    show_events: bool = True,
    show_levels: bool = True,
    show_range: bool = True,
    min_confidence: float = 0.5,
    height: int = 900,
) -> go.Figure:
    """
    创建完整的威科夫分析图表
    
    Args:
        df: OHLCV 数据
        analysis: 威科夫分析结果
        title: 图表标题（默认使用 symbol + timeframe）
        show_volume: 显示成交量
        show_ema: 显示 EMA 均线
        show_events: 显示事件标注
        show_levels: 显示支撑/阻力位
        show_range: 显示交易区间
        min_confidence: 最小事件置信度
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    logger.info(f"创建分析图表: {analysis.symbol} {analysis.timeframe}")
    
    # 默认标题
    if title is None:
        title = f"威科夫分析 - {analysis.symbol} ({analysis.timeframe})"
        if analysis.market_structure:
            title += f" | {analysis.market_structure}"
    
    # 创建基础图表
    fig = create_candlestick_chart(df, title=title, show_volume=show_volume, height=height)
    
    # 添加 EMA
    if show_ema:
        fig = add_ema_lines(fig, df, periods=[50])
    
    # 添加事件标注
    if show_events and analysis.events:
        fig = add_events_to_chart(fig, analysis.events, df, min_confidence=min_confidence)
    
    # 添加支撑/阻力位
    if show_levels and analysis.levels:
        fig = add_support_resistance(
            fig,
            support_levels=analysis.levels.support,
            resistance_levels=analysis.levels.resistance,
            df=df,
        )
    
    # 添加交易区间
    if show_range and analysis.range and analysis.range.low and analysis.range.high:
        fig = add_range_highlight(
            fig,
            range_low=analysis.range.low,
            range_high=analysis.range.high,
            df=df,
            label="威科夫区间",
        )
    
    # 添加分析信息注释
    info_text = f"市场结构: {analysis.market_structure or '未知'}"
    if analysis.events:
        info_text += f"<br>检测事件: {len(analysis.events)} 个"
    
    fig.add_annotation(
        x=0.01, y=0.99,
        xref="paper", yref="paper",
        text=info_text,
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        align="left",
    )
    
    return fig


def save_chart(
    fig: go.Figure,
    output_path: str | Path,
    format: Literal["html", "png", "svg", "pdf"] = "html",
    width: int = 1600,
    height: int = 900,
) -> Path:
    """
    保存图表到文件
    
    Args:
        fig: Plotly Figure 对象
        output_path: 输出路径（不含扩展名）
        format: 输出格式
        width: 图片宽度（仅对图片格式有效）
        height: 图片高度（仅对图片格式有效）
        
    Returns:
        保存的文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 添加扩展名
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")
    
    if format == "html":
        fig.write_html(str(output_path), include_plotlyjs="cdn")
    else:
        fig.write_image(str(output_path), width=width, height=height, scale=2)
    
    logger.info(f"图表已保存: {output_path}")
    return output_path

