"""
状态机可视化模块

创建威科夫状态机转换图和阶段进度图
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wyckoff_ai.logging import get_logger
from wyckoff_ai.wyckoff.state_machine import (
    STATE_DESCRIPTIONS,
    WyckoffState,
    StateMachineResult,
    StateTransition,
)

logger = get_logger("visualization.state_diagram")

# 状态颜色映射
STATE_COLORS = {
    WyckoffState.UNKNOWN: "#9E9E9E",
    WyckoffState.ACC_PHASE_A: "#81C784",
    WyckoffState.ACC_PHASE_B: "#66BB6A",
    WyckoffState.ACC_PHASE_C: "#4CAF50",
    WyckoffState.ACC_PHASE_D: "#43A047",
    WyckoffState.ACC_PHASE_E: "#388E3C",
    WyckoffState.DIST_PHASE_A: "#E57373",
    WyckoffState.DIST_PHASE_B: "#EF5350",
    WyckoffState.DIST_PHASE_C: "#F44336",
    WyckoffState.DIST_PHASE_D: "#E53935",
    WyckoffState.MARKUP: "#2196F3",
    WyckoffState.MARKDOWN: "#FF5722",
    WyckoffState.RANGE: "#FFC107",
}

# 状态位置（用于图表布局）
STATE_POSITIONS = {
    # 吸筹序列（左侧，从下到上）
    WyckoffState.ACC_PHASE_A: (0, 1),
    WyckoffState.ACC_PHASE_B: (0, 2),
    WyckoffState.ACC_PHASE_C: (0, 3),
    WyckoffState.ACC_PHASE_D: (0, 4),
    WyckoffState.ACC_PHASE_E: (0, 5),
    
    # 派发序列（右侧，从上到下）
    WyckoffState.DIST_PHASE_A: (2, 5),
    WyckoffState.DIST_PHASE_B: (2, 4),
    WyckoffState.DIST_PHASE_C: (2, 3),
    WyckoffState.DIST_PHASE_D: (2, 2),
    
    # 趋势状态（中间）
    WyckoffState.MARKUP: (1, 6),
    WyckoffState.MARKDOWN: (1, 0),
    WyckoffState.RANGE: (1, 3),
    WyckoffState.UNKNOWN: (1, -1),
}


def create_state_diagram(
    result: StateMachineResult | None = None,
    current_state: WyckoffState | None = None,
    transitions: list[StateTransition] | None = None,
    title: str = "威科夫状态机",
    height: int = 700,
) -> go.Figure:
    """
    创建状态机转换图
    
    Args:
        result: 状态机分析结果
        current_state: 当前状态（如果没有 result）
        transitions: 状态转换历史
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    logger.debug("创建状态机图")
    
    if result:
        current_state = result.current_state
        transitions = result.transition_history
    
    fig = go.Figure()
    
    # 绘制所有状态节点
    for state, (x, y) in STATE_POSITIONS.items():
        color = STATE_COLORS.get(state, "#9E9E9E")
        is_current = (state == current_state)
        
        # 节点样式
        size = 50 if is_current else 35
        line_width = 4 if is_current else 1
        
        # 状态名称
        state_name = STATE_DESCRIPTIONS.get(state, state.value)
        
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=line_width, color="white"),
                    symbol="circle",
                ),
                text=[state_name.split(" - ")[0] if " - " in state_name else state_name[:8]],
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                hovertemplate=f"<b>{state_name}</b><extra></extra>",
                showlegend=False,
            )
        )
    
    # 绘制状态转换箭头
    if transitions:
        for trans in transitions:
            from_pos = STATE_POSITIONS.get(trans.from_state)
            to_pos = STATE_POSITIONS.get(trans.to_state)
            
            if from_pos and to_pos:
                # 箭头线
                fig.add_annotation(
                    x=to_pos[0],
                    y=to_pos[1],
                    ax=from_pos[0],
                    ay=from_pos[1],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="#FFD700",
                )
    
    # 添加序列标签
    fig.add_annotation(
        x=0, y=6.5,
        text="<b>吸筹序列</b>",
        showarrow=False,
        font=dict(size=14, color="#4CAF50"),
    )
    fig.add_annotation(
        x=2, y=6.5,
        text="<b>派发序列</b>",
        showarrow=False,
        font=dict(size=14, color="#F44336"),
    )
    
    # 添加当前状态信息
    if current_state:
        state_desc = STATE_DESCRIPTIONS.get(current_state, current_state.value)
        fig.add_annotation(
            x=0.5, y=-0.1,
            xref="paper", yref="paper",
            text=f"<b>当前状态:</b> {state_desc}",
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=8,
        )
    
    # 样式设置
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        template="plotly_dark",
        height=height,
        showlegend=False,
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.5, 2.5],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-2, 8],
        ),
        margin=dict(l=40, r=40, t=60, b=80),
    )
    
    return fig


def create_phase_progress_chart(
    result: StateMachineResult,
    title: str = "阶段进度",
    height: int = 300,
) -> go.Figure:
    """
    创建阶段进度条图
    
    Args:
        result: 状态机分析结果
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    logger.debug("创建阶段进度图")
    
    progress = result.phase_progress
    
    fig = go.Figure()
    
    # 进度条背景
    fig.add_trace(
        go.Bar(
            x=[1],
            y=["进度"],
            orientation="h",
            marker=dict(color="rgba(128,128,128,0.3)"),
            showlegend=False,
            hoverinfo="none",
        )
    )
    
    # 当前进度
    color = "#4CAF50" if "ACC" in result.current_state.value else "#F44336"
    fig.add_trace(
        go.Bar(
            x=[progress.progress],
            y=["进度"],
            orientation="h",
            marker=dict(color=color),
            showlegend=False,
            text=[f"{progress.progress:.0%}"],
            textposition="inside",
            textfont=dict(size=16, color="white"),
            hovertemplate=f"进度: {progress.progress:.1%}<extra></extra>",
        )
    )
    
    # 阶段标签
    phases = ["A", "B", "C", "D", "E"]
    for i, phase in enumerate(phases):
        x_pos = (i + 0.5) / 5
        fig.add_annotation(
            x=x_pos, y=1.2,
            xref="paper", yref="paper",
            text=f"Phase {phase}",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
        # 分隔线
        if i > 0:
            fig.add_vline(
                x=i/5, 
                line=dict(color="gray", width=1, dash="dot"),
            )
    
    # 当前阶段标记（从 current_state 提取阶段）
    current_state = progress.current_state.value if hasattr(progress.current_state, 'value') else str(progress.current_state)
    # 从状态名中提取阶段（如 "accumulation_phase_a" -> "A"）
    current_phase = None
    if "_phase_" in current_state:
        phase_letter = current_state.split("_phase_")[-1].upper()
        if phase_letter in ["A", "B", "C", "D", "E"]:
            current_phase = phase_letter
    if current_phase:
        phase_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(current_phase, 0)
        x_pos = (phase_idx + 0.5) / 5
        fig.add_annotation(
            x=x_pos, y=-0.3,
            xref="paper", yref="paper",
            text="▲",
            showarrow=False,
            font=dict(size=20, color=color),
        )
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        template="plotly_dark",
        height=height,
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        margin=dict(l=20, r=20, t=50, b=50),
        bargap=0,
    )
    
    return fig


def create_timeline_chart(
    transitions: list[StateTransition],
    title: str = "状态转换时间线",
    height: int = 400,
) -> go.Figure:
    """
    创建状态转换时间线图
    
    Args:
        transitions: 状态转换列表
        title: 图表标题
        height: 图表高度
        
    Returns:
        Plotly Figure 对象
    """
    logger.debug(f"创建时间线图: {len(transitions)} 个转换")
    
    if not transitions:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="暂无状态转换记录",
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_dark",
            height=height,
        )
        return fig
    
    fig = go.Figure()
    
    # 时间线
    times = [trans.timestamp for trans in transitions]
    states = [trans.to_state.value for trans in transitions]
    events = [trans.trigger_event for trans in transitions]
    confidences = [trans.confidence for trans in transitions]
    
    # 状态点
    colors = [STATE_COLORS.get(trans.to_state, "#9E9E9E") for trans in transitions]
    
    fig.add_trace(
        go.Scatter(
            x=times,
            y=states,
            mode="lines+markers",
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color="white"),
            ),
            line=dict(color="rgba(255,255,255,0.3)", width=2),
            text=[f"触发: {e}<br>置信度: {c:.0%}" for e, c in zip(events, confidences)],
            hovertemplate="<b>%{y}</b><br>时间: %{x}<br>%{text}<extra></extra>",
        )
    )
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        template="plotly_dark",
        height=height,
        xaxis=dict(title="时间"),
        yaxis=dict(title="状态", categoryorder="array"),
        margin=dict(l=150, r=40, t=60, b=60),
    )
    
    return fig


def create_combined_state_view(
    result: StateMachineResult,
    title: str = "威科夫状态分析",
    height: int = 800,
) -> go.Figure:
    """
    创建组合状态视图（状态图 + 进度条 + 时间线）
    
    Args:
        result: 状态机分析结果
        title: 图表标题
        height: 总高度
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.6, 0.4],
        column_widths=[0.5, 0.5],
        subplot_titles=("状态机图", "阶段进度", "状态转换时间线", "偏向分析"),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter", "colspan": 2}, None],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    # 1. 状态机图（简化版）
    for state, (x, y) in STATE_POSITIONS.items():
        color = STATE_COLORS.get(state, "#9E9E9E")
        is_current = (state == result.current_state)
        size = 30 if is_current else 20
        
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=size, color=color, line=dict(width=2 if is_current else 0, color="white")),
                showlegend=False,
                hoverinfo="none",
            ),
            row=1, col=1,
        )
    
    # 2. 进度条
    progress = result.phase_progress.progress
    color = "#4CAF50" if result.bias == "bullish" else "#F44336" if result.bias == "bearish" else "#FFC107"
    
    fig.add_trace(
        go.Bar(x=[progress], y=[""], orientation="h", marker=dict(color=color), showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=[1-progress], y=[""], orientation="h", marker=dict(color="rgba(128,128,128,0.3)"), showlegend=False),
        row=1, col=2,
    )
    
    # 3. 时间线
    if result.transition_history:
        times = [t.timestamp for t in result.transition_history]
        states = [t.to_state.value for t in result.transition_history]
        colors = [STATE_COLORS.get(t.to_state, "#9E9E9E") for t in result.transition_history]
        
        fig.add_trace(
            go.Scatter(
                x=times, y=states,
                mode="lines+markers",
                marker=dict(size=10, color=colors),
                line=dict(color="rgba(255,255,255,0.3)"),
                showlegend=False,
            ),
            row=2, col=1,
        )
    
    # 样式
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        template="plotly_dark",
        height=height,
        showlegend=False,
    )
    
    return fig

