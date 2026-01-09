"""
概率化剧本引擎

基于多维度分析（状态机、事件序列、量价、历史统计）生成概率化交易剧本，
提供实用的交易指导（入场、止损、目标、仓位等）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from wyckoff_ai.schemas import WyckoffAnalysis, WyckoffEvent, StateMachineInfo
    from wyckoff_ai.wyckoff.sequence import SequenceAnalysis


@dataclass
class TradingSignal:
    """交易信号"""
    entry_price: float | None = None
    entry_condition: str = ""
    stop_loss: float | None = None
    targets: list[float] = field(default_factory=list)
    risk_reward_ratio: float = 0.0
    position_size_pct: float = 0.0  # 建议仓位百分比（0-100）
    time_horizon: str = ""  # 预期持仓时间
    confirmation_signals: list[str] = field(default_factory=list)
    invalidation_signals: list[str] = field(default_factory=list)


@dataclass
class ProbabilisticScenario:
    """概率化剧本"""
    name: str
    bias: Literal["bullish", "bearish", "neutral"]
    probability: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    
    # 概率来源分解
    probability_breakdown: dict[str, float] = field(default_factory=dict)
    
    # 交易信号
    signal: TradingSignal | None = None
    
    # 描述
    description: str = ""
    key_events: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    
    # 风险评估
    risk_level: Literal["low", "medium", "high", "extreme"] = "medium"
    risk_factors: list[str] = field(default_factory=list)


def calculate_scenario_probability(
    analysis: "WyckoffAnalysis",
    sequence: "SequenceAnalysis | None" = None,
    current_price: float | None = None,
) -> list[ProbabilisticScenario]:
    """
    计算概率化剧本
    
    Args:
        analysis: 威科夫分析结果
        sequence: 序列分析结果（可选）
        current_price: 当前价格（可选，用于计算具体点位）
    
    Returns:
        按概率排序的剧本列表
    """
    scenarios: list[ProbabilisticScenario] = []
    
    # 获取当前价格
    if current_price is None and analysis.events:
        current_price = analysis.events[-1].price
    elif current_price is None:
        if analysis.range.mid is not None:
            current_price = analysis.range.mid
        else:
            return scenarios
    
    # 获取状态机信息
    sm = analysis.state_machine
    
    # 获取最近事件
    recent_events = analysis.events[-10:] if analysis.events else []
    recent_types = {e.type for e in recent_events}
    
    # 获取历史统计
    fstats = analysis.event_forward_stats
    
    # === 剧本1: 吸筹完成，准备上涨 ===
    if _should_generate_accumulation_scenario(analysis, sm, recent_types):
        prob, breakdown = _calculate_accumulation_probability(
            analysis, sm, sequence, recent_events, fstats
        )
        if prob > 0.3:  # 只生成概率>30%的剧本
            scenario = _build_accumulation_scenario(
                analysis, sm, prob, breakdown, current_price, recent_events
            )
            scenarios.append(scenario)
    
    # === 剧本2: 派发完成，准备下跌 ===
    if _should_generate_distribution_scenario(analysis, sm, recent_types):
        prob, breakdown = _calculate_distribution_probability(
            analysis, sm, sequence, recent_events, fstats
        )
        if prob > 0.3:
            scenario = _build_distribution_scenario(
                analysis, sm, prob, breakdown, current_price, recent_events
            )
            scenarios.append(scenario)
    
    # === 剧本3: 区间突破（向上）===
    if _should_generate_breakout_up_scenario(analysis, sm, recent_types):
        prob, breakdown = _calculate_breakout_up_probability(
            analysis, sm, recent_events, fstats
        )
        if prob > 0.25:
            scenario = _build_breakout_up_scenario(
                analysis, sm, prob, breakdown, current_price, recent_events
            )
            scenarios.append(scenario)
    
    # === 剧本4: 区间突破（向下）===
    if _should_generate_breakout_down_scenario(analysis, sm, recent_types):
        prob, breakdown = _calculate_breakout_down_probability(
            analysis, sm, recent_events, fstats
        )
        if prob > 0.25:
            scenario = _build_breakout_down_scenario(
                analysis, sm, prob, breakdown, current_price, recent_events
            )
            scenarios.append(scenario)
    
    # === 剧本5: 区间震荡延续 ===
    if _should_generate_range_continuation_scenario(analysis, sm, recent_types):
        prob, breakdown = _calculate_range_continuation_probability(
            analysis, sm, recent_events
        )
        if prob > 0.2:
            scenario = _build_range_continuation_scenario(
                analysis, sm, prob, breakdown, current_price
            )
            scenarios.append(scenario)
    
    # === 剧本6: 趋势延续 ===
    if _should_generate_trend_continuation_scenario(analysis, sm, recent_types):
        prob, breakdown = _calculate_trend_continuation_probability(
            analysis, sm, recent_events, fstats
        )
        if prob > 0.3:
            scenario = _build_trend_continuation_scenario(
                analysis, sm, prob, breakdown, current_price, recent_events
            )
            scenarios.append(scenario)
    
    # 按概率排序
    scenarios.sort(key=lambda s: s.probability, reverse=True)
    
    return scenarios


def _should_generate_accumulation_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_types: set[str],
) -> bool:
    """判断是否应该生成吸筹剧本"""
    if not analysis.range.duration_bars or analysis.range.low is None:
        return False
    
    # 状态机在吸筹阶段
    if sm and "accumulation" in sm.current_state:
        return True
    
    # 有吸筹相关事件
    if any(t in recent_types for t in ("SC", "SPRING", "TEST", "SOS", "LPS")):
        return True
    
    return False


def _should_generate_distribution_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_types: set[str],
) -> bool:
    """判断是否应该生成派发剧本"""
    if not analysis.range.duration_bars or analysis.range.high is None:
        return False
    
    # 状态机在派发阶段
    if sm and "distribution" in sm.current_state:
        return True
    
    # 有派发相关事件
    if any(t in recent_types for t in ("BC", "UT", "UTAD", "SOW", "LPSY")):
        return True
    
    return False


def _should_generate_breakout_up_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_types: set[str],
) -> bool:
    """判断是否应该生成向上突破剧本"""
    if not analysis.range.duration_bars or analysis.range.high is None:
        return False
    
    # 有突破相关事件
    if "SOS" in recent_types or "JAC" in recent_types:
        return True
    
    # 状态机在突破阶段
    if sm and sm.current_state in ("accumulation_phase_d", "accumulation_phase_e"):
        return True
    
    return False


def _should_generate_breakout_down_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_types: set[str],
) -> bool:
    """判断是否应该生成向下突破剧本"""
    if not analysis.range.duration_bars or analysis.range.low is None:
        return False
    
    # 有突破相关事件
    if "SOW" in recent_types:
        return True
    
    # 状态机在突破阶段
    if sm and sm.current_state == "distribution_phase_d":
        return True
    
    return False


def _should_generate_range_continuation_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_types: set[str],
) -> bool:
    """判断是否应该生成区间延续剧本"""
    if not analysis.range.duration_bars or analysis.range.duration_bars < 20:
        return False
    
    # 状态机在构筑阶段
    if sm and sm.current_state in ("accumulation_phase_b", "distribution_phase_b"):
        return True
    
    # 没有明确的突破信号
    if not any(t in recent_types for t in ("SOS", "SOW", "JAC")):
        return True
    
    return False


def _should_generate_trend_continuation_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_types: set[str],
) -> bool:
    """判断是否应该生成趋势延续剧本"""
    # 状态机在趋势状态
    if sm and sm.current_state in ("markup", "markdown"):
        return True
    
    # 市场结构是趋势
    if analysis.market_structure in ("markup", "markdown"):
        return True
    
    return False


def _calculate_accumulation_probability(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    sequence: "SequenceAnalysis | None",
    recent_events: list["WyckoffEvent"],
    fstats: dict,
) -> tuple[float, dict[str, float]]:
    """计算吸筹剧本概率"""
    breakdown: dict[str, float] = {}
    total_prob = 0.0
    
    # 1. 状态机因素（权重 0.4）
    if sm:
        if "accumulation_phase_c" in sm.current_state:
            breakdown["state_machine"] = 0.7
        elif "accumulation_phase_d" in sm.current_state:
            breakdown["state_machine"] = 0.85
        elif "accumulation_phase_e" in sm.current_state:
            breakdown["state_machine"] = 0.9
        elif "accumulation_phase_b" in sm.current_state:
            breakdown["state_machine"] = 0.6
        else:
            breakdown["state_machine"] = 0.4
        total_prob += breakdown["state_machine"] * 0.4
    else:
        breakdown["state_machine"] = 0.3
        total_prob += 0.3 * 0.4
    
    # 2. 事件序列因素（权重 0.3）
    if sequence:
        breakdown["sequence"] = sequence.accumulation_score
        total_prob += sequence.accumulation_score * 0.3
    else:
        # 基于事件类型判断
        event_score = 0.0
        if "SC" in {e.type for e in recent_events}:
            event_score += 0.3
        if "SPRING" in {e.type for e in recent_events} or "TEST" in {e.type for e in recent_events}:
            event_score += 0.3
        if "SOS" in {e.type for e in recent_events}:
            event_score += 0.4
        breakdown["sequence"] = min(1.0, event_score)
        total_prob += breakdown["sequence"] * 0.3
    
    # 3. 历史统计因素（权重 0.2）
    hist_score = 0.5  # 默认中性
    if fstats:
        sos_stats = fstats.get("SOS", {})
        spring_stats = fstats.get("SPRING", {})
        if sos_stats.get("win12", 0) > 0.6:
            hist_score = 0.7
        elif spring_stats.get("win12", 0) > 0.6:
            hist_score = 0.65
    breakdown["history"] = hist_score
    total_prob += hist_score * 0.2
    
    # 4. 市场结构因素（权重 0.1）
    if analysis.market_structure == "accumulation":
        breakdown["structure"] = 0.8
    elif analysis.market_structure == "range":
        breakdown["structure"] = 0.6
    else:
        breakdown["structure"] = 0.4
    total_prob += breakdown["structure"] * 0.1
    
    return min(1.0, total_prob), breakdown


def _calculate_distribution_probability(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    sequence: "SequenceAnalysis | None",
    recent_events: list["WyckoffEvent"],
    fstats: dict,
) -> tuple[float, dict[str, float]]:
    """计算派发剧本概率"""
    breakdown: dict[str, float] = {}
    total_prob = 0.0
    
    # 1. 状态机因素
    if sm:
        if "distribution_phase_c" in sm.current_state:
            breakdown["state_machine"] = 0.7
        elif "distribution_phase_d" in sm.current_state:
            breakdown["state_machine"] = 0.85
        elif "distribution_phase_b" in sm.current_state:
            breakdown["state_machine"] = 0.6
        else:
            breakdown["state_machine"] = 0.4
        total_prob += breakdown["state_machine"] * 0.4
    else:
        breakdown["state_machine"] = 0.3
        total_prob += 0.3 * 0.4
    
    # 2. 事件序列因素
    if sequence:
        breakdown["sequence"] = sequence.distribution_score
        total_prob += sequence.distribution_score * 0.3
    else:
        event_score = 0.0
        if "BC" in {e.type for e in recent_events}:
            event_score += 0.3
        if "UT" in {e.type for e in recent_events} or "UTAD" in {e.type for e in recent_events}:
            event_score += 0.3
        if "SOW" in {e.type for e in recent_events}:
            event_score += 0.4
        breakdown["sequence"] = min(1.0, event_score)
        total_prob += breakdown["sequence"] * 0.3
    
    # 3. 历史统计因素
    hist_score = 0.5
    if fstats:
        sow_stats = fstats.get("SOW", {})
        utad_stats = fstats.get("UTAD", {})
        if sow_stats.get("win12", 0) > 0.6:
            hist_score = 0.7
        elif utad_stats.get("win12", 0) > 0.6:
            hist_score = 0.65
    breakdown["history"] = hist_score
    total_prob += hist_score * 0.2
    
    # 4. 市场结构因素
    if analysis.market_structure == "distribution":
        breakdown["structure"] = 0.8
    elif analysis.market_structure == "range":
        breakdown["structure"] = 0.6
    else:
        breakdown["structure"] = 0.4
    total_prob += breakdown["structure"] * 0.1
    
    return min(1.0, total_prob), breakdown


def _calculate_breakout_up_probability(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_events: list["WyckoffEvent"],
    fstats: dict,
) -> tuple[float, dict[str, float]]:
    """计算向上突破概率"""
    breakdown: dict[str, float] = {}
    total_prob = 0.0
    
    # 检查是否有SOS事件
    has_sos = any(e.type == "SOS" for e in recent_events)
    has_jac = any(e.type == "JAC" for e in recent_events)
    
    if has_jac:
        breakdown["breakout_signal"] = 0.9
    elif has_sos:
        breakdown["breakout_signal"] = 0.75
    else:
        breakdown["breakout_signal"] = 0.4
    
    total_prob += breakdown["breakout_signal"] * 0.5
    
    # 状态机因素
    if sm and sm.current_state in ("accumulation_phase_d", "accumulation_phase_e"):
        breakdown["state_machine"] = 0.8
    else:
        breakdown["state_machine"] = 0.5
    total_prob += breakdown["state_machine"] * 0.3
    
    # 历史统计
    hist_score = 0.5
    if fstats:
        sos_stats = fstats.get("SOS", {})
        if sos_stats.get("r24_med", 0) > 0.01:
            hist_score = 0.7
    breakdown["history"] = hist_score
    total_prob += hist_score * 0.2
    
    return min(1.0, total_prob), breakdown


def _calculate_breakout_down_probability(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_events: list["WyckoffEvent"],
    fstats: dict,
) -> tuple[float, dict[str, float]]:
    """计算向下突破概率"""
    breakdown: dict[str, float] = {}
    total_prob = 0.0
    
    has_sow = any(e.type == "SOW" for e in recent_events)
    
    if has_sow:
        breakdown["breakout_signal"] = 0.8
    else:
        breakdown["breakout_signal"] = 0.3
    
    total_prob += breakdown["breakout_signal"] * 0.5
    
    if sm and sm.current_state == "distribution_phase_d":
        breakdown["state_machine"] = 0.8
    else:
        breakdown["state_machine"] = 0.5
    total_prob += breakdown["state_machine"] * 0.3
    
    hist_score = 0.5
    if fstats:
        sow_stats = fstats.get("SOW", {})
        if sow_stats.get("r24_med", 0) < -0.01:
            hist_score = 0.7
    breakdown["history"] = hist_score
    total_prob += hist_score * 0.2
    
    return min(1.0, total_prob), breakdown


def _calculate_range_continuation_probability(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_events: list["WyckoffEvent"],
) -> tuple[float, dict[str, float]]:
    """计算区间延续概率"""
    breakdown: dict[str, float] = {}
    total_prob = 0.0
    
    # 区间持续时间越长，延续概率越高
    if analysis.range.duration_bars:
        duration_score = min(1.0, analysis.range.duration_bars / 100)
        breakdown["duration"] = duration_score
        total_prob += duration_score * 0.4
    
    # 状态机在构筑阶段
    if sm and sm.current_state in ("accumulation_phase_b", "distribution_phase_b"):
        breakdown["state_machine"] = 0.7
    else:
        breakdown["state_machine"] = 0.5
    total_prob += breakdown["state_machine"] * 0.4
    
    # 没有突破信号
    has_breakout = any(e.type in ("SOS", "SOW", "JAC") for e in recent_events)
    breakdown["no_breakout"] = 0.8 if not has_breakout else 0.3
    total_prob += breakdown["no_breakout"] * 0.2
    
    return min(1.0, total_prob), breakdown


def _calculate_trend_continuation_probability(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    recent_events: list["WyckoffEvent"],
    fstats: dict,
) -> tuple[float, dict[str, float]]:
    """计算趋势延续概率"""
    breakdown: dict[str, float] = {}
    total_prob = 0.0
    
    # 市场结构
    if analysis.market_structure == "markup":
        breakdown["structure"] = 0.7
    elif analysis.market_structure == "markdown":
        breakdown["structure"] = 0.7
    else:
        breakdown["structure"] = 0.4
    total_prob += breakdown["structure"] * 0.5
    
    # 状态机
    if sm and sm.current_state in ("markup", "markdown"):
        breakdown["state_machine"] = 0.8
    else:
        breakdown["state_machine"] = 0.5
    total_prob += breakdown["state_machine"] * 0.5
    
    return min(1.0, total_prob), breakdown


def _build_accumulation_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    probability: float,
    breakdown: dict[str, float],
    current_price: float,
    recent_events: list["WyckoffEvent"],
) -> ProbabilisticScenario:
    """构建吸筹剧本"""
    rng = analysis.range
    
    # 计算交易信号
    signal = TradingSignal()
    
    # 入场点位
    if rng.low is not None and rng.high is not None:
        # 如果有LPS，在LPS附近入场
        lps_events = [e for e in recent_events if e.type == "LPS"]
        if lps_events:
            signal.entry_price = lps_events[-1].price
            signal.entry_condition = f"在LPS回踩点 {signal.entry_price:.2f} 附近入场"
        else:
            # 否则在区间中下沿附近
            signal.entry_price = rng.low + (rng.mid - rng.low) * 0.3
            signal.entry_condition = f"在区间中下沿 {signal.entry_price:.2f} 附近入场"
        
        # 止损位
        signal.stop_loss = rng.low * 0.998  # 区间下沿下方0.2%
        
        # 目标位
        signal.targets = [
            rng.mid,  # 第一目标：区间中轴
            rng.high,  # 第二目标：区间上沿
            rng.high * 1.02,  # 第三目标：突破后
        ]
        
        # 风险收益比
        if signal.entry_price and signal.stop_loss:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = signal.targets[0] - signal.entry_price if signal.targets else 0
            if risk > 0:
                signal.risk_reward_ratio = reward / risk
    
    # 仓位建议（基于概率和风险）
    if probability > 0.7:
        signal.position_size_pct = 15.0  # 高概率，中等仓位
    elif probability > 0.5:
        signal.position_size_pct = 10.0  # 中等概率，轻仓
    else:
        signal.position_size_pct = 5.0  # 低概率，极轻仓
    
    # 时间窗口
    signal.time_horizon = "1-3周" if analysis.timeframe in ("1d", "4h") else "3-7天"
    
    # 确认信号
    signal.confirmation_signals = [
        "价格回踩不破区间下沿",
        "出现SPRING/TEST后快速收回",
        "SOS突破区间上沿",
        "成交量在突破时放大",
    ]
    
    # 失效条件
    signal.invalidation_signals = [
        f"跌破区间下沿 {rng.low:.2f}",
        "出现SOW信号",
        "连续3根K线收在区间下沿下方",
    ]
    
    # 关键事件
    key_events = [e.type for e in recent_events if e.type in ("SC", "SPRING", "TEST", "SOS", "LPS")]
    
    # 证据
    evidence = []
    if sm:
        evidence.append(f"状态机: {sm.state_description} (置信度 {sm.state_confidence*100:.0f}%)")
    evidence.append(f"概率分解: 状态机{breakdown.get('state_machine', 0)*100:.0f}% + 序列{breakdown.get('sequence', 0)*100:.0f}%")
    
    # 风险等级
    risk_level = "medium"
    risk_factors = []
    if probability < 0.5:
        risk_level = "high"
        risk_factors.append("概率较低，需谨慎")
    if not signal.stop_loss:
        risk_level = "high"
        risk_factors.append("止损位不明确")
    
    return ProbabilisticScenario(
        name="吸筹完成，准备上涨",
        bias="bullish",
        probability=probability,
        confidence=min(probability, 0.85),
        probability_breakdown=breakdown,
        signal=signal,
        description="市场已完成吸筹阶段，需求开始主导，预期将突破区间上沿进入上涨趋势",
        key_events=key_events,
        evidence=evidence,
        risk_level=risk_level,
        risk_factors=risk_factors,
    )


def _build_distribution_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    probability: float,
    breakdown: dict[str, float],
    current_price: float,
    recent_events: list["WyckoffEvent"],
) -> ProbabilisticScenario:
    """构建派发剧本"""
    rng = analysis.range
    
    signal = TradingSignal()
    
    if rng.low is not None and rng.high is not None:
        # 入场点位
        lpsy_events = [e for e in recent_events if e.type == "LPSY"]
        if lpsy_events:
            signal.entry_price = lpsy_events[-1].price
            signal.entry_condition = f"在LPSY反抽点 {signal.entry_price:.2f} 附近做空"
        else:
            signal.entry_price = rng.high - (rng.high - rng.mid) * 0.3
            signal.entry_condition = f"在区间中上沿 {signal.entry_price:.2f} 附近做空"
        
        # 止损位
        signal.stop_loss = rng.high * 1.002  # 区间上沿上方0.2%
        
        # 目标位
        signal.targets = [
            rng.mid,  # 第一目标：区间中轴
            rng.low,  # 第二目标：区间下沿
            rng.low * 0.98,  # 第三目标：跌破后
        ]
        
        # 风险收益比
        if signal.entry_price and signal.stop_loss:
            risk = abs(signal.stop_loss - signal.entry_price)
            reward = signal.entry_price - signal.targets[0] if signal.targets else 0
            if risk > 0:
                signal.risk_reward_ratio = reward / risk
    
    # 仓位建议
    if probability > 0.7:
        signal.position_size_pct = 15.0
    elif probability > 0.5:
        signal.position_size_pct = 10.0
    else:
        signal.position_size_pct = 5.0
    
    signal.time_horizon = "1-3周" if analysis.timeframe in ("1d", "4h") else "3-7天"
    
    signal.confirmation_signals = [
        "价格反抽不过区间上沿",
        "出现UT/UTAD后快速回落",
        "SOW跌破区间下沿",
        "成交量在突破时放大",
    ]
    
    signal.invalidation_signals = [
        f"突破区间上沿 {rng.high:.2f}",
        "出现SOS信号",
        "连续3根K线收在区间上沿上方",
    ]
    
    key_events = [e.type for e in recent_events if e.type in ("BC", "UT", "UTAD", "SOW", "LPSY")]
    
    evidence = []
    if sm:
        evidence.append(f"状态机: {sm.state_description} (置信度 {sm.state_confidence*100:.0f}%)")
    evidence.append(f"概率分解: 状态机{breakdown.get('state_machine', 0)*100:.0f}% + 序列{breakdown.get('sequence', 0)*100:.0f}%")
    
    risk_level = "medium"
    risk_factors = []
    if probability < 0.5:
        risk_level = "high"
        risk_factors.append("概率较低，需谨慎")
    
    return ProbabilisticScenario(
        name="派发完成，准备下跌",
        bias="bearish",
        probability=probability,
        confidence=min(probability, 0.85),
        probability_breakdown=breakdown,
        signal=signal,
        description="市场已完成派发阶段，供应开始主导，预期将跌破区间下沿进入下跌趋势",
        key_events=key_events,
        evidence=evidence,
        risk_level=risk_level,
        risk_factors=risk_factors,
    )


def _build_breakout_up_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    probability: float,
    breakdown: dict[str, float],
    current_price: float,
    recent_events: list["WyckoffEvent"],
) -> ProbabilisticScenario:
    """构建向上突破剧本"""
    rng = analysis.range
    
    signal = TradingSignal()
    
    if rng.high is not None:
        # 入场：突破后回踩
        signal.entry_price = rng.high * 1.001  # 略高于区间上沿
        signal.entry_condition = f"突破 {rng.high:.2f} 后回踩确认时入场"
        
        # 止损：区间上沿下方
        signal.stop_loss = rng.high * 0.998
        
        # 目标：基于区间高度
        if rng.low is not None:
            range_height = rng.high - rng.low
            signal.targets = [
                rng.high + range_height * 0.5,  # 第一目标：0.5倍区间高度
                rng.high + range_height,  # 第二目标：1倍区间高度
                rng.high + range_height * 1.5,  # 第三目标：1.5倍区间高度
            ]
        
        if signal.entry_price and signal.stop_loss and signal.targets:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = signal.targets[0] - signal.entry_price
            if risk > 0:
                signal.risk_reward_ratio = reward / risk
    
    signal.position_size_pct = 12.0 if probability > 0.6 else 8.0
    signal.time_horizon = "1-2周"
    
    signal.confirmation_signals = [
        "突破后回踩不破区间上沿",
        "成交量放大确认",
        "连续2根K线收在区间上方",
    ]
    
    signal.invalidation_signals = [
        "回踩跌破区间上沿",
        "出现假突破（UT形态）",
    ]
    
    key_events = [e.type for e in recent_events if e.type in ("SOS", "JAC", "LPS")]
    
    return ProbabilisticScenario(
        name="区间向上突破",
        bias="bullish",
        probability=probability,
        confidence=min(probability, 0.8),
        probability_breakdown=breakdown,
        signal=signal,
        description="价格已突破区间上沿，预期继续上涨",
        key_events=key_events,
        evidence=[f"突破信号强度: {breakdown.get('breakout_signal', 0)*100:.0f}%"],
        risk_level="medium",
        risk_factors=[],
    )


def _build_breakout_down_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    probability: float,
    breakdown: dict[str, float],
    current_price: float,
    recent_events: list["WyckoffEvent"],
) -> ProbabilisticScenario:
    """构建向下突破剧本"""
    rng = analysis.range
    
    signal = TradingSignal()
    
    if rng.low is not None:
        signal.entry_price = rng.low * 0.999
        signal.entry_condition = f"跌破 {rng.low:.2f} 后反抽确认时做空"
        
        signal.stop_loss = rng.low * 1.002
        
        if rng.high is not None:
            range_height = rng.high - rng.low
            signal.targets = [
                rng.low - range_height * 0.5,
                rng.low - range_height,
                rng.low - range_height * 1.5,
            ]
        
        if signal.entry_price and signal.stop_loss and signal.targets:
            risk = abs(signal.stop_loss - signal.entry_price)
            reward = signal.entry_price - signal.targets[0]
            if risk > 0:
                signal.risk_reward_ratio = reward / risk
    
    signal.position_size_pct = 12.0 if probability > 0.6 else 8.0
    signal.time_horizon = "1-2周"
    
    signal.confirmation_signals = [
        "跌破后反抽不过区间下沿",
        "成交量放大确认",
        "连续2根K线收在区间下方",
    ]
    
    signal.invalidation_signals = [
        "反抽突破区间下沿",
        "出现假跌破（SPRING形态）",
    ]
    
    key_events = [e.type for e in recent_events if e.type in ("SOW", "LPSY")]
    
    return ProbabilisticScenario(
        name="区间向下突破",
        bias="bearish",
        probability=probability,
        confidence=min(probability, 0.8),
        probability_breakdown=breakdown,
        signal=signal,
        description="价格已跌破区间下沿，预期继续下跌",
        key_events=key_events,
        evidence=[f"突破信号强度: {breakdown.get('breakout_signal', 0)*100:.0f}%"],
        risk_level="medium",
        risk_factors=[],
    )


def _build_range_continuation_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    probability: float,
    breakdown: dict[str, float],
    current_price: float,
) -> ProbabilisticScenario:
    """构建区间延续剧本"""
    rng = analysis.range
    
    signal = TradingSignal()
    
    if rng.low is not None and rng.high is not None:
        # 区间交易策略
        signal.entry_price = None  # 动态入场
        signal.entry_condition = f"在区间 [{rng.low:.2f}, {rng.high:.2f}] 内高抛低吸"
        
        signal.stop_loss = None  # 动态止损
        
        signal.targets = [
            rng.mid,  # 第一目标：中轴
            rng.high if current_price < rng.mid else rng.low,  # 第二目标：对侧边界
        ]
    
    signal.position_size_pct = 8.0  # 区间交易轻仓
    signal.time_horizon = "1-2周"
    
    signal.confirmation_signals = [
        "价格在区间内震荡",
        "成交量萎缩",
        "无明显突破信号",
    ]
    
    signal.invalidation_signals = [
        "突破区间上沿或下沿",
        "出现明确的SOS或SOW信号",
    ]
    
    return ProbabilisticScenario(
        name="区间震荡延续",
        bias="neutral",
        probability=probability,
        confidence=min(probability, 0.7),
        probability_breakdown=breakdown,
        signal=signal,
        description="市场继续在区间内震荡，适合区间交易策略",
        key_events=[],
        evidence=[f"区间持续时间: {rng.duration_bars} 根K线"],
        risk_level="low",
        risk_factors=[],
    )


def _build_trend_continuation_scenario(
    analysis: "WyckoffAnalysis",
    sm: "StateMachineInfo | None",
    probability: float,
    breakdown: dict[str, float],
    current_price: float,
    recent_events: list["WyckoffEvent"],
) -> ProbabilisticScenario:
    """构建趋势延续剧本"""
    signal = TradingSignal()
    
    # 趋势延续策略：回调买入/反弹做空
    signal.entry_price = None
    signal.entry_condition = "在趋势回调/反弹时入场"
    
    signal.position_size_pct = 10.0
    signal.time_horizon = "2-4周"
    
    signal.confirmation_signals = [
        "趋势结构完整",
        "回调/反弹幅度有限",
        "成交量健康",
    ]
    
    signal.invalidation_signals = [
        "出现反向高潮信号（SC/BC）",
        "趋势结构被破坏",
    ]
    
    bias = "bullish" if analysis.market_structure == "markup" else "bearish"
    
    return ProbabilisticScenario(
        name="趋势延续",
        bias=bias,
        probability=probability,
        confidence=min(probability, 0.75),
        probability_breakdown=breakdown,
        signal=signal,
        description="当前趋势将继续，适合趋势跟踪策略",
        key_events=[],
        evidence=[f"市场结构: {analysis.market_structure}"],
        risk_level="medium",
        risk_factors=[],
    )

