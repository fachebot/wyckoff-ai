"""
多时间框架（MTF）威科夫分析模块

核心理念：
- 大级别定方向（1d/4h）：确定主趋势和结构
- 中级别找结构（4h/1h）：识别威科夫阶段和事件
- 小级别找入场（1h/15m）：寻找精确入场点

级别共振：
- 当多个时间框架的结论一致时，信号更可靠
- 大级别优先：小级别信号需要与大级别方向一致
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.features import compute_features
from wyckoff_ai.schemas import WyckoffAnalysis, WyckoffEvent
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff
from wyckoff_ai.wyckoff.sequence import SequenceAnalysis, analyze_sequence


# 时间框架层级定义
TIMEFRAME_HIERARCHY = {
    "1M": {"level": 1, "name": "月线", "weight": 1.5},
    "1w": {"level": 2, "name": "周线", "weight": 1.4},
    "1d": {"level": 3, "name": "日线", "weight": 1.3},
    "4h": {"level": 4, "name": "4小时", "weight": 1.2},
    "1h": {"level": 5, "name": "1小时", "weight": 1.0},
    "30m": {"level": 6, "name": "30分钟", "weight": 0.9},
    "15m": {"level": 7, "name": "15分钟", "weight": 0.8},
    "5m": {"level": 8, "name": "5分钟", "weight": 0.7},
}

# 预设的时间框架组合
MTF_PRESETS = {
    "swing": ["1d", "4h", "1h"],      # 波段交易
    "intraday": ["4h", "1h", "15m"],  # 日内交易
    "scalp": ["1h", "15m", "5m"],     # 短线交易
    "position": ["1w", "1d", "4h"],   # 仓位交易
}


@dataclass
class TimeframeAnalysis:
    """单个时间框架的分析结果"""
    timeframe: str
    timeframe_name: str
    weight: float
    analysis: WyckoffAnalysis
    sequence: SequenceAnalysis
    bias: Literal["bullish", "bearish", "neutral"]
    bias_strength: float  # 0.0 - 1.0
    key_events: list[WyckoffEvent]
    summary: str


@dataclass
class ResonanceResult:
    """级别共振结果"""
    aligned: bool  # 是否所有级别一致
    alignment_score: float  # 0.0 - 1.0
    dominant_bias: Literal["bullish", "bearish", "neutral"]
    conflicts: list[str]
    notes: list[str]


@dataclass
class MTFAnalysisResult:
    """多时间框架分析完整结果"""
    symbol: str
    exchange: str
    timeframes: list[str]
    
    # 各时间框架分析
    tf_analyses: list[TimeframeAnalysis]
    
    # 综合结论
    resonance: ResonanceResult
    overall_bias: Literal["bullish", "bearish", "neutral"]
    overall_confidence: float
    
    # 交易建议
    structure_phase: str
    entry_timeframe: str
    entry_events: list[str]
    stop_reference: float | None
    target_reference: float | None
    
    # 风险评估
    risk_level: Literal["low", "medium", "high"]
    risk_factors: list[str]
    
    # 综合备注
    summary: str
    action_plan: list[str]


def _get_tf_info(timeframe: str) -> dict:
    """获取时间框架信息"""
    return TIMEFRAME_HIERARCHY.get(timeframe, {"level": 99, "name": timeframe, "weight": 1.0})


def _determine_bias(analysis: WyckoffAnalysis, sequence: SequenceAnalysis) -> tuple[str, float]:
    """
    综合分析结果和序列分析，确定偏向和强度
    """
    # 基于市场结构
    structure_bias = 0.0
    if analysis.market_structure == "accumulation":
        structure_bias = 0.6
    elif analysis.market_structure == "distribution":
        structure_bias = -0.6
    elif analysis.market_structure == "markup":
        structure_bias = 0.8
    elif analysis.market_structure == "markdown":
        structure_bias = -0.8
    
    # 基于序列分析
    sequence_bias = 0.0
    if sequence.primary_bias == "bullish":
        sequence_bias = sequence.accumulation_score * 0.8
    elif sequence.primary_bias == "bearish":
        sequence_bias = -sequence.distribution_score * 0.8
    
    # 基于最近事件
    event_bias = 0.0
    recent_events = analysis.events[-5:] if analysis.events else []
    for e in recent_events:
        if e.type in ("SOS", "LPS", "SPRING", "TEST", "JAC", "BUEC"):
            event_bias += e.confidence * 0.2
        elif e.type in ("SOW", "LPSY", "UT", "UTAD"):
            event_bias -= e.confidence * 0.2
        elif e.type == "SC":
            event_bias += e.confidence * 0.15  # SC 后可能反弹
        elif e.type == "BC":
            event_bias -= e.confidence * 0.15  # BC 后可能回调
    
    # 综合计算
    total_bias = structure_bias * 0.4 + sequence_bias * 0.35 + event_bias * 0.25
    
    if total_bias > 0.15:
        return "bullish", min(1.0, abs(total_bias))
    elif total_bias < -0.15:
        return "bearish", min(1.0, abs(total_bias))
    else:
        return "neutral", 0.5


def _analyze_single_timeframe(
    symbol: str,
    exchange: str,
    timeframe: str,
    limit: int = 300,
    strict: bool = False,
) -> TimeframeAnalysis:
    """
    分析单个时间框架
    """
    tf_info = _get_tf_info(timeframe)
    
    # 获取数据
    fr = fetch_ohlcv_binance_spot(symbol=symbol, timeframe=timeframe, limit=limit)
    
    # 计算特征
    features = compute_features(fr.df)
    
    # 威科夫检测
    cfg = DetectionConfig(strict=strict, lookback_bars=min(limit, 220))
    analysis = detect_wyckoff(
        features,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        cfg=cfg,
    )
    
    # 序列分析
    sequence = analyze_sequence(analysis.events)
    
    # 确定偏向
    bias, bias_strength = _determine_bias(analysis, sequence)
    
    # 提取关键事件（最近的高置信度事件）
    key_events = [e for e in analysis.events if e.confidence >= 0.6][-5:]
    
    # 生成摘要
    summary_parts = [f"{tf_info['name']}"]
    if analysis.market_structure != "unknown":
        summary_parts.append(f"结构:{analysis.market_structure}")
    if sequence.current_stage:
        summary_parts.append(f"阶段:{sequence.current_stage}")
    summary_parts.append(f"偏向:{bias}({bias_strength*100:.0f}%)")
    
    return TimeframeAnalysis(
        timeframe=timeframe,
        timeframe_name=tf_info["name"],
        weight=tf_info["weight"],
        analysis=analysis,
        sequence=sequence,
        bias=bias,
        bias_strength=bias_strength,
        key_events=key_events,
        summary=" | ".join(summary_parts),
    )


def _calculate_resonance(tf_analyses: list[TimeframeAnalysis]) -> ResonanceResult:
    """
    计算多时间框架共振
    """
    if not tf_analyses:
        return ResonanceResult(
            aligned=False,
            alignment_score=0.0,
            dominant_bias="neutral",
            conflicts=[],
            notes=["无分析数据"],
        )
    
    # 加权计算偏向
    bullish_weight = 0.0
    bearish_weight = 0.0
    neutral_weight = 0.0
    total_weight = 0.0
    
    biases = []
    for tfa in tf_analyses:
        w = tfa.weight
        total_weight += w
        biases.append((tfa.timeframe_name, tfa.bias))
        
        if tfa.bias == "bullish":
            bullish_weight += w * tfa.bias_strength
        elif tfa.bias == "bearish":
            bearish_weight += w * tfa.bias_strength
        else:
            neutral_weight += w * 0.5
    
    # 确定主导偏向
    if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
        dominant_bias = "bullish"
    elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
        dominant_bias = "bearish"
    else:
        dominant_bias = "neutral"
    
    # 检查冲突
    conflicts = []
    unique_biases = set(b for _, b in biases if b != "neutral")
    if len(unique_biases) > 1:
        for tf_name, bias in biases:
            if bias != dominant_bias and bias != "neutral":
                conflicts.append(f"{tf_name}({bias}) 与主导方向({dominant_bias})冲突")
    
    # 计算一致性得分
    aligned_count = sum(1 for _, b in biases if b == dominant_bias or b == "neutral")
    alignment_score = aligned_count / len(biases) if biases else 0.0
    
    # 生成备注
    notes = []
    if alignment_score >= 0.8:
        notes.append("多级别高度共振，信号可靠性较高")
    elif alignment_score >= 0.6:
        notes.append("多级别基本一致，可考虑顺势操作")
    else:
        notes.append("多级别存在分歧，建议观望或轻仓")
    
    # 大级别优先原则
    if tf_analyses:
        largest_tf = tf_analyses[0]  # 假设已按级别排序
        if largest_tf.bias != dominant_bias and largest_tf.bias != "neutral":
            notes.append(f"注意：大级别({largest_tf.timeframe_name})偏向为{largest_tf.bias}")
    
    return ResonanceResult(
        aligned=len(conflicts) == 0,
        alignment_score=alignment_score,
        dominant_bias=dominant_bias,
        conflicts=conflicts,
        notes=notes,
    )


def _generate_trading_plan(
    tf_analyses: list[TimeframeAnalysis],
    resonance: ResonanceResult,
) -> dict[str, Any]:
    """
    生成交易计划
    """
    if not tf_analyses:
        return {
            "structure_phase": "未知",
            "entry_timeframe": "",
            "entry_events": [],
            "stop_reference": None,
            "target_reference": None,
            "risk_level": "high",
            "risk_factors": ["数据不足"],
            "action_plan": ["等待更多数据"],
        }
    
    # 找到最大级别的分析（定方向）
    largest_tf = tf_analyses[0]
    # 找到最小级别的分析（找入场）
    smallest_tf = tf_analyses[-1]
    
    # 结构阶段
    structure_phase = largest_tf.sequence.current_stage
    
    # 入场时间框架
    entry_timeframe = smallest_tf.timeframe
    
    # 入场事件
    entry_events = []
    if resonance.dominant_bias == "bullish":
        entry_events = ["SPRING", "TEST", "LPS", "SOS", "BUEC"]
    elif resonance.dominant_bias == "bearish":
        entry_events = ["UT", "UTAD", "LPSY", "SOW"]
    
    # 止损/目标参考
    stop_reference = None
    target_reference = None
    
    # 从最大级别获取关键价位
    if largest_tf.analysis.range.low is not None:
        if resonance.dominant_bias == "bullish":
            stop_reference = largest_tf.analysis.range.low
            target_reference = largest_tf.analysis.range.high
        else:
            stop_reference = largest_tf.analysis.range.high
            target_reference = largest_tf.analysis.range.low
    
    # 风险评估
    risk_factors = []
    
    # 检查共振情况
    if not resonance.aligned:
        risk_factors.append("多级别存在方向分歧")
    
    # 检查是否在区间边缘
    for tfa in tf_analyses:
        rng = tfa.analysis.range
        if rng.low is not None and rng.high is not None:
            latest_close = None
            # 尝试获取最新收盘价
            for e in reversed(tfa.analysis.events):
                latest_close = e.price
                break
            if latest_close:
                range_height = rng.high - rng.low
                if latest_close > rng.high - range_height * 0.1:
                    risk_factors.append(f"{tfa.timeframe_name}接近区间上沿")
                elif latest_close < rng.low + range_height * 0.1:
                    risk_factors.append(f"{tfa.timeframe_name}接近区间下沿")
    
    # 检查波动率
    for tfa in tf_analyses:
        if tfa.analysis.risk_notes:
            for note in tfa.analysis.risk_notes:
                if "波动" in note:
                    risk_factors.append(f"{tfa.timeframe_name}波动偏高")
                    break
    
    # 确定风险等级
    if len(risk_factors) >= 3:
        risk_level = "high"
    elif len(risk_factors) >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # 生成行动计划
    action_plan = []
    
    if resonance.dominant_bias == "bullish":
        action_plan.append(f"大方向看多，在{smallest_tf.timeframe_name}寻找回调入场机会")
        action_plan.append(f"关注入场信号：{', '.join(entry_events[:3])}")
        if stop_reference:
            action_plan.append(f"止损参考区间下沿 {stop_reference:.2f}")
    elif resonance.dominant_bias == "bearish":
        action_plan.append(f"大方向看空，在{smallest_tf.timeframe_name}寻找反弹做空机会")
        action_plan.append(f"关注入场信号：{', '.join(entry_events[:3])}")
        if stop_reference:
            action_plan.append(f"止损参考区间上沿 {stop_reference:.2f}")
    else:
        action_plan.append("方向不明确，建议观望")
        action_plan.append("等待多级别形成共振后再行动")
    
    if risk_level == "high":
        action_plan.append("⚠️ 风险较高，建议轻仓或观望")
    
    return {
        "structure_phase": structure_phase,
        "entry_timeframe": entry_timeframe,
        "entry_events": entry_events,
        "stop_reference": stop_reference,
        "target_reference": target_reference,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "action_plan": action_plan,
    }


def analyze_mtf(
    symbol: str,
    exchange: str = "binance",
    timeframes: list[str] | None = None,
    preset: str | None = None,
    limit: int = 300,
    strict: bool = False,
) -> MTFAnalysisResult:
    """
    多时间框架威科夫分析
    
    Args:
        symbol: 交易对，如 "BTC/USDT"
        exchange: 交易所
        timeframes: 时间框架列表，如 ["1d", "4h", "1h"]
        preset: 预设组合，如 "swing", "intraday", "scalp", "position"
        limit: 每个时间框架拉取的K线数量
        strict: 是否使用严格模式
    
    Returns:
        MTFAnalysisResult: 多时间框架分析结果
    """
    # 确定时间框架
    if timeframes is None:
        if preset and preset in MTF_PRESETS:
            timeframes = MTF_PRESETS[preset]
        else:
            timeframes = MTF_PRESETS["swing"]  # 默认波段交易
    
    # 按级别排序（大到小）
    timeframes = sorted(timeframes, key=lambda tf: _get_tf_info(tf)["level"])
    
    # 分析各时间框架
    tf_analyses = []
    for tf in timeframes:
        try:
            tfa = _analyze_single_timeframe(
                symbol=symbol,
                exchange=exchange,
                timeframe=tf,
                limit=limit,
                strict=strict,
            )
            tf_analyses.append(tfa)
        except Exception as e:
            # 单个时间框架失败不应影响整体
            print(f"Warning: Failed to analyze {tf}: {e}")
            continue
    
    # 计算共振
    resonance = _calculate_resonance(tf_analyses)
    
    # 生成交易计划
    trading_plan = _generate_trading_plan(tf_analyses, resonance)
    
    # 计算整体置信度
    if tf_analyses:
        overall_confidence = sum(tfa.bias_strength * tfa.weight for tfa in tf_analyses)
        overall_confidence /= sum(tfa.weight for tfa in tf_analyses)
        overall_confidence *= resonance.alignment_score
    else:
        overall_confidence = 0.0
    
    # 生成总结
    summary_parts = []
    summary_parts.append(f"分析了 {len(tf_analyses)} 个时间框架")
    summary_parts.append(f"主导方向: {resonance.dominant_bias}")
    summary_parts.append(f"共振得分: {resonance.alignment_score*100:.0f}%")
    if trading_plan["structure_phase"]:
        summary_parts.append(f"当前阶段: {trading_plan['structure_phase']}")
    
    return MTFAnalysisResult(
        symbol=symbol,
        exchange=exchange,
        timeframes=timeframes,
        tf_analyses=tf_analyses,
        resonance=resonance,
        overall_bias=resonance.dominant_bias,
        overall_confidence=overall_confidence,
        structure_phase=trading_plan["structure_phase"],
        entry_timeframe=trading_plan["entry_timeframe"],
        entry_events=trading_plan["entry_events"],
        stop_reference=trading_plan["stop_reference"],
        target_reference=trading_plan["target_reference"],
        risk_level=trading_plan["risk_level"],
        risk_factors=trading_plan["risk_factors"],
        summary=" | ".join(summary_parts),
        action_plan=trading_plan["action_plan"],
    )

