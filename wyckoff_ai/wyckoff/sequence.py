"""
威科夫事件序列验证模块

经典威科夫序列：
- 吸筹（Accumulation）: PS → SC → AR → ST → Spring/Test → SOS → LPS → JAC → BUEC
- 派发（Distribution）: PSY → BC → AR → ST → UT/UTAD → SOW → LPSY

本模块用于：
1. 验证检测到的事件是否符合经典序列
2. 计算序列完整度评分
3. 识别当前所处阶段
4. 预测下一个可能出现的事件
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from wyckoff_ai.schemas import WyckoffEvent


class PhaseType(str, Enum):
    """威科夫阶段类型"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


# 经典威科夫序列定义
# 每个元素是一个阶段，包含该阶段可能出现的事件类型
ACCUMULATION_SEQUENCE = [
    {"phase": "A", "events": ["PSY", "SC"], "description": "停止阶段：初步支撑/卖出高潮"},
    {"phase": "A", "events": ["AR"], "description": "停止阶段：自动反弹"},
    {"phase": "B", "events": ["ST"], "description": "构筑阶段：二次测试"},
    {"phase": "C", "events": ["SPRING", "TEST"], "description": "测试阶段：弹簧/测试"},
    {"phase": "D", "events": ["SOS", "LPS"], "description": "突破阶段：强势信号/最后支撑点"},
    {"phase": "E", "events": ["JAC", "BUEC"], "description": "离开阶段：跳跃/回踩确认"},
]

DISTRIBUTION_SEQUENCE = [
    {"phase": "A", "events": ["PSY", "BC"], "description": "停止阶段：初步供应/买入高潮"},
    {"phase": "A", "events": ["AR"], "description": "停止阶段：自动反应"},
    {"phase": "B", "events": ["ST"], "description": "构筑阶段：二次测试"},
    {"phase": "C", "events": ["UT", "UTAD"], "description": "测试阶段：上冲回落"},
    {"phase": "D", "events": ["SOW", "LPSY"], "description": "突破阶段：弱势信号/最后供应点"},
]

# 事件类型分类
BULLISH_EVENTS = {"SC", "AR", "ST", "SPRING", "TEST", "SOS", "LPS", "JAC", "BUEC", "PSY"}
BEARISH_EVENTS = {"BC", "AR", "ST", "UT", "UTAD", "SOW", "LPSY", "PSY"}
CLIMAX_EVENTS = {"SC", "BC"}
REACTION_EVENTS = {"AR"}
TEST_EVENTS = {"ST", "TEST", "SPRING", "UT", "UTAD"}
STRENGTH_EVENTS = {"SOS", "SOW", "LPS", "LPSY", "JAC", "BUEC"}


@dataclass
class SequenceMatch:
    """序列匹配结果"""
    phase_type: PhaseType
    current_phase: str  # A, B, C, D, E
    phase_description: str
    completeness: float  # 0.0 - 1.0
    matched_events: list[str]
    missing_events: list[str]
    next_expected: list[str]
    confidence: float


@dataclass
class SequenceAnalysis:
    """序列分析完整结果"""
    primary_bias: Literal["bullish", "bearish", "neutral"]
    accumulation_score: float  # 0.0 - 1.0
    distribution_score: float  # 0.0 - 1.0
    accumulation_match: SequenceMatch | None
    distribution_match: SequenceMatch | None
    current_stage: str
    stage_description: str
    next_expected_events: list[str]
    sequence_notes: list[str] = field(default_factory=list)


def _match_sequence(
    events: list[WyckoffEvent],
    sequence_def: list[dict],
    phase_type: PhaseType,
) -> SequenceMatch:
    """
    将事件列表与序列定义进行匹配
    """
    event_types = [e.type for e in events]
    event_set = set(event_types)
    
    matched = []
    missing = []
    current_phase = "A"
    phase_desc = ""
    max_phase_idx = -1
    
    for idx, step in enumerate(sequence_def):
        step_events = step["events"]
        step_matched = [e for e in step_events if e in event_set]
        step_missing = [e for e in step_events if e not in event_set]
        
        if step_matched:
            matched.extend(step_matched)
            if idx > max_phase_idx:
                max_phase_idx = idx
                current_phase = step["phase"]
                phase_desc = step["description"]
        else:
            missing.extend(step_missing)
    
    # 计算完整度
    total_steps = len(sequence_def)
    matched_steps = max_phase_idx + 1 if max_phase_idx >= 0 else 0
    completeness = matched_steps / total_steps if total_steps > 0 else 0.0
    
    # 预测下一个事件
    next_expected = []
    if max_phase_idx + 1 < len(sequence_def):
        next_step = sequence_def[max_phase_idx + 1]
        next_expected = next_step["events"]
    
    # 计算置信度（基于匹配事件数量和顺序）
    confidence = min(1.0, len(set(matched)) / max(1, len(sequence_def))) * 0.7
    # 如果事件顺序正确，加分
    if _check_event_order(event_types, sequence_def):
        confidence += 0.3
    
    return SequenceMatch(
        phase_type=phase_type,
        current_phase=current_phase,
        phase_description=phase_desc,
        completeness=completeness,
        matched_events=list(set(matched)),
        missing_events=list(set(missing)),
        next_expected=next_expected,
        confidence=min(1.0, confidence),
    )


def _check_event_order(event_types: list[str], sequence_def: list[dict]) -> bool:
    """
    检查事件顺序是否符合序列定义
    """
    # 构建序列中每个事件的阶段索引
    event_phase_map: dict[str, int] = {}
    for idx, step in enumerate(sequence_def):
        for e in step["events"]:
            if e not in event_phase_map:
                event_phase_map[e] = idx
    
    # 检查事件出现顺序
    last_phase = -1
    violations = 0
    for et in event_types:
        if et in event_phase_map:
            phase = event_phase_map[et]
            if phase < last_phase:
                violations += 1
            last_phase = max(last_phase, phase)
    
    return violations == 0


def _calculate_bias_score(events: list[WyckoffEvent]) -> tuple[float, float]:
    """
    计算多空偏向得分
    返回 (bullish_score, bearish_score)
    """
    bullish_score = 0.0
    bearish_score = 0.0
    
    for e in events:
        conf = e.confidence
        et = e.type
        
        # 高潮事件
        if et == "SC":
            bullish_score += conf * 1.5  # SC 后看涨
        elif et == "BC":
            bearish_score += conf * 1.5  # BC 后看跌
        
        # 测试事件
        elif et in ("SPRING", "TEST"):
            bullish_score += conf * 1.2
        elif et in ("UT", "UTAD"):
            bearish_score += conf * 1.2
        
        # 强势/弱势信号
        elif et == "SOS":
            bullish_score += conf * 1.5
        elif et == "SOW":
            bearish_score += conf * 1.5
        elif et == "LPS":
            bullish_score += conf * 1.0
        elif et == "LPSY":
            bearish_score += conf * 1.0
        
        # 突破事件
        elif et in ("JAC", "BUEC"):
            bullish_score += conf * 1.3
        
        # ST 和 AR 需要根据上下文判断
        elif et == "ST":
            # ST 是中性的，取决于之前的高潮类型
            pass
        elif et == "AR":
            pass
    
    # 归一化
    total = bullish_score + bearish_score
    if total > 0:
        bullish_score /= total
        bearish_score /= total
    else:
        bullish_score = 0.5
        bearish_score = 0.5
    
    return bullish_score, bearish_score


def analyze_sequence(events: list[WyckoffEvent]) -> SequenceAnalysis:
    """
    分析事件序列，判断当前处于威科夫哪个阶段
    
    Args:
        events: 按时间排序的威科夫事件列表
    
    Returns:
        SequenceAnalysis: 序列分析结果
    """
    if not events:
        return SequenceAnalysis(
            primary_bias="neutral",
            accumulation_score=0.0,
            distribution_score=0.0,
            accumulation_match=None,
            distribution_match=None,
            current_stage="未知",
            stage_description="未检测到有效事件",
            next_expected_events=[],
            sequence_notes=["暂无足够事件进行序列分析"],
        )
    
    # 匹配两种序列
    acc_match = _match_sequence(events, ACCUMULATION_SEQUENCE, PhaseType.ACCUMULATION)
    dist_match = _match_sequence(events, DISTRIBUTION_SEQUENCE, PhaseType.DISTRIBUTION)
    
    # 计算多空得分
    bullish_score, bearish_score = _calculate_bias_score(events)
    
    # 综合判断
    acc_score = acc_match.completeness * acc_match.confidence * bullish_score
    dist_score = dist_match.completeness * dist_match.confidence * bearish_score
    
    # 确定主要偏向
    if acc_score > dist_score * 1.2:
        primary_bias = "bullish"
        primary_match = acc_match
    elif dist_score > acc_score * 1.2:
        primary_bias = "bearish"
        primary_match = dist_match
    else:
        primary_bias = "neutral"
        primary_match = acc_match if acc_score >= dist_score else dist_match
    
    # 生成阶段描述
    current_stage = f"{primary_match.phase_type.value} - Phase {primary_match.current_phase}"
    stage_description = primary_match.phase_description
    
    # 生成注释
    notes = []
    
    # 检查是否有高潮事件
    event_types = {e.type for e in events}
    if "SC" in event_types:
        notes.append("检测到卖出高潮(SC)，可能是吸筹起点")
    if "BC" in event_types:
        notes.append("检测到买入高潮(BC)，可能是派发起点")
    
    # 检查序列完整度
    if primary_match.completeness >= 0.8:
        notes.append(f"序列完整度高({primary_match.completeness*100:.0f}%)，结构清晰")
    elif primary_match.completeness >= 0.5:
        notes.append(f"序列完整度中等({primary_match.completeness*100:.0f}%)，需继续观察")
    else:
        notes.append(f"序列完整度较低({primary_match.completeness*100:.0f}%)，结构尚不明确")
    
    # 检查关键事件
    if "SPRING" in event_types or "TEST" in event_types:
        notes.append("已出现 Spring/Test，关注后续 SOS 确认")
    if "SOS" in event_types and "LPS" in event_types:
        notes.append("SOS + LPS 组合出现，吸筹结构较为完整")
    if "UTAD" in event_types or ("UT" in event_types and "SOW" in event_types):
        notes.append("UTAD/UT + SOW 组合出现，派发结构较为完整")
    
    return SequenceAnalysis(
        primary_bias=primary_bias,
        accumulation_score=acc_score,
        distribution_score=dist_score,
        accumulation_match=acc_match,
        distribution_match=dist_match,
        current_stage=current_stage,
        stage_description=stage_description,
        next_expected_events=primary_match.next_expected,
        sequence_notes=notes,
    )


def get_phase_description(phase_type: PhaseType, phase: str) -> str:
    """获取阶段的详细描述"""
    descriptions = {
        (PhaseType.ACCUMULATION, "A"): "Phase A (停止下跌): 供应耗尽，出现SC/AR，价格停止下跌",
        (PhaseType.ACCUMULATION, "B"): "Phase B (构筑区间): ST确认底部，在AR-SC区间内震荡",
        (PhaseType.ACCUMULATION, "C"): "Phase C (测试): Spring/Test测试供应，准备突破",
        (PhaseType.ACCUMULATION, "D"): "Phase D (突破): SOS确认需求控制，LPS回踩确认",
        (PhaseType.ACCUMULATION, "E"): "Phase E (离开): JAC突破区间，BUEC回踩确认，进入上涨",
        (PhaseType.DISTRIBUTION, "A"): "Phase A (停止上涨): 需求耗尽，出现BC/AR，价格停止上涨",
        (PhaseType.DISTRIBUTION, "B"): "Phase B (构筑区间): ST确认顶部，在AR-BC区间内震荡",
        (PhaseType.DISTRIBUTION, "C"): "Phase C (测试): UT/UTAD测试需求，准备下跌",
        (PhaseType.DISTRIBUTION, "D"): "Phase D (突破): SOW确认供应控制，LPSY回抽确认",
    }
    return descriptions.get((phase_type, phase), "未知阶段")

