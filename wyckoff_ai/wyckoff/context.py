"""
事件上下文验证模块

核心功能：
1. 验证新事件是否符合历史事件的逻辑顺序
2. 检测事件间的时间和价格关系
3. 过滤矛盾和低质量事件
4. 为事件提供上下文增强的置信度

事件逻辑关系：
- SC 后应该出现 AR（1-25根内）
- AR 后应该出现 ST（5-60根内）
- ST 后可能出现 SPRING 或 直接 SOS
- SPRING 后应该出现 SOS（3-30根内）
- SOS 后应该出现 LPS（3-25根内）

- BC 后应该出现 AR（1-25根内）
- AR 后应该出现 ST（5-60根内）
- ST 后可能出现 UT/UTAD
- UT 后可能出现 SOW
- SOW 后应该出现 LPSY
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from wyckoff_ai.schemas import WyckoffEvent


@dataclass
class EventRelationship:
    """事件关系定义"""
    predecessor: str           # 前置事件
    successor: str             # 后续事件
    min_bars: int              # 最小间隔
    max_bars: int              # 最大间隔
    price_relation: str        # 价格关系："higher", "lower", "near", "any"
    confidence_boost: float    # 满足时的置信度加成
    is_required: bool = False  # 是否必须


# 事件关系表
EVENT_RELATIONSHIPS = [
    # 吸筹序列
    EventRelationship("SC", "AR", 1, 25, "higher", 0.15, True),
    EventRelationship("AR", "ST", 5, 60, "lower", 0.12, False),
    EventRelationship("ST", "SPRING", 3, 40, "lower", 0.10, False),
    EventRelationship("ST", "SOS", 5, 60, "higher", 0.10, False),
    EventRelationship("SPRING", "TEST", 3, 15, "near", 0.10, False),
    EventRelationship("SPRING", "SOS", 3, 30, "higher", 0.18, True),
    EventRelationship("TEST", "SOS", 3, 25, "higher", 0.15, False),
    EventRelationship("SOS", "LPS", 3, 25, "near", 0.12, False),
    EventRelationship("LPS", "JAC", 1, 15, "higher", 0.15, False),
    EventRelationship("JAC", "BUEC", 2, 16, "near", 0.12, False),
    
    # 派发序列
    EventRelationship("BC", "AR", 1, 25, "lower", 0.15, True),
    EventRelationship("AR", "ST", 5, 60, "higher", 0.12, False),
    EventRelationship("ST", "UT", 3, 40, "higher", 0.10, False),
    EventRelationship("ST", "UTAD", 5, 50, "higher", 0.12, False),
    EventRelationship("UT", "SOW", 3, 30, "lower", 0.18, False),
    EventRelationship("UTAD", "SOW", 3, 25, "lower", 0.20, True),
    EventRelationship("SOW", "LPSY", 3, 21, "near", 0.12, False),
]

# 矛盾事件对（不应该在短期内同时出现）
CONFLICTING_EVENTS = [
    ("SOS", "SOW", 15),    # SOS 和 SOW 不应该在15根内同时出现
    ("SPRING", "UT", 10),  # SPRING 和 UT 不应该在10根内同时出现
    ("LPS", "LPSY", 10),
    ("JAC", "SOW", 10),
]

# 事件类型的预期前置事件
EXPECTED_PREDECESSORS = {
    "AR": ["SC", "BC"],
    "ST": ["AR", "SC", "BC"],
    "SPRING": ["ST", "AR"],
    "TEST": ["SPRING"],
    "SOS": ["SPRING", "TEST", "ST", "LPS"],
    "LPS": ["SOS"],
    "JAC": ["SOS", "LPS"],
    "BUEC": ["JAC", "SOS"],
    "UT": ["ST", "AR"],
    "UTAD": ["ST", "UT"],
    "SOW": ["UT", "UTAD", "ST"],
    "LPSY": ["SOW"],
}


@dataclass
class ContextValidation:
    """上下文验证结果"""
    is_valid: bool
    confidence_adjustment: float
    supporting_events: list[str]
    conflicting_events: list[str]
    missing_predecessors: list[str]
    time_valid: bool
    price_valid: bool
    notes: list[str] = field(default_factory=list)


def _get_event_bar_index(event: WyckoffEvent, df: pd.DataFrame) -> int | None:
    """获取事件对应的K线索引"""
    try:
        ts = pd.Timestamp(event.ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        
        # 在 DataFrame 中查找
        for idx, row in df.iterrows():
            row_ts = row["timestamp"]
            if hasattr(row_ts, "tz_localize") and row_ts.tzinfo is None:
                row_ts = row_ts.tz_localize("UTC")
            if row_ts == ts:
                return df.index.get_loc(idx)
    except Exception:
        pass
    return None


def validate_event_context(
    new_event: WyckoffEvent,
    existing_events: list[WyckoffEvent],
    df: pd.DataFrame,
    lookback_events: int = 10,
) -> ContextValidation:
    """
    验证新事件是否符合上下文逻辑
    
    Args:
        new_event: 待验证的新事件
        existing_events: 已有的事件列表
        df: K线数据
        lookback_events: 回看的事件数量
    
    Returns:
        ContextValidation: 验证结果
    """
    notes = []
    conf_adj = 0.0
    supporting = []
    conflicting = []
    missing_pred = []
    time_valid = True
    price_valid = True
    
    # 获取新事件的K线索引
    new_idx = _get_event_bar_index(new_event, df)
    
    # 获取最近的事件
    recent_events = existing_events[-lookback_events:] if existing_events else []
    recent_types = [e.type for e in recent_events]
    
    # 1. 检查预期的前置事件
    expected_preds = EXPECTED_PREDECESSORS.get(new_event.type, [])
    if expected_preds:
        has_predecessor = any(t in recent_types for t in expected_preds)
        if has_predecessor:
            matched = [t for t in expected_preds if t in recent_types]
            supporting.extend(matched)
            conf_adj += 0.10
            notes.append(f"前置事件 {matched} 已出现")
        else:
            missing_pred = expected_preds
            conf_adj -= 0.08
            notes.append(f"缺少预期前置事件 {expected_preds}")
    
    # 2. 检查事件关系
    for rel in EVENT_RELATIONSHIPS:
        if rel.successor == new_event.type:
            # 找前置事件
            for prev_event in reversed(recent_events):
                if prev_event.type == rel.predecessor:
                    prev_idx = _get_event_bar_index(prev_event, df)
                    
                    if prev_idx is not None and new_idx is not None:
                        bars_diff = new_idx - prev_idx
                        
                        # 检查时间间隔
                        if rel.min_bars <= bars_diff <= rel.max_bars:
                            supporting.append(f"{rel.predecessor}->{new_event.type}")
                            conf_adj += rel.confidence_boost
                            notes.append(f"{rel.predecessor} 后 {bars_diff} 根出现 {new_event.type}，符合预期")
                        elif bars_diff < rel.min_bars:
                            time_valid = False
                            conf_adj -= 0.05
                            notes.append(f"{new_event.type} 出现过早（距 {rel.predecessor} 仅 {bars_diff} 根）")
                        elif bars_diff > rel.max_bars:
                            notes.append(f"{new_event.type} 可能与 {rel.predecessor} 无关（间隔 {bars_diff} 根）")
                        
                        # 检查价格关系
                        if rel.price_relation == "higher" and new_event.price <= prev_event.price:
                            price_valid = False
                            conf_adj -= 0.05
                            notes.append(f"{new_event.type} 价格应高于 {rel.predecessor}")
                        elif rel.price_relation == "lower" and new_event.price >= prev_event.price:
                            price_valid = False
                            conf_adj -= 0.05
                            notes.append(f"{new_event.type} 价格应低于 {rel.predecessor}")
                    
                    break  # 只检查最近的匹配
    
    # 3. 检查矛盾事件
    for e1, e2, max_dist in CONFLICTING_EVENTS:
        if new_event.type == e1:
            opposite = e2
        elif new_event.type == e2:
            opposite = e1
        else:
            continue
        
        for prev_event in recent_events:
            if prev_event.type == opposite:
                prev_idx = _get_event_bar_index(prev_event, df)
                if prev_idx is not None and new_idx is not None:
                    dist = abs(new_idx - prev_idx)
                    if dist < max_dist:
                        conflicting.append(f"{opposite}(距离{dist}根)")
                        conf_adj -= 0.15
                        notes.append(f"⚠️ {new_event.type} 与 {opposite} 在短期内同时出现，信号矛盾")
    
    # 4. 特殊规则
    # SC/BC 是起点，不需要前置
    if new_event.type in ("SC", "BC", "PSY"):
        if not supporting and not conflicting:
            notes.append(f"{new_event.type} 可作为序列起点")
            conf_adj += 0.05
    
    # SPRING 成功后应该看到价格反弹
    if new_event.type == "SPRING" and new_idx is not None:
        if new_idx + 5 < len(df):
            future = df.iloc[new_idx + 1:new_idx + 6]
            if (future["close"] > new_event.price).any():
                supporting.append("SPRING后有反弹")
                conf_adj += 0.10
                notes.append("SPRING 后价格反弹，确认有效")
    
    # UT 成功后应该看到价格回落
    if new_event.type in ("UT", "UTAD") and new_idx is not None:
        if new_idx + 5 < len(df):
            future = df.iloc[new_idx + 1:new_idx + 6]
            if (future["close"] < new_event.price).any():
                supporting.append("UT后有回落")
                conf_adj += 0.10
                notes.append("UT 后价格回落，确认有效")
    
    # 判断是否有效
    is_valid = (
        len(conflicting) == 0 and
        conf_adj >= -0.15  # 允许一定的负调整
    )
    
    return ContextValidation(
        is_valid=is_valid,
        confidence_adjustment=conf_adj,
        supporting_events=supporting,
        conflicting_events=conflicting,
        missing_predecessors=missing_pred,
        time_valid=time_valid,
        price_valid=price_valid,
        notes=notes,
    )


def filter_conflicting_events(
    events: list[WyckoffEvent],
    df: pd.DataFrame,
) -> list[WyckoffEvent]:
    """
    过滤矛盾的事件，保留高置信度的
    
    Args:
        events: 事件列表
        df: K线数据
    
    Returns:
        过滤后的事件列表
    """
    if len(events) <= 1:
        return events
    
    # 按时间排序
    sorted_events = sorted(events, key=lambda e: e.ts)
    
    filtered = []
    for i, event in enumerate(sorted_events):
        keep = True
        
        # 检查与已保留事件的冲突
        for kept_event in filtered:
            for e1, e2, max_dist in CONFLICTING_EVENTS:
                if (event.type == e1 and kept_event.type == e2) or \
                   (event.type == e2 and kept_event.type == e1):
                    # 计算距离
                    idx1 = _get_event_bar_index(event, df)
                    idx2 = _get_event_bar_index(kept_event, df)
                    
                    if idx1 is not None and idx2 is not None:
                        dist = abs(idx1 - idx2)
                        if dist < max_dist:
                            # 保留置信度高的
                            if event.confidence <= kept_event.confidence:
                                keep = False
                                break
        
        if keep:
            filtered.append(event)
    
    return filtered


def enhance_event_with_context(
    event: WyckoffEvent,
    existing_events: list[WyckoffEvent],
    df: pd.DataFrame,
) -> tuple[WyckoffEvent, ContextValidation]:
    """
    使用上下文增强事件
    
    Args:
        event: 原始事件
        existing_events: 已有事件
        df: K线数据
    
    Returns:
        (增强后的事件, 验证结果)
    """
    validation = validate_event_context(event, existing_events, df)
    
    # 调整置信度
    new_confidence = max(0.0, min(1.0, event.confidence + validation.confidence_adjustment))
    
    # 增强证据
    enhanced_evidence = list(event.evidence)
    
    if validation.supporting_events:
        enhanced_evidence.append(f"上下文支持: {', '.join(validation.supporting_events)}")
    
    if validation.conflicting_events:
        enhanced_evidence.append(f"⚠️ 存在矛盾: {', '.join(validation.conflicting_events)}")
    
    # 创建增强后的事件
    enhanced_event = WyckoffEvent(
        type=event.type,
        ts=event.ts,
        price=event.price,
        confidence=new_confidence,
        evidence=enhanced_evidence,
    )
    
    return enhanced_event, validation


def calculate_sequence_coherence(events: list[WyckoffEvent]) -> float:
    """
    计算事件序列的连贯性得分
    
    Args:
        events: 事件列表
    
    Returns:
        连贯性得分 0.0 - 1.0
    """
    if len(events) < 2:
        return 0.5
    
    score = 0.0
    total_checks = 0
    
    for i in range(1, len(events)):
        current = events[i]
        previous = events[i - 1]
        
        # 检查是否有定义的关系
        for rel in EVENT_RELATIONSHIPS:
            if rel.predecessor == previous.type and rel.successor == current.type:
                score += 1.0
                total_checks += 1
                break
        else:
            # 没有直接关系，检查是否是合理的序列
            expected_preds = EXPECTED_PREDECESSORS.get(current.type, [])
            if previous.type in expected_preds:
                score += 0.7
            total_checks += 1
    
    return score / total_checks if total_checks > 0 else 0.5


def get_next_expected_events(
    events: list[WyckoffEvent],
    bias: Literal["bullish", "bearish", "neutral"],
) -> list[dict]:
    """
    根据当前事件序列预测下一个可能的事件
    
    Args:
        events: 当前事件列表
        bias: 偏向
    
    Returns:
        预期事件列表，包含类型和概率
    """
    if not events:
        if bias == "bullish":
            return [
                {"type": "SC", "probability": 0.4, "description": "等待卖出高潮"},
                {"type": "PSY", "probability": 0.3, "description": "或初步支撑"},
            ]
        elif bias == "bearish":
            return [
                {"type": "BC", "probability": 0.4, "description": "等待买入高潮"},
                {"type": "PSY", "probability": 0.3, "description": "或初步供应"},
            ]
        return []
    
    last_event = events[-1]
    recent_types = {e.type for e in events[-5:]}
    
    predictions = []
    
    # 根据最后事件预测
    for rel in EVENT_RELATIONSHIPS:
        if rel.predecessor == last_event.type:
            # 检查是否已出现
            if rel.successor not in recent_types:
                prob = 0.6 if rel.is_required else 0.4
                predictions.append({
                    "type": rel.successor,
                    "probability": prob,
                    "description": f"在 {rel.predecessor} 后 {rel.min_bars}-{rel.max_bars} 根内",
                })
    
    # 根据偏向调整
    if bias == "bullish":
        for p in predictions:
            if p["type"] in ("SOS", "LPS", "JAC", "SPRING"):
                p["probability"] *= 1.2
            elif p["type"] in ("SOW", "LPSY"):
                p["probability"] *= 0.5
    elif bias == "bearish":
        for p in predictions:
            if p["type"] in ("SOW", "LPSY", "UT", "UTAD"):
                p["probability"] *= 1.2
            elif p["type"] in ("SOS", "LPS"):
                p["probability"] *= 0.5
    
    # 归一化并排序
    total_prob = sum(p["probability"] for p in predictions)
    if total_prob > 0:
        for p in predictions:
            p["probability"] = min(0.9, p["probability"] / total_prob)
    
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    
    return predictions[:5]

