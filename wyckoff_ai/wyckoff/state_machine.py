"""
威科夫状态机模型

实现基于有限状态机（FSM）的威科夫阶段识别，
通过事件序列驱动状态转换，提供更准确的阶段判断。

状态机结构：
    UNKNOWN -> ACC_A/DIST_A (高潮事件触发)
    ACC_A -> ACC_B (ST事件)
    ACC_B -> ACC_C (SPRING/TEST事件)
    ACC_C -> ACC_D (SOS事件)
    ACC_D -> ACC_E (JAC/BUEC事件)
    ACC_E -> MARKUP (离开区间)
    
    DIST_A -> DIST_B (ST事件)
    DIST_B -> DIST_C (UT/UTAD事件)
    DIST_C -> DIST_D (SOW事件)
    DIST_D -> MARKDOWN (离开区间)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import pandas as pd

from wyckoff_ai.schemas import WyckoffEvent


class WyckoffState(str, Enum):
    """威科夫市场状态"""
    # 未知/初始状态
    UNKNOWN = "unknown"
    
    # 吸筹阶段
    ACC_PHASE_A = "accumulation_phase_a"   # 停止下跌：PSY, SC, AR
    ACC_PHASE_B = "accumulation_phase_b"   # 构筑区间：ST, 横盘
    ACC_PHASE_C = "accumulation_phase_c"   # 测试：SPRING, TEST
    ACC_PHASE_D = "accumulation_phase_d"   # 突破：SOS, LPS
    ACC_PHASE_E = "accumulation_phase_e"   # 离开：JAC, BUEC
    
    # 派发阶段
    DIST_PHASE_A = "distribution_phase_a"  # 停止上涨：PSY, BC, AR
    DIST_PHASE_B = "distribution_phase_b"  # 构筑区间：ST, 横盘
    DIST_PHASE_C = "distribution_phase_c"  # 测试：UT, UTAD
    DIST_PHASE_D = "distribution_phase_d"  # 突破：SOW, LPSY
    
    # 趋势状态
    MARKUP = "markup"       # 上涨趋势
    MARKDOWN = "markdown"   # 下跌趋势
    RANGE = "range"         # 横盘整理（方向未明）


# 状态的中文描述
STATE_DESCRIPTIONS = {
    WyckoffState.UNKNOWN: "未知状态",
    WyckoffState.ACC_PHASE_A: "吸筹 Phase A - 停止下跌",
    WyckoffState.ACC_PHASE_B: "吸筹 Phase B - 构筑区间",
    WyckoffState.ACC_PHASE_C: "吸筹 Phase C - 测试供应",
    WyckoffState.ACC_PHASE_D: "吸筹 Phase D - 确认需求",
    WyckoffState.ACC_PHASE_E: "吸筹 Phase E - 离开区间",
    WyckoffState.DIST_PHASE_A: "派发 Phase A - 停止上涨",
    WyckoffState.DIST_PHASE_B: "派发 Phase B - 构筑区间",
    WyckoffState.DIST_PHASE_C: "派发 Phase C - 测试需求",
    WyckoffState.DIST_PHASE_D: "派发 Phase D - 确认供应",
    WyckoffState.MARKUP: "上涨趋势",
    WyckoffState.MARKDOWN: "下跌趋势",
    WyckoffState.RANGE: "横盘整理",
}

# 状态的详细说明
STATE_DETAILS = {
    WyckoffState.ACC_PHASE_A: {
        "description": "供应耗尽阶段，出现恐慌性抛售后市场开始企稳",
        "key_events": ["PSY", "SC", "AR"],
        "characteristics": ["放量下跌后反弹", "卖出高潮伴随巨量", "自动反弹确认买盘入场"],
        "next_phase": WyckoffState.ACC_PHASE_B,
    },
    WyckoffState.ACC_PHASE_B: {
        "description": "区间构筑阶段，供需双方在区间内博弈",
        "key_events": ["ST"],
        "characteristics": ["价格在SC-AR区间内震荡", "二次测试确认底部", "成交量逐渐萎缩"],
        "next_phase": WyckoffState.ACC_PHASE_C,
    },
    WyckoffState.ACC_PHASE_C: {
        "description": "最终测试阶段，主力最后一次测试供应",
        "key_events": ["SPRING", "TEST"],
        "characteristics": ["假跌破下沿", "快速收回", "放量确认需求"],
        "next_phase": WyckoffState.ACC_PHASE_D,
    },
    WyckoffState.ACC_PHASE_D: {
        "description": "需求确认阶段，价格开始突破区间",
        "key_events": ["SOS", "LPS"],
        "characteristics": ["强势信号突破区间上沿", "回踩不破支撑", "需求主导"],
        "next_phase": WyckoffState.ACC_PHASE_E,
    },
    WyckoffState.ACC_PHASE_E: {
        "description": "离开区间阶段，正式进入上涨趋势",
        "key_events": ["JAC", "BUEC"],
        "characteristics": ["突破后回踩确认", "回踩后继续上涨", "成交量健康"],
        "next_phase": WyckoffState.MARKUP,
    },
    WyckoffState.DIST_PHASE_A: {
        "description": "需求耗尽阶段，出现买入高潮后市场开始见顶",
        "key_events": ["PSY", "BC", "AR"],
        "characteristics": ["放量上涨后回落", "买入高潮伴随巨量", "自动反应确认卖盘入场"],
        "next_phase": WyckoffState.DIST_PHASE_B,
    },
    WyckoffState.DIST_PHASE_B: {
        "description": "区间构筑阶段，供需双方在区间内博弈",
        "key_events": ["ST"],
        "characteristics": ["价格在BC-AR区间内震荡", "二次测试确认顶部", "成交量逐渐萎缩"],
        "next_phase": WyckoffState.DIST_PHASE_C,
    },
    WyckoffState.DIST_PHASE_C: {
        "description": "最终测试阶段，主力最后一次测试需求",
        "key_events": ["UT", "UTAD"],
        "characteristics": ["假突破上沿", "快速回落", "放量确认供应"],
        "next_phase": WyckoffState.DIST_PHASE_D,
    },
    WyckoffState.DIST_PHASE_D: {
        "description": "供应确认阶段，价格开始跌破区间",
        "key_events": ["SOW", "LPSY"],
        "characteristics": ["弱势信号跌破区间下沿", "反抽不破阻力", "供应主导"],
        "next_phase": WyckoffState.MARKDOWN,
    },
}


@dataclass
class StateTransition:
    """状态转换记录"""
    from_state: WyckoffState
    to_state: WyckoffState
    trigger_event: WyckoffEvent
    timestamp: str
    confidence: float
    reason: str


@dataclass
class PhaseProgress:
    """阶段进度"""
    current_state: WyckoffState
    state_description: str
    progress: float  # 0.0 - 1.0
    events_in_phase: list[str]
    missing_events: list[str]
    next_expected: list[str]
    time_in_phase_bars: int
    notes: list[str]


@dataclass 
class StateMachineResult:
    """状态机分析结果"""
    current_state: WyckoffState
    state_description: str
    state_confidence: float
    
    # 阶段进度
    phase_progress: PhaseProgress
    
    # 偏向判断
    bias: Literal["bullish", "bearish", "neutral"]
    bias_confidence: float
    
    # 转换历史
    transition_history: list[StateTransition]
    
    # 预测
    next_probable_states: list[tuple[WyckoffState, float]]  # (state, probability)
    predicted_events: list[dict]
    
    # 风险评估
    risk_level: Literal["low", "medium", "high", "extreme"]
    risk_factors: list[str]
    
    # 交易建议
    action_suggestion: str
    key_levels: dict[str, float]


class WyckoffStateMachine:
    """
    威科夫状态机
    
    通过事件序列驱动状态转换，实现准确的阶段识别。
    """
    
    def __init__(self):
        self.current_state: WyckoffState = WyckoffState.UNKNOWN
        self.state_confidence: float = 0.0
        self.transition_history: list[StateTransition] = []
        self.events_in_current_phase: list[WyckoffEvent] = []
        self.phase_start_idx: int = 0
        self.last_climax_type: str | None = None  # "SC" or "BC"
        
        # 状态持续时间（K线数）
        self.bars_in_state: int = 0
        
        # 关键价位
        self.range_high: float | None = None
        self.range_low: float | None = None
        self.climax_price: float | None = None
        
        # 定义状态转换规则
        self._init_transition_rules()
    
    def _init_transition_rules(self):
        """初始化状态转换规则"""
        # 转换规则格式: (当前状态, 触发事件) -> (目标状态, 基础置信度)
        self.transition_rules: dict[tuple[WyckoffState, str], tuple[WyckoffState, float]] = {
            # 从 UNKNOWN 开始
            (WyckoffState.UNKNOWN, "SC"): (WyckoffState.ACC_PHASE_A, 0.7),
            (WyckoffState.UNKNOWN, "BC"): (WyckoffState.DIST_PHASE_A, 0.7),
            (WyckoffState.UNKNOWN, "PSY"): (WyckoffState.RANGE, 0.4),
            
            # 从横盘状态转换
            (WyckoffState.RANGE, "SC"): (WyckoffState.ACC_PHASE_A, 0.75),
            (WyckoffState.RANGE, "BC"): (WyckoffState.DIST_PHASE_A, 0.75),
            (WyckoffState.RANGE, "SPRING"): (WyckoffState.ACC_PHASE_C, 0.65),
            (WyckoffState.RANGE, "UT"): (WyckoffState.DIST_PHASE_C, 0.65),
            (WyckoffState.RANGE, "UTAD"): (WyckoffState.DIST_PHASE_C, 0.70),
            
            # 吸筹阶段转换
            (WyckoffState.ACC_PHASE_A, "AR"): (WyckoffState.ACC_PHASE_A, 0.8),  # 停留，确认
            (WyckoffState.ACC_PHASE_A, "ST"): (WyckoffState.ACC_PHASE_B, 0.75),
            (WyckoffState.ACC_PHASE_B, "ST"): (WyckoffState.ACC_PHASE_B, 0.7),  # 可多次ST
            (WyckoffState.ACC_PHASE_B, "SPRING"): (WyckoffState.ACC_PHASE_C, 0.8),
            (WyckoffState.ACC_PHASE_B, "TEST"): (WyckoffState.ACC_PHASE_C, 0.75),
            (WyckoffState.ACC_PHASE_C, "TEST"): (WyckoffState.ACC_PHASE_C, 0.7),  # 可多次TEST
            (WyckoffState.ACC_PHASE_C, "SOS"): (WyckoffState.ACC_PHASE_D, 0.85),
            (WyckoffState.ACC_PHASE_C, "LPS"): (WyckoffState.ACC_PHASE_D, 0.7),
            (WyckoffState.ACC_PHASE_D, "LPS"): (WyckoffState.ACC_PHASE_D, 0.75),  # 可多次LPS
            (WyckoffState.ACC_PHASE_D, "SOS"): (WyckoffState.ACC_PHASE_D, 0.8),   # 再次SOS
            (WyckoffState.ACC_PHASE_D, "JAC"): (WyckoffState.ACC_PHASE_E, 0.85),
            (WyckoffState.ACC_PHASE_D, "BUEC"): (WyckoffState.ACC_PHASE_E, 0.8),
            (WyckoffState.ACC_PHASE_E, "BUEC"): (WyckoffState.ACC_PHASE_E, 0.75),
            
            # 吸筹失败/回退
            (WyckoffState.ACC_PHASE_B, "SC"): (WyckoffState.ACC_PHASE_A, 0.6),  # 回测
            (WyckoffState.ACC_PHASE_C, "SC"): (WyckoffState.ACC_PHASE_A, 0.5),  # 失败
            (WyckoffState.ACC_PHASE_D, "SOW"): (WyckoffState.DIST_PHASE_D, 0.6),  # 反转信号
            
            # 派发阶段转换
            (WyckoffState.DIST_PHASE_A, "AR"): (WyckoffState.DIST_PHASE_A, 0.8),  # 停留，确认
            (WyckoffState.DIST_PHASE_A, "ST"): (WyckoffState.DIST_PHASE_B, 0.75),
            (WyckoffState.DIST_PHASE_B, "ST"): (WyckoffState.DIST_PHASE_B, 0.7),
            (WyckoffState.DIST_PHASE_B, "UT"): (WyckoffState.DIST_PHASE_C, 0.8),
            (WyckoffState.DIST_PHASE_B, "UTAD"): (WyckoffState.DIST_PHASE_C, 0.85),
            (WyckoffState.DIST_PHASE_C, "UT"): (WyckoffState.DIST_PHASE_C, 0.7),
            (WyckoffState.DIST_PHASE_C, "SOW"): (WyckoffState.DIST_PHASE_D, 0.85),
            (WyckoffState.DIST_PHASE_C, "LPSY"): (WyckoffState.DIST_PHASE_D, 0.7),
            (WyckoffState.DIST_PHASE_D, "LPSY"): (WyckoffState.DIST_PHASE_D, 0.75),
            (WyckoffState.DIST_PHASE_D, "SOW"): (WyckoffState.DIST_PHASE_D, 0.8),
            
            # 派发失败/回退
            (WyckoffState.DIST_PHASE_B, "BC"): (WyckoffState.DIST_PHASE_A, 0.6),
            (WyckoffState.DIST_PHASE_C, "BC"): (WyckoffState.DIST_PHASE_A, 0.5),
            (WyckoffState.DIST_PHASE_D, "SOS"): (WyckoffState.ACC_PHASE_D, 0.6),  # 反转信号
            
            # 趋势状态
            (WyckoffState.MARKUP, "BC"): (WyckoffState.DIST_PHASE_A, 0.7),  # 见顶
            (WyckoffState.MARKDOWN, "SC"): (WyckoffState.ACC_PHASE_A, 0.7),  # 见底
        }
        
        # 每个状态期望的事件
        self.expected_events: dict[WyckoffState, list[str]] = {
            WyckoffState.ACC_PHASE_A: ["SC", "AR", "PSY"],
            WyckoffState.ACC_PHASE_B: ["ST"],
            WyckoffState.ACC_PHASE_C: ["SPRING", "TEST"],
            WyckoffState.ACC_PHASE_D: ["SOS", "LPS"],
            WyckoffState.ACC_PHASE_E: ["JAC", "BUEC"],
            WyckoffState.DIST_PHASE_A: ["BC", "AR", "PSY"],
            WyckoffState.DIST_PHASE_B: ["ST"],
            WyckoffState.DIST_PHASE_C: ["UT", "UTAD"],
            WyckoffState.DIST_PHASE_D: ["SOW", "LPSY"],
        }
    
    def reset(self):
        """重置状态机"""
        self.current_state = WyckoffState.UNKNOWN
        self.state_confidence = 0.0
        self.transition_history = []
        self.events_in_current_phase = []
        self.phase_start_idx = 0
        self.last_climax_type = None
        self.bars_in_state = 0
        self.range_high = None
        self.range_low = None
        self.climax_price = None
    
    def process_event(
        self, 
        event: WyckoffEvent, 
        current_idx: int,
        df: pd.DataFrame | None = None,
    ) -> StateTransition | None:
        """
        处理新事件，可能触发状态转换
        
        Args:
            event: 威科夫事件
            current_idx: 当前K线索引
            df: 价格数据（可选，用于上下文判断）
        
        Returns:
            如果发生转换，返回 StateTransition；否则返回 None
        """
        event_type = event.type
        
        # 查找转换规则
        rule_key = (self.current_state, event_type)
        
        if rule_key in self.transition_rules:
            new_state, base_conf = self.transition_rules[rule_key]
            
            # 计算调整后的置信度
            adjusted_conf = self._adjust_confidence(
                base_conf, event, new_state, df, current_idx
            )
            
            # 只有置信度足够高才转换
            if adjusted_conf >= 0.5:
                transition = self._do_transition(
                    new_state, event, adjusted_conf, current_idx
                )
                return transition
        
        # 即使不转换，也记录事件
        self.events_in_current_phase.append(event)
        self.bars_in_state = current_idx - self.phase_start_idx
        
        # 更新关键价位
        self._update_key_levels(event)
        
        return None
    
    def _adjust_confidence(
        self,
        base_conf: float,
        event: WyckoffEvent,
        target_state: WyckoffState,
        df: pd.DataFrame | None,
        current_idx: int,
    ) -> float:
        """根据上下文调整置信度"""
        conf = base_conf
        
        # 事件本身的置信度
        conf *= event.confidence
        
        # 时间因素：在当前阶段停留时间越长，转换置信度越高
        if self.bars_in_state > 10:
            conf += 0.05
        if self.bars_in_state > 20:
            conf += 0.05
        
        # 事件序列验证
        recent_types = [e.type for e in self.events_in_current_phase[-5:]]
        
        # 吸筹序列验证
        if target_state == WyckoffState.ACC_PHASE_B:
            if "SC" in recent_types or "AR" in recent_types:
                conf += 0.1
        elif target_state == WyckoffState.ACC_PHASE_C:
            if "ST" in recent_types:
                conf += 0.1
        elif target_state == WyckoffState.ACC_PHASE_D:
            if "SPRING" in recent_types or "TEST" in recent_types:
                conf += 0.15
        
        # 派发序列验证
        elif target_state == WyckoffState.DIST_PHASE_B:
            if "BC" in recent_types or "AR" in recent_types:
                conf += 0.1
        elif target_state == WyckoffState.DIST_PHASE_C:
            if "ST" in recent_types:
                conf += 0.1
        elif target_state == WyckoffState.DIST_PHASE_D:
            if "UT" in recent_types or "UTAD" in recent_types:
                conf += 0.15
        
        # 价格位置验证
        if df is not None and current_idx < len(df):
            conf = self._validate_price_context(conf, event, target_state, df, current_idx)
        
        return min(1.0, conf)
    
    def _validate_price_context(
        self,
        conf: float,
        event: WyckoffEvent,
        target_state: WyckoffState,
        df: pd.DataFrame,
        current_idx: int,
    ) -> float:
        """验证价格上下文"""
        row = df.iloc[current_idx]
        close = float(row["close"])
        
        # 如果有区间边界，验证事件位置
        if self.range_high is not None and self.range_low is not None:
            range_mid = (self.range_high + self.range_low) / 2
            range_width = self.range_high - self.range_low
            
            # SPRING 应该在区间下半部分
            if event.type == "SPRING":
                if close < range_mid:
                    conf += 0.1
                else:
                    conf -= 0.1
            
            # UT/UTAD 应该在区间上半部分
            elif event.type in ("UT", "UTAD"):
                if close > range_mid:
                    conf += 0.1
                else:
                    conf -= 0.1
            
            # SOS 应该突破区间上沿
            elif event.type == "SOS":
                if close > self.range_high:
                    conf += 0.15
                elif close > range_mid:
                    conf += 0.05
            
            # SOW 应该跌破区间下沿
            elif event.type == "SOW":
                if close < self.range_low:
                    conf += 0.15
                elif close < range_mid:
                    conf += 0.05
        
        return conf
    
    def _do_transition(
        self,
        new_state: WyckoffState,
        trigger_event: WyckoffEvent,
        confidence: float,
        current_idx: int,
    ) -> StateTransition:
        """执行状态转换"""
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            trigger_event=trigger_event,
            timestamp=trigger_event.ts,
            confidence=confidence,
            reason=self._get_transition_reason(new_state, trigger_event),
        )
        
        # 更新状态
        old_state = self.current_state
        self.current_state = new_state
        self.state_confidence = confidence
        self.transition_history.append(transition)
        
        # 重置阶段相关变量
        self.events_in_current_phase = [trigger_event]
        self.phase_start_idx = current_idx
        self.bars_in_state = 0
        
        # 记录高潮类型
        if trigger_event.type == "SC":
            self.last_climax_type = "SC"
            self.climax_price = trigger_event.price
        elif trigger_event.type == "BC":
            self.last_climax_type = "BC"
            self.climax_price = trigger_event.price
        
        return transition
    
    def _get_transition_reason(self, new_state: WyckoffState, event: WyckoffEvent) -> str:
        """生成转换原因描述"""
        reasons = {
            WyckoffState.ACC_PHASE_A: f"检测到 {event.type}，可能开始吸筹",
            WyckoffState.ACC_PHASE_B: f"{event.type} 确认底部，进入构筑阶段",
            WyckoffState.ACC_PHASE_C: f"{event.type} 测试供应，准备突破",
            WyckoffState.ACC_PHASE_D: f"{event.type} 确认需求主导，进入突破阶段",
            WyckoffState.ACC_PHASE_E: f"{event.type} 离开区间，进入上涨",
            WyckoffState.DIST_PHASE_A: f"检测到 {event.type}，可能开始派发",
            WyckoffState.DIST_PHASE_B: f"{event.type} 确认顶部，进入构筑阶段",
            WyckoffState.DIST_PHASE_C: f"{event.type} 测试需求，准备下跌",
            WyckoffState.DIST_PHASE_D: f"{event.type} 确认供应主导，进入下跌阶段",
            WyckoffState.MARKUP: "进入上涨趋势",
            WyckoffState.MARKDOWN: "进入下跌趋势",
            WyckoffState.RANGE: f"{event.type} 触发，进入横盘整理",
        }
        return reasons.get(new_state, f"{event.type} 触发状态转换")
    
    def _update_key_levels(self, event: WyckoffEvent):
        """更新关键价位"""
        price = event.price
        
        # 初始化区间边界
        if self.range_high is None:
            self.range_high = price
            self.range_low = price
        
        # SC/BC 事件更新区间边界
        if event.type == "SC":
            self.range_low = min(self.range_low, price)
        elif event.type == "BC":
            self.range_high = max(self.range_high, price)
        elif event.type == "AR":
            if self.last_climax_type == "SC":
                self.range_high = max(self.range_high, price)
            elif self.last_climax_type == "BC":
                self.range_low = min(self.range_low, price)
    
    def get_phase_progress(self) -> PhaseProgress:
        """获取当前阶段进度"""
        # 当前阶段的事件类型
        current_events = [e.type for e in self.events_in_current_phase]
        current_event_set = set(current_events)
        
        # 期望的事件
        expected = self.expected_events.get(self.current_state, [])
        
        # 已出现和缺失的事件
        events_in_phase = list(current_event_set & set(expected))
        missing_events = list(set(expected) - current_event_set)
        
        # 计算进度
        if expected:
            progress = len(events_in_phase) / len(expected)
        else:
            progress = 0.0
        
        # 获取下一阶段期望事件
        state_detail = STATE_DETAILS.get(self.current_state, {})
        next_phase = state_detail.get("next_phase")
        next_expected = self.expected_events.get(next_phase, []) if next_phase else []
        
        # 生成备注
        notes = []
        if self.bars_in_state > 30:
            notes.append(f"已在当前阶段停留 {self.bars_in_state} 根K线")
        if progress >= 0.8:
            notes.append("阶段即将完成，关注转换信号")
        if missing_events:
            notes.append(f"等待事件: {', '.join(missing_events)}")
        
        return PhaseProgress(
            current_state=self.current_state,
            state_description=STATE_DESCRIPTIONS.get(self.current_state, "未知"),
            progress=progress,
            events_in_phase=events_in_phase,
            missing_events=missing_events,
            next_expected=next_expected,
            time_in_phase_bars=self.bars_in_state,
            notes=notes,
        )
    
    def get_bias(self) -> tuple[Literal["bullish", "bearish", "neutral"], float]:
        """获取当前偏向"""
        if self.current_state in (
            WyckoffState.ACC_PHASE_A, WyckoffState.ACC_PHASE_B,
            WyckoffState.ACC_PHASE_C, WyckoffState.ACC_PHASE_D,
            WyckoffState.ACC_PHASE_E, WyckoffState.MARKUP,
        ):
            # 吸筹阶段，偏向看涨
            # 阶段越后，置信度越高
            phase_conf = {
                WyckoffState.ACC_PHASE_A: 0.5,
                WyckoffState.ACC_PHASE_B: 0.55,
                WyckoffState.ACC_PHASE_C: 0.65,
                WyckoffState.ACC_PHASE_D: 0.8,
                WyckoffState.ACC_PHASE_E: 0.9,
                WyckoffState.MARKUP: 0.95,
            }
            return "bullish", phase_conf.get(self.current_state, 0.5)
        
        elif self.current_state in (
            WyckoffState.DIST_PHASE_A, WyckoffState.DIST_PHASE_B,
            WyckoffState.DIST_PHASE_C, WyckoffState.DIST_PHASE_D,
            WyckoffState.MARKDOWN,
        ):
            phase_conf = {
                WyckoffState.DIST_PHASE_A: 0.5,
                WyckoffState.DIST_PHASE_B: 0.55,
                WyckoffState.DIST_PHASE_C: 0.65,
                WyckoffState.DIST_PHASE_D: 0.8,
                WyckoffState.MARKDOWN: 0.95,
            }
            return "bearish", phase_conf.get(self.current_state, 0.5)
        
        return "neutral", 0.5
    
    def predict_next_states(self) -> list[tuple[WyckoffState, float]]:
        """预测下一个可能的状态"""
        predictions = []
        
        # 查找当前状态的所有可能转换
        for (state, event_type), (target, prob) in self.transition_rules.items():
            if state == self.current_state:
                # 根据当前阶段进度调整概率
                progress = self.get_phase_progress().progress
                adjusted_prob = prob * (0.5 + progress * 0.5)
                predictions.append((target, adjusted_prob))
        
        # 去重并按概率排序
        state_probs: dict[WyckoffState, float] = {}
        for state, prob in predictions:
            if state not in state_probs:
                state_probs[state] = prob
            else:
                state_probs[state] = max(state_probs[state], prob)
        
        sorted_predictions = sorted(state_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:3]
    
    def predict_events(self) -> list[dict]:
        """预测下一个可能的事件"""
        predictions = []
        
        # 当前阶段缺失的事件
        progress = self.get_phase_progress()
        for event_type in progress.missing_events:
            predictions.append({
                "type": event_type,
                "probability": 0.7,
                "description": f"当前阶段({STATE_DESCRIPTIONS[self.current_state]})期望事件",
            })
        
        # 下一阶段的首发事件
        for event_type in progress.next_expected[:2]:
            predictions.append({
                "type": event_type,
                "probability": 0.5 * progress.progress,
                "description": f"下一阶段首发事件",
            })
        
        # 按概率排序
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions[:5]
    
    def assess_risk(self) -> tuple[Literal["low", "medium", "high", "extreme"], list[str]]:
        """评估当前风险"""
        factors = []
        risk_score = 0
        
        # 阶段风险
        high_risk_states = {
            WyckoffState.ACC_PHASE_A: "处于吸筹初期，方向尚未确认",
            WyckoffState.ACC_PHASE_C: "测试阶段，存在失败可能",
            WyckoffState.DIST_PHASE_A: "处于派发初期，方向尚未确认",
            WyckoffState.DIST_PHASE_C: "测试阶段，存在失败可能",
            WyckoffState.UNKNOWN: "状态不明，风险较高",
            WyckoffState.RANGE: "横盘状态，方向不明",
        }
        
        if self.current_state in high_risk_states:
            factors.append(high_risk_states[self.current_state])
            risk_score += 2
        
        # 置信度风险
        if self.state_confidence < 0.6:
            factors.append(f"状态置信度较低({self.state_confidence:.0%})")
            risk_score += 1
        
        # 时间风险
        if self.bars_in_state > 50:
            factors.append(f"在当前阶段停留过久({self.bars_in_state}根K线)")
            risk_score += 1
        
        # 转换历史风险
        recent_transitions = self.transition_history[-5:]
        if len(recent_transitions) >= 3:
            # 频繁转换可能表示市场混乱
            factors.append("近期状态转换频繁")
            risk_score += 1
        
        # 回退风险
        for t in recent_transitions:
            if "失败" in t.reason or "回测" in t.reason:
                factors.append("近期出现阶段回退")
                risk_score += 2
                break
        
        # 确定风险级别
        if risk_score >= 5:
            level = "extreme"
        elif risk_score >= 3:
            level = "high"
        elif risk_score >= 1:
            level = "medium"
        else:
            level = "low"
        
        return level, factors
    
    def get_action_suggestion(self) -> str:
        """获取交易建议"""
        bias, bias_conf = self.get_bias()
        risk_level, _ = self.assess_risk()
        progress = self.get_phase_progress()
        
        suggestions = {
            # 吸筹阶段建议
            (WyckoffState.ACC_PHASE_A, "low"): "观望等待，等待 ST 确认底部",
            (WyckoffState.ACC_PHASE_A, "medium"): "观望等待，SC 后可能有 AR 反弹",
            (WyckoffState.ACC_PHASE_B, "low"): "观望或小仓试探，等待 SPRING/TEST",
            (WyckoffState.ACC_PHASE_B, "medium"): "观望等待，区间内不追涨杀跌",
            (WyckoffState.ACC_PHASE_C, "low"): "SPRING/TEST 后可分批建仓，止损区间下方",
            (WyckoffState.ACC_PHASE_C, "medium"): "等待 SOS 确认后再入场",
            (WyckoffState.ACC_PHASE_D, "low"): "SOS 确认后可加仓，LPS 是加仓点",
            (WyckoffState.ACC_PHASE_D, "medium"): "持有观察，等待 JAC 突破",
            (WyckoffState.ACC_PHASE_E, "low"): "持有，JAC/BUEC 后可追加仓位",
            
            # 派发阶段建议
            (WyckoffState.DIST_PHASE_A, "low"): "观望等待，等待 ST 确认顶部",
            (WyckoffState.DIST_PHASE_A, "medium"): "考虑减仓，BC 后可能有 AR 回落",
            (WyckoffState.DIST_PHASE_B, "low"): "减仓或清仓，等待 UT/UTAD 确认",
            (WyckoffState.DIST_PHASE_B, "medium"): "观望等待，区间内不追涨",
            (WyckoffState.DIST_PHASE_C, "low"): "UT/UTAD 后考虑做空，止损区间上方",
            (WyckoffState.DIST_PHASE_C, "medium"): "等待 SOW 确认后再做空",
            (WyckoffState.DIST_PHASE_D, "low"): "SOW 确认后可加空，LPSY 是加仓点",
            
            # 趋势状态
            (WyckoffState.MARKUP, "low"): "趋势持有，回调不破前低可加仓",
            (WyckoffState.MARKUP, "medium"): "持有观察，注意 BC 信号",
            (WyckoffState.MARKDOWN, "low"): "空头趋势，反弹不破前高可加空",
            (WyckoffState.MARKDOWN, "medium"): "空头持有，注意 SC 信号",
            
            # 其他状态
            (WyckoffState.RANGE, "low"): "横盘观望，等待方向明确",
            (WyckoffState.RANGE, "medium"): "横盘观望，不宜重仓",
            (WyckoffState.UNKNOWN, "high"): "状态不明，建议观望",
        }
        
        key = (self.current_state, risk_level)
        if key in suggestions:
            return suggestions[key]
        
        # 默认建议
        if risk_level in ("high", "extreme"):
            return "风险较高，建议观望或轻仓"
        elif bias == "bullish" and bias_conf > 0.7:
            return "偏多，可考虑在回调时建仓"
        elif bias == "bearish" and bias_conf > 0.7:
            return "偏空，可考虑在反弹时做空或减仓"
        else:
            return "方向不明，建议观望等待信号"
    
    def analyze(self, events: list[WyckoffEvent], df: pd.DataFrame | None = None) -> StateMachineResult:
        """
        完整分析事件序列
        
        Args:
            events: 按时间排序的事件列表
            df: 价格数据
        
        Returns:
            StateMachineResult: 分析结果
        """
        self.reset()
        
        # 处理所有事件
        for idx, event in enumerate(events):
            self.process_event(event, idx, df)
        
        # 获取各项分析结果
        phase_progress = self.get_phase_progress()
        bias, bias_conf = self.get_bias()
        next_states = self.predict_next_states()
        predicted_events = self.predict_events()
        risk_level, risk_factors = self.assess_risk()
        action = self.get_action_suggestion()
        
        # 构建关键价位
        key_levels = {}
        if self.range_high is not None:
            key_levels["range_high"] = self.range_high
        if self.range_low is not None:
            key_levels["range_low"] = self.range_low
        if self.climax_price is not None:
            key_levels["climax_price"] = self.climax_price
        
        return StateMachineResult(
            current_state=self.current_state,
            state_description=STATE_DESCRIPTIONS.get(self.current_state, "未知"),
            state_confidence=self.state_confidence,
            phase_progress=phase_progress,
            bias=bias,
            bias_confidence=bias_conf,
            transition_history=self.transition_history,
            next_probable_states=next_states,
            predicted_events=predicted_events,
            risk_level=risk_level,
            risk_factors=risk_factors,
            action_suggestion=action,
            key_levels=key_levels,
        )


def analyze_with_state_machine(events: list[WyckoffEvent], df: pd.DataFrame | None = None) -> StateMachineResult:
    """
    使用状态机分析事件序列
    
    Args:
        events: 按时间排序的威科夫事件列表
        df: 价格数据（可选）
    
    Returns:
        StateMachineResult: 状态机分析结果
    """
    sm = WyckoffStateMachine()
    return sm.analyze(events, df)

