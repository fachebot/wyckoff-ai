from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


WyckoffEventType = Literal[
    "SC",      # Selling Climax
    "BC",      # Buying Climax (新增)
    "AR",      # Automatic Rally/Reaction
    "ST",      # Secondary Test
    "SOS",     # Sign of Strength
    "LPS",     # Last Point of Support
    "UT",      # Upthrust
    "UTAD",    # Upthrust After Distribution
    "SOW",     # Sign of Weakness
    "LPSY",    # Last Point of Supply
    "SPRING",  # Spring
    "TEST",    # Test
    "PSY",     # Preliminary Supply/Support (新增)
    "JAC",     # Jump Across the Creek (新增)
    "BUEC",    # Backup to Edge of Creek (新增)
]

MarketStructure = Literal[
    "range",
    "markup",
    "markdown",
    "accumulation",
    "distribution",
    "unknown",
]


class RangeInfo(BaseModel):
    low: float | None = None
    high: float | None = None
    mid: float | None = None
    duration_bars: int = 0
    start_ts: str | None = None
    end_ts: str | None = None


class WyckoffEvent(BaseModel):
    type: WyckoffEventType
    ts: str
    price: float
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class KeyLevels(BaseModel):
    support: list[float] = Field(default_factory=list)
    resistance: list[float] = Field(default_factory=list)
    pivot: list[float] = Field(default_factory=list)


class Scenario(BaseModel):
    bias: Literal["bullish", "bearish", "neutral"]
    confirmation: str
    invalidation: str
    notes: str | None = None


class TradingSignal(BaseModel):
    """交易信号"""
    entry_price: float | None = None
    entry_condition: str = ""
    stop_loss: float | None = None
    targets: list[float] = Field(default_factory=list)
    risk_reward_ratio: float = 0.0
    position_size_pct: float = 0.0  # 建议仓位百分比（0-100）
    time_horizon: str = ""
    confirmation_signals: list[str] = Field(default_factory=list)
    invalidation_signals: list[str] = Field(default_factory=list)


class ProbabilisticScenario(BaseModel):
    """概率化剧本"""
    name: str
    bias: Literal["bullish", "bearish", "neutral"]
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    
    # 概率来源分解
    probability_breakdown: dict[str, float] = Field(default_factory=dict)
    
    # 交易信号
    signal: TradingSignal | None = None
    
    # 描述
    description: str = ""
    key_events: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    
    # 风险评估
    risk_level: Literal["low", "medium", "high", "extreme"] = "medium"
    risk_factors: list[str] = Field(default_factory=list)


class StateMachineInfo(BaseModel):
    """状态机分析结果"""
    current_state: str
    state_description: str
    state_confidence: float = Field(ge=0.0, le=1.0)
    
    # 阶段进度
    phase_progress: float = Field(ge=0.0, le=1.0)
    events_in_phase: list[str] = Field(default_factory=list)
    missing_events: list[str] = Field(default_factory=list)
    next_expected_events: list[str] = Field(default_factory=list)
    time_in_phase_bars: int = 0
    
    # 偏向判断
    bias: Literal["bullish", "bearish", "neutral"] = "neutral"
    bias_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # 转换历史（最近5次）
    recent_transitions: list[dict] = Field(default_factory=list)
    
    # 预测
    next_probable_states: list[dict] = Field(default_factory=list)
    predicted_events: list[dict] = Field(default_factory=list)
    
    # 风险评估
    risk_level: Literal["low", "medium", "high", "extreme"] = "medium"
    risk_factors: list[str] = Field(default_factory=list)
    
    # 交易建议
    action_suggestion: str = ""
    
    # 关键价位（从状态机获取）
    key_levels: dict[str, float] = Field(default_factory=dict)
    
    # 阶段备注
    phase_notes: list[str] = Field(default_factory=list)


class WyckoffAnalysis(BaseModel):
    symbol: str
    exchange: str
    timeframe: str
    asof_ts: str

    market_structure: MarketStructure = "unknown"
    range: RangeInfo = Field(default_factory=RangeInfo)
    events: list[WyckoffEvent] = Field(default_factory=list)
    levels: KeyLevels = Field(default_factory=KeyLevels)
    scenarios: list[Scenario] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)

    # Optional: regime/state info for more stable range recognition
    regime_method: str | None = None
    regime_hint: str | None = None

    # Optional: event -> forward return stats computed on the same lookback window
    # Example: {"SOS": {"n": 4, "r12_med": 0.012, "r24_med": 0.018, "r48_med": 0.031, "win12": 0.75}}
    event_forward_stats: dict[str, dict[str, float]
                              ] = Field(default_factory=dict)
    
    # 状态机分析结果
    state_machine: StateMachineInfo | None = None
    
    # 概率化剧本
    probabilistic_scenarios: list[ProbabilisticScenario] = Field(default_factory=list)
