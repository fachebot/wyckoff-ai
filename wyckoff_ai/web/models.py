"""
API 数据模型定义
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OHLCVBar(BaseModel):
    """单根 K 线数据"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class WyckoffEventResponse(BaseModel):
    """威科夫事件响应"""
    type: str
    timestamp: str
    price: float
    confidence: float
    evidence: list[str] = []
    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    color: str = "#FFFFFF"


class KeyLevelsResponse(BaseModel):
    """关键价位响应"""
    support: list[float] = []
    resistance: list[float] = []
    pivot: list[float] = []


class RangeResponse(BaseModel):
    """区间响应"""
    low: float | None = None
    high: float | None = None
    mid: float | None = None


class StateInfoResponse(BaseModel):
    """状态机信息响应"""
    current_state: str
    state_description: str
    phase_progress: float
    bias: Literal["bullish", "bearish", "neutral"]
    bias_confidence: float
    next_expected_events: list[str] = []
    action_suggestion: str = ""


class ScenarioResponse(BaseModel):
    """交易剧本响应"""
    name: str
    bias: Literal["bullish", "bearish", "neutral"]
    probability: float
    confidence: float
    description: str = ""
    entry_price: float | None = None
    stop_loss: float | None = None
    targets: list[float] = []
    risk_reward_ratio: float = 0
    risk_level: Literal["low", "medium", "high", "extreme"] = "medium"


class AnalysisResponse(BaseModel):
    """完整分析响应"""
    symbol: str
    timeframe: str
    exchange: str = "binance"
    timestamp: str
    
    # K 线数据
    ohlcv: list[OHLCVBar]
    
    # 威科夫分析
    market_structure: str
    events: list[WyckoffEventResponse]
    levels: KeyLevelsResponse
    range: RangeResponse | None = None
    
    # 状态机
    state_info: StateInfoResponse | None = None
    
    # 交易剧本
    scenarios: list[ScenarioResponse] = []
    
    # 元信息
    analysis_time_ms: int = 0


class AnalysisRequest(BaseModel):
    """分析请求"""
    symbol: str = Field(default="BTC/USDT", description="交易对")
    timeframe: str = Field(default="1h", description="时间周期")
    limit: int = Field(default=200, ge=10, le=1000, description="K线数量")


class SymbolInfo(BaseModel):
    """交易对信息"""
    symbol: str
    base: str
    quote: str
    display_name: str


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    message: str
    details: dict = {}

