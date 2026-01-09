"""
API 数据模型定义
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KlineRequest(BaseModel):
    """K线数据请求"""
    symbol: str = Field(default="BTC/USDT", description="交易对")
    timeframe: str = Field(default="1h", description="时间周期")
    limit: int = Field(default=200, ge=10, le=1000, description="K线数量")


class AnalyzeRequest(BaseModel):
    """分析请求"""
    symbol: str = Field(default="BTC/USDT", description="交易对")
    timeframe: str = Field(default="1h", description="时间周期")
    limit: int = Field(default=200, ge=10, le=1000, description="K线数量")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="最小置信度")


class BacktestRequest(BaseModel):
    """回测请求"""
    symbol: str = Field(default="BTC/USDT", description="交易对")
    timeframe: str = Field(default="1h", description="时间周期")
    limit: int = Field(default=500, ge=100, le=1000, description="K线数量")
    initial_capital: float = Field(default=100000, ge=1000, description="初始资金")
    position_size_pct: float = Field(default=10, ge=1, le=100, description="仓位百分比")
    stop_loss_atr: float = Field(default=2.0, ge=0.5, le=5.0, description="止损ATR倍数")
    take_profit_atr: float = Field(default=3.0, ge=1.0, le=10.0, description="止盈ATR倍数")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="最小置信度")


class KlineData(BaseModel):
    """K线数据"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class WyckoffEventData(BaseModel):
    """威科夫事件"""
    type: str
    timestamp: str
    price: float
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    direction: Literal["bullish", "bearish", "neutral"] = "neutral"


class LevelData(BaseModel):
    """关键价位"""
    support: list[float] = Field(default_factory=list)
    resistance: list[float] = Field(default_factory=list)


class RangeData(BaseModel):
    """交易区间"""
    low: float | None = None
    high: float | None = None
    mid: float | None = None


class StateMachineData(BaseModel):
    """状态机数据"""
    current_state: str
    state_description: str
    progress: float
    bias: str
    bias_confidence: float
    next_expected_events: list[str] = Field(default_factory=list)


class ScenarioData(BaseModel):
    """交易剧本"""
    name: str
    bias: str
    probability: float
    description: str
    entry_price: float | None = None
    stop_loss: float | None = None
    targets: list[float] = Field(default_factory=list)
    risk_level: str = "medium"


class AnalysisResponse(BaseModel):
    """分析响应"""
    symbol: str
    timeframe: str
    timestamp: str
    market_structure: str
    
    # K线数据
    klines: list[KlineData]
    
    # 事件
    events: list[WyckoffEventData]
    
    # 关键价位
    levels: LevelData
    
    # 交易区间
    range: RangeData
    
    # 状态机
    state_machine: StateMachineData | None = None
    
    # 概率化剧本
    scenarios: list[ScenarioData] = Field(default_factory=list)


class BacktestTradeData(BaseModel):
    """回测交易记录"""
    trade_id: int
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str | None = None
    exit_price: float | None = None
    pnl: float = 0
    pnl_pct: float = 0
    status: str
    trigger_event: str


class BacktestMetricsData(BaseModel):
    """回测指标"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_pnl: float


class BacktestResponse(BaseModel):
    """回测响应"""
    symbol: str
    timeframe: str
    total_bars: int
    
    # 资金曲线
    equity_curve: list[float]
    
    # 交易记录
    trades: list[BacktestTradeData]
    
    # 指标
    metrics: BacktestMetricsData
    
    # 按事件统计
    stats_by_event: dict[str, dict]


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: str | None = None
    code: str | None = None

