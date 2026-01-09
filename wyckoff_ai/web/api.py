"""
威科夫分析 REST API

提供 K 线数据、威科夫分析、事件检测等 API 接口
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.exceptions import WyckoffError
from wyckoff_ai.features import compute_features
from wyckoff_ai.logging import get_logger
from wyckoff_ai.schemas import WyckoffAnalysis, WyckoffEvent
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff
from wyckoff_ai.wyckoff.state_machine import WyckoffStateMachine, StateMachineResult

logger = get_logger("web.api")

router = APIRouter(prefix="/api", tags=["analysis"])


# ============== 请求/响应模型 ==============

class OHLCVBar(BaseModel):
    """K 线数据"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class EventResponse(BaseModel):
    """威科夫事件"""
    type: str
    timestamp: str
    price: float
    confidence: float
    evidence: list[str]
    direction: Literal["bullish", "bearish", "neutral"]


class LevelResponse(BaseModel):
    """关键价位"""
    support: list[float]
    resistance: list[float]


class RangeResponse(BaseModel):
    """交易区间"""
    low: float | None
    high: float | None
    mid: float | None


class StateResponse(BaseModel):
    """状态机状态"""
    current_state: str
    state_description: str
    phase_progress: float
    bias: Literal["bullish", "bearish", "neutral"]
    bias_confidence: float
    next_expected_events: list[str]
    action_suggestion: str


class PredictionResponse(BaseModel):
    """预测信息"""
    bias: Literal["bullish", "bearish", "neutral"]
    probability: float
    confidence: float
    description: str
    entry_price: float | None = None
    stop_loss: float | None = None
    targets: list[float] = []


class AnalysisResponse(BaseModel):
    """完整分析响应"""
    symbol: str
    timeframe: str
    timestamp: str
    
    # K 线数据
    ohlcv: list[OHLCVBar]
    
    # 威科夫分析
    market_structure: str
    events: list[EventResponse]
    levels: LevelResponse
    range: RangeResponse | None
    
    # 状态机
    state: StateResponse | None
    
    # 预测
    predictions: list[PredictionResponse]
    
    # 元数据
    bars_count: int
    events_count: int


class QuickAnalysisResponse(BaseModel):
    """快速分析响应（不含 K 线数据）"""
    symbol: str
    timeframe: str
    timestamp: str
    market_structure: str
    bias: Literal["bullish", "bearish", "neutral"]
    confidence: float
    current_state: str
    recent_events: list[EventResponse]
    action: str


# ============== 事件方向映射 ==============

EVENT_DIRECTION = {
    "SC": "bullish", "AR": "bullish", "ST": "bullish", "SOS": "bullish",
    "LPS": "bullish", "SPRING": "bullish", "JAC": "bullish", "BUEC": "bullish",
    "TEST": "bullish",
    "BC": "bearish", "SOW": "bearish", "LPSY": "bearish", "UT": "bearish",
    "UTAD": "bearish",
    "PSY": "neutral", "TR": "neutral",
}


# ============== API 端点 ==============

@router.get("/symbols")
async def get_supported_symbols():
    """获取支持的交易对列表"""
    # 常用交易对
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
        "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    ]
    return {"symbols": symbols}


@router.get("/timeframes")
async def get_supported_timeframes():
    """获取支持的时间周期列表"""
    timeframes = [
        {"value": "5m", "label": "5 分钟"},
        {"value": "15m", "label": "15 分钟"},
        {"value": "30m", "label": "30 分钟"},
        {"value": "1h", "label": "1 小时"},
        {"value": "2h", "label": "2 小时"},
        {"value": "4h", "label": "4 小时"},
        {"value": "1d", "label": "1 天"},
    ]
    return {"timeframes": timeframes}


@router.get("/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1h", description="时间周期"),
    limit: int = Query(default=200, ge=10, le=1000, description="K 线数量"),
):
    """获取 K 线数据"""
    try:
        # 转换 symbol 格式（URL 中用 - 代替 /）
        symbol = symbol.replace("-", "/").upper()
        
        result = fetch_ohlcv_binance_spot(symbol=symbol, timeframe=timeframe, limit=limit)
        
        bars = []
        for _, row in result.df.iterrows():
            if not row["is_gap"]:
                bars.append(OHLCVBar(
                    timestamp=row["timestamp"].isoformat(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                ))
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bars,
            "count": len(bars),
        }
    except WyckoffError as e:
        logger.error(f"获取 K 线失败: {e}")
        raise HTTPException(status_code=400, detail=str(e.message))
    except Exception as e:
        logger.exception(f"获取 K 线时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{symbol}", response_model=AnalysisResponse)
async def analyze_symbol(
    symbol: str,
    timeframe: str = Query(default="1h", description="时间周期"),
    limit: int = Query(default=200, ge=50, le=1000, description="K 线数量"),
    min_confidence: float = Query(default=0.5, ge=0, le=1, description="最小事件置信度"),
):
    """
    完整威科夫分析
    
    返回 K 线数据、事件检测、状态机分析、预测等完整信息
    """
    try:
        # 转换 symbol 格式
        symbol = symbol.replace("-", "/").upper()
        logger.info(f"开始分析: {symbol} {timeframe}")
        
        # 获取数据
        result = fetch_ohlcv_binance_spot(symbol=symbol, timeframe=timeframe, limit=limit)
        df = compute_features(result.df)
        
        # 威科夫分析
        cfg = DetectionConfig(
            lookback_bars=min(limit, 500),
            min_confidence_threshold=min_confidence,
        )
        analysis = detect_wyckoff(
            df,
            symbol=symbol,
            exchange="binance",
            timeframe=timeframe,
            cfg=cfg,
        )
        
        # 状态机分析
        sm = WyckoffStateMachine()
        sm_result = sm.analyze(analysis.events, df)
        
        # 构建响应
        bars = []
        for _, row in result.df.iterrows():
            if not row["is_gap"]:
                bars.append(OHLCVBar(
                    timestamp=row["timestamp"].isoformat(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                ))
        
        events = [
            EventResponse(
                type=e.type,
                timestamp=e.ts,
                price=e.price,
                confidence=e.confidence,
                evidence=e.evidence,
                direction=EVENT_DIRECTION.get(e.type, "neutral"),
            )
            for e in analysis.events
            if e.confidence >= min_confidence
        ]
        
        levels = LevelResponse(
            support=analysis.levels.support if analysis.levels else [],
            resistance=analysis.levels.resistance if analysis.levels else [],
        )
        
        range_info = None
        if analysis.range and analysis.range.low and analysis.range.high:
            range_info = RangeResponse(
                low=analysis.range.low,
                high=analysis.range.high,
                mid=analysis.range.mid,
            )
        
        state = None
        if sm_result:
            state = StateResponse(
                current_state=sm_result.current_state.value,
                state_description=sm_result.state_description,
                phase_progress=sm_result.phase_progress.progress,
                bias=sm_result.bias,
                bias_confidence=sm_result.bias_confidence,
                next_expected_events=sm_result.phase_progress.next_expected,
                action_suggestion=sm_result.action_suggestion,
            )
        
        # 构建预测
        predictions = []
        if analysis.probabilistic_scenarios:
            for scenario in analysis.probabilistic_scenarios[:3]:
                pred = PredictionResponse(
                    bias=scenario.bias,
                    probability=scenario.probability,
                    confidence=scenario.confidence,
                    description=scenario.description,
                )
                if scenario.signal:
                    pred.entry_price = scenario.signal.entry_price
                    pred.stop_loss = scenario.signal.stop_loss
                    pred.targets = scenario.signal.targets
                predictions.append(pred)
        
        logger.info(f"分析完成: {len(events)} 个事件, 状态={state.current_state if state else 'N/A'}")
        
        return AnalysisResponse(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow().isoformat(),
            ohlcv=bars,
            market_structure=analysis.market_structure,
            events=events,
            levels=levels,
            range=range_info,
            state=state,
            predictions=predictions,
            bars_count=len(bars),
            events_count=len(events),
        )
        
    except WyckoffError as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e.message))
    except Exception as e:
        logger.exception(f"分析时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick/{symbol}", response_model=QuickAnalysisResponse)
async def quick_analysis(
    symbol: str,
    timeframe: str = Query(default="1h", description="时间周期"),
):
    """
    快速分析
    
    返回简要分析结果，适合快速查看市场状态
    """
    try:
        symbol = symbol.replace("-", "/").upper()
        
        # 获取数据（较少数量）
        result = fetch_ohlcv_binance_spot(symbol=symbol, timeframe=timeframe, limit=200)
        df = compute_features(result.df)
        
        # 威科夫分析
        cfg = DetectionConfig(lookback_bars=200, min_confidence_threshold=0.6)
        analysis = detect_wyckoff(
            df,
            symbol=symbol,
            exchange="binance",
            timeframe=timeframe,
            cfg=cfg,
        )
        
        # 状态机分析
        sm = WyckoffStateMachine()
        sm_result = sm.analyze(analysis.events, df)
        
        # 最近事件（最多 5 个）
        recent_events = [
            EventResponse(
                type=e.type,
                timestamp=e.ts,
                price=e.price,
                confidence=e.confidence,
                evidence=e.evidence[:2],  # 只取前 2 个证据
                direction=EVENT_DIRECTION.get(e.type, "neutral"),
            )
            for e in analysis.events[-5:]
        ]
        
        return QuickAnalysisResponse(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow().isoformat(),
            market_structure=analysis.market_structure,
            bias=sm_result.bias if sm_result else "neutral",
            confidence=sm_result.bias_confidence if sm_result else 0.5,
            current_state=sm_result.current_state.value if sm_result else "unknown",
            recent_events=recent_events,
            action=sm_result.action_suggestion if sm_result else "观望",
        )
        
    except WyckoffError as e:
        raise HTTPException(status_code=400, detail=str(e.message))
    except Exception as e:
        logger.exception(f"快速分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }
