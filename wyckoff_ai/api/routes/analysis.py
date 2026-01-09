"""
分析 API 路由
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from wyckoff_ai.api.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    KlineData,
    KlineRequest,
    LevelData,
    RangeData,
    ScenarioData,
    StateMachineData,
    WyckoffEventData,
)
from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.exceptions import WyckoffError
from wyckoff_ai.features import compute_features
from wyckoff_ai.logging import get_logger
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff
from wyckoff_ai.wyckoff.state_machine import WyckoffStateMachine

logger = get_logger("api.analysis")

router = APIRouter(prefix="/api", tags=["analysis"])

# 事件方向映射
EVENT_DIRECTION = {
    "SC": "bullish", "AR": "bullish", "ST": "bullish", "SOS": "bullish",
    "LPS": "bullish", "SPRING": "bullish", "JAC": "bullish", "BUEC": "bullish",
    "TEST": "bullish",
    "BC": "bearish", "SOW": "bearish", "LPSY": "bearish", "UT": "bearish",
    "UTAD": "bearish",
    "PSY": "neutral", "TR": "neutral",
}


@router.get("/klines")
async def get_klines(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 200,
) -> dict:
    """
    获取 K 线数据
    """
    try:
        logger.info(f"获取K线: {symbol} {timeframe} limit={limit}")
        result = fetch_ohlcv_binance_spot(symbol=symbol, timeframe=timeframe, limit=min(limit, 1000))
        
        klines = []
        for _, row in result.df.iterrows():
            if not row["is_gap"]:
                klines.append({
                    "timestamp": row["timestamp"].isoformat(),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(klines),
            "klines": klines,
        }
    except WyckoffError as e:
        logger.error(f"获取K线失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("获取K线时发生错误")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    """
    执行威科夫分析
    
    返回 K 线数据、事件检测结果、状态机分析和概率化剧本
    """
    try:
        logger.info(f"分析请求: {request.symbol} {request.timeframe}")
        
        # 获取数据
        result = fetch_ohlcv_binance_spot(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=min(request.limit, 1000),
        )
        df = compute_features(result.df)
        
        # 检测事件
        cfg = DetectionConfig(
            lookback_bars=min(request.limit, 500),
            min_confidence_threshold=request.min_confidence,
        )
        analysis = detect_wyckoff(
            df,
            symbol=request.symbol,
            exchange="binance",
            timeframe=request.timeframe,
            cfg=cfg,
        )
        
        # 状态机分析
        sm = WyckoffStateMachine()
        sm_result = sm.analyze(analysis.events, df)
        
        # 构建 K 线数据
        klines = []
        for _, row in df.iterrows():
            if not row.get("is_gap", False):
                klines.append(KlineData(
                    timestamp=row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                ))
        
        # 构建事件数据
        events = []
        for evt in analysis.events:
            events.append(WyckoffEventData(
                type=evt.type,
                timestamp=evt.ts,
                price=evt.price,
                confidence=evt.confidence,
                evidence=evt.evidence,
                direction=EVENT_DIRECTION.get(evt.type, "neutral"),
            ))
        
        # 构建关键价位
        levels = LevelData(
            support=analysis.levels.support if analysis.levels else [],
            resistance=analysis.levels.resistance if analysis.levels else [],
        )
        
        # 构建区间
        range_data = RangeData(
            low=analysis.range.low if analysis.range else None,
            high=analysis.range.high if analysis.range else None,
            mid=analysis.range.mid if analysis.range else None,
        )
        
        # 构建状态机数据
        state_machine_data = None
        if sm_result:
            state_machine_data = StateMachineData(
                current_state=sm_result.current_state.value,
                state_description=sm_result.state_description,
                progress=sm_result.phase_progress.progress,
                bias=sm_result.bias,
                bias_confidence=sm_result.bias_confidence,
                next_expected_events=sm_result.phase_progress.next_expected,
            )
        
        # 构建概率化剧本
        scenarios = []
        if analysis.probabilistic_scenarios:
            for ps in analysis.probabilistic_scenarios[:3]:  # 只取前3个
                scenarios.append(ScenarioData(
                    name=ps.name,
                    bias=ps.bias,
                    probability=ps.probability,
                    description=ps.description,
                    entry_price=ps.signal.entry_price if ps.signal else None,
                    stop_loss=ps.signal.stop_loss if ps.signal else None,
                    targets=ps.signal.targets if ps.signal else [],
                    risk_level=ps.risk_level,
                ))
        
        return AnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            timestamp=datetime.utcnow().isoformat(),
            market_structure=analysis.market_structure or "unknown",
            klines=klines,
            events=events,
            levels=levels,
            range=range_data,
            state_machine=state_machine_data,
            scenarios=scenarios,
        )
        
    except WyckoffError as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("分析时发生错误")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols")
async def get_symbols() -> dict:
    """
    获取支持的交易对列表
    """
    # 常用的加密货币交易对
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT",
        "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT",
    ]
    
    timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"]
    
    return {
        "symbols": symbols,
        "timeframes": timeframes,
    }

