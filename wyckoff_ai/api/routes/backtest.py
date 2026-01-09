"""
回测 API 路由
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from wyckoff_ai.api.schemas import (
    BacktestMetricsData,
    BacktestRequest,
    BacktestResponse,
    BacktestTradeData,
)
from wyckoff_ai.backtest import BacktestConfig, BacktestEngine, calculate_metrics
from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.exceptions import WyckoffError
from wyckoff_ai.features import compute_features
from wyckoff_ai.logging import get_logger
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff

logger = get_logger("api.backtest")

router = APIRouter(prefix="/api", tags=["backtest"])


@router.post("/backtest")
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    运行策略回测
    """
    try:
        logger.info(f"回测请求: {request.symbol} {request.timeframe}")
        
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
        
        if not analysis.events:
            raise HTTPException(status_code=400, detail="未检测到任何事件，无法回测")
        
        # 配置回测
        bt_config = BacktestConfig(
            initial_capital=request.initial_capital,
            position_size_pct=request.position_size_pct,
            stop_loss_atr=request.stop_loss_atr,
            take_profit_atr=request.take_profit_atr,
            min_confidence=request.min_confidence,
        )
        
        # 运行回测
        engine = BacktestEngine(bt_config)
        bt_result = engine.run(df, analysis.events)
        
        # 计算指标
        metrics = calculate_metrics(bt_result)
        
        # 构建交易记录
        trades = []
        for trade in bt_result.trades:
            trades.append(BacktestTradeData(
                trade_id=trade.trade_id,
                direction=trade.direction.value,
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                status=trade.status.value,
                trigger_event=trade.entry_reason.split()[0] if trade.entry_reason else "",
            ))
        
        # 构建指标
        metrics_data = BacktestMetricsData(
            total_trades=metrics.trade_metrics.total_trades,
            winning_trades=metrics.trade_metrics.winning_trades,
            losing_trades=metrics.trade_metrics.losing_trades,
            win_rate=metrics.trade_metrics.win_rate,
            total_return_pct=metrics.total_return_pct,
            max_drawdown_pct=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            profit_factor=metrics.trade_metrics.profit_factor,
            avg_trade_pnl=metrics.trade_metrics.avg_trade,
        )
        
        return BacktestResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            total_bars=bt_result.total_bars,
            equity_curve=bt_result.equity_curve,
            trades=trades,
            metrics=metrics_data,
            stats_by_event=bt_result.stats_by_event,
        )
        
    except WyckoffError as e:
        logger.error(f"回测失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("回测时发生错误")
        raise HTTPException(status_code=500, detail=str(e))

