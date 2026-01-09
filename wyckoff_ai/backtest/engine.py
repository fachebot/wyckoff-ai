"""
回测引擎

模拟交易执行，支持：
- 基于威科夫事件的交易信号
- 止损止盈管理
- 仓位管理
- 交易记录
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from wyckoff_ai.schemas import WyckoffEvent, WyckoffAnalysis, ProbabilisticScenario


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"
    TARGET_HIT = "target_hit"


@dataclass
class Trade:
    """单笔交易记录"""
    trade_id: int
    direction: TradeDirection
    
    # 入场信息
    entry_time: str
    entry_price: float
    entry_reason: str
    entry_event: WyckoffEvent | None = None
    
    # 出场信息
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    
    # 止损止盈
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    
    # 仓位
    position_size: float = 1.0  # 合约数/股数
    position_value: float = 0.0  # 入场时的仓位价值
    
    # 状态
    status: TradeStatus = TradeStatus.OPEN
    
    # 收益
    pnl: float = 0.0  # 绝对收益
    pnl_pct: float = 0.0  # 百分比收益
    
    # MFE/MAE
    mfe: float = 0.0
    mae: float = 0.0
    
    # 持仓时间
    bars_held: int = 0
    
    # 备注
    notes: list[str] = field(default_factory=list)
    
    def close(
        self,
        exit_time: str,
        exit_price: float,
        exit_reason: str,
        status: TradeStatus = TradeStatus.CLOSED,
    ):
        """平仓"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = status
        
        # 计算收益
        if self.direction == TradeDirection.LONG:
            self.pnl = (exit_price - self.entry_price) * self.position_size
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:
            self.pnl = (self.entry_price - exit_price) * self.position_size
            self.pnl_pct = (self.entry_price / exit_price - 1) * 100


@dataclass
class BacktestConfig:
    """回测配置"""
    # 初始资金
    initial_capital: float = 100000.0
    
    # 仓位管理
    position_size_pct: float = 10.0  # 每笔交易使用的资金比例
    max_positions: int = 3  # 最大同时持仓数
    
    # 止损止盈
    stop_loss_atr: float = 2.0  # 止损距离（ATR倍数）
    take_profit_atr: float = 3.0  # 止盈距离（ATR倍数）
    use_trailing_stop: bool = False  # 是否使用移动止损
    trailing_stop_atr: float = 1.5  # 移动止损距离
    
    # 交易规则
    min_confidence: float = 0.6  # 最低置信度要求
    trade_bullish_events: bool = True  # 是否交易看涨事件
    trade_bearish_events: bool = True  # 是否交易看跌事件
    
    # 事件过滤
    allowed_events: list[str] | None = None  # 允许交易的事件类型
    blocked_events: list[str] | None = None  # 禁止交易的事件类型
    
    # 时间限制
    max_bars_in_trade: int = 48  # 最大持仓K线数
    
    # 手续费
    commission_pct: float = 0.1  # 手续费比例
    slippage_pct: float = 0.05  # 滑点比例


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    
    # 交易记录
    trades: list[Trade] = field(default_factory=list)
    
    # 资金曲线
    equity_curve: list[float] = field(default_factory=list)
    equity_timestamps: list[str] = field(default_factory=list)
    
    # 统计指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    avg_bars_in_trade: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    
    # 按事件类型统计
    stats_by_event: dict[str, dict] = field(default_factory=dict)
    
    # 时间信息
    start_time: str = ""
    end_time: str = ""
    total_bars: int = 0


class BacktestEngine:
    """回测引擎"""
    
    # 事件方向映射
    BULLISH_EVENTS = {"SC", "SPRING", "TEST", "SOS", "LPS", "JAC", "BUEC"}
    BEARISH_EVENTS = {"BC", "UT", "UTAD", "SOW", "LPSY"}
    
    def __init__(self, config: BacktestConfig | None = None):
        """初始化回测引擎"""
        self.config = config or BacktestConfig()
        
        # 状态
        self._trades: list[Trade] = []
        self._open_trades: list[Trade] = []
        self._equity: float = self.config.initial_capital
        self._equity_curve: list[float] = []
        self._equity_timestamps: list[str] = []
        self._trade_counter: int = 0
        
        # 当前K线数据
        self._current_bar: pd.Series | None = None
        self._current_idx: int = 0
        self._df: pd.DataFrame | None = None
    
    def reset(self):
        """重置引擎状态"""
        self._trades = []
        self._open_trades = []
        self._equity = self.config.initial_capital
        self._equity_curve = []
        self._equity_timestamps = []
        self._trade_counter = 0
        self._current_bar = None
        self._current_idx = 0
        self._df = None
    
    def _should_trade_event(self, event: WyckoffEvent) -> bool:
        """判断是否应该交易该事件"""
        # 置信度过滤
        if event.confidence < self.config.min_confidence:
            return False
        
        # 事件类型过滤
        if self.config.allowed_events is not None:
            if event.type not in self.config.allowed_events:
                return False
        
        if self.config.blocked_events is not None:
            if event.type in self.config.blocked_events:
                return False
        
        # 方向过滤
        if event.type in self.BULLISH_EVENTS and not self.config.trade_bullish_events:
            return False
        if event.type in self.BEARISH_EVENTS and not self.config.trade_bearish_events:
            return False
        
        # 持仓数量限制
        if len(self._open_trades) >= self.config.max_positions:
            return False
        
        return True
    
    def _get_trade_direction(self, event: WyckoffEvent) -> TradeDirection | None:
        """获取交易方向"""
        if event.type in self.BULLISH_EVENTS:
            return TradeDirection.LONG
        elif event.type in self.BEARISH_EVENTS:
            return TradeDirection.SHORT
        return None
    
    def _open_trade(
        self,
        direction: TradeDirection,
        price: float,
        atr: float,
        time: str,
        reason: str,
        event: WyckoffEvent | None = None,
    ) -> Trade:
        """开仓"""
        self._trade_counter += 1
        
        # 计算仓位大小
        position_value = self._equity * (self.config.position_size_pct / 100)
        position_size = position_value / price
        
        # 计算止损止盈
        if direction == TradeDirection.LONG:
            stop_loss = price - self.config.stop_loss_atr * atr
            take_profit = price + self.config.take_profit_atr * atr
        else:
            stop_loss = price + self.config.stop_loss_atr * atr
            take_profit = price - self.config.take_profit_atr * atr
        
        # 应用滑点
        slippage = price * (self.config.slippage_pct / 100)
        if direction == TradeDirection.LONG:
            entry_price = price + slippage
        else:
            entry_price = price - slippage
        
        # 扣除手续费
        commission = position_value * (self.config.commission_pct / 100)
        self._equity -= commission
        
        trade = Trade(
            trade_id=self._trade_counter,
            direction=direction,
            entry_time=time,
            entry_price=entry_price,
            entry_reason=reason,
            entry_event=event,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            position_value=position_value,
        )
        
        self._trades.append(trade)
        self._open_trades.append(trade)
        
        return trade
    
    def _close_trade(
        self,
        trade: Trade,
        price: float,
        time: str,
        reason: str,
        status: TradeStatus = TradeStatus.CLOSED,
    ):
        """平仓"""
        # 应用滑点
        slippage = price * (self.config.slippage_pct / 100)
        if trade.direction == TradeDirection.LONG:
            exit_price = price - slippage
        else:
            exit_price = price + slippage
        
        trade.close(time, exit_price, reason, status)
        
        # 扣除手续费
        commission = trade.position_value * (self.config.commission_pct / 100)
        
        # 更新权益
        self._equity += trade.pnl - commission
        
        # 从持仓列表移除
        if trade in self._open_trades:
            self._open_trades.remove(trade)
    
    def _update_open_trades(self, bar: pd.Series, time: str):
        """更新持仓状态（检查止损止盈）"""
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        
        trades_to_close = []
        
        for trade in self._open_trades:
            trade.bars_held += 1
            
            # 更新MFE/MAE
            if trade.direction == TradeDirection.LONG:
                current_mfe = (high / trade.entry_price - 1) * 100
                current_mae = (trade.entry_price / low - 1) * 100 if low > 0 else 0
            else:
                current_mfe = (trade.entry_price / low - 1) * 100 if low > 0 else 0
                current_mae = (high / trade.entry_price - 1) * 100
            
            trade.mfe = max(trade.mfe, current_mfe)
            trade.mae = max(trade.mae, current_mae)
            
            # 检查止损
            if trade.stop_loss is not None:
                if trade.direction == TradeDirection.LONG and low <= trade.stop_loss:
                    trades_to_close.append((trade, trade.stop_loss, "止损触发", TradeStatus.STOPPED))
                    continue
                elif trade.direction == TradeDirection.SHORT and high >= trade.stop_loss:
                    trades_to_close.append((trade, trade.stop_loss, "止损触发", TradeStatus.STOPPED))
                    continue
            
            # 检查止盈
            if trade.take_profit is not None:
                if trade.direction == TradeDirection.LONG and high >= trade.take_profit:
                    trades_to_close.append((trade, trade.take_profit, "止盈触发", TradeStatus.TARGET_HIT))
                    continue
                elif trade.direction == TradeDirection.SHORT and low <= trade.take_profit:
                    trades_to_close.append((trade, trade.take_profit, "止盈触发", TradeStatus.TARGET_HIT))
                    continue
            
            # 检查最大持仓时间
            if trade.bars_held >= self.config.max_bars_in_trade:
                trades_to_close.append((trade, close, f"超过最大持仓时间({self.config.max_bars_in_trade}根)", TradeStatus.CLOSED))
                continue
            
            # 移动止损
            if self.config.use_trailing_stop and trade.trailing_stop is not None:
                atr = float(bar.get("atr_14", 0) or 0)
                if trade.direction == TradeDirection.LONG:
                    new_trailing = close - self.config.trailing_stop_atr * atr
                    if new_trailing > trade.trailing_stop:
                        trade.trailing_stop = new_trailing
                        trade.stop_loss = new_trailing
                else:
                    new_trailing = close + self.config.trailing_stop_atr * atr
                    if new_trailing < trade.trailing_stop:
                        trade.trailing_stop = new_trailing
                        trade.stop_loss = new_trailing
        
        # 执行平仓
        for trade, price, reason, status in trades_to_close:
            self._close_trade(trade, price, time, reason, status)
    
    def _process_events(
        self,
        events: list[WyckoffEvent],
        bar: pd.Series,
        bar_time: str,
    ):
        """处理事件，决定是否开仓"""
        for event in events:
            if not self._should_trade_event(event):
                continue
            
            direction = self._get_trade_direction(event)
            if direction is None:
                continue
            
            # 检查是否已有同方向持仓
            same_direction = [
                t for t in self._open_trades
                if t.direction == direction
            ]
            if same_direction:
                continue  # 不重复建仓
            
            price = float(bar["close"])
            atr = float(bar.get("atr_14", 0) or 0)
            if atr <= 0:
                atr = price * 0.02  # 默认2%
            
            self._open_trade(
                direction=direction,
                price=price,
                atr=atr,
                time=bar_time,
                reason=f"{event.type} (置信度 {event.confidence:.0%})",
                event=event,
            )
    
    def run(
        self,
        df: pd.DataFrame,
        events: list[WyckoffEvent],
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            df: K线数据，需包含 timestamp, open, high, low, close, atr_14
            events: 威科夫事件列表
        
        Returns:
            BacktestResult: 回测结果
        """
        self.reset()
        self._df = df
        
        # 建立事件时间戳映射
        events_by_ts: dict[str, list[WyckoffEvent]] = {}
        for event in events:
            ts = event.ts
            if ts not in events_by_ts:
                events_by_ts[ts] = []
            events_by_ts[ts].append(event)
        
        # 遍历K线
        for idx, row in df.iterrows():
            self._current_idx = int(df.index.get_loc(idx))
            self._current_bar = row
            
            ts = row["timestamp"]
            if hasattr(ts, "isoformat"):
                time_str = ts.isoformat()
            else:
                time_str = str(ts)
            
            # 更新持仓状态（止损止盈检查）
            self._update_open_trades(row, time_str)
            
            # 处理当前K线的事件
            current_events = events_by_ts.get(time_str, [])
            if current_events:
                self._process_events(current_events, row, time_str)
            
            # 记录权益曲线
            # 计算持仓市值
            open_value = 0.0
            for trade in self._open_trades:
                close = float(row["close"])
                if trade.direction == TradeDirection.LONG:
                    open_value += (close - trade.entry_price) * trade.position_size
                else:
                    open_value += (trade.entry_price - close) * trade.position_size
            
            total_equity = self._equity + open_value
            self._equity_curve.append(total_equity)
            self._equity_timestamps.append(time_str)
        
        # 平掉所有未平仓位
        for trade in list(self._open_trades):
            last_bar = df.iloc[-1]
            last_close = float(last_bar["close"])
            last_ts = last_bar["timestamp"]
            if hasattr(last_ts, "isoformat"):
                last_time = last_ts.isoformat()
            else:
                last_time = str(last_ts)
            
            self._close_trade(trade, last_close, last_time, "回测结束强制平仓")
        
        # 计算统计指标
        return self._calculate_result(df)
    
    def _calculate_result(self, df: pd.DataFrame) -> BacktestResult:
        """计算回测结果统计"""
        trades = self._trades
        
        result = BacktestResult(
            config=self.config,
            trades=trades,
            equity_curve=self._equity_curve,
            equity_timestamps=self._equity_timestamps,
            total_trades=len(trades),
            total_bars=len(df),
        )
        
        if not trades:
            return result
        
        # 时间信息
        result.start_time = self._equity_timestamps[0] if self._equity_timestamps else ""
        result.end_time = self._equity_timestamps[-1] if self._equity_timestamps else ""
        
        # 胜负统计
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades) if trades else 0.0
        
        # 收益统计
        total_pnl = sum(t.pnl for t in trades)
        result.total_pnl = total_pnl
        result.total_pnl_pct = (self._equity / self.config.initial_capital - 1) * 100
        
        if wins:
            result.avg_win = sum(t.pnl for t in wins) / len(wins)
        if losses:
            result.avg_loss = abs(sum(t.pnl for t in losses) / len(losses))
        
        # 盈亏比
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0
        
        # 最大回撤
        peak = self.config.initial_capital
        max_dd = 0.0
        max_dd_pct = 0.0
        
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        result.max_drawdown = max_dd
        result.max_drawdown_pct = max_dd_pct
        
        # 风险调整收益
        if len(self._equity_curve) > 1:
            returns = np.diff(self._equity_curve) / np.array(self._equity_curve[:-1])
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(returns) > 0:
                avg_return = float(np.mean(returns))
                std_return = float(np.std(returns))
                
                # Sharpe (假设无风险利率为0，年化系数根据时间框架调整)
                if std_return > 0:
                    result.sharpe_ratio = avg_return / std_return * np.sqrt(252)
                
                # Sortino (只考虑下行风险)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_std = float(np.std(negative_returns))
                    if downside_std > 0:
                        result.sortino_ratio = avg_return / downside_std * np.sqrt(252)
        
        # Calmar
        if result.max_drawdown_pct > 0:
            annual_return = result.total_pnl_pct  # 简化处理
            result.calmar_ratio = annual_return / result.max_drawdown_pct
        
        # 持仓统计
        if trades:
            result.avg_bars_in_trade = sum(t.bars_held for t in trades) / len(trades)
            result.avg_mfe = sum(t.mfe for t in trades) / len(trades)
            result.avg_mae = sum(t.mae for t in trades) / len(trades)
        
        # 按事件类型统计
        by_event: dict[str, list[Trade]] = {}
        for trade in trades:
            if trade.entry_event:
                et = trade.entry_event.type
                if et not in by_event:
                    by_event[et] = []
                by_event[et].append(trade)
        
        for event_type, event_trades in by_event.items():
            wins_e = [t for t in event_trades if t.pnl > 0]
            losses_e = [t for t in event_trades if t.pnl <= 0]
            
            result.stats_by_event[event_type] = {
                "count": len(event_trades),
                "wins": len(wins_e),
                "losses": len(losses_e),
                "win_rate": len(wins_e) / len(event_trades) if event_trades else 0,
                "total_pnl": sum(t.pnl for t in event_trades),
                "avg_pnl": sum(t.pnl for t in event_trades) / len(event_trades) if event_trades else 0,
                "avg_mfe": sum(t.mfe for t in event_trades) / len(event_trades) if event_trades else 0,
                "avg_mae": sum(t.mae for t in event_trades) / len(event_trades) if event_trades else 0,
            }
        
        return result
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        events: list[WyckoffEvent],
        n_splits: int = 5,
        train_ratio: float = 0.7,
    ) -> list[BacktestResult]:
        """
        Walk-forward 回测
        
        将数据分成多个时间段，每个时间段用前部分训练、后部分测试
        
        Args:
            df: K线数据
            events: 事件列表
            n_splits: 分割数量
            train_ratio: 训练集比例
        
        Returns:
            每个测试期的回测结果列表
        """
        n = len(df)
        split_size = n // n_splits
        results = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n)
            
            # 测试期数据
            test_df = df.iloc[start_idx:end_idx]
            
            # 筛选测试期内的事件
            test_start_ts = test_df.iloc[0]["timestamp"]
            test_end_ts = test_df.iloc[-1]["timestamp"]
            
            test_events = []
            for event in events:
                try:
                    event_ts = pd.Timestamp(event.ts)
                    if test_start_ts <= event_ts <= test_end_ts:
                        test_events.append(event)
                except Exception:
                    pass
            
            if test_events:
                result = self.run(test_df, test_events)
                results.append(result)
        
        return results

