"""
事件后验评估模块

评估威科夫事件发生后的价格表现，计算：
- MFE (Maximum Favorable Excursion) - 最大有利偏移
- MAE (Maximum Adverse Excursion) - 最大不利偏移
- 收益率统计
- 胜率和盈亏比
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from wyckoff_ai.schemas import WyckoffEvent


@dataclass
class EventPerformance:
    """单个事件的表现评估"""
    event: WyckoffEvent
    event_idx: int
    
    # 价格数据
    entry_price: float
    exit_price: float | None = None
    
    # MFE/MAE（百分比）
    mfe_pct: float = 0.0  # 最大有利偏移
    mae_pct: float = 0.0  # 最大不利偏移
    mfe_bars: int = 0     # MFE发生的K线数
    mae_bars: int = 0     # MAE发生的K线数
    
    # 收益率（不同持仓周期）
    return_6: float | None = None    # 6根K线后收益
    return_12: float | None = None   # 12根K线后收益
    return_24: float | None = None   # 24根K线后收益
    return_48: float | None = None   # 48根K线后收益
    
    # 方向判断
    expected_direction: Literal["long", "short", "neutral"] = "neutral"
    actual_direction: Literal["up", "down", "flat"] = "flat"
    is_correct: bool = False
    
    # 止损/止盈是否触发
    stop_hit: bool = False
    target_hit: bool = False
    stop_price: float | None = None
    target_price: float | None = None
    
    # 备注
    notes: list[str] = field(default_factory=list)


@dataclass
class EventTypeStats:
    """某类事件的统计"""
    event_type: str
    count: int = 0
    
    # 收益率统计
    avg_return_6: float = 0.0
    avg_return_12: float = 0.0
    avg_return_24: float = 0.0
    avg_return_48: float = 0.0
    
    median_return_6: float = 0.0
    median_return_12: float = 0.0
    median_return_24: float = 0.0
    median_return_48: float = 0.0
    
    # MFE/MAE 统计
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    median_mfe: float = 0.0
    median_mae: float = 0.0
    
    # 胜率（方向判断正确率）
    win_rate_6: float = 0.0
    win_rate_12: float = 0.0
    win_rate_24: float = 0.0
    win_rate_48: float = 0.0
    
    # 盈亏比
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # 方向正确率
    direction_accuracy: float = 0.0
    
    # 置信度与表现相关性
    confidence_correlation: float = 0.0


class EventEvaluator:
    """事件评估器"""
    
    # 事件的期望方向
    BULLISH_EVENTS = {"SC", "SPRING", "TEST", "SOS", "LPS", "JAC", "BUEC"}
    BEARISH_EVENTS = {"BC", "UT", "UTAD", "SOW", "LPSY"}
    NEUTRAL_EVENTS = {"AR", "ST", "PSY"}
    
    def __init__(
        self,
        df: pd.DataFrame,
        events: list[WyckoffEvent],
        *,
        horizons: tuple[int, ...] = (6, 12, 24, 48),
        stop_atr_mult: float = 2.0,
        target_atr_mult: float = 3.0,
    ):
        """
        初始化事件评估器
        
        Args:
            df: K线数据，需包含 timestamp, open, high, low, close, atr_14
            events: 威科夫事件列表
            horizons: 评估周期（K线数）
            stop_atr_mult: 止损距离（ATR倍数）
            target_atr_mult: 目标距离（ATR倍数）
        """
        self.df = df
        self.events = events
        self.horizons = horizons
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        
        # 建立时间戳到索引的映射
        self._ts_to_idx: dict[str, int] = {}
        for i, row in df.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "isoformat"):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)
            self._ts_to_idx[ts_str] = int(df.index.get_loc(i))
    
    def _get_expected_direction(self, event_type: str) -> Literal["long", "short", "neutral"]:
        """获取事件的期望方向"""
        if event_type in self.BULLISH_EVENTS:
            return "long"
        elif event_type in self.BEARISH_EVENTS:
            return "short"
        return "neutral"
    
    def _find_event_idx(self, event: WyckoffEvent) -> int | None:
        """找到事件对应的K线索引"""
        ts_str = event.ts
        if ts_str in self._ts_to_idx:
            return self._ts_to_idx[ts_str]
        
        # 尝试解析时间戳
        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            
            for i, row in self.df.iterrows():
                row_ts = row["timestamp"]
                if hasattr(row_ts, "tz_localize") and row_ts.tzinfo is None:
                    row_ts = row_ts.tz_localize("UTC")
                if row_ts == ts:
                    return int(self.df.index.get_loc(i))
        except Exception:
            pass
        
        return None
    
    def evaluate_single_event(self, event: WyckoffEvent) -> EventPerformance | None:
        """
        评估单个事件的表现
        
        Returns:
            EventPerformance 或 None（如果找不到对应K线）
        """
        idx = self._find_event_idx(event)
        if idx is None:
            return None
        
        n = len(self.df)
        if idx >= n - 1:
            return None  # 没有后续数据
        
        # 入场价格
        entry_price = float(self.df.iloc[idx]["close"])
        entry_atr = float(self.df.iloc[idx].get("atr_14", 0) or 0)
        
        # 期望方向
        expected_dir = self._get_expected_direction(event.type)
        
        # 计算止损和目标
        if expected_dir == "long":
            stop_price = entry_price - self.stop_atr_mult * entry_atr
            target_price = entry_price + self.target_atr_mult * entry_atr
        elif expected_dir == "short":
            stop_price = entry_price + self.stop_atr_mult * entry_atr
            target_price = entry_price - self.target_atr_mult * entry_atr
        else:
            stop_price = None
            target_price = None
        
        # 获取后续价格数据
        max_horizon = max(self.horizons) if self.horizons else 48
        end_idx = min(idx + max_horizon + 1, n)
        future = self.df.iloc[idx + 1: end_idx]
        
        if future.empty:
            return None
        
        # 计算各周期收益率
        returns: dict[int, float | None] = {}
        for h in self.horizons:
            h_idx = idx + h
            if h_idx < n:
                future_price = float(self.df.iloc[h_idx]["close"])
                ret = (future_price / entry_price - 1) * 100  # 百分比
                if expected_dir == "short":
                    ret = -ret  # 做空时收益反向
                returns[h] = ret
            else:
                returns[h] = None
        
        # 计算 MFE/MAE
        highs = future["high"].astype(float).to_numpy()
        lows = future["low"].astype(float).to_numpy()
        
        if expected_dir == "long":
            # 多头：高点是有利，低点是不利
            mfe_price = float(np.nanmax(highs))
            mae_price = float(np.nanmin(lows))
            mfe_pct = (mfe_price / entry_price - 1) * 100
            mae_pct = (entry_price / mae_price - 1) * 100 if mae_price > 0 else 0
            mfe_bars = int(np.nanargmax(highs)) + 1 if len(highs) > 0 else 0
            mae_bars = int(np.nanargmin(lows)) + 1 if len(lows) > 0 else 0
        elif expected_dir == "short":
            # 空头：低点是有利，高点是不利
            mfe_price = float(np.nanmin(lows))
            mae_price = float(np.nanmax(highs))
            mfe_pct = (entry_price / mfe_price - 1) * 100 if mfe_price > 0 else 0
            mae_pct = (mae_price / entry_price - 1) * 100
            mfe_bars = int(np.nanargmin(lows)) + 1 if len(lows) > 0 else 0
            mae_bars = int(np.nanargmax(highs)) + 1 if len(highs) > 0 else 0
        else:
            # 中性：使用绝对值
            mfe_pct = max(
                (float(np.nanmax(highs)) / entry_price - 1) * 100,
                (entry_price / float(np.nanmin(lows)) - 1) * 100 if np.nanmin(lows) > 0 else 0
            )
            mae_pct = 0
            mfe_bars = 0
            mae_bars = 0
        
        # 判断实际方向（24根K线后）
        if returns.get(24) is not None:
            ret_24 = returns[24]
            if ret_24 > 0.5:
                actual_dir = "up" if expected_dir != "short" else "down"
            elif ret_24 < -0.5:
                actual_dir = "down" if expected_dir != "short" else "up"
            else:
                actual_dir = "flat"
        else:
            actual_dir = "flat"
        
        # 判断方向是否正确
        is_correct = False
        if expected_dir == "long" and actual_dir == "up":
            is_correct = True
        elif expected_dir == "short" and actual_dir == "down":
            is_correct = True
        elif expected_dir == "neutral":
            is_correct = True  # 中性事件不判断对错
        
        # 检查止损/止盈是否触发
        stop_hit = False
        target_hit = False
        if stop_price is not None and target_price is not None:
            for _, row in future.iterrows():
                low = float(row["low"])
                high = float(row["high"])
                
                if expected_dir == "long":
                    if low <= stop_price:
                        stop_hit = True
                        break
                    if high >= target_price:
                        target_hit = True
                        break
                elif expected_dir == "short":
                    if high >= stop_price:
                        stop_hit = True
                        break
                    if low <= target_price:
                        target_hit = True
                        break
        
        # 生成备注
        notes = []
        if mfe_pct > 2:
            notes.append(f"MFE较大({mfe_pct:.1f}%在{mfe_bars}根后)")
        if mae_pct > 2:
            notes.append(f"MAE较大({mae_pct:.1f}%在{mae_bars}根后)")
        if target_hit:
            notes.append("达到目标位")
        if stop_hit:
            notes.append("触发止损")
        
        return EventPerformance(
            event=event,
            event_idx=idx,
            entry_price=entry_price,
            exit_price=float(self.df.iloc[end_idx - 1]["close"]) if end_idx > idx + 1 else None,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            mfe_bars=mfe_bars,
            mae_bars=mae_bars,
            return_6=returns.get(6),
            return_12=returns.get(12),
            return_24=returns.get(24),
            return_48=returns.get(48),
            expected_direction=expected_dir,
            actual_direction=actual_dir,
            is_correct=is_correct,
            stop_hit=stop_hit,
            target_hit=target_hit,
            stop_price=stop_price,
            target_price=target_price,
            notes=notes,
        )
    
    def evaluate_all(self) -> list[EventPerformance]:
        """评估所有事件"""
        results = []
        for event in self.events:
            perf = self.evaluate_single_event(event)
            if perf is not None:
                results.append(perf)
        return results
    
    def get_stats_by_type(self, performances: list[EventPerformance] | None = None) -> dict[str, EventTypeStats]:
        """
        按事件类型统计表现
        
        Returns:
            事件类型 -> EventTypeStats 的字典
        """
        if performances is None:
            performances = self.evaluate_all()
        
        # 按类型分组
        by_type: dict[str, list[EventPerformance]] = {}
        for perf in performances:
            t = perf.event.type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(perf)
        
        stats: dict[str, EventTypeStats] = {}
        
        for event_type, perfs in by_type.items():
            n = len(perfs)
            if n == 0:
                continue
            
            # 收集各周期收益
            ret_6 = [p.return_6 for p in perfs if p.return_6 is not None]
            ret_12 = [p.return_12 for p in perfs if p.return_12 is not None]
            ret_24 = [p.return_24 for p in perfs if p.return_24 is not None]
            ret_48 = [p.return_48 for p in perfs if p.return_48 is not None]
            
            mfes = [p.mfe_pct for p in perfs]
            maes = [p.mae_pct for p in perfs]
            confs = [p.event.confidence for p in perfs]
            
            # 计算胜率
            def win_rate(returns: list[float]) -> float:
                if not returns:
                    return 0.0
                wins = sum(1 for r in returns if r > 0)
                return wins / len(returns)
            
            # 计算盈亏比
            wins = [r for r in ret_24 if r > 0]
            losses = [r for r in ret_24 if r < 0]
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = abs(float(np.mean(losses))) if losses else 0.0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0
            
            # 方向正确率
            correct = sum(1 for p in perfs if p.is_correct)
            direction_accuracy = correct / n
            
            # 置信度与收益相关性
            if len(ret_24) >= 3 and len(confs) >= 3:
                conf_arr = np.array([p.event.confidence for p in perfs if p.return_24 is not None])
                ret_arr = np.array(ret_24)
                if len(conf_arr) == len(ret_arr) and len(conf_arr) > 2:
                    try:
                        corr = float(np.corrcoef(conf_arr, ret_arr)[0, 1])
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0
                else:
                    corr = 0.0
            else:
                corr = 0.0
            
            stats[event_type] = EventTypeStats(
                event_type=event_type,
                count=n,
                avg_return_6=float(np.mean(ret_6)) if ret_6 else 0.0,
                avg_return_12=float(np.mean(ret_12)) if ret_12 else 0.0,
                avg_return_24=float(np.mean(ret_24)) if ret_24 else 0.0,
                avg_return_48=float(np.mean(ret_48)) if ret_48 else 0.0,
                median_return_6=float(np.median(ret_6)) if ret_6 else 0.0,
                median_return_12=float(np.median(ret_12)) if ret_12 else 0.0,
                median_return_24=float(np.median(ret_24)) if ret_24 else 0.0,
                median_return_48=float(np.median(ret_48)) if ret_48 else 0.0,
                avg_mfe=float(np.mean(mfes)) if mfes else 0.0,
                avg_mae=float(np.mean(maes)) if maes else 0.0,
                median_mfe=float(np.median(mfes)) if mfes else 0.0,
                median_mae=float(np.median(maes)) if maes else 0.0,
                win_rate_6=win_rate(ret_6),
                win_rate_12=win_rate(ret_12),
                win_rate_24=win_rate(ret_24),
                win_rate_48=win_rate(ret_48),
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                direction_accuracy=direction_accuracy,
                confidence_correlation=corr,
            )
        
        return stats


def evaluate_events(
    df: pd.DataFrame,
    events: list[WyckoffEvent],
    **kwargs,
) -> tuple[list[EventPerformance], dict[str, EventTypeStats]]:
    """
    便捷函数：评估事件并返回结果和统计
    
    Args:
        df: K线数据
        events: 威科夫事件列表
        **kwargs: 传递给 EventEvaluator 的参数
    
    Returns:
        (事件表现列表, 按类型统计)
    """
    evaluator = EventEvaluator(df, events, **kwargs)
    performances = evaluator.evaluate_all()
    stats = evaluator.get_stats_by_type(performances)
    return performances, stats

