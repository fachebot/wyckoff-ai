"""
量价关系分析模块

核心概念：
1. 努力与结果（Effort vs Result）
   - 大努力小结果：放量但价格变化小 -> 吸筹/派发信号
   - 小努力大结果：缩量但价格变化大 -> 趋势延续或假突破

2. 量价背离（Volume Divergence）
   - 价格新高但成交量递减 -> 顶部信号
   - 价格新低但成交量递减 -> 底部信号

3. 吸筹/派发特征
   - 下跌放量上涨缩量 -> 派发
   - 上涨放量下跌缩量 -> 吸筹
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class EffortResultType(str, Enum):
    """努力与结果类型"""
    ABSORPTION = "absorption"      # 吸收：大努力小结果，有人在吸筹/派发
    EASY_MOVE = "easy_move"        # 轻松移动：小努力大结果
    CLIMAX = "climax"              # 高潮：大努力大结果
    NO_DEMAND = "no_demand"        # 无需求：小努力小结果（上涨时）
    NO_SUPPLY = "no_supply"        # 无供应：小努力小结果（下跌时）
    NORMAL = "normal"              # 正常


class DivergenceType(str, Enum):
    """背离类型"""
    BULLISH = "bullish"            # 看涨背离（价格新低，量能递减）
    BEARISH = "bearish"            # 看跌背离（价格新高，量能递减）
    HIDDEN_BULLISH = "hidden_bullish"  # 隐藏看涨（价格更高低点，量能更低）
    HIDDEN_BEARISH = "hidden_bearish"  # 隐藏看跌
    NONE = "none"


@dataclass
class EffortResultAnalysis:
    """努力与结果分析结果"""
    type: EffortResultType
    effort_score: float      # 努力得分（成交量相对强度）
    result_score: float      # 结果得分（价格变化幅度）
    imbalance: float         # 不平衡度（正=结果>努力，负=努力>结果）
    interpretation: str      # 解读


@dataclass
class DivergenceSignal:
    """背离信号"""
    type: DivergenceType
    strength: float          # 强度 0-1
    price_swing: float       # 价格波动幅度
    volume_change: float     # 量能变化比例
    bars_apart: int          # 两个极值点相隔K线数
    description: str


@dataclass
class AccumulationDistributionScore:
    """吸筹/派发评分"""
    score: float             # -1 (派发) 到 +1 (吸筹)
    up_volume_ratio: float   # 上涨时的成交量占比
    down_volume_ratio: float # 下跌时的成交量占比
    trend: Literal["accumulation", "distribution", "neutral"]
    evidence: list[str]


def analyze_effort_result(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
) -> EffortResultAnalysis:
    """
    分析单根K线的努力与结果关系
    
    Args:
        df: OHLCV + 特征数据
        idx: 要分析的K线索引
        lookback: 用于计算基准的回看周期
    
    Returns:
        EffortResultAnalysis: 分析结果
    """
    if idx < lookback or idx >= len(df):
        return EffortResultAnalysis(
            type=EffortResultType.NORMAL,
            effort_score=0.0,
            result_score=0.0,
            imbalance=0.0,
            interpretation="数据不足",
        )
    
    row = df.iloc[idx]
    recent = df.iloc[max(0, idx - lookback):idx]
    
    # 努力：成交量相对强度
    vol_z = float(row.get("vol_z_20", 0) or 0)
    effort_score = vol_z
    
    # 结果：价格变化相对于 ATR
    spread = float(row.get("spread", 0) or 0)
    atr = float(row.get("atr_14", 1) or 1)
    body = float(row.get("body", 0) or 0)
    
    # 使用 body/ATR 作为结果（排除影线的干扰）
    result_score = body / atr if atr > 0 else 0
    
    # 方向
    is_up = float(row.get("close", 0)) > float(row.get("open", 0))
    
    # 不平衡度
    imbalance = result_score - effort_score
    
    # 判断类型
    if effort_score > 1.5 and result_score < 0.6:
        # 大努力小结果
        er_type = EffortResultType.ABSORPTION
        interp = "放量但价格变化小，可能有大资金在吸收筹码" if is_up else "放量但跌幅有限，可能有大资金在承接"
    elif effort_score < 0.5 and result_score > 1.2:
        # 小努力大结果
        er_type = EffortResultType.EASY_MOVE
        interp = "缩量上涨，阻力小" if is_up else "缩量下跌，支撑弱"
    elif effort_score > 1.5 and result_score > 1.2:
        # 大努力大结果
        er_type = EffortResultType.CLIMAX
        interp = "放量大涨，可能是突破或高潮" if is_up else "放量大跌，可能是恐慌或出货"
    elif effort_score < 0.5 and result_score < 0.5:
        if is_up:
            er_type = EffortResultType.NO_DEMAND
            interp = "缩量小涨，需求不足"
        else:
            er_type = EffortResultType.NO_SUPPLY
            interp = "缩量小跌，供应不足"
    else:
        er_type = EffortResultType.NORMAL
        interp = "量价关系正常"
    
    return EffortResultAnalysis(
        type=er_type,
        effort_score=effort_score,
        result_score=result_score,
        imbalance=imbalance,
        interpretation=interp,
    )


def detect_volume_divergence(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 30,
    min_swing: float = 0.02,
) -> DivergenceSignal:
    """
    检测量价背离
    
    Args:
        df: OHLCV + 特征数据
        idx: 当前K线索引
        lookback: 回看周期
        min_swing: 最小波动幅度（相对价格）
    
    Returns:
        DivergenceSignal: 背离信号
    """
    if idx < lookback or idx >= len(df):
        return DivergenceSignal(
            type=DivergenceType.NONE,
            strength=0.0,
            price_swing=0.0,
            volume_change=0.0,
            bars_apart=0,
            description="数据不足",
        )
    
    window = df.iloc[max(0, idx - lookback):idx + 1]
    
    high = window["high"].astype(float)
    low = window["low"].astype(float)
    close = window["close"].astype(float)
    vol = window["volume"].astype(float)
    vol_z = window["vol_z_20"].astype(float) if "vol_z_20" in window.columns else vol / vol.rolling(20).mean()
    
    current_price = float(close.iloc[-1])
    current_vol_z = float(vol_z.iloc[-1]) if not pd.isna(vol_z.iloc[-1]) else 0
    
    # 找到 lookback 期内的高点和低点
    high_idx = high.idxmax()
    low_idx = low.idxmin()
    
    # 检测看跌背离：价格新高但量能递减
    if high.iloc[-1] >= high.max() * 0.995:  # 接近或创新高
        # 找前一个高点
        prev_highs = high[:-5].nlargest(3)
        if len(prev_highs) > 0:
            prev_high_idx = prev_highs.index[0]
            prev_high_price = float(high.loc[prev_high_idx])
            prev_high_vol = float(vol_z.loc[prev_high_idx]) if prev_high_idx in vol_z.index else 0
            
            price_change = (float(high.iloc[-1]) - prev_high_price) / prev_high_price
            vol_change = current_vol_z - prev_high_vol
            bars = len(window) - window.index.get_loc(prev_high_idx) - 1
            
            # 价格更高但量能更低
            if price_change > 0 and vol_change < -0.3:
                strength = min(1.0, abs(vol_change) / 2)
                return DivergenceSignal(
                    type=DivergenceType.BEARISH,
                    strength=strength,
                    price_swing=price_change,
                    volume_change=vol_change,
                    bars_apart=bars,
                    description=f"看跌背离：价格上涨{price_change*100:.1f}%但量能下降",
                )
    
    # 检测看涨背离：价格新低但量能递减
    if low.iloc[-1] <= low.min() * 1.005:  # 接近或创新低
        # 找前一个低点
        prev_lows = low[:-5].nsmallest(3)
        if len(prev_lows) > 0:
            prev_low_idx = prev_lows.index[0]
            prev_low_price = float(low.loc[prev_low_idx])
            prev_low_vol = float(vol_z.loc[prev_low_idx]) if prev_low_idx in vol_z.index else 0
            
            price_change = (float(low.iloc[-1]) - prev_low_price) / prev_low_price
            vol_change = current_vol_z - prev_low_vol
            bars = len(window) - window.index.get_loc(prev_low_idx) - 1
            
            # 价格更低但量能更低
            if price_change < 0 and vol_change < -0.3:
                strength = min(1.0, abs(vol_change) / 2)
                return DivergenceSignal(
                    type=DivergenceType.BULLISH,
                    strength=strength,
                    price_swing=price_change,
                    volume_change=vol_change,
                    bars_apart=bars,
                    description=f"看涨背离：价格下跌{abs(price_change)*100:.1f}%但量能下降",
                )
    
    return DivergenceSignal(
        type=DivergenceType.NONE,
        strength=0.0,
        price_swing=0.0,
        volume_change=0.0,
        bars_apart=0,
        description="无明显背离",
    )


def calculate_accumulation_distribution(
    df: pd.DataFrame,
    lookback: int = 30,
) -> AccumulationDistributionScore:
    """
    计算吸筹/派发评分
    
    原理：
    - 吸筹：上涨时放量，下跌时缩量
    - 派发：下跌时放量，上涨时缩量
    
    Args:
        df: OHLCV + 特征数据
        lookback: 回看周期
    
    Returns:
        AccumulationDistributionScore: 评分结果
    """
    if len(df) < lookback:
        return AccumulationDistributionScore(
            score=0.0,
            up_volume_ratio=0.5,
            down_volume_ratio=0.5,
            trend="neutral",
            evidence=["数据不足"],
        )
    
    recent = df.tail(lookback).copy()
    
    # 计算每根K线的方向
    recent["is_up"] = recent["close"] > recent["open"]
    recent["is_down"] = recent["close"] < recent["open"]
    
    # 获取成交量
    vol = recent["volume"].astype(float)
    vol_z = recent["vol_z_20"].astype(float) if "vol_z_20" in recent.columns else vol / vol.mean()
    
    # 上涨K线的成交量
    up_mask = recent["is_up"]
    down_mask = recent["is_down"]
    
    up_volume = vol[up_mask].sum()
    down_volume = vol[down_mask].sum()
    total_volume = up_volume + down_volume
    
    if total_volume == 0:
        return AccumulationDistributionScore(
            score=0.0,
            up_volume_ratio=0.5,
            down_volume_ratio=0.5,
            trend="neutral",
            evidence=["成交量为零"],
        )
    
    up_ratio = up_volume / total_volume
    down_ratio = down_volume / total_volume
    
    # 计算加权得分（考虑波动幅度）
    up_spread = (recent.loc[up_mask, "spread"] * vol[up_mask]).sum() if up_mask.any() else 0
    down_spread = (recent.loc[down_mask, "spread"] * vol[down_mask]).sum() if down_mask.any() else 0
    
    # 吸筹得分：上涨放量程度 - 下跌放量程度
    # 范围 -1 到 +1
    score = (up_ratio - down_ratio)
    
    # 证据收集
    evidence = []
    
    # 检测连续模式
    up_vol_avg = vol_z[up_mask].mean() if up_mask.any() else 0
    down_vol_avg = vol_z[down_mask].mean() if down_mask.any() else 0
    
    if up_vol_avg > down_vol_avg + 0.3:
        evidence.append(f"上涨平均量能({up_vol_avg:.2f})高于下跌({down_vol_avg:.2f})")
        score += 0.1
    elif down_vol_avg > up_vol_avg + 0.3:
        evidence.append(f"下跌平均量能({down_vol_avg:.2f})高于上涨({up_vol_avg:.2f})")
        score -= 0.1
    
    # 检测最近趋势
    recent_5 = df.tail(5)
    recent_up = (recent_5["close"] > recent_5["open"]).sum()
    recent_down = (recent_5["close"] < recent_5["open"]).sum()
    
    if recent_up >= 4:
        evidence.append("近期连续上涨")
    elif recent_down >= 4:
        evidence.append("近期连续下跌")
    
    # 确定趋势
    if score > 0.15:
        trend = "accumulation"
        evidence.append("整体呈现吸筹特征")
    elif score < -0.15:
        trend = "distribution"
        evidence.append("整体呈现派发特征")
    else:
        trend = "neutral"
        evidence.append("量价关系中性")
    
    return AccumulationDistributionScore(
        score=max(-1.0, min(1.0, score)),
        up_volume_ratio=up_ratio,
        down_volume_ratio=down_ratio,
        trend=trend,
        evidence=evidence,
    )


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算量价分析相关特征，添加到 DataFrame
    """
    out = df.copy()
    n = len(out)
    
    # 基础列
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    
    # 1. 量价相关性（滚动）
    out["vol_price_corr_20"] = close.rolling(20).corr(volume)
    
    # 2. OBV (On Balance Volume)
    price_change = close.diff()
    obv = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
    out["obv"] = np.cumsum(obv)
    out["obv_ma_20"] = out["obv"].rolling(20).mean()
    out["obv_divergence"] = out["obv"] - out["obv_ma_20"]
    
    # 3. 成交量加权平均价 (VWAP) - 简化版
    typical_price = (high + low + close) / 3
    out["vwap_20"] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    out["price_vs_vwap"] = (close - out["vwap_20"]) / out["vwap_20"]
    
    # 4. 吸筹/派发线 (A/D Line)
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    out["ad_line"] = (clv * volume).cumsum()
    
    # 5. 量能潮 (Volume Flow)
    out["vol_flow_20"] = np.where(
        close > close.shift(1),
        volume.rolling(20).sum(),
        -volume.rolling(20).sum()
    )
    
    # 6. 努力结果比
    body = (close - out["open"]).abs()
    spread = (high - low).abs()
    atr = out["atr_14"] if "atr_14" in out.columns else spread.rolling(14).mean()
    vol_z = out["vol_z_20"] if "vol_z_20" in out.columns else (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    
    effort = vol_z.clip(lower=0.1)  # 避免除零
    result = (body / atr).clip(lower=0.1)
    out["effort_result_ratio"] = result / effort
    
    # 7. 量价背离指标
    # 价格创新高/新低时的量能变化
    price_high_20 = high.rolling(20).max()
    price_low_20 = low.rolling(20).min()
    vol_at_high = volume.where(high >= price_high_20 * 0.995).rolling(5).mean()
    vol_at_low = volume.where(low <= price_low_20 * 1.005).rolling(5).mean()
    
    out["vol_at_highs"] = vol_at_high
    out["vol_at_lows"] = vol_at_low
    
    # 8. 上涨/下跌成交量比率
    up_vol = volume.where(close > out["open"], 0)
    down_vol = volume.where(close < out["open"], 0)
    out["up_down_vol_ratio"] = up_vol.rolling(10).sum() / down_vol.rolling(10).sum().replace(0, np.nan)
    
    return out


def get_volume_context_for_event(
    df: pd.DataFrame,
    idx: int,
    event_type: str,
) -> dict:
    """
    获取特定事件的量价上下文分析
    
    Args:
        df: OHLCV + 特征数据
        idx: 事件发生的K线索引
        event_type: 事件类型
    
    Returns:
        dict: 量价上下文信息
    """
    context = {
        "confirms": [],       # 确认因素
        "warns": [],          # 警告因素
        "confidence_adj": 0.0,  # 置信度调整
    }
    
    if idx < 20 or idx >= len(df):
        return context
    
    # 分析努力与结果
    er = analyze_effort_result(df, idx)
    
    # 分析背离
    div = detect_volume_divergence(df, idx)
    
    # 计算吸筹/派发
    window = df.iloc[max(0, idx - 30):idx + 1]
    ad = calculate_accumulation_distribution(window)
    
    # 根据事件类型判断量价是否支持
    if event_type == "SC":
        # 卖出高潮应该是大努力（放量）
        if er.type == EffortResultType.CLIMAX:
            context["confirms"].append("放量下跌符合SC特征")
            context["confidence_adj"] += 0.15
        elif er.type == EffortResultType.ABSORPTION:
            context["confirms"].append("放量但跌幅有限，可能有承接")
            context["confidence_adj"] += 0.10
        
        # 看涨背离增强SC信号
        if div.type == DivergenceType.BULLISH:
            context["confirms"].append(f"量价看涨背离({div.strength*100:.0f}%)")
            context["confidence_adj"] += 0.15
    
    elif event_type == "BC":
        # 买入高潮应该是大努力（放量）
        if er.type == EffortResultType.CLIMAX:
            context["confirms"].append("放量上涨符合BC特征")
            context["confidence_adj"] += 0.15
        elif er.type == EffortResultType.ABSORPTION:
            context["confirms"].append("放量但涨幅有限，可能有出货")
            context["confidence_adj"] += 0.10
        
        # 看跌背离增强BC信号
        if div.type == DivergenceType.BEARISH:
            context["confirms"].append(f"量价看跌背离({div.strength*100:.0f}%)")
            context["confidence_adj"] += 0.15
    
    elif event_type == "SPRING":
        # Spring 应该是缩量下探后反弹
        if er.type == EffortResultType.NO_SUPPLY:
            context["confirms"].append("缩量下跌，供应不足")
            context["confidence_adj"] += 0.15
        elif er.effort_score < 0.8:
            context["confirms"].append("下探时量能不大")
            context["confidence_adj"] += 0.08
        
        # 如果是吸筹趋势
        if ad.trend == "accumulation":
            context["confirms"].append("整体呈吸筹特征")
            context["confidence_adj"] += 0.10
    
    elif event_type in ("UT", "UTAD"):
        # UT 应该是放量上冲后回落
        if er.type == EffortResultType.ABSORPTION:
            context["confirms"].append("放量上冲但未能维持")
            context["confidence_adj"] += 0.12
        
        # 看跌背离增强UT信号
        if div.type == DivergenceType.BEARISH:
            context["confirms"].append(f"量价看跌背离({div.strength*100:.0f}%)")
            context["confidence_adj"] += 0.12
        
        # 如果是派发趋势
        if ad.trend == "distribution":
            context["confirms"].append("整体呈派发特征")
            context["confidence_adj"] += 0.10
    
    elif event_type == "SOS":
        # SOS 应该是放量突破
        if er.effort_score > 1.0:
            context["confirms"].append("放量突破")
            context["confidence_adj"] += 0.15
        else:
            context["warns"].append("突破时量能不足")
            context["confidence_adj"] -= 0.10
    
    elif event_type == "SOW":
        # SOW 应该是放量下跌
        if er.effort_score > 1.0:
            context["confirms"].append("放量下跌")
            context["confidence_adj"] += 0.15
        else:
            context["warns"].append("下跌时量能不足")
            context["confidence_adj"] -= 0.10
    
    elif event_type == "ST":
        # ST 应该是缩量回测
        if er.effort_score < 0.8:
            context["confirms"].append("缩量回测")
            context["confidence_adj"] += 0.10
        elif er.effort_score > 1.5:
            context["warns"].append("回测时量能偏大")
            context["confidence_adj"] -= 0.08
    
    elif event_type == "LPS":
        # LPS 应该是缩量回踩
        if er.effort_score < 0.6:
            context["confirms"].append("缩量回踩，需求仍在")
            context["confidence_adj"] += 0.12
    
    elif event_type == "LPSY":
        # LPSY 应该是缩量反弹
        if er.effort_score < 0.6:
            context["confirms"].append("缩量反弹，供应仍在")
            context["confidence_adj"] += 0.12
    
    return context

