from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wyckoff_ai.exceptions import FeatureComputationError, InsufficientDataError
from wyckoff_ai.logging import get_logger, log_execution_time
from wyckoff_ai.regime import RegimeConfig, add_regime_columns

logger = get_logger("features")


def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    """
    Rolling linear regression slope vs x=0..window-1, in units of y per bar.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        return float(((x - x_mean) * (arr - y_mean)).sum() / denom)

    return y.rolling(window).apply(lambda a: slope(a.values), raw=False)


@log_execution_time(logger_name="features")
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: timestamp, open, high, low, close, volume, is_gap
    Output: adds feature columns, keeps original columns.
    Note: gaps remain NaN; most features will be NaN on those rows.
    
    Raises:
        InsufficientDataError: 数据量不足
        FeatureComputationError: 特征计算失败
    """
    logger.debug(f"开始计算特征，输入 {len(df)} 行数据")
    
    # 验证输入
    if df.empty:
        raise InsufficientDataError("输入数据为空", required=1, actual=0)
    
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise FeatureComputationError(
            f"缺少必要列: {missing_cols}",
            feature="input_validation",
        )
    
    try:
        out = df.copy()

        o = out["open"].astype(float)
        h = out["high"].astype(float)
        l = out["low"].astype(float)
        c = out["close"].astype(float)
        v = out["volume"].astype(float)
    except Exception as e:
        raise FeatureComputationError(
            f"数据类型转换失败: {e}",
            feature="type_conversion",
            cause=e,
        )

    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(
        axis=1
    )
    out["tr"] = tr
    out["atr_14"] = tr.rolling(14).mean()

    out["ret_1"] = c.pct_change()
    out["log_ret_1"] = np.log(c / prev_c)

    out["spread"] = (h - l).abs()
    out["body"] = (c - o).abs()
    out["upper_wick"] = (h - np.maximum(o, c)).clip(lower=0)
    out["lower_wick"] = (np.minimum(o, c) - l).clip(lower=0)
    out["close_pos_in_range"] = np.where(
        out["spread"] > 0, (c - l) / out["spread"], np.nan
    )

    vol_mean = v.rolling(20).mean()
    vol_std = v.rolling(20).std(ddof=0)
    out["vol_z_20"] = (v - vol_mean) / vol_std.replace({0: np.nan})

    # Effort vs result: high effort (volume) but poor result (small body or close off highs/lows)
    # +1 means effort with positive result, -1 means effort with negative result
    er = (out["body"] / out["spread"].replace({0: np.nan})).clip(0, 1)
    dir_ = np.sign(c - o).replace({0: np.nan})
    out["effort_result"] = (out["vol_z_20"] * er * dir_).astype(float)

    # Trend proxy: slope of close + EMA
    out["ema_50"] = c.ewm(span=50, adjust=False).mean()
    out["slope_50"] = _rolling_slope(c, 50)

    # Pivot highs/lows (fractal): center is max/min within k bars on both sides
    k = 3
    roll_high = h.rolling(2 * k + 1, center=True).max()
    roll_low = l.rolling(2 * k + 1, center=True).min()
    out["pivot_high"] = np.where(h == roll_high, h, np.nan)
    out["pivot_low"] = np.where(l == roll_low, l, np.nan)

    # Multi-scale pivots for "分型级别嵌套"（轻量：固定几个尺度）
    for kk in (5, 8):
        rh = h.rolling(2 * kk + 1, center=True).max()
        rl = l.rolling(2 * kk + 1, center=True).min()
        out[f"pivot_high_{kk}"] = np.where(h == rh, h, np.nan)
        out[f"pivot_low_{kk}"] = np.where(l == rl, l, np.nan)

    # Donchian channel width (range-ness)
    dc_n = 50
    dc_high = h.rolling(dc_n).max()
    dc_low = l.rolling(dc_n).min()
    out["donchian_high_50"] = dc_high
    out["donchian_low_50"] = dc_low
    out["donchian_width_50"] = (dc_high - dc_low) / c.replace({0: np.nan})

    # Additional horizons help detect composite structures
    for nn in (100, 200):
        dh = h.rolling(nn).max()
        dl = l.rolling(nn).min()
        out[f"donchian_high_{nn}"] = dh
        out[f"donchian_low_{nn}"] = dl
        out[f"donchian_width_{nn}"] = (dh - dl) / c.replace({0: np.nan})

    # Simple range flag: narrow channel + low atr
    # 放宽阈值以适应高波动资产（如 BTC）
    atr_pct = out["atr_14"] / c.replace({0: np.nan})
    out["atr_pct_14"] = atr_pct
    # 多级别区间检测：宽松 + 严格
    out["is_range_like"] = (out["donchian_width_50"] < 0.12) & (atr_pct < 0.035)
    out["is_range_strict"] = (out["donchian_width_50"] < 0.06) & (atr_pct < 0.015)
    
    # 基于波动率收敛的区间检测（相对历史）
    dcw_ma = out["donchian_width_50"].rolling(30).mean()
    atr_ma = atr_pct.rolling(30).mean()
    out["is_range_relative"] = (
        (out["donchian_width_50"] < dcw_ma * 1.2) &
        (atr_pct < atr_ma * 1.2) &
        (out["donchian_width_50"] < 0.18)
    )

    # A few helpful booleans
    out["is_up_bar"] = c > o
    out["is_down_bar"] = c < o

    # === 量价分析特征 ===
    
    # 1. OBV (On Balance Volume)
    price_change = c.diff()
    obv = np.where(price_change > 0, v, np.where(price_change < 0, -v, 0))
    out["obv"] = np.cumsum(obv)
    out["obv_ma_20"] = out["obv"].rolling(20).mean()
    out["obv_divergence"] = out["obv"] - out["obv_ma_20"]
    
    # 2. 量价相关性（滚动）
    out["vol_price_corr_20"] = c.rolling(20).corr(v)
    
    # 3. 上涨/下跌成交量比率
    up_vol = v.where(c > o, 0)
    down_vol = v.where(c < o, 0)
    out["up_down_vol_ratio"] = up_vol.rolling(10).sum() / down_vol.rolling(10).sum().replace(0, np.nan)
    
    # 4. 努力结果比（改进版）
    effort = out["vol_z_20"].clip(lower=0.1)
    result = (out["body"] / out["atr_14"].replace(0, np.nan)).clip(lower=0.1)
    out["effort_result_ratio"] = result / effort
    
    # 5. 量价背离检测特征
    # 价格创新高/新低时的量能水平
    price_high_20 = h.rolling(20).max()
    price_low_20 = l.rolling(20).min()
    vol_ma_5 = v.rolling(5).mean()
    
    # 新高时的量能相对于之前新高时的量能
    at_high = h >= price_high_20 * 0.998
    at_low = l <= price_low_20 * 1.002
    
    out["vol_at_highs"] = vol_ma_5.where(at_high)
    out["vol_at_lows"] = vol_ma_5.where(at_low)
    
    # 6. 背离标记（简化版）
    # 看跌背离：价格创新高但量能递减
    vol_z_ma = out["vol_z_20"].rolling(5).mean()
    vol_declining = vol_z_ma < vol_z_ma.shift(5)
    out["bearish_divergence"] = at_high & vol_declining
    
    # 看涨背离：价格创新低但量能递减
    out["bullish_divergence"] = at_low & vol_declining
    
    # 7. 吸筹/派发指标
    # 基于收盘位置的资金流向
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)  # Close Location Value
    out["ad_line"] = (clv * v).cumsum()  # Accumulation/Distribution Line
    out["ad_line_ma_20"] = out["ad_line"].rolling(20).mean()
    
    # 8. 相对强度指标
    # 上涨成交量 vs 下跌成交量的累积
    up_vol_cum = up_vol.rolling(20).sum()
    down_vol_cum = down_vol.rolling(20).sum()
    total_vol = up_vol_cum + down_vol_cum
    out["accumulation_score"] = (up_vol_cum - down_vol_cum) / total_vol.replace(0, np.nan)

    # Clean infinities
    for col in ["log_ret_1", "effort_result_ratio", "up_down_vol_ratio"]:
        if col in out.columns:
            out[col] = out[col].replace([math.inf, -math.inf], np.nan)

    # Regime / state inference (no extra deps). Safe no-op if not enough clean rows.
    try:
        out = add_regime_columns(out, RegimeConfig(method="kmeans"))
    except Exception as e:
        logger.warning(f"Regime 计算失败，跳过: {e}")
        out["regime_id"] = None
        out["regime_hint"] = None
        out["is_range_regime"] = False

    feature_count = len(out.columns) - len(df.columns)
    logger.info(f"特征计算完成，新增 {feature_count} 个特征列")
    
    return out


