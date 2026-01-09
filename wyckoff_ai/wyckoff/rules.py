from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wyckoff_ai.schemas import (
    KeyLevels,
    ProbabilisticScenario,
    RangeInfo,
    Scenario,
    StateMachineInfo,
    TradingSignal,
    WyckoffAnalysis,
    WyckoffEvent,
)
from wyckoff_ai.wyckoff.volume_analysis import (
    get_volume_context_for_event,
    analyze_effort_result,
    detect_volume_divergence,
    calculate_accumulation_distribution,
    EffortResultType,
    DivergenceType,
)
from wyckoff_ai.wyckoff.context import (
    validate_event_context,
    enhance_event_with_context,
    filter_conflicting_events,
    calculate_sequence_coherence,
)
from wyckoff_ai.wyckoff.state_machine import (
    WyckoffStateMachine,
    analyze_with_state_machine,
)
from wyckoff_ai.wyckoff.scenario_engine import (
    calculate_scenario_probability,
    ProbabilisticScenario as ProbScenario,
    TradingSignal as TradingSignalData,
)


@dataclass(frozen=True)
class DetectionConfig:
    strict: bool = False
    lookback_bars: int = 220
    # event scan controls - 放宽最小区间长度
    min_range_bars: int = 20  # 从 48 降低到 20，更容易识别短期区间
    # 区间段比例法（walk-forward）：前段用于定义箱体，后段扫描事件
    build_ratio: float = 0.60  # 从 0.70 降低到 0.60，增加扫描期
    min_scan_bars: int = 8  # 从 12 降低到 8
    # Keep reasonably high so "list every UT/SPRING inside a range" won't be truncated.
    max_events: int = 2000
    min_event_separation_bars: int = 5  # 从 8 降低到 5，允许更密集的事件
    
    # 事件检测阈值（可调整敏感度）
    sc_vol_z_threshold: float = 1.5  # SC 放量阈值（原 2.0）
    sc_spread_atr_ratio: float = 1.3  # SC 波动阈值（原 1.6）
    bc_vol_z_threshold: float = 1.5  # BC 放量阈值
    bc_spread_atr_ratio: float = 1.3  # BC 波动阈值
    spring_vol_z_threshold: float = 0.8  # SPRING 放量阈值（原 1.2）
    ut_vol_z_threshold: float = 1.2  # UT 放量阈值（原 1.5）
    sow_sos_vol_z_threshold: float = 0.8  # SOS/SOW 放量阈值（原 1.0）
    
    # 量价分析和上下文验证开关
    use_volume_analysis: bool = True     # 是否使用量价分析增强
    use_context_validation: bool = True  # 是否使用上下文验证
    filter_conflicts: bool = True        # 是否过滤矛盾事件
    min_confidence_threshold: float = 0.45  # 最低置信度阈值
    
    # 状态机开关
    use_state_machine: bool = True       # 是否使用状态机进行阶段识别


def _merge_bool_gaps(base: np.ndarray, *, max_gap: int = 3) -> np.ndarray:
    """
    Fill short False gaps inside a True segment to reduce jitter.
    """
    s = base.copy().astype(bool)
    gap = 0
    for i in range(len(s)):
        if s[i]:
            gap = 0
            continue
        gap += 1
        if gap <= max_gap:
            left_true = (i - gap) >= 0 and bool(s[i - gap])
            right_true = (i + 1) < len(s) and bool(base[i + 1])
            if left_true and right_true:
                s[i] = True
    return s


def _segments_from_mask(mask: np.ndarray, *, min_len: int) -> list[tuple[int, int]]:
    """
    Return inclusive segments [start, end] where mask is True, with length >= min_len.
    """
    segs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if (j - i + 1) >= min_len:
            segs.append((i, j))
        i = j + 1
    return segs


def _dedup_events(
    events_with_idx: list[tuple[int, WyckoffEvent]],
    *,
    min_sep: int,
    no_sep_types: set[str] | None = None,
) -> list[tuple[int, WyckoffEvent]]:
    """
    Dedup by (type, ts), and enforce minimal separation (bars) per type.
    """
    no_sep_types = no_sep_types or set()
    # sort by idx asc
    events_with_idx = sorted(events_with_idx, key=lambda x: x[0])
    out: list[tuple[int, WyckoffEvent]] = []
    last_by_type: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for idx, ev in events_with_idx:
        key = (ev.type, ev.ts)
        if key in seen:
            continue
        if ev.type not in no_sep_types:
            prev = last_by_type.get(ev.type)
            if prev is not None and (idx - prev) < min_sep:
                continue
        out.append((idx, ev))
        seen.add(key)
        last_by_type[ev.type] = idx
    return out


def _forward_stats(
    d: pd.DataFrame,
    events_with_idx: list[tuple[int, WyckoffEvent]],
    *,
    horizons: tuple[int, ...] = (12, 24, 48),
) -> dict[str, dict[str, float]]:
    close = d["close"].astype(float).to_numpy()
    stats: dict[str, dict[str, float]] = {}
    by_type: dict[str, list[int]] = {}
    for idx, ev in events_with_idx:
        by_type.setdefault(ev.type, []).append(int(idx))
    for t, idxs in by_type.items():
        rows: dict[int, list[float]] = {h: [] for h in horizons}
        wins: dict[int, int] = {h: 0 for h in horizons}
        n = 0
        for i in idxs:
            if not np.isfinite(close[i]):
                continue
            n += 1
            for h in horizons:
                j = i + h
                if j >= len(close) or not np.isfinite(close[j]) or close[i] == 0:
                    continue
                r = float(close[j] / close[i] - 1.0)
                rows[h].append(r)
                if r > 0:
                    wins[h] += 1
        if n <= 0:
            continue
        s: dict[str, float] = {"n": float(n)}
        for h in horizons:
            arr = np.array(rows[h], dtype=float)
            if arr.size > 0:
                s[f"r{h}_med"] = float(np.nanmedian(arr))
                s[f"win{h}"] = float(wins[h] / max(1, arr.size))
        stats[t] = s
    return stats


def _asof_ts(df: pd.DataFrame) -> str:
    ts = df["timestamp"].iloc[-1]
    return ts.isoformat()


def _find_latest_range(features: pd.DataFrame, min_bars: int = 20) -> RangeInfo:
    """
    多策略区间检测：
    1. 基于 regime 聚类
    2. 基于 is_range_like 特征
    3. 基于相对波动率收敛
    4. 基于价格振幅收窄
    """
    n = len(features)
    if n < min_bars:
        return RangeInfo()
    
    # 策略1: regime-based
    base_reg = np.zeros(n, dtype=bool)
    if "is_range_regime" in features.columns:
        base_reg = features["is_range_regime"].fillna(False).to_numpy()
    
    # 策略2: is_range_like (已放宽的特征)
    base_like = features["is_range_like"].fillna(False).to_numpy() if "is_range_like" in features.columns else np.zeros(n, dtype=bool)
    
    # 策略3: 相对波动率收敛
    base_rel = features["is_range_relative"].fillna(False).to_numpy() if "is_range_relative" in features.columns else np.zeros(n, dtype=bool)
    
    # 策略4: 基于价格振幅的简单检测（新增）
    if "high" in features.columns and "low" in features.columns:
        h = features["high"].astype(float)
        l = features["low"].astype(float)
        c = features["close"].astype(float)
        # 计算滚动振幅比例
        roll_high = h.rolling(min_bars).max()
        roll_low = l.rolling(min_bars).min()
        amplitude = (roll_high - roll_low) / c
        # 振幅小于 15% 视为潜在区间
        base_amplitude = (amplitude < 0.15).fillna(False).to_numpy()
    else:
        base_amplitude = np.zeros(n, dtype=bool)
    
    # 组合策略：任一为 True 即可（提高敏感度）
    combined = base_reg | base_like | base_rel | base_amplitude
    
    # Merge small gaps inside a range segment to reduce jitter
    s = _merge_bool_gaps(combined, max_gap=5)  # 增加 gap 容忍度

    if len(s) == 0 or not np.any(s):
        # 降级策略：如果没有识别到区间，尝试找最近的低波动段
        return _find_range_by_volatility(features, min_bars)

    # walk backwards to find latest contiguous True segment
    end = len(s) - 1
    while end >= 0 and not s[end]:
        end -= 1
    if end < 0:
        return _find_range_by_volatility(features, min_bars)

    start = end
    while start >= 0 and s[start]:
        start -= 1
    start += 1

    dur = end - start + 1
    if dur < min_bars:
        # 尝试降级策略
        return _find_range_by_volatility(features, min_bars)

    seg = features.iloc[start: end + 1]
    low = float(seg["low"].min())
    high = float(seg["high"].max())
    mid = (low + high) / 2.0
    return RangeInfo(
        low=low,
        high=high,
        mid=mid,
        duration_bars=int(dur),
        start_ts=seg["timestamp"].iloc[0].isoformat(),
        end_ts=seg["timestamp"].iloc[-1].isoformat(),
    )


def _find_range_by_volatility(features: pd.DataFrame, min_bars: int = 20) -> RangeInfo:
    """
    降级策略：基于波动率最低的滑动窗口来识别潜在区间
    """
    if len(features) < min_bars:
        return RangeInfo()
    
    c = features["close"].astype(float)
    h = features["high"].astype(float)
    l = features["low"].astype(float)
    
    # 计算滚动波动率（使用 ATR%）
    if "atr_pct_14" in features.columns:
        vol = features["atr_pct_14"].astype(float)
    else:
        tr = pd.concat([
            (h - l).abs(),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        vol = tr.rolling(14).mean() / c
    
    # 找波动率最低的连续区间
    vol_smooth = vol.rolling(min_bars).mean()
    if vol_smooth.isna().all():
        return RangeInfo()
    
    # 找最近的低波动区间（波动率低于中位数的 1.2 倍）
    vol_threshold = vol_smooth.median() * 1.2
    low_vol_mask = vol_smooth <= vol_threshold
    
    if not low_vol_mask.any():
        return RangeInfo()
    
    # 从后往前找连续的低波动段
    mask_arr = low_vol_mask.fillna(False).to_numpy()
    s = _merge_bool_gaps(mask_arr, max_gap=3)
    
    end = len(s) - 1
    while end >= 0 and not s[end]:
        end -= 1
    if end < 0:
        return RangeInfo()
    
    start = end
    while start >= 0 and s[start]:
        start -= 1
    start += 1
    
    dur = end - start + 1
    if dur < min_bars:
        return RangeInfo()
    
    seg = features.iloc[start: end + 1]
    low_val = float(seg["low"].min())
    high_val = float(seg["high"].max())
    mid_val = (low_val + high_val) / 2.0
    return RangeInfo(
        low=low_val,
        high=high_val,
        mid=mid_val,
        duration_bars=int(dur),
        start_ts=seg["timestamp"].iloc[0].isoformat(),
        end_ts=seg["timestamp"].iloc[-1].isoformat(),
    )


def _trend_label(features: pd.DataFrame) -> str:
    # Very simple: slope_50 sign + position vs ema_50
    slope = features["slope_50"].iloc[-1]
    c = features["close"].iloc[-1]
    ema = features["ema_50"].iloc[-1]
    if pd.isna(slope) or pd.isna(ema) or pd.isna(c):
        return "unknown"
    if slope > 0 and c >= ema:
        return "up"
    if slope < 0 and c <= ema:
        return "down"
    return "sideways"


def _confidence(strict: bool, base: float, evidence: list[str]) -> float:
    """
    计算置信度，基于基础分数和证据数量。
    
    证据权重规则：
    - 前 2 条证据各加 0.05
    - 第 3-4 条证据各加 0.04
    - 第 5+ 条证据各加 0.02
    - 包含"确认"/"站稳"/"放量"关键词的证据额外加 0.02
    """
    bonus = 0.0
    for i, ev in enumerate(evidence):
        if i < 2:
            bonus += 0.05
        elif i < 4:
            bonus += 0.04
        else:
            bonus += 0.02
        # 关键证据加分
        key_words = ["确认", "站稳", "放量", "反弹", "回落", "跳空"]
        if any(kw in ev for kw in key_words):
            bonus += 0.02
    
    conf = min(1.0, max(0.0, base + bonus))
    if strict:
        # Strict mode: 降低置信度但不要过度惩罚
        conf = max(0.0, conf - 0.10)
    return float(conf)


def detect_wyckoff(
    features: pd.DataFrame,
    *,
    symbol: str,
    exchange: str,
    timeframe: str,
    cfg: DetectionConfig | None = None,
) -> WyckoffAnalysis:
    cfg = cfg or DetectionConfig()
    df = features.dropna(subset=["close"]).copy()
    if len(df) < 120:
        return WyckoffAnalysis(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            asof_ts=_asof_ts(features),
            market_structure="unknown",
            risk_notes=["数据长度不足（建议至少 200 根）"],
        )

    lb = min(cfg.lookback_bars, len(df))
    d = df.iloc[-lb:].copy()

    rng = _find_latest_range(d)
    trend = _trend_label(d)

    market_structure = "unknown"
    if rng.duration_bars > 0:
        # crude: use prior trend before range to label accumulation/distribution
        pre = d.iloc[: max(1, len(d) - rng.duration_bars)]
        pre_trend = _trend_label(pre) if len(pre) >= 60 else "unknown"
        if pre_trend == "down":
            market_structure = "accumulation"
        elif pre_trend == "up":
            market_structure = "distribution"
        else:
            market_structure = "range"
    else:
        market_structure = "markup" if trend == "up" else (
            "markdown" if trend == "down" else "unknown")

    risk_notes: list[str] = []
    events_with_idx: list[tuple[int, WyckoffEvent]] = []
    sc_pos: int | None = None
    st_pos: int | None = None
    spring_pos: int | None = None

    # Helper series
    atr = d["atr_14"]
    volz = d["vol_z_20"]
    spread = d["spread"]
    close = d["close"]
    high = d["high"]
    low = d["low"]

    # --- SC (Selling Climax) candidates across the whole window ---
    down_ctx = d["slope_50"] < 0
    # 多级别检测：严格 + 宽松
    sc_mask_strict = (
        down_ctx
        & (volz > cfg.sc_vol_z_threshold * 1.3)
        & (spread > cfg.sc_spread_atr_ratio * 1.2 * atr)
        & (d["close_pos_in_range"] > 0.50)
    )
    sc_mask_loose = (
        down_ctx
        & (volz > cfg.sc_vol_z_threshold)
        & (spread > cfg.sc_spread_atr_ratio * atr)
        & (d["close_pos_in_range"] > 0.45)
    )
    # 增加：基于价格创新低 + 量价背离的 SC 候选
    price_new_low = low == low.rolling(20).min()
    vol_divergence = volz.rolling(5).mean() < volz.shift(5).rolling(5).mean()
    sc_mask_divergence = down_ctx & price_new_low & vol_divergence & (spread > atr)
    
    sc_mask = sc_mask_strict | sc_mask_loose | sc_mask_divergence
    sc_all = np.where(sc_mask.fillna(False).to_numpy())[0].tolist()
    # keep spaced SCs to avoid over-labeling
    last_keep = -10_000
    sc_kept: list[int] = []
    for i in sc_all:
        if i - last_keep < 15:  # 从 20 降低到 15
            continue
        sc_kept.append(int(i))
        last_keep = int(i)
    for i in sc_kept:
        ev = ["下跌趋势背景(slope_50<0)"]
        if float(volz.iloc[i] or 0) > cfg.sc_vol_z_threshold:
            ev.append(f"放量(vol_z_20>{cfg.sc_vol_z_threshold:.1f})")
        if float(spread.iloc[i] or 0) > cfg.sc_spread_atr_ratio * float(atr.iloc[i] or 0):
            ev.append(f"大波动(spread>{cfg.sc_spread_atr_ratio:.1f}*ATR)")
        if bool(price_new_low.iloc[i]):
            ev.append("价格创20日新低")
        if i + 5 < len(d):
            rebound = close.iloc[i + 1: i +
                                 6].max() > (close.iloc[i] + 0.6 * atr.iloc[i])
            if bool(rebound):
                ev.append("随后快速反弹(<=5根)")
        # 根据证据数量调整基础置信度
        base_conf = 0.50 + 0.05 * min(len(ev) - 1, 4)
        conf = _confidence(cfg.strict, base_conf, ev)
        if (not cfg.strict) or conf >= 0.50:
            events_with_idx.append(
                (
                    int(i),
                    WyckoffEvent(
                        type="SC",
                        ts=d["timestamp"].iloc[i].isoformat(),
                        price=float(low.iloc[i]),
                        confidence=conf,
                        evidence=ev,
                    ),
                )
            )

        # AR after this SC
        j0 = i + 1
        j1 = min(len(d), i + 26)
        if j0 < j1:
            seg = d.iloc[j0:j1]
            j_label = seg["high"].idxmax()
            jj = int(d.index.get_loc(j_label))
            ev_ar = ["SC 后 1~25 根内形成反弹高点"]
            if close.iloc[jj] > close.iloc[i]:
                ev_ar.append("价格抬升(AR高于SC收盘)")
            conf_ar = _confidence(cfg.strict, 0.50, ev_ar)
            if (not cfg.strict) or conf_ar >= 0.55:
                events_with_idx.append(
                    (
                        int(jj),
                        WyckoffEvent(
                            type="AR",
                            ts=d["timestamp"].iloc[jj].isoformat(),
                            price=float(high.iloc[jj]),
                            confidence=conf_ar,
                            evidence=ev_ar,
                        ),
                    )
                )

        # ST after this SC
        sc_low = float(low.iloc[i])
        sc_volz = float(volz.iloc[i] or 0)
        buf = float(np.nanmedian(atr.tail(50))) * 0.35
        w0 = i + 5
        w1 = min(len(d), i + 61)
        if w0 < w1:
            seg = d.iloc[w0:w1]
            cond = (
                (seg["low"] <= (sc_low + buf))
                & (seg["spread"] <= 1.25 * seg["atr_14"])
                & (seg["close_pos_in_range"] >= 0.50)
            )
            if "vol_z_20" in seg.columns:
                cond = cond & (seg["vol_z_20"] <= max(1.2, sc_volz - 0.6))
            idx = np.where(cond.fillna(False).to_numpy())[0]
            if len(idx) > 0:
                j = int(idx[-1])
                jj = int(seg.index[j])
                st_pos = int(d.index.get_loc(jj))
                ev_st = ["SC 后回测低位(ST候选)", "低位接盘迹象(收盘不贴地)"]
                if float(seg["vol_z_20"].iloc[j] or 0) < sc_volz:
                    ev_st.append("相对缩量(vol_z_20低于SC)")
                if float(seg["spread"].iloc[j] or 0) < float(spread.iloc[i] or 0):
                    ev_st.append("波动收敛(spread较SC收敛)")
                conf_st = _confidence(cfg.strict, 0.52, ev_st)
                if (not cfg.strict) or conf_st >= 0.55:
                    events_with_idx.append(
                        (
                            int(st_pos),
                            WyckoffEvent(
                                type="ST",
                                ts=d["timestamp"].iloc[st_pos].isoformat(),
                                price=float(low.iloc[st_pos]),
                                confidence=conf_st,
                                evidence=ev_st,
                            ),
                        )
                    )

    # SC/AR/ST above are detected across the whole window; do not re-detect only latest.

    # --- BC (Buying Climax) candidates across the whole window ---
    # BC 是上涨趋势末期的放量冲高后回落
    up_ctx = d["slope_50"] > 0
    # 多级别检测
    bc_mask_strict = (
        up_ctx
        & (volz > cfg.bc_vol_z_threshold * 1.3)
        & (spread > cfg.bc_spread_atr_ratio * 1.2 * atr)
        & (d["close_pos_in_range"] < 0.50)
    )
    bc_mask_loose = (
        up_ctx
        & (volz > cfg.bc_vol_z_threshold)
        & (spread > cfg.bc_spread_atr_ratio * atr)
        & (d["close_pos_in_range"] < 0.55)
    )
    # 基于价格创新高 + 量价背离
    price_new_high = high == high.rolling(20).max()
    bc_mask_divergence = up_ctx & price_new_high & vol_divergence & (spread > atr)
    
    bc_mask = bc_mask_strict | bc_mask_loose | bc_mask_divergence
    bc_all = np.where(bc_mask.fillna(False).to_numpy())[0].tolist()
    last_keep = -10_000
    bc_kept: list[int] = []
    for i in bc_all:
        if i - last_keep < 15:
            continue
        bc_kept.append(int(i))
        last_keep = int(i)
    for i in bc_kept:
        ev = ["上涨趋势背景(slope_50>0)"]
        if float(volz.iloc[i] or 0) > cfg.bc_vol_z_threshold:
            ev.append(f"放量(vol_z_20>{cfg.bc_vol_z_threshold:.1f})")
        if float(spread.iloc[i] or 0) > cfg.bc_spread_atr_ratio * float(atr.iloc[i] or 0):
            ev.append(f"大波动(spread>{cfg.bc_spread_atr_ratio:.1f}*ATR)")
        if bool(price_new_high.iloc[i]):
            ev.append("价格创20日新高")
        if float(d["close_pos_in_range"].iloc[i] or 0) < 0.50:
            ev.append("收盘偏弱(上影线)")
        if i + 5 < len(d):
            pullback = close.iloc[i + 1: i + 6].min() < (close.iloc[i] - 0.6 * atr.iloc[i])
            if bool(pullback):
                ev.append("随后快速回落(<=5根)")
        base_conf = 0.50 + 0.05 * min(len(ev) - 1, 4)
        conf = _confidence(cfg.strict, base_conf, ev)
        if (not cfg.strict) or conf >= 0.50:
            events_with_idx.append(
                (
                    int(i),
                    WyckoffEvent(
                        type="BC",
                        ts=d["timestamp"].iloc[i].isoformat(),
                        price=float(high.iloc[i]),
                        confidence=conf,
                        evidence=ev,
                    ),
                )
            )

        # AR (Automatic Reaction) after BC
        j0 = i + 1
        j1 = min(len(d), i + 26)
        if j0 < j1:
            seg = d.iloc[j0:j1]
            j_label = seg["low"].idxmin()
            jj = int(d.index.get_loc(j_label))
            ev_ar = ["BC 后 1~25 根内形成回调低点"]
            if close.iloc[jj] < close.iloc[i]:
                ev_ar.append("价格回落(AR低于BC收盘)")
            conf_ar = _confidence(cfg.strict, 0.50, ev_ar)
            if (not cfg.strict) or conf_ar >= 0.50:
                events_with_idx.append(
                    (
                        int(jj),
                        WyckoffEvent(
                            type="AR",
                            ts=d["timestamp"].iloc[jj].isoformat(),
                            price=float(low.iloc[jj]),
                            confidence=conf_ar,
                            evidence=ev_ar,
                        ),
                    )
                )

    # --- PSY (Preliminary Supply/Support) 检测 ---
    # PSY 是趋势末期的初步信号，通常伴随放量但未形成完整的高潮
    # PSY-S (供应): 上涨中出现的初步卖压
    psy_s_mask = (
        up_ctx &
        (volz > 1.0) &
        (d["close_pos_in_range"] < 0.55) &
        (d["is_down_bar"].fillna(False))
    )
    psy_s_all = np.where(psy_s_mask.fillna(False).to_numpy())[0].tolist()
    last_keep = -10_000
    for i in psy_s_all:
        if i - last_keep < 20:
            continue
        last_keep = int(i)
        ev = ["上涨趋势中出现卖压(PSY)", "放量阴线", "收盘偏弱"]
        if float(spread.iloc[i] or 0) > atr.iloc[i]:
            ev.append("波动放大(spread>ATR)")
        conf = _confidence(cfg.strict, 0.45, ev)
        if (not cfg.strict) or conf >= 0.45:
            events_with_idx.append(
                (
                    int(i),
                    WyckoffEvent(
                        type="PSY",
                        ts=d["timestamp"].iloc[i].isoformat(),
                        price=float(high.iloc[i]),
                        confidence=conf,
                        evidence=ev,
                    ),
                )
            )

    # PSY-D (支撑): 下跌中出现的初步买盘
    psy_d_mask = (
        down_ctx &
        (volz > 1.0) &
        (d["close_pos_in_range"] > 0.45) &
        (d["is_up_bar"].fillna(False))
    )
    psy_d_all = np.where(psy_d_mask.fillna(False).to_numpy())[0].tolist()
    last_keep = -10_000
    for i in psy_d_all:
        if i - last_keep < 20:
            continue
        last_keep = int(i)
        ev = ["下跌趋势中出现买盘(PSY)", "放量阳线", "收盘偏强"]
        if float(spread.iloc[i] or 0) > atr.iloc[i]:
            ev.append("波动放大(spread>ATR)")
        conf = _confidence(cfg.strict, 0.45, ev)
        if (not cfg.strict) or conf >= 0.45:
            events_with_idx.append(
                (
                    int(i),
                    WyckoffEvent(
                        type="PSY",
                        ts=d["timestamp"].iloc[i].isoformat(),
                        price=float(low.iloc[i]),
                        confidence=conf,
                        evidence=ev,
                    ),
                )
            )

    # --- Range-based events (SOS/UT/SOW) ---
    levels = KeyLevels()
    if rng.duration_bars > 0 and rng.low is not None and rng.high is not None:
        levels.support.extend([rng.low, rng.mid])
        levels.resistance.extend([rng.high, rng.mid])

    # Scan ALL range-like segments in the window for range-based events.
    base_like = d["is_range_like"].fillna(False).to_numpy(
    ) if "is_range_like" in d.columns else np.zeros(len(d), dtype=bool)
    if "is_range_regime" in d.columns:
        base_reg = d["is_range_regime"].fillna(False).to_numpy()
        base = base_reg if bool(np.any(base_reg)) else base_like
    else:
        base = base_like
    mask = _merge_bool_gaps(base, max_gap=3)
    segs = _segments_from_mask(mask, min_len=int(cfg.min_range_bars))
    for a, b in segs:
        seg = d.iloc[a:b + 1]
        if seg.empty:
            continue

        # --- 区间段比例法（walk-forward） ---
        # 用 build 期定义箱体（避免用到段内未来信息），只在 scan 期扫描事件。
        seg_len = int(b - a + 1)
        # ensure we leave enough scan bars
        build_len = int(min(seg_len - int(cfg.min_scan_bars),
                        max(int(cfg.min_range_bars), int(seg_len * float(cfg.build_ratio)))))
        if build_len < int(cfg.min_range_bars) or (seg_len - build_len) < int(cfg.min_scan_bars):
            continue
        build_end = a + build_len - 1
        scan_start = build_end + 1
        scan_end = b
        build = d.iloc[a:build_end + 1]
        scan = d.iloc[scan_start:scan_end + 1]
        if build.empty or scan.empty:
            continue

        r_low = float(build["low"].min())
        r_high = float(build["high"].max())
        r_mid = (r_low + r_high) / 2.0
        buffer_ = float(np.nanmedian(build["atr_14"].tail(50))) * 0.25
        upper = r_high + buffer_
        lower = r_low - buffer_

        # SPRING within this segment（放宽检测条件）
        atr_med = float(np.nanmedian(build["atr_14"].tail(50)))
        low_thr = r_low - max(buffer_, atr_med * 0.08)  # 从 0.15 降低到 0.08
        # 多级别检测
        spring_mask_strict = (scan["low"] < low_thr) & (scan["close"] > r_low)
        # 宽松：低点接近下沿 + 收盘不破
        spring_mask_loose = (scan["low"] <= r_low) & (scan["close"] > r_low - buffer_ * 0.5)
        # 假跌破后快速收回
        spring_mask_false_break = (
            (scan["low"] < r_low) & 
            (scan["close"] > r_low) & 
            (scan["close_pos_in_range"] > 0.40)
        )
        spring_mask = spring_mask_strict | spring_mask_loose | spring_mask_false_break
        spring_idx = np.where(spring_mask.fillna(False).to_numpy())[0].tolist()
        for j in spring_idx:
            i = scan_start + int(j)
            row = d.iloc[i]
            ev = ["下破/触及区间下沿但收回(SPRING形态)"]
            if float(row.get("vol_z_20", 0) or 0) > cfg.spring_vol_z_threshold:
                ev.append(f"放量下探(vol_z_20>{cfg.spring_vol_z_threshold:.1f})")
            if float(row.get("close_pos_in_range", 0) or 0) > 0.45:
                ev.append("收盘位置较强(不贴地)")
            # 下影线检测
            lower_wick = float(row.get("lower_wick", 0) or 0)
            spread_val = float(row.get("spread", 0) or 0)
            if spread_val > 0 and lower_wick / spread_val > 0.4:
                ev.append("明显下影线(>40%)")
            # rebound confirmation
            j0 = i + 1
            j1 = min(len(d), i + 9)
            base_conf = 0.48
            if j0 < j1:
                fut = d.iloc[j0:j1]
                rebound = (fut["close"] > r_mid).any() or (fut["high"] > (
                    float(row["close"]) + 0.4 * float(row["atr_14"] or 0))).any()
                if bool(rebound):
                    ev.append("随后快速反弹(<=8根)")
                    base_conf = 0.55
            conf = _confidence(cfg.strict, base_conf, ev)
            if (not cfg.strict) or conf >= 0.50:
                events_with_idx.append(
                    (
                        int(i),
                        WyckoffEvent(
                            type="SPRING",
                            ts=row["timestamp"].isoformat(),
                            price=float(row["low"]),
                            confidence=conf,
                            evidence=ev,
                        ),
                    )
                )

            # TEST after this SPRING (3~15 bars), prefer low effort + tight spread
            t0 = i + 3
            t1 = min(len(d), i + 16)
            if t0 < t1:
                fut = d.iloc[t0:t1]
                spr_low = float(row["low"])
                tol = max(buffer_, float(np.nanmedian(
                    build["atr_14"].tail(50))) * 0.25)
                cond = (
                    (fut["low"] >= (spr_low - tol))
                    & (fut["low"] <= (spr_low + tol))
                    & (fut["vol_z_20"] < 0.25)
                    & (fut["spread"] <= 1.05 * fut["atr_14"])
                    & (fut["close_pos_in_range"] >= 0.55)
                )
                idx2 = np.where(cond.fillna(False).to_numpy())[0]
                if len(idx2) > 0:
                    k = int(idx2[-1])
                    kk = fut.index[k]
                    pos = int(d.index.get_loc(kk))
                    ev2 = ["SPRING 后回测(TEST)", "缩量(vol_z_20<0.25)",
                           "波动收敛(spread<=1.05*ATR)"]
                    conf2 = _confidence(cfg.strict, 0.56, ev2)
                    if (not cfg.strict) or conf2 >= 0.58:
                        events_with_idx.append(
                            (
                                int(pos),
                                WyckoffEvent(
                                    type="TEST",
                                    ts=d["timestamp"].iloc[pos].isoformat(),
                                    price=float(d["low"].iloc[pos]),
                                    confidence=conf2,
                                    evidence=ev2,
                                ),
                            )
                        )

        # SOS / SOW inside segment（仅扫描 scan 期）- 放宽检测条件
        # Compare with previous bar (global) but require current within scan period
        for i in range(max(scan_start, a + 1), scan_end + 1):
            last = d.iloc[i]
            prev = d.iloc[i - 1]
            
            # SOS 检测（多级别）
            sos_strict = float(last["close"]) > upper and float(prev["close"]) <= upper
            sos_loose = float(last["close"]) > r_high and float(last["high"]) > upper
            sos_momentum = (
                float(last["close"]) > r_high and 
                float(last.get("slope_50", 0) or 0) > 0 and
                float(last.get("vol_z_20", 0) or 0) > 0.5
            )
            
            if sos_strict or sos_loose or sos_momentum:
                ev = ["上破区间上沿"]
                if float(last["close"]) > upper:
                    ev[0] = "有效上破区间上沿(含ATR缓冲)"
                if float(last.get("vol_z_20", 0) or 0) > cfg.sow_sos_vol_z_threshold:
                    ev.append(f"放量配合(vol_z_20>{cfg.sow_sos_vol_z_threshold:.1f})")
                if float(last.get("is_up_bar", False)):
                    ev.append("阳线收盘")
                # 检查后续确认
                if i + 3 < len(d):
                    fut = d.iloc[i+1:i+4]
                    if (fut["close"] > r_high).all():
                        ev.append("后续站稳上沿上方")
                base_conf = 0.52 if sos_strict else 0.48
                conf = _confidence(cfg.strict, base_conf, ev)
                if (not cfg.strict) or conf >= 0.50:
                    events_with_idx.append(
                        (
                            int(i),
                            WyckoffEvent(
                                type="SOS",
                                ts=last["timestamp"].isoformat(),
                                price=float(last["close"]),
                                confidence=conf,
                                evidence=ev,
                            ),
                        )
                    )

            # SOW 检测（多级别）
            sow_strict = float(last["close"]) < lower and float(prev["close"]) >= lower
            sow_loose = float(last["close"]) < r_low and float(last["low"]) < lower
            sow_momentum = (
                float(last["close"]) < r_low and 
                float(last.get("slope_50", 0) or 0) < 0 and
                float(last.get("vol_z_20", 0) or 0) > 0.5
            )
            
            if sow_strict or sow_loose or sow_momentum:
                ev = ["下破区间下沿"]
                if float(last["close"]) < lower:
                    ev[0] = "有效下破区间下沿(含ATR缓冲)"
                if float(last.get("vol_z_20", 0) or 0) > cfg.sow_sos_vol_z_threshold:
                    ev.append(f"放量走弱(vol_z_20>{cfg.sow_sos_vol_z_threshold:.1f})")
                if float(last.get("is_down_bar", False)):
                    ev.append("阴线收盘")
                base_conf = 0.52 if sow_strict else 0.48
                conf = _confidence(cfg.strict, base_conf, ev)
                if (not cfg.strict) or conf >= 0.50:
                    events_with_idx.append(
                        (
                            int(i),
                            WyckoffEvent(
                                type="SOW",
                                ts=last["timestamp"].isoformat(),
                                price=float(last["close"]),
                                confidence=conf,
                                evidence=ev,
                            ),
                        )
                    )

        # UT / UTAD inside segment（仅扫描 scan 期）- 放宽检测条件
        high_thr = r_high + max(buffer_, atr_med * 0.08)  # 从 0.15 降低到 0.08
        weak_mid = r_mid - max(buffer_, atr_med * 0.10)
        weak_low = r_low - max(buffer_, atr_med * 0.05)
        
        # 多级别检测
        ut_mask_strict = (scan["high"] > high_thr) & (scan["close"] < r_high)
        ut_mask_loose = (scan["high"] >= r_high) & (scan["close"] < r_high + buffer_ * 0.5)
        # 假突破：上冲后收盘回区间内
        ut_mask_false_break = (
            (scan["high"] > r_high) & 
            (scan["close"] < r_high) & 
            (scan["close_pos_in_range"] < 0.60)
        )
        ut_mask = ut_mask_strict | ut_mask_loose | ut_mask_false_break
        ut_idx = np.where(ut_mask.fillna(False).to_numpy())[0].tolist()
        for j in ut_idx:
            i = scan_start + int(j)
            row = d.iloc[i]
            ev = ["上冲区间上沿但收回(UT形态)"]
            if float(row.get("vol_z_20", 0) or 0) > cfg.ut_vol_z_threshold:
                ev.append(f"放量上冲(vol_z_20>{cfg.ut_vol_z_threshold:.1f})")
            if float(row.get("spread", 0) or 0) > 1.1 * float(row.get("atr_14", 0) or 0):
                ev.append("大波动(spread>1.1*ATR)")
            # 上影线检测
            upper_wick = float(row.get("upper_wick", 0) or 0)
            spread_val = float(row.get("spread", 0) or 0)
            if spread_val > 0 and upper_wick / spread_val > 0.4:
                ev.append("明显上影线(>40%)")
            j0 = i + 1
            j1 = min(len(d), i + 13)
            weak = False
            if j0 < j1:
                fut = d.iloc[j0:j1]
                weak = (fut["close"] < weak_mid).any() or (fut["low"] < weak_low).any()
            if weak:
                ev.append("随后出现结构性走弱(回落破中轴/下沿)")
                base_conf = 0.55
            else:
                base_conf = 0.48
            conf = _confidence(cfg.strict, base_conf, ev)
            if (not cfg.strict) or conf >= 0.48:
                events_with_idx.append(
                    (
                        int(i),
                        WyckoffEvent(
                            type="UTAD" if market_structure == "distribution" else "UT",
                            ts=row["timestamp"].isoformat(),
                            price=float(row["high"]),
                            confidence=conf,
                            evidence=ev,
                        ),
                    )
                )

        # LPS after SOS: pullback near upper with lower vol
        sos_positions = [
            idx for idx, ev in events_with_idx if ev.type == "SOS" and scan_start <= idx <= scan_end]
        for i_s in sos_positions[-3:]:
            i0 = i_s + 3
            i1 = min(len(d), i_s + 26)
            if i0 >= i1:
                continue
            fut = d.iloc[i0:i1]
            buf2 = max(buffer_, float(
                np.nanmedian(build["atr_14"].tail(50))) * 0.20)
            # elementwise threshold: use np.maximum instead of python max (which breaks on Series)
            near_thr = np.maximum(buf2, (fut["atr_14"] * 0.4).astype(float))
            near = (fut["low"] - r_high).abs() <= near_thr
            cond = near & (fut["vol_z_20"] <
                           0.5) & fut["is_up_bar"].fillna(False)
            idx = np.where(cond.fillna(False).to_numpy())[0]
            if len(idx) > 0:
                j = int(idx[-1])
                jj = fut.index[j]
                pos = int(d.index.get_loc(jj))
                ev = ["SOS 后回踩上沿附近止跌(LPS)", "缩量(vol_z_20<0.5)", "阳线收盘"]
                conf = _confidence(cfg.strict, 0.55, ev)
                if (not cfg.strict) or conf >= 0.60:
                    events_with_idx.append(
                        (
                            int(pos),
                            WyckoffEvent(
                                type="LPS",
                                ts=d["timestamp"].iloc[pos].isoformat(),
                                price=float(d["low"].iloc[pos]),
                                confidence=conf,
                                evidence=ev,
                            ),
                        )
                    )

        # LPSY after SOW: weak rally to breakdown area then fails
        sow_positions = [
            idx for idx, ev in events_with_idx if ev.type == "SOW" and scan_start <= idx <= scan_end]
        for i_sow in sow_positions[-3:]:
            i0 = i_sow + 3
            i1 = min(len(d), i_sow + 21)
            if i0 >= i1:
                continue
            fut = d.iloc[i0:i1]
            near_break = fut["high"] >= (
                r_low - max(buffer_, float(np.nanmedian(build["atr_14"].tail(50))) * 0.15))
            fail_reclaim = fut["close"] < r_low
            low_effort = fut["vol_z_20"] < 0.6
            bear_bar = fut["is_down_bar"].fillna(False)
            cond = near_break & fail_reclaim & low_effort & bear_bar
            idx = np.where(cond.fillna(False).to_numpy())[0]
            if len(idx) > 0:
                j = int(idx[-1])
                jj = fut.index[j]
                pos = int(d.index.get_loc(jj))
                ev = ["SOW 后弱势回抽失败(LPSY)", "缩量(vol_z_20<0.6)", "收盘无法收复下沿"]
                conf = _confidence(cfg.strict, 0.55, ev)
                if (not cfg.strict) or conf >= 0.50:
                    events_with_idx.append(
                        (
                            int(pos),
                            WyckoffEvent(
                                type="LPSY",
                                ts=d["timestamp"].iloc[pos].isoformat(),
                                price=float(d["high"].iloc[pos]),
                                confidence=conf,
                                evidence=ev,
                            ),
                        )
                    )

        # JAC (Jump Across the Creek) 检测
        # JAC 是从区间向上突破的强势跳跃，通常伴随放量
        sos_positions_jac = [
            idx for idx, ev in events_with_idx if ev.type == "SOS" and scan_start <= idx <= scan_end]
        for i_sos in sos_positions_jac[-3:]:
            row = d.iloc[i_sos]
            # JAC 需要明显的放量和突破幅度
            if float(row.get("vol_z_20", 0) or 0) > 1.0:
                gap_up = float(row["low"]) > float(d.iloc[i_sos - 1]["high"]) if i_sos > 0 else False
                strong_close = float(row.get("close_pos_in_range", 0) or 0) > 0.70
                if gap_up or strong_close:
                    ev_jac = ["强势上破区间(JAC)", "放量突破(vol_z_20>1)"]
                    if gap_up:
                        ev_jac.append("跳空高开")
                    if strong_close:
                        ev_jac.append("收盘强势(上方)")
                    conf_jac = _confidence(cfg.strict, 0.55, ev_jac)
                    if (not cfg.strict) or conf_jac >= 0.52:
                        events_with_idx.append(
                            (
                                int(i_sos),
                                WyckoffEvent(
                                    type="JAC",
                                    ts=row["timestamp"].isoformat(),
                                    price=float(row["close"]),
                                    confidence=conf_jac,
                                    evidence=ev_jac,
                                ),
                            )
                        )

        # BUEC (Backup to Edge of Creek) 检测
        # BUEC 是 JAC 后回踩区间上沿确认支撑
        jac_positions = [
            idx for idx, ev in events_with_idx if ev.type == "JAC" and scan_start <= idx <= scan_end]
        for i_jac in jac_positions[-3:]:
            i0 = i_jac + 2
            i1 = min(len(d), i_jac + 16)
            if i0 >= i1:
                continue
            fut = d.iloc[i0:i1]
            # 回踩到区间上沿附近
            near_upper = (fut["low"] - r_high).abs() <= buffer_ * 1.5
            # 不破区间上沿
            hold_upper = fut["close"] >= r_high
            # 缩量
            low_vol = fut["vol_z_20"] < 0.5
            cond = near_upper & hold_upper & low_vol
            idx_buec = np.where(cond.fillna(False).to_numpy())[0]
            if len(idx_buec) > 0:
                j = int(idx_buec[-1])
                jj = fut.index[j]
                pos = int(d.index.get_loc(jj))
                ev_buec = ["JAC 后回踩确认(BUEC)", "回踩区间上沿", "缩量确认", "站稳不破"]
                conf_buec = _confidence(cfg.strict, 0.55, ev_buec)
                if (not cfg.strict) or conf_buec >= 0.52:
                    events_with_idx.append(
                        (
                            int(pos),
                            WyckoffEvent(
                                type="BUEC",
                                ts=d["timestamp"].iloc[pos].isoformat(),
                                price=float(d["low"].iloc[pos]),
                                confidence=conf_buec,
                                evidence=ev_buec,
                            ),
                        )
                    )

        # Composite structure hint on this segment
        if "donchian_width_200" in d.columns and "donchian_width_50" in d.columns:
            w50 = float(d["donchian_width_50"].iloc[b] or 0)
            w200 = float(d["donchian_width_200"].iloc[b] or 0)
            if w50 > 0 and w200 > 0 and w50 < 0.07 and w200 > 0.12:
                risk_notes.append(
                    "检测到复合结构迹象：小级别区间(50)嵌套在更宽的大级别通道(200)内，注意假突破与结构分层。")

    # Finalize events: dedup + cap
    # For UT/UTAD/SPRING we want to list every occurrence inside range segments
    # 新增事件类型也需要特殊处理
    events_with_idx = _dedup_events(
        events_with_idx,
        min_sep=int(cfg.min_event_separation_bars),
        no_sep_types={"UT", "UTAD", "SPRING", "PSY", "JAC", "BUEC"},
    )
    if len(events_with_idx) > int(cfg.max_events):
        events_with_idx = events_with_idx[-int(cfg.max_events):]
    
    # === 量价分析增强 ===
    if cfg.use_volume_analysis:
        enhanced_events_with_idx = []
        for idx, event in events_with_idx:
            # 获取量价上下文
            vol_context = get_volume_context_for_event(d, idx, event.type)
            
            # 调整置信度
            new_conf = event.confidence + vol_context["confidence_adj"]
            new_conf = max(0.0, min(1.0, new_conf))
            
            # 增强证据
            new_evidence = list(event.evidence)
            for confirm in vol_context["confirms"]:
                new_evidence.append(f"[量价]{confirm}")
            for warn in vol_context["warns"]:
                new_evidence.append(f"[量价警告]{warn}")
            
            enhanced_event = WyckoffEvent(
                type=event.type,
                ts=event.ts,
                price=event.price,
                confidence=new_conf,
                evidence=new_evidence,
            )
            enhanced_events_with_idx.append((idx, enhanced_event))
        events_with_idx = enhanced_events_with_idx
    
    # === 上下文验证增强 ===
    if cfg.use_context_validation:
        validated_events_with_idx = []
        validated_events = []  # 用于累积已验证的事件
        
        for idx, event in events_with_idx:
            # 验证上下文
            validation = validate_event_context(event, validated_events, d)
            
            # 调整置信度
            new_conf = event.confidence + validation.confidence_adjustment
            new_conf = max(0.0, min(1.0, new_conf))
            
            # 增强证据
            new_evidence = list(event.evidence)
            if validation.supporting_events:
                new_evidence.append(f"[上下文支持]{', '.join(validation.supporting_events[:3])}")
            if validation.conflicting_events:
                new_evidence.append(f"[上下文冲突]{', '.join(validation.conflicting_events[:2])}")
            
            # 只保留有效且置信度足够的事件
            if validation.is_valid and new_conf >= cfg.min_confidence_threshold:
                enhanced_event = WyckoffEvent(
                    type=event.type,
                    ts=event.ts,
                    price=event.price,
                    confidence=new_conf,
                    evidence=new_evidence,
                )
                validated_events_with_idx.append((idx, enhanced_event))
                validated_events.append(enhanced_event)
        
        events_with_idx = validated_events_with_idx
    
    # === 过滤矛盾事件 ===
    if cfg.filter_conflicts:
        events_only = [e for _, e in events_with_idx]
        filtered_events = filter_conflicting_events(events_only, d)
        # 重建 events_with_idx
        filtered_set = {(e.type, e.ts) for e in filtered_events}
        events_with_idx = [
            (idx, e) for idx, e in events_with_idx
            if (e.type, e.ts) in filtered_set
        ]
    
    events = [e for _i, e in events_with_idx]
    # Pivot levels (last ~10 pivots)
    piv_hi = d["pivot_high"].dropna().tail(8).tolist()
    piv_lo = d["pivot_low"].dropna().tail(8).tolist()
    levels.pivot = sorted({float(x) for x in (piv_hi + piv_lo)})

    # Scenarios (based on structure + recent events + local forward stats)
    scenarios: list[Scenario] = []
    fstats = _forward_stats(d, events_with_idx)
    recent = [ev for idx, ev in events_with_idx if idx >= (len(d) - 60)]
    recent_types = {e.type for e in recent}
    if rng.duration_bars > 0 and rng.low is not None and rng.high is not None:
        bull_notes = "偏趋势跟随；严格模式下可等待回踩确认。"
        if any(t in recent_types for t in ("SOS", "LPS", "SPRING", "TEST")) and fstats:
            s = fstats.get("SOS") or fstats.get("SPRING") or {}
            if "r24_med" in s:
                bull_notes = f"同窗统计：相关事件后 24 根中位收益约 {float(s['r24_med'])*100:.2f}%（样本 n={int(s.get('n', 0))}）。"
        bear_notes = "偏防守；注意假跌破与消息波动。"
        if any(t in recent_types for t in ("UTAD", "SOW", "LPSY")) and fstats:
            s = fstats.get("SOW") or fstats.get("UTAD") or {}
            if "r24_med" in s:
                bear_notes = f"同窗统计：相关事件后 24 根中位收益约 {float(s['r24_med'])*100:.2f}%（样本 n={int(s.get('n', 0))}）。"
        scenarios.append(
            Scenario(
                bias="bullish",
                confirmation=f"收盘有效站上 {rng.high:.2f} 并回踩不破（SOS→LPS/TEST）",
                invalidation=f"回落跌破区间中轴 {rng.mid:.2f} 或下沿 {rng.low:.2f}（结构转弱）",
                notes=bull_notes,
            )
        )
        scenarios.append(
            Scenario(
                bias="bearish",
                confirmation=f"收盘有效跌破 {rng.low:.2f} 且反抽不过（SOW→LPSY / UTAD→走弱）",
                invalidation=f"重新站回区间中轴 {rng.mid:.2f} 上方",
                notes=bear_notes,
            )
        )
    else:
        scenarios.append(
            Scenario(
                bias="neutral",
                confirmation="等待形成清晰区间或出现结构性拐点（SC/AR/ST 或 SOS/SOW/SPRING/UTAD）",
                invalidation="无",
                notes="当前结构不够明确，减少主观推断。",
            )
        )

    if int(features["is_gap"].fillna(False).sum()) > 0:
        risk_notes.append("存在缺口K线（已标记 is_gap=true），特征与事件置信度需打折。")
    if float(d["atr_pct_14"].iloc[-1] or 0) > 0.03:
        risk_notes.append("波动偏高（ATR% 较大），假突破/扫损风险上升。")
    if float(d["volume"].tail(50).median() or 0) == 0:
        risk_notes.append("近期成交量中位数为 0，数据质量可能异常。")

    # 状态机分析
    sm_info = None
    if cfg.use_state_machine and events:
        sm_result = analyze_with_state_machine(events, d)
        
        # 转换为 schema
        sm_info = StateMachineInfo(
            current_state=sm_result.current_state.value,
            state_description=sm_result.state_description,
            state_confidence=sm_result.state_confidence,
            phase_progress=sm_result.phase_progress.progress,
            events_in_phase=sm_result.phase_progress.events_in_phase,
            missing_events=sm_result.phase_progress.missing_events,
            next_expected_events=sm_result.phase_progress.next_expected,
            time_in_phase_bars=sm_result.phase_progress.time_in_phase_bars,
            bias=sm_result.bias,
            bias_confidence=sm_result.bias_confidence,
            recent_transitions=[
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "trigger": t.trigger_event.type,
                    "confidence": t.confidence,
                    "reason": t.reason,
                }
                for t in sm_result.transition_history[-5:]
            ],
            next_probable_states=[
                {"state": s.value, "probability": p}
                for s, p in sm_result.next_probable_states
            ],
            predicted_events=sm_result.predicted_events,
            risk_level=sm_result.risk_level,
            risk_factors=sm_result.risk_factors,
            action_suggestion=sm_result.action_suggestion,
            key_levels=sm_result.key_levels,
            phase_notes=sm_result.phase_progress.notes,
        )
        
        # 将状态机的风险因素添加到主风险备注
        for factor in sm_result.risk_factors:
            if factor not in risk_notes:
                risk_notes.append(f"[状态机] {factor}")

    # 概率化剧本分析
    probabilistic_scenarios: list[ProbabilisticScenario] = []
    
    # 构建临时分析对象用于计算概率化剧本
    temp_analysis = WyckoffAnalysis(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        asof_ts=_asof_ts(features),
        market_structure=market_structure,
        range=rng,
        events=events,
        levels=levels,
        scenarios=scenarios,
        risk_notes=risk_notes,
        state_machine=sm_info,
        event_forward_stats=fstats,
    )
    
    try:
        # 获取序列分析
        from wyckoff_ai.wyckoff.sequence import analyze_sequence
        sequence_result = analyze_sequence(events) if events else None
    except Exception:
        sequence_result = None
    
    # 获取当前价格
    current_price = None
    if events:
        current_price = events[-1].price
    elif rng.mid is not None:
        current_price = rng.mid
    
    if current_price is not None:
        try:
            # 计算概率化剧本
            prob_scenarios = calculate_scenario_probability(
                analysis=temp_analysis,
                sequence=sequence_result,
                current_price=current_price,
            )
            
            # 转换为 schema
            for ps in prob_scenarios:
                signal_schema = None
                if ps.signal:
                    signal_schema = TradingSignal(
                        entry_price=ps.signal.entry_price,
                        entry_condition=ps.signal.entry_condition,
                        stop_loss=ps.signal.stop_loss,
                        targets=ps.signal.targets,
                        risk_reward_ratio=ps.signal.risk_reward_ratio,
                        position_size_pct=ps.signal.position_size_pct,
                        time_horizon=ps.signal.time_horizon,
                        confirmation_signals=ps.signal.confirmation_signals,
                        invalidation_signals=ps.signal.invalidation_signals,
                    )
                
                probabilistic_scenarios.append(
                    ProbabilisticScenario(
                        name=ps.name,
                        bias=ps.bias,
                        probability=ps.probability,
                        confidence=ps.confidence,
                        probability_breakdown=ps.probability_breakdown,
                        signal=signal_schema,
                        description=ps.description,
                        key_events=ps.key_events,
                        evidence=ps.evidence,
                        risk_level=ps.risk_level,
                        risk_factors=ps.risk_factors,
                    )
                )
        except Exception as e:
            # 如果计算失败，记录但不中断流程
            risk_notes.append(f"[概率化剧本] 计算失败: {str(e)}")

    return WyckoffAnalysis(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        asof_ts=_asof_ts(features),
        market_structure=market_structure,  # type: ignore[arg-type]
        range=rng,
        events=events,
        levels=levels,
        scenarios=scenarios,
        risk_notes=risk_notes,
        regime_method="kmeans" if "regime_id" in features.columns else None,
        regime_hint=str(
            d.get("regime_hint").iloc[-1]) if "regime_hint" in d.columns else None,
        event_forward_stats=fstats,
        state_machine=sm_info,
        probabilistic_scenarios=probabilistic_scenarios,
    )
