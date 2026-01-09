from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    """
    轻量状态识别配置（不依赖 sklearn/hmmlearn）。
    - method:
        - "none": 不做状态识别
        - "kmeans": KMeans 聚类（对特征做稳健标准化）
        - "cusum": 简易变点检测（基于单变量信号，如 log_ret_1 / atr_pct_14）
    """

    method: str = "kmeans"
    k: int = 4
    max_iter: int = 40
    seed: int = 7

    # cusum
    cusum_signal: str = "log_ret_1"
    cusum_threshold: float = 6.0
    cusum_drift: float = 0.0


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    使用 median/MAD 的稳健标准化，避免极端值把尺度撑爆。
    """
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return (x - med) / scale


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)
    i0 = int(rng.integers(0, n))
    centers[0] = X[i0]
    # distances to nearest center
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        ij = int(rng.choice(n, p=probs))
        centers[j] = X[ij]
        d2 = np.minimum(d2, np.sum((X - centers[j]) ** 2, axis=1))
    return centers


def _kmeans_fit_predict(X: np.ndarray, k: int, *, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 labels, centers。X 需无 NaN 且已标准化。
    """
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(X, k, rng)
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        # assign
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            m = labels == j
            if not m.any():
                centers[j] = X[int(rng.integers(0, X.shape[0]))]
            else:
                centers[j] = X[m].mean(axis=0)
    return labels, centers


def _cusum_change_points(x: np.ndarray, *, threshold: float, drift: float) -> np.ndarray:
    """
    双边 CUSUM 变点检测（离线、简易版），输出触发的索引（从 0 开始）。
    """
    gp = 0.0
    gn = 0.0
    cps: list[int] = []
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            continue
        gp = max(0.0, gp + xi - drift)
        gn = min(0.0, gn + xi + drift)
        if gp > threshold or abs(gn) > threshold:
            cps.append(i)
            gp = 0.0
            gn = 0.0
    return np.array(cps, dtype=int)


def add_regime_columns(features: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """
    在 features 上添加：
    - regime_id: int（聚类或段 ID）
    - regime_hint: str（range / trend / volatile 等）
    - is_range_regime: bool（更稳的区间候选）
    """
    cfg = cfg or RegimeConfig()
    out = features.copy()
    n = len(out)
    out["regime_id"] = np.nan
    out["regime_hint"] = None
    out["is_range_regime"] = False

    if cfg.method == "none" or n == 0:
        return out

    if cfg.method == "cusum":
        sig = out.get(cfg.cusum_signal)
        if sig is None:
            return out
        x = sig.astype(float).to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cps = _cusum_change_points(x, threshold=cfg.cusum_threshold, drift=cfg.cusum_drift)
        # build segment ids
        seg_id = np.zeros(n, dtype=int)
        cur = 0
        last = 0
        for cp in cps:
            seg_id[last:cp + 1] = cur
            cur += 1
            last = cp + 1
        seg_id[last:] = cur
        out["regime_id"] = seg_id
        # hint: 以段内 slope/atr/donchian 粗分
        out = _infer_regime_hints(out, by="regime_id")
        return out

    if cfg.method == "kmeans":
        cols = [
            "slope_50",
            "atr_pct_14",
            "donchian_width_50",
            "vol_z_20",
            "log_ret_1",
        ]
        use = [c for c in cols if c in out.columns]
        if len(use) < 3:
            return out

        Xraw = out[use].astype(float).to_numpy()
        ok = np.isfinite(Xraw).all(axis=1)
        if ok.sum() < max(80, cfg.k * 15):
            return out

        X = _robust_zscore(Xraw[ok])
        labels, _centers = _kmeans_fit_predict(X, int(cfg.k), max_iter=int(cfg.max_iter), seed=int(cfg.seed))
        rid = np.full(n, np.nan)
        rid[ok] = labels.astype(float)
        out["regime_id"] = rid
        out = _infer_regime_hints(out, by="regime_id")
        return out

    # unknown method -> no-op
    return out


def _infer_regime_hints(df: pd.DataFrame, *, by: str) -> pd.DataFrame:
    out = df.copy()
    if by not in out.columns:
        return out

    slope = out.get("slope_50")
    atrp = out.get("atr_pct_14")
    dcw = out.get("donchian_width_50")
    if slope is None or atrp is None or dcw is None:
        return out

    # 先按 regime_id 聚合，再把 hint 映射回去
    tmp = out[[by, "slope_50", "atr_pct_14", "donchian_width_50"]].copy()
    tmp = tmp.dropna(subset=[by])
    if tmp.empty:
        return out

    g = tmp.groupby(by, dropna=True)
    stats = g.agg(
        slope_med=("slope_50", "median"),
        slope_abs_med=("slope_50", lambda x: float(np.nanmedian(np.abs(x)))),
        atrp_med=("atr_pct_14", "median"),
        dcw_med=("donchian_width_50", "median"),
    )

    hint: dict[float, str] = {}
    is_range: dict[float, bool] = {}
    for rid, row in stats.iterrows():
        slope_abs = float(row["slope_abs_med"])
        atrp_m = float(row["atrp_med"])
        dcw_m = float(row["dcw_med"])
        slope_m = float(row["slope_med"])

        # range: 低斜率 + 窄通道 + 低 ATR%（放宽阈值）
        # 严格区间：传统低波动横盘
        is_strict_range = (slope_abs <= 0.015) and (dcw_m <= 0.07) and (atrp_m <= 0.02)
        # 宽松区间：相对低波动或波动收敛
        is_loose_range = (slope_abs <= 0.04) and (dcw_m <= 0.15) and (atrp_m <= 0.04)
        # 构筑区间：有方向但波动收敛
        is_building_range = (slope_abs <= 0.06) and (dcw_m <= 0.12) and (atrp_m <= 0.035)
        
        if is_strict_range:
            hint[rid] = "range"
            is_range[rid] = True
        elif is_loose_range or is_building_range:
            hint[rid] = "range_forming"
            is_range[rid] = True
        # trend: 斜率显著，且通道相对更宽
        elif slope_m > 0.03:
            hint[rid] = "trend_up"
            is_range[rid] = False
        elif slope_m < -0.03:
            hint[rid] = "trend_down"
            is_range[rid] = False
        # 弱趋势/过渡态
        elif slope_m > 0.01:
            hint[rid] = "weak_up"
            is_range[rid] = True  # 弱趋势中也可能有区间特征
        elif slope_m < -0.01:
            hint[rid] = "weak_down"
            is_range[rid] = True
        else:
            hint[rid] = "consolidation"
            is_range[rid] = True

    def map_hint(x: object) -> str | None:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            return hint.get(float(x))
        except Exception:
            return None

    def map_is_range(x: object) -> bool:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return False
        try:
            return bool(is_range.get(float(x), False))
        except Exception:
            return False

    out["regime_hint"] = out[by].map(map_hint)
    out["is_range_regime"] = out[by].map(map_is_range).fillna(False).astype(bool)
    return out


