"""
Binance 数据获取模块

直接使用 Binance REST API 获取 K 线数据，不依赖 ccxt。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from wyckoff_ai.exceptions import DataFetchError, DataValidationError, ExchangeAPIError
from wyckoff_ai.logging import get_logger

logger = get_logger("data.binance")

# Binance API 基础 URL
BINANCE_API_BASE = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"

# 请求超时时间（秒）
REQUEST_TIMEOUT = 30


@dataclass(frozen=True)
class FetchResult:
    df: pd.DataFrame
    gaps: int


def _symbol_to_binance(symbol: str) -> str:
    """
    将通用交易对格式转换为 Binance 格式

    例如: BTC/USDT -> BTCUSDT
    """
    return symbol.replace("/", "").replace("-", "").upper()


def _timeframe_to_binance(timeframe: str) -> str:
    """
    将时间周期格式转换为 Binance API 格式

    支持: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    # Binance 支持的时间周期格式与我们使用的基本一致
    valid_intervals = [
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    ]

    if timeframe in valid_intervals:
        return timeframe

    raise DataValidationError(
        f"不支持的时间周期: {timeframe}",
        field="timeframe",
        expected=", ".join(valid_intervals),
        actual=timeframe,
    )


def _timeframe_to_pandas_freq(timeframe: str) -> str:
    """将时间周期转换为 pandas freq 格式"""
    unit = timeframe[-1]
    try:
        n = int(timeframe[:-1])
    except ValueError:
        raise DataValidationError(
            f"无效的时间周期格式: {timeframe}",
            field="timeframe",
            expected="例如 1m, 5m, 1h, 4h, 1d",
            actual=timeframe,
        )

    if unit == "m":
        return f"{n}min"
    if unit == "h":
        return f"{n}h"
    if unit == "d":
        return f"{n}d"
    if unit == "w":
        return f"{n}W"
    if unit == "M":
        return f"{n}MS"  # Month Start

    raise DataValidationError(
        f"不支持的时间周期单位: {unit}",
        field="timeframe",
        expected="m, h, d, w, M",
        actual=unit,
    )


def fetch_ohlcv_binance_spot(
    symbol: str,
    timeframe: str,
    limit: int = 720,
) -> FetchResult:
    """
    从 Binance 现货市场获取 K 线数据

    直接调用 Binance REST API，不依赖 ccxt。

    Args:
        symbol: 交易对，如 "BTC/USDT" 或 "BTCUSDT"
        timeframe: 时间周期，如 "1m", "5m", "1h", "4h", "1d"
        limit: 返回的 K 线数量，最大 1000

    Returns:
        FetchResult: 包含 DataFrame 和缺口数量
        DataFrame 列: timestamp, open, high, low, close, volume, is_gap

    Raises:
        DataFetchError: 数据获取失败
        ExchangeAPIError: 交易所 API 错误
        DataValidationError: 参数验证失败
    """
    logger.debug(
        f"开始获取数据: symbol={symbol}, timeframe={timeframe}, limit={limit}")

    # 参数验证
    if not symbol:
        raise DataValidationError("交易对不能为空", field="symbol")
    if limit <= 0:
        raise DataValidationError(
            "limit 必须为正整数",
            field="limit",
            expected="> 0",
            actual=limit,
        )
    if limit > 1000:
        logger.warning(
            f"Binance API 最大支持 1000 条数据，已将 limit 从 {limit} 调整为 1000")
        limit = 1000

    # 转换格式
    binance_symbol = _symbol_to_binance(symbol)
    binance_interval = _timeframe_to_binance(timeframe)

    # 构建请求 URL
    url = f"{BINANCE_API_BASE}{BINANCE_KLINES_ENDPOINT}"
    params = {
        "symbol": binance_symbol,
        "interval": binance_interval,
        "limit": limit,
    }

    logger.debug(f"请求 URL: {url}, 参数: {params}")

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error("请求超时")
        raise ExchangeAPIError(
            f"Binance API 请求超时（{REQUEST_TIMEOUT}秒）",
            exchange="binance",
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"连接错误: {e}")
        raise ExchangeAPIError(
            f"无法连接到 Binance API: {e}",
            exchange="binance",
            cause=e,
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP 错误: {e}")
        # 解析 Binance 错误响应
        error_msg = str(e)
        try:
            error_data = response.json()
            if "msg" in error_data:
                error_msg = error_data["msg"]
        except Exception:
            pass
        raise ExchangeAPIError(
            f"Binance API 错误: {error_msg}",
            exchange="binance",
            cause=e,
        )
    except Exception as e:
        logger.error(f"未知错误: {e}")
        raise DataFetchError(
            f"获取数据时发生错误: {e}",
            exchange="binance",
            symbol=symbol,
            timeframe=timeframe,
            cause=e,
        )

    if not data:
        raise DataFetchError(
            "未获取到任何K线数据",
            exchange="binance",
            symbol=symbol,
            timeframe=timeframe,
        )

    logger.debug(f"获取到 {len(data)} 条原始数据")

    # 解析 K 线数据
    # Binance 返回格式: [open_time, open, high, low, close, volume, close_time, ...]
    rows = []
    for kline in data:
        rows.append({
            "timestamp_ms": kline[0],  # Open time
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"]).sort_values(
        "timestamp").reset_index(drop=True)

    # Reindex to mark gaps
    freq = _timeframe_to_pandas_freq(timeframe)
    full_index = pd.date_range(
        start=df["timestamp"].iloc[0], end=df["timestamp"].iloc[-1], freq=freq, tz="UTC"
    )
    df2 = df.set_index("timestamp").reindex(full_index)
    gaps = int(df2["close"].isna().sum())
    df2["is_gap"] = df2["close"].isna()

    # Keep OHLCV numeric; gaps remain NaN; caller may ffill if needed
    df2.index.name = "timestamp"
    df2 = df2.reset_index()

    if gaps > 0:
        logger.warning(f"检测到 {gaps} 个数据缺口")

    logger.info(f"数据获取完成: {len(df2)} 条K线, {gaps} 个缺口")

    return FetchResult(df=df2, gaps=gaps)
