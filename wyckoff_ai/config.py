"""
配置管理模块

支持多种配置来源（优先级从高到低）：
1. CLI 参数
2. 环境变量
3. 配置文件（TOML）
4. 默认值

配置文件查找顺序：
1. 当前目录 wyckoff.toml
2. 用户目录 ~/.wyckoff/config.toml
3. 项目默认配置
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Python 3.11+ 内置 tomllib，3.10 需要 tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

from wyckoff_ai.logging import get_logger

logger = get_logger("config")


# ============== 配置模型 ==============

class ExchangeConfig(BaseModel):
    """交易所配置"""
    name: str = Field(default="binance", description="交易所名称")
    rate_limit: bool = Field(default=True, description="是否启用速率限制")
    default_type: str = Field(default="spot", description="默认市场类型")
    timeout: int = Field(default=30000, description="请求超时时间(ms)")
    
    @field_validator("name")
    @classmethod
    def validate_exchange_name(cls, v: str) -> str:
        allowed = ["binance", "okx", "bybit"]
        if v.lower() not in allowed:
            raise ValueError(f"不支持的交易所: {v}，支持: {allowed}")
        return v.lower()


class AnalysisConfig(BaseModel):
    """分析配置"""
    default_symbol: str = Field(default="BTC/USDT", description="默认交易对")
    default_timeframe: str = Field(default="1h", description="默认时间周期")
    default_limit: int = Field(default=720, ge=50, le=5000, description="默认K线数量")
    lookback_bars: int | None = Field(default=None, description="回看K线数（默认等于limit）")
    strict_mode: bool = Field(default=False, description="严格模式")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="最小置信度")
    
    # Regime 配置
    regime_method: Literal["kmeans", "cusum", "none"] = Field(
        default="kmeans", description="状态识别方法"
    )
    regime_k: int = Field(default=4, ge=2, le=10, description="KMeans 聚类数")
    
    @field_validator("default_timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        allowed = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        if v not in allowed:
            raise ValueError(f"不支持的时间周期: {v}，支持: {allowed}")
        return v


class BacktestConfig(BaseModel):
    """回测配置"""
    initial_capital: float = Field(default=100000.0, gt=0, description="初始资金")
    position_size_pct: float = Field(default=10.0, ge=1, le=100, description="仓位百分比")
    stop_loss_atr: float = Field(default=2.0, ge=0.5, le=10, description="止损ATR倍数")
    take_profit_atr: float = Field(default=3.0, ge=0.5, le=20, description="止盈ATR倍数")
    max_bars_in_trade: int = Field(default=48, ge=1, le=500, description="最大持仓K线数")
    commission_pct: float = Field(default=0.075, ge=0, le=1, description="手续费百分比")
    slippage_pct: float = Field(default=0.01, ge=0, le=1, description="滑点百分比")
    allowed_events: list[str] | None = Field(
        default=None, description="允许交易的事件类型"
    )


class OutputConfig(BaseModel):
    """输出配置"""
    output_dir: str = Field(default="out", description="输出目录")
    export_csv: bool = Field(default=False, description="是否导出CSV")
    report_format: Literal["markdown", "html", "json"] = Field(
        default="markdown", description="报告格式"
    )
    locale: str = Field(default="zh_CN", description="语言区域")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="日志级别"
    )
    file: str | None = Field(default=None, description="日志文件路径")
    show_path: bool = Field(default=False, description="显示文件路径")
    rich_traceback: bool = Field(default=True, description="使用Rich异常追踪")


class LLMConfig(BaseModel):
    """LLM 配置"""
    provider: Literal["openai", "azure", "anthropic", "none"] = Field(
        default="none", description="LLM 提供商"
    )
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    temperature: float = Field(default=0.7, ge=0, le=2, description="温度参数")
    max_tokens: int = Field(default=2000, ge=100, le=8000, description="最大token数")
    api_key: str | None = Field(default=None, description="API密钥（建议使用环境变量）")


class WyckoffConfig(BaseSettings):
    """
    威科夫分析工具主配置
    
    支持从以下来源加载配置：
    - 环境变量（前缀 WYCKOFF_）
    - TOML 配置文件
    - 默认值
    """
    model_config = SettingsConfigDict(
        env_prefix="WYCKOFF_",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # 子配置
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    @model_validator(mode="after")
    def validate_config(self) -> "WyckoffConfig":
        """验证配置一致性"""
        # 确保止盈大于止损
        if self.backtest.take_profit_atr <= self.backtest.stop_loss_atr:
            logger.warning(
                f"止盈({self.backtest.take_profit_atr}x ATR) "
                f"应大于止损({self.backtest.stop_loss_atr}x ATR)"
            )
        return self


# ============== 配置加载 ==============

def find_config_file() -> Path | None:
    """
    查找配置文件
    
    查找顺序：
    1. 当前目录 wyckoff.toml
    2. 用户目录 ~/.wyckoff/config.toml
    3. 环境变量 WYCKOFF_CONFIG_FILE 指定的路径
    """
    # 环境变量优先
    env_path = os.getenv("WYCKOFF_CONFIG_FILE")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        logger.warning(f"环境变量指定的配置文件不存在: {env_path}")
    
    # 当前目录
    local_config = Path("wyckoff.toml")
    if local_config.exists():
        return local_config
    
    # 用户目录
    user_config = Path.home() / ".wyckoff" / "config.toml"
    if user_config.exists():
        return user_config
    
    return None


def load_toml_file(path: Path) -> dict[str, Any]:
    """加载 TOML 配置文件"""
    if tomllib is None:
        logger.warning("Python < 3.11 需要安装 tomli 包来读取 TOML 文件")
        return {}
    
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        logger.info(f"已加载配置文件: {path}")
        return data
    except Exception as e:
        logger.error(f"读取配置文件失败: {path}, 错误: {e}")
        return {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并两个配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_file: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> WyckoffConfig:
    """
    加载配置
    
    优先级（从低到高）：
    1. 默认值
    2. 配置文件
    3. 环境变量
    4. CLI 参数
    
    Args:
        config_file: 指定配置文件路径
        cli_overrides: CLI 参数覆盖
        
    Returns:
        配置对象
    """
    config_data: dict[str, Any] = {}
    
    # 1. 查找并加载配置文件
    if config_file:
        file_path = Path(config_file)
        if file_path.exists():
            config_data = load_toml_file(file_path)
        else:
            logger.warning(f"指定的配置文件不存在: {config_file}")
    else:
        found_file = find_config_file()
        if found_file:
            config_data = load_toml_file(found_file)
    
    # 2. 应用 CLI 覆盖
    if cli_overrides:
        config_data = merge_configs(config_data, cli_overrides)
    
    # 3. 创建配置对象（Pydantic 自动处理环境变量）
    try:
        config = WyckoffConfig(**config_data)
        logger.debug(f"配置加载完成: {config.model_dump()}")
        return config
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        raise


# ============== 全局配置实例 ==============

_global_config: WyckoffConfig | None = None


def get_config() -> WyckoffConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: WyckoffConfig) -> None:
    """设置全局配置实例"""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """重置全局配置"""
    global _global_config
    _global_config = None


# ============== 便捷函数 ==============

def get_output_dir() -> Path:
    """获取输出目录"""
    config = get_config()
    out_dir = Path(os.getenv("WYCKOFF_OUT_DIR", config.output.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_default_symbol() -> str:
    """获取默认交易对"""
    return os.getenv("WYCKOFF_SYMBOL", get_config().analysis.default_symbol)


def get_default_timeframe() -> str:
    """获取默认时间周期"""
    return os.getenv("WYCKOFF_TIMEFRAME", get_config().analysis.default_timeframe)

