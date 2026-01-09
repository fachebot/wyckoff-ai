"""
自定义异常类

提供结构化的异常层次，便于错误处理和用户友好的错误消息
"""
from __future__ import annotations

from typing import Any


class WyckoffError(Exception):
    """
    威科夫分析工具的基础异常类
    
    所有自定义异常都应继承此类
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"[{detail_str}]")
        return " ".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式，便于JSON序列化"""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# ============== 数据相关异常 ==============

class DataError(WyckoffError):
    """数据相关的基础异常"""
    pass


class DataFetchError(DataError):
    """数据获取失败"""
    
    def __init__(
        self,
        message: str = "数据获取失败",
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        cause: Exception | None = None,
    ):
        details = {}
        if exchange:
            details["exchange"] = exchange
        if symbol:
            details["symbol"] = symbol
        if timeframe:
            details["timeframe"] = timeframe
        
        super().__init__(
            message,
            code="DATA_FETCH_ERROR",
            details=details,
            cause=cause,
        )


class DataValidationError(DataError):
    """数据验证失败"""
    
    def __init__(
        self,
        message: str = "数据验证失败",
        *,
        field: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ):
        details = {}
        if field:
            details["field"] = field
        if expected is not None:
            details["expected"] = str(expected)
        if actual is not None:
            details["actual"] = str(actual)
        
        super().__init__(
            message,
            code="DATA_VALIDATION_ERROR",
            details=details,
        )


class InsufficientDataError(DataError):
    """数据量不足"""
    
    def __init__(
        self,
        message: str = "数据量不足",
        *,
        required: int | None = None,
        actual: int | None = None,
    ):
        details = {}
        if required is not None:
            details["required"] = required
        if actual is not None:
            details["actual"] = actual
        
        super().__init__(
            message,
            code="INSUFFICIENT_DATA",
            details=details,
        )


# ============== 分析相关异常 ==============

class AnalysisError(WyckoffError):
    """分析相关的基础异常"""
    pass


class FeatureComputationError(AnalysisError):
    """特征计算失败"""
    
    def __init__(
        self,
        message: str = "特征计算失败",
        *,
        feature: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"feature": feature} if feature else {}
        super().__init__(
            message,
            code="FEATURE_COMPUTATION_ERROR",
            details=details,
            cause=cause,
        )


class EventDetectionError(AnalysisError):
    """事件检测失败"""
    
    def __init__(
        self,
        message: str = "事件检测失败",
        *,
        event_type: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"event_type": event_type} if event_type else {}
        super().__init__(
            message,
            code="EVENT_DETECTION_ERROR",
            details=details,
            cause=cause,
        )


class StateMachineError(AnalysisError):
    """状态机处理失败"""
    
    def __init__(
        self,
        message: str = "状态机处理失败",
        *,
        current_state: str | None = None,
        event_type: str | None = None,
        cause: Exception | None = None,
    ):
        details = {}
        if current_state:
            details["current_state"] = current_state
        if event_type:
            details["event_type"] = event_type
        
        super().__init__(
            message,
            code="STATE_MACHINE_ERROR",
            details=details,
            cause=cause,
        )


# ============== 回测相关异常 ==============

class BacktestError(WyckoffError):
    """回测相关的基础异常"""
    pass


class BacktestConfigError(BacktestError):
    """回测配置错误"""
    
    def __init__(
        self,
        message: str = "回测配置错误",
        *,
        param: str | None = None,
        value: Any = None,
    ):
        details = {}
        if param:
            details["param"] = param
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message,
            code="BACKTEST_CONFIG_ERROR",
            details=details,
        )


class TradeExecutionError(BacktestError):
    """交易执行错误"""
    
    def __init__(
        self,
        message: str = "交易执行错误",
        *,
        trade_id: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"trade_id": trade_id} if trade_id else {}
        super().__init__(
            message,
            code="TRADE_EXECUTION_ERROR",
            details=details,
            cause=cause,
        )


# ============== 配置相关异常 ==============

class ConfigError(WyckoffError):
    """配置相关的基础异常"""
    pass


class InvalidConfigError(ConfigError):
    """无效配置"""
    
    def __init__(
        self,
        message: str = "无效配置",
        *,
        config_key: str | None = None,
        config_value: Any = None,
    ):
        details = {}
        if config_key:
            details["key"] = config_key
        if config_value is not None:
            details["value"] = str(config_value)
        
        super().__init__(
            message,
            code="INVALID_CONFIG",
            details=details,
        )


class MissingConfigError(ConfigError):
    """缺少必要配置"""
    
    def __init__(
        self,
        message: str = "缺少必要配置",
        *,
        config_key: str | None = None,
    ):
        details = {"key": config_key} if config_key else {}
        super().__init__(
            message,
            code="MISSING_CONFIG",
            details=details,
        )


# ============== 输出相关异常 ==============

class OutputError(WyckoffError):
    """输出相关的基础异常"""
    pass


class ReportGenerationError(OutputError):
    """报告生成失败"""
    
    def __init__(
        self,
        message: str = "报告生成失败",
        *,
        report_type: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"report_type": report_type} if report_type else {}
        super().__init__(
            message,
            code="REPORT_GENERATION_ERROR",
            details=details,
            cause=cause,
        )


class FileWriteError(OutputError):
    """文件写入失败"""
    
    def __init__(
        self,
        message: str = "文件写入失败",
        *,
        file_path: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"file_path": file_path} if file_path else {}
        super().__init__(
            message,
            code="FILE_WRITE_ERROR",
            details=details,
            cause=cause,
        )


# ============== 外部服务异常 ==============

class ExternalServiceError(WyckoffError):
    """外部服务相关的基础异常"""
    pass


class ExchangeAPIError(ExternalServiceError):
    """交易所 API 错误"""
    
    def __init__(
        self,
        message: str = "交易所 API 错误",
        *,
        exchange: str | None = None,
        status_code: int | None = None,
        cause: Exception | None = None,
    ):
        details = {}
        if exchange:
            details["exchange"] = exchange
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(
            message,
            code="EXCHANGE_API_ERROR",
            details=details,
            cause=cause,
        )


class LLMError(ExternalServiceError):
    """LLM 服务错误"""
    
    def __init__(
        self,
        message: str = "LLM 服务错误",
        *,
        provider: str | None = None,
        cause: Exception | None = None,
    ):
        details = {"provider": provider} if provider else {}
        super().__init__(
            message,
            code="LLM_ERROR",
            details=details,
            cause=cause,
        )

