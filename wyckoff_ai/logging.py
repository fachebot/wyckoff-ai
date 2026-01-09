"""
日志配置模块

提供统一的日志配置和格式化，支持：
- 控制台输出（彩色）
- 文件输出（可选）
- 不同日志级别
- 结构化日志上下文
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# 日志级别类型
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# 默认日志格式
DEFAULT_FORMAT = "%(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 自定义主题
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim",
    "repr.number": "bold cyan",
    "repr.str": "green",
})

# 模块级logger缓存
_loggers: dict[str, logging.Logger] = {}
_configured = False


def setup_logging(
    level: LogLevel = "INFO",
    log_file: str | Path | None = None,
    show_path: bool = False,
    rich_traceback: bool = True,
) -> None:
    """
    配置全局日志设置
    
    Args:
        level: 日志级别
        log_file: 可选的日志文件路径
        show_path: 是否在日志中显示文件路径
        rich_traceback: 是否使用 rich 的异常追踪格式
    """
    global _configured
    
    # 获取根logger
    root_logger = logging.getLogger("wyckoff_ai")
    root_logger.setLevel(getattr(logging, level))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 创建 Rich 控制台处理器
    console = Console(theme=CUSTOM_THEME, stderr=True)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=show_path,
        rich_tracebacks=rich_traceback,
        tracebacks_show_locals=level == "DEBUG",
        markup=True,
    )
    rich_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(rich_handler)
    
    # 可选：添加文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        root_logger.addHandler(file_handler)
    
    # 抑制第三方库的日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称，如果为None则返回根logger
        
    Returns:
        配置好的logger实例
    """
    global _configured
    
    # 确保已配置
    if not _configured:
        setup_logging()
    
    full_name = f"wyckoff_ai.{name}" if name else "wyckoff_ai"
    
    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)
    
    return _loggers[full_name]


class LogContext:
    """
    日志上下文管理器，用于添加额外的上下文信息
    
    Example:
        with LogContext(symbol="BTC/USDT", timeframe="1h"):
            logger.info("开始分析")
    """
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        context = self.context
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_execution_time(func=None, *, logger_name: str | None = None):
    """
    装饰器：记录函数执行时间
    
    Example:
        @log_execution_time
        def slow_function():
            ...
            
        @log_execution_time(logger_name="analysis")
        def analyze():
            ...
    """
    import functools
    import time
    
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            log = get_logger(logger_name or fn.__module__.split(".")[-1])
            start = time.perf_counter()
            
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.debug(f"{fn.__name__} 执行完成，耗时 {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.error(f"{fn.__name__} 执行失败，耗时 {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


# 便捷函数
def debug(msg: str, *args, **kwargs):
    """记录 DEBUG 级别日志"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """记录 INFO 级别日志"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """记录 WARNING 级别日志"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """记录 ERROR 级别日志"""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """记录 CRITICAL 级别日志"""
    get_logger().critical(msg, *args, **kwargs)

