"""
Structured logging configuration with context tracking.
"""

import logging
import time
import os
from typing import Dict, Any, Optional
from datetime import datetime

class StructuredLogger:
    """
    Logger that provides structured context with each log entry.
    Maintains context across multiple log calls.
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def add_context(self, **kwargs) -> None:
        """
        Add context to be included in all log messages.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context data."""
        self.context = {}
    
    def _format_context(self, extra_context=None) -> str:
        """
        Format context for logging.
        
        Args:
            extra_context: Additional context for this log entry
            
        Returns:
            Formatted context string
        """
        context = self.context.copy()
        if extra_context:
            context.update(extra_context)
        
        timestamp = datetime.now().isoformat()
        context_str = ' '.join(f"{k}={v}" for k, v in context.items())
        return f"[{timestamp}] {context_str}"
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message with context."""
        self.logger.debug(f"{msg} {self._format_context(kwargs)}")
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message with context."""
        self.logger.info(f"{msg} {self._format_context(kwargs)}")
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message with context."""
        self.logger.warning(f"{msg} {self._format_context(kwargs)}")
    
    def error(self, msg: str, **kwargs) -> None:
        """Log error message with context."""
        self.logger.error(f"{msg} {self._format_context(kwargs)}")
    
    def exception(self, msg: str, exc_info=True, **kwargs) -> None:
        """Log exception with context."""
        self.logger.exception(f"{msg} {self._format_context(kwargs)}", exc_info=exc_info)


def configure_logging(
    level=logging.INFO,
    log_file=None,
    console=True,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
):
    """
    Configure application-wide logging.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        console: Whether to log to console
        log_format: Log format string
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger