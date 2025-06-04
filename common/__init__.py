"""
Common utilities module for RSS reader.

This module now contains only true shared utilities:
- config: Environment variable management
- errors: Custom exceptions and retry decorators  
- http: HTTP session creation
- logging: Structured logging utilities
- performance: Performance tracking decorators
- batch_processing: Generic batch processing utilities

Note: Archive and content extraction functionality has been moved to:
- content.archive: Archive services and paywall handling
- content.extractors: Source extraction and content processing

For backward compatibility, use:
- common.archive_compat: Drop-in replacement for old archive module
- common.source_extractor_compat: Drop-in replacement for old extractor module
"""

# Re-export core utilities for convenience
from .config import get_env_var
from .http import create_http_session
from .logging import configure_logging, StructuredLogger
from .performance import track_performance
from .batch_processing import BatchProcessor
from .errors import (
    APIError, 
    RateLimitError, 
    AuthenticationError, 
    ConnectionError, 
    retry_with_backoff
)

__all__ = [
    'get_env_var',
    'create_http_session', 
    'configure_logging',
    'StructuredLogger',
    'track_performance',
    'BatchProcessor',
    'APIError',
    'RateLimitError', 
    'AuthenticationError',
    'ConnectionError',
    'retry_with_backoff'
]