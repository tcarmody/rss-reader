"""
Standardized error handling for the application.
Common error types and retry mechanism with backoff.
"""

import logging
import time
import functools
import random
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger(__name__)

# Base exception class
class ApplicationError(Exception):
    """Base exception for all application errors."""
    pass

# Specific error types
class APIError(ApplicationError):
    """Errors related to API communication."""
    pass

class RateLimitError(APIError):
    """API rate limit exceeded."""
    pass

class AuthenticationError(APIError):
    """API authentication failed."""
    pass

class ConnectionError(APIError):
    """Connection to external service failed."""
    pass

class ProcessingError(ApplicationError):
    """Error during data processing."""
    pass

class ConfigurationError(ApplicationError):
    """Error in configuration or setup."""
    pass

def retry_with_backoff(
    max_retries=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    jitter=0.1,
    retryable_exceptions=(ConnectionError, RateLimitError)
):
    """
    Decorator that implements retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries before giving up
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor by which the backoff time increases
        jitter: Random factor to add to backoff (0-1)
        retryable_exceptions: Tuple of exceptions that should trigger retries
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
                        raise
                    
                    # Calculate backoff with jitter
                    jitter_amount = backoff * jitter * random.uniform(-1, 1)
                    sleep_time = max(0.1, backoff + jitter_amount)
                    
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}, waiting {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    backoff *= backoff_factor
                    
        return wrapper
    return decorator