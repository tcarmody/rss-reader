import time
import random
import threading
import logging
from datetime import datetime, timedelta
from functools import wraps

class RateLimiter:
    """Rate limiter to manage API usage."""
    
    def __init__(self, requests_per_minute=50, max_burst=10):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Target RPM limit
            max_burst: Maximum burst of requests allowed
        """
        self.rpm_limit = requests_per_minute
        self.max_burst = max_burst
        self.current_tokens = max_burst
        self.last_update = datetime.now()
        self.lock = threading.RLock()
        self.logger = logging.getLogger("RateLimiter")
    
    def acquire(self, timeout=None):
        """
        Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Refill tokens based on time elapsed
                now = datetime.now()
                elapsed = (now - self.last_update).total_seconds()
                new_tokens = elapsed * (self.rpm_limit / 60.0)
                
                self.current_tokens = min(self.max_burst, self.current_tokens + new_tokens)
                self.last_update = now
                
                if self.current_tokens >= 1:
                    # Consume a token
                    self.current_tokens -= 1
                    return True
            
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                self.logger.warning(f"Rate limit timeout after {timeout}s")
                return False
            
            # Wait before trying again
            wait_time = min(0.1, self.get_wait_time())
            time.sleep(wait_time)
    
    def get_wait_time(self):
        """Get estimated wait time in seconds until next token is available."""
        with self.lock:
            if self.current_tokens >= 1:
                return 0
            
            tokens_needed = 1 - self.current_tokens
            return tokens_needed / (self.rpm_limit / 60.0)


# Enhanced retry decorator
def adaptive_retry(
    max_retries=3, 
    initial_backoff=1, 
    max_backoff=60,
    rate_limiter=None,
    retryable_exceptions=None
):
    """
    Enhanced retry decorator with adaptive backoff and rate limiting.
    
    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        rate_limiter: Rate limiter instance or function to get it
        retryable_exceptions: Exceptions that trigger retry
    """
    # Default exceptions if none provided
    if retryable_exceptions is None:
        # Import here to avoid circular imports
        from summarizer import APIConnectionError, APIRateLimitError
        retryable_exceptions = (APIConnectionError, APIRateLimitError)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from self if it exists, otherwise create one
            self = args[0] if args else None
            logger = getattr(self, 'logger', logging.getLogger(__name__))
            
            # Resolve rate limiter if it's a function
            actual_rate_limiter = rate_limiter(self) if callable(rate_limiter) and self else rate_limiter
            
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    # Rate limit if provided
                    if actual_rate_limiter:
                        wait_time = actual_rate_limiter.get_wait_time()
                        if wait_time > 0:
                            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                            time.sleep(wait_time)
                        
                        actual_rate_limiter.acquire()
                    
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded", 
                                     error_type=type(e).__name__,
                                     final_attempt=True)
                        raise
                    
                    # Calculate backoff with jitter (avoid synchronized retries)
                    jitter = random.uniform(0.8, 1.2)
                    current_backoff = min(max_backoff, backoff * jitter)
                    
                    # If rate limited, use a more aggressive backoff
                    if "RateLimit" in type(e).__name__:
                        current_backoff = max(current_backoff, 5)  # At least 5 seconds for rate limits
                    
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}",
                                  error_type=type(e).__name__,
                                  backoff_time=current_backoff,
                                  retry_count=retries)
                    
                    time.sleep(current_backoff)
                    backoff = min(max_backoff, backoff * 2)  # Exponential backoff
        return wrapper
    return decorator