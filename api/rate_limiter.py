"""
Rate limiting implementation for API calls with both synchronous and asynchronous support.
"""

import logging
import threading
import asyncio
import time
from typing import Optional, Callable, Any, Union

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Thread-safe rate limiter for API calls.
    
    Features:
    - Configurable requests per minute/second
    - Blocking and non-blocking modes
    - Adaptive backoff support
    - Synchronous and asynchronous interfaces
    """
    
    def __init__(self, requests_per_minute: int = 60, requests_per_second: Optional[int] = None):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_second: Optional maximum requests per second
        """
        self.lock = threading.RLock()
        self.async_lock = None  # Will be initialized on first async use
        
        # Save RPM for async wrapper
        self.requests_per_minute = requests_per_minute
        
        # Calculate delay between requests
        if requests_per_second:
            self.delay = 1.0 / requests_per_second
        else:
            self.delay = 60.0 / requests_per_minute
        
        self.last_request_time = 0
        self.request_count = 0
        
        self.rpm_window_start = time.time()
        self.rpm_limit = requests_per_minute
        self.rpm_count = 0
        
        logger.info(f"Initialized RateLimiter with {requests_per_minute} RPM ({1.0/self.delay:.2f} RPS)")
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            blocking: Whether to block until permission is granted
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if permission granted, False otherwise
        """
        with self.lock:
            # Check if we need to enforce RPM limit
            current_time = time.time()
            
            # Reset RPM window if needed
            if current_time - self.rpm_window_start > 60:
                self.rpm_window_start = current_time
                self.rpm_count = 0
            
            # Check if we're over RPM limit
            if self.rpm_count >= self.rpm_limit:
                if not blocking:
                    return False
                
                # Calculate wait time
                wait_time = 60 - (current_time - self.rpm_window_start)
                
                if timeout is not None and wait_time > timeout:
                    return False
                
                if wait_time > 0:
                    self.lock.release()
                    time.sleep(wait_time)
                    self.lock.acquire()
                    
                    # Reset RPM window
                    current_time = time.time()
                    self.rpm_window_start = current_time
                    self.rpm_count = 0
            
            # Check rate limit
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                if not blocking:
                    return False
                
                # Calculate wait time
                wait_time = self.delay - time_since_last
                
                if timeout is not None and wait_time > timeout:
                    return False
                
                if wait_time > 0:
                    self.lock.release()
                    time.sleep(wait_time)
                    self.lock.acquire()
                    current_time = time.time()
            
            # Update state
            self.last_request_time = current_time
            self.request_count += 1
            self.rpm_count += 1
            
            return True
    
    def wait(self) -> None:
        """Wait until it's safe to make a request."""
        self.acquire(blocking=True)
    
    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """
        Asynchronously acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if permission granted, False otherwise
        """
        # Initialize async lock if this is first async call
        if self.async_lock is None:
            self.async_lock = asyncio.Lock()
        
        # Use timeout if provided
        if timeout is not None:
            try:
                async with asyncio.timeout(timeout):
                    return await self._acquire_async_impl()
            except asyncio.TimeoutError:
                return False
        else:
            return await self._acquire_async_impl()
    
    async def _acquire_async_impl(self) -> bool:
        """
        Internal implementation of async acquire.
        
        Returns:
            True when permission is granted
        """
        async with self.async_lock:
            # Check if we need to enforce RPM limit
            current_time = time.time()
            
            # Reset RPM window if needed
            if current_time - self.rpm_window_start > 60:
                self.rpm_window_start = current_time
                self.rpm_count = 0
            
            # Check if we're over RPM limit
            if self.rpm_count >= self.rpm_limit:
                # Calculate wait time
                wait_time = 60 - (current_time - self.rpm_window_start)
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
                    # Reset RPM window
                    current_time = time.time()
                    self.rpm_window_start = current_time
                    self.rpm_count = 0
            
            # Check rate limit
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                # Calculate wait time
                wait_time = self.delay - time_since_last
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    current_time = time.time()
            
            # Update state
            self.last_request_time = current_time
            self.request_count += 1
            self.rpm_count += 1
            
            return True
    
    async def wait_async(self) -> None:
        """Asynchronously wait until it's safe to make a request."""
        await self.acquire_async()


def adaptive_retry(max_retries=3, initial_backoff=2, max_backoff=60, rate_limiter=None):
    """
    Create a retry decorator with adaptive backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        rate_limiter: Optional rate limiter or function to get one
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while True:
                # Apply rate limiting if provided
                limiter = None
                if callable(rate_limiter):
                    limiter = rate_limiter(*args)
                elif rate_limiter is not None:
                    limiter = rate_limiter
                
                if limiter:
                    limiter.wait()
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
                        raise
                    
                    # Adaptive backoff
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}, waiting {backoff}s")
                    time.sleep(backoff)
                    
                    # Increase backoff, but not beyond max
                    backoff = min(backoff * 2, max_backoff)
                    
                    # Add jitter
                    backoff *= (0.8 + 0.4 * time.time() % 1)
        
        return wrapper
    
    return decorator


def async_adaptive_retry(max_retries=3, initial_backoff=2, max_backoff=60, rate_limiter=None):
    """
    Create an async retry decorator with adaptive backoff.
    
    Args:
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        rate_limiter: Optional rate limiter or function to get one
        
    Returns:
        Async decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while True:
                # Apply rate limiting if provided
                limiter = None
                if callable(rate_limiter):
                    limiter = rate_limiter(*args)
                elif rate_limiter is not None:
                    limiter = rate_limiter
                
                if limiter:
                    await limiter.wait_async()
                
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
                        raise
                    
                    # Adaptive backoff
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}, waiting {backoff}s")
                    await asyncio.sleep(backoff)
                    
                    # Increase backoff, but not beyond max
                    backoff = min(backoff * 2, max_backoff)
                    
                    # Add jitter
                    backoff *= (0.8 + 0.4 * time.time() % 1)
        
        return wrapper
    
    return decorator