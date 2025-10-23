"""
Rate limiting implementation for API calls with asynchronous support.
"""

import logging
import asyncio
import time
from typing import Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Async-only rate limiter for API calls.

    Features:
    - Configurable requests per minute/second
    - Async/await interface only
    - RPM and RPS limiting
    """

    def __init__(self, requests_per_minute: int = 60, requests_per_second: Optional[int] = None):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_second: Optional maximum requests per second
        """
        self.lock = asyncio.Lock()
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

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Asynchronously acquire permission to make a request.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if permission granted, False otherwise
        """
        # Use timeout if provided
        if timeout is not None:
            try:
                async with asyncio.timeout(timeout):
                    return await self._acquire_impl()
            except asyncio.TimeoutError:
                return False
        else:
            return await self._acquire_impl()

    async def _acquire_impl(self) -> bool:
        """
        Internal implementation of async acquire.

        Returns:
            True when permission is granted
        """
        async with self.lock:
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