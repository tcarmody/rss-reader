"""
Simple in-memory cache implementation.
"""

import logging
import time
import threading
from typing import Any, Dict, Optional, Tuple

from cache.base import CacheInterface

logger = logging.getLogger(__name__)

class MemoryCache(CacheInterface):
    """
    Simple in-memory cache with TTL support.
    
    Features:
    - Time-based expiration
    - Thread-safe operations
    - Size limiting with LRU eviction
    - Usage statistics
    """
    
    def __init__(self, max_size: int = 256, default_ttl: int = 86400):
        """
        Initialize the memory cache.
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default TTL in seconds (1 day)
        """
        self.cache = {}  # key -> (value, expiry_time, access_time)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        
        logger.info(f"Initialized MemoryCache with max_size={max_size}, default_ttl={default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self.lock:
            if key in self.cache:
                value, expiry_time, _ = self.cache[key]
                
                # Check if expired
                if expiry_time is not None and time.time() > expiry_time:
                    # Remove expired item
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Update access time
                self.cache[key] = (value, expiry_time, time.time())
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        with self.lock:
            # Calculate expiry time
            expiry_time = None
            if ttl is not None:
                expiry_time = time.time() + ttl
            elif self.default_ttl > 0:
                expiry_time = time.time() + self.default_ttl
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Store the value
            self.cache[key] = (value, expiry_time, time.time())
            self.sets += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all values from the cache."""
        with self.lock:
            self.cache.clear()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'sets': self.sets,
                'evictions': self.evictions,
                'hit_rate': hit_rate
            }
    
    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self.cache:
            return
            
        # Find the least recently used key
        lru_key = min(self.cache.items(), key=lambda x: x[1][2])[0]
        
        # Remove it
        if lru_key in self.cache:
            del self.cache[lru_key]
            self.evictions += 1