import pickle
import os
import hashlib
import logging
from functools import lru_cache
from datetime import datetime, timedelta

class TieredSummaryCache:
    """Multi-level caching system with memory and disk caching."""
    
    def __init__(self, cache_dir="./cache", memory_size=128, ttl_days=30):
        """
        Initialize the cache system.
        
        Args:
            cache_dir: Directory for disk cache
            memory_size: Size of LRU memory cache
            ttl_days: Time-to-live for cached entries in days
        """
        self.cache_dir = cache_dir
        self.ttl_days = ttl_days
        self.logger = logging.getLogger("TieredSummaryCache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize LRU memory cache
        self.memory_cache = lru_cache(maxsize=memory_size)(self._cache_wrapper)
        
    def _cache_wrapper(self, key):
        """Wrapper for LRU cache functionality."""
        return None
        
    def _get_cache_path(self, key_hash):
        """Get the file path for a cache entry."""
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _hash_key(self, key):
        """Create a hash for the cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key):
        """
        Get item from cache, checking memory first, then disk.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        # Try memory cache first (fastest)
        mem_result = self.memory_cache(key)
        if mem_result is not None:
            self.logger.debug("Cache hit (memory)")
            return mem_result
            
        # Try disk cache
        key_hash = self._hash_key(key)
        cache_path = self._get_cache_path(key_hash)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Check if expired
                if (datetime.now() - cached_data['timestamp']) > timedelta(days=self.ttl_days):
                    os.remove(cache_path)  # Remove expired cache
                    self.logger.debug("Cache expired (disk)")
                    return None
                    
                # Update memory cache with disk result
                result = cached_data['data']
                self._update_memory_cache(key, result)
                self.logger.debug("Cache hit (disk)")
                return result
            except (pickle.PickleError, OSError) as e:
                # Handle corrupt cache
                self.logger.warning(f"Corrupt cache file: {cache_path}, error: {str(e)}")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                return None
        
        self.logger.debug("Cache miss")
        return None
    
    def _update_memory_cache(self, key, value):
        """Update the memory cache with a new value."""
        # Force update by calling and ignoring result, then setting a new value
        self.memory_cache(key)
        self.memory_cache.__wrapped__.__dict__['cache_dict'][key] = value
    
    def set(self, key, value):
        """
        Set item in both memory and disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Update memory cache
        self._update_memory_cache(key, value)
        
        # Update disk cache
        key_hash = self._hash_key(key)
        cache_path = self._get_cache_path(key_hash)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'data': value
                }, f)
            self.logger.debug(f"Added entry to cache: {key_hash[:8]}")
        except (pickle.PickleError, OSError) as e:
            # Log error but continue
            self.logger.warning(f"Failed to write to disk cache: {str(e)}")
            
    def clear(self, older_than_days=None):
        """
        Clear cache entries.
        
        Args:
            older_than_days: Only clear entries older than this many days
                            If None, clear all entries
        """
        # Clear memory cache
        self.memory_cache.cache_clear()
        self.logger.info("Cleared memory cache")
        
        # Clear disk cache
        if older_than_days is None:
            # Clear all
            count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
            self.logger.info(f"Cleared entire disk cache ({count} entries)")
        else:
            # Clear only old entries
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            cached_data = pickle.load(f)
                        if cached_data['timestamp'] < cutoff_time:
                            os.remove(file_path)
                            count += 1
                    except (pickle.PickleError, OSError):
                        # Remove corrupt cache files
                        os.remove(file_path)
                        count += 1
            self.logger.info(f"Cleared {count} entries older than {older_than_days} days")