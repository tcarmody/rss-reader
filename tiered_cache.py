"""Tiered caching system with memory and disk storage."""

import os
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from cachetools import LRUCache

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
        
        # Initialize memory cache using cachetools.LRUCache
        self.memory_cache = LRUCache(maxsize=memory_size)
        
    def _get_cache_path(self, key_hash):
        """Get the file path for a cache entry."""
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _hash_key(self, key):
        """Create a hash for the cache key."""
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key):
        """
        Get item from cache, checking memory first, then disk.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            self.logger.debug("Cache hit (memory)")
            return self.memory_cache[key]
            
        # Try disk cache
        key_hash = self._hash_key(key)
        cache_path = self._get_cache_path(key_hash)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    timestamp = cache_data['timestamp']
                    value = cache_data['value']
                    
                    # Check if entry is expired
                    expiry_date = timestamp + timedelta(days=self.ttl_days)
                    if datetime.now() > expiry_date:
                        self.logger.debug("Cache expired (disk)")
                        os.remove(cache_path)
                        return None
                    
                    # Add to memory cache for faster future access
                    self.memory_cache[key] = value
                    self.logger.debug("Cache hit (disk)")
                    return value
            except (pickle.PickleError, KeyError, EOFError) as e:
                self.logger.error(f"Error reading cache file: {e}")
                os.remove(cache_path)
                return None
        
        self.logger.debug("Cache miss")
        return None
    
    def set(self, key, value):
        """
        Set item in both memory and disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Update memory cache
        self.memory_cache[key] = value
        
        # Update disk cache
        key_hash = self._hash_key(key)
        cache_path = self._get_cache_path(key_hash)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'value': value
                }, f)
            self.logger.debug("Cache updated")
        except Exception as e:
            self.logger.error(f"Error writing to cache: {e}")
    
    def clear(self):
        """Clear both memory and disk caches."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except OSError as e:
                    self.logger.error(f"Error removing cache file {filename}: {e}")
        
        self.logger.info("Cache cleared")
    
    def remove(self, key):
        """
        Remove an item from both memory and disk cache.
        
        Args:
            key: Cache key to remove
        """
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from disk cache
        key_hash = self._hash_key(key)
        cache_path = self._get_cache_path(key_hash)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                self.logger.debug(f"Removed cache entry for {key}")
            except OSError as e:
                self.logger.error(f"Error removing cache file: {e}")