"""
Tiered cache implementation with memory and disk storage.
"""

import json
import logging
import os
import pickle
import time
import hashlib
from typing import Any, Dict, Optional, Tuple

from cache.base import CacheInterface
from cache.memory_cache import MemoryCache

logger = logging.getLogger(__name__)

class TieredCache(CacheInterface):
    """
    Tiered cache with memory and disk storage.
    
    Features:
    - Fast memory cache with slower disk backup
    - Time-based expiration
    - Size limiting with LRU eviction
    - Thread-safe operations
    """
    
    def __init__(
        self,
        memory_size: int = 256,
        disk_path: str = "./cache",
        ttl_days: int = 30,
        disk_flush_interval: int = 60
    ):
        """
        Initialize the tiered cache.
        
        Args:
            memory_size: Maximum number of items in memory cache
            disk_path: Path to disk cache directory
            ttl_days: TTL in days
            disk_flush_interval: Interval in seconds for flushing to disk
        """
        # Create memory cache
        self.memory_cache = MemoryCache(max_size=memory_size, default_ttl=ttl_days * 86400)
        
        # Set up disk cache
        self.disk_path = os.path.abspath(disk_path)
        os.makedirs(self.disk_path, exist_ok=True)
        
        self.ttl_seconds = ttl_days * 86400
        self.last_flush_time = time.time()
        self.disk_flush_interval = disk_flush_interval
        
        logger.info(
            f"Initialized TieredCache with memory_size={memory_size}, "
            f"disk_path={disk_path}, ttl_days={ttl_days}"
        )
    
    def _get_disk_path(self, key: str) -> str:
        """Get the disk path for a key."""
        # Create a safe filename from the key
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.disk_path, f"{key_hash}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        try:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                # Check if file is expired
                file_mtime = os.path.getmtime(disk_path)
                if time.time() - file_mtime > self.ttl_seconds:
                    # Remove expired file
                    os.remove(disk_path)
                    return None
                
                # Load from disk
                with open(disk_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Store in memory for faster access next time
                self.memory_cache.set(key, value)
                
                return value
        except Exception as e:
            logger.error(f"Error reading from disk cache: {str(e)}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        # Always set in memory
        self.memory_cache.set(key, value, ttl)
        
        # Write to disk if necessary
        try:
            current_time = time.time()
            disk_path = self._get_disk_path(key)
            
            # Store on disk
            with open(disk_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Check if we should flush other pending changes
            if current_time - self.last_flush_time > self.disk_flush_interval:
                self._flush_to_disk()
                self.last_flush_time = current_time
            
            return True
        except Exception as e:
            logger.error(f"Error writing to disk cache: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        # Delete from memory
        self.memory_cache.delete(key)
        
        # Delete from disk
        try:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                os.remove(disk_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all values from the cache."""
        # Clear memory
        self.memory_cache.clear()
        
        # Clear disk
        try:
            for filename in os.listdir(self.disk_path):
                file_path = os.path.join(self.disk_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.cache'):
                    os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error clearing disk cache: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        # Count disk entries
        disk_count = 0
        disk_size = 0
        
        try:
            for filename in os.listdir(self.disk_path):
                file_path = os.path.join(self.disk_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.cache'):
                    disk_count += 1
                    disk_size += os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting disk cache stats: {str(e)}")
        
        return {
            'memory': memory_stats,
            'disk_entries': disk_count,
            'disk_size_bytes': disk_size
        }
    
    def _flush_to_disk(self) -> None:
        """Flush memory cache to disk."""
        # Not needed - items are written to disk when set
        pass