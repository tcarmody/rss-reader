"""Caching mechanism for article summaries."""

import os
import time
import json
import hashlib
import logging
import threading


class SummaryCache:
    """
    A caching mechanism for article summaries to reduce redundant API calls
    and improve processing speed.

    This class provides thread-safe caching of article summaries with features like:
    - File-based persistent storage
    - Automatic expiration of old entries
    - Maximum cache size enforcement
    - MD5 hashing for cache keys

    Example:
        cache = SummaryCache()
        summary = cache.get("article text")
        if not summary:
            summary = generate_summary("article text")
            cache.set("article text", summary)
    """
    def __init__(self, cache_dir='.cache', cache_duration=7*24*60*60, max_cache_size=500):
        """
        Initialize the summary cache with configurable settings.

        Args:
            cache_dir: Directory to store cache files
            cache_duration: How long to keep summaries (in seconds)
            max_cache_size: Maximum number of entries in cache
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        self.max_cache_size = max_cache_size
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'summary_cache.json')
        self.cache = {}
        self.lock = threading.RLock()  # Use RLock for thread safety
        self._load_cache()

    def _load_cache(self):
        """Load the cache from disk, creating an empty one if it doesn't exist."""
        with self.lock:
            try:
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)
                        # Convert any string values to dict format
                        for key, value in data.items():
                            if isinstance(value, str):
                                self.cache[key] = {
                                    'summary': value,
                                    'timestamp': time.time()
                                }
                            else:
                                self.cache[key] = value
                    # Clean up expired entries
                    self._cleanup_cache()
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save the current cache to disk in JSON format."""
        with self.lock:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                logging.error(f"Error saving cache: {e}")

    def _cleanup_cache(self):
        """Remove expired entries and enforce maximum cache size."""
        with self.lock:
            current_time = time.time()
            # Remove expired entries
            self.cache = {
                k: v for k, v in self.cache.items()
                if isinstance(v, dict) and current_time - v.get('timestamp', 0) < self.cache_duration
            }

            # If still too large, remove oldest entries
            if len(self.cache) > self.max_cache_size:
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].get('timestamp', 0) if isinstance(x[1], dict) else 0
                )
                self.cache = dict(sorted_items[-self.max_cache_size:])

    def get(self, text):
        """
        Retrieve cached summary for a given text.
        
        Args:
            text: The text to look up in the cache
            
        Returns:
            Cached entry or None if not found or expired
        """
        with self.lock:
            key = self._hash_text(text)
            if key in self.cache:
                entry = self.cache[key]
                if isinstance(entry, dict) and time.time() - entry.get('timestamp', 0) < self.cache_duration:
                    return entry
                else:
                    del self.cache[key]
            return None

    def set(self, text, summary):
        """
        Cache a summary for a given text.
        
        Args:
            text: The text to use as the cache key
            summary: The summary to cache (string or dict)
        """
        with self.lock:
            key = self._hash_text(text)
            if isinstance(summary, str):
                summary = {'summary': summary}
            summary['timestamp'] = time.time()
            self.cache[key] = summary
            if len(self.cache) > self.max_cache_size:
                self._cleanup_cache()
            self._save_cache()

    def _hash_text(self, text):
        """
        Generate a hash for the given text to use as a cache key.
        
        Args:
            text: Text to hash
            
        Returns:
            MD5 hash of the text
        """
        # Convert text to string if it's not already
        if not isinstance(text, str):
            text = str(text)
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def clear_cache(self):
        """Completely clear the cache from memory and disk."""
        with self.lock:
            self.cache = {}
            try:
                os.remove(self.cache_file)
            except FileNotFoundError:
                pass
            self._save_cache()
