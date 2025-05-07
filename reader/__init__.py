"""
RSS Reader package for fetching and processing feeds with enhanced clustering.

This package provides classes and utilities for fetching, parsing, and
processing RSS feeds with intelligent summarization and article clustering.
"""

from reader.base_reader import RSSReader
from reader.enhanced_reader import EnhancedRSSReader

__all__ = ['RSSReader', 'EnhancedRSSReader']