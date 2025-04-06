"""
RSS Reader and Summarizer

This package provides a comprehensive RSS feed reader that fetches articles,
summarizes them using the Anthropic Claude API, and clusters similar articles together.

Example usage:
    # Basic usage
    from rss_reader.reader import RSSReader
    reader = RSSReader()
    output_file = reader.process_feeds()
    
    # Custom usage with specific feeds
    from rss_reader.reader import RSSReader
    feeds = ['https://example.com/rss', 'https://example2.com/rss']
    reader = RSSReader(feeds=feeds, batch_size=10)
    output_file = reader.process_feeds()
"""

# Version info
__version__ = '1.0.0'
