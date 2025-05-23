"""
Enhanced RSS Reader with clustering and batch processing capabilities.
"""

import os
import logging
import asyncio
from typing import List, Dict, Optional, Any

from reader.base_reader import RSSReader  # You'll need to create this
from summarization.fast_summarizer import FastSummarizer
from summarization.article_summarizer import ArticleSummarizer
from common.logging import StructuredLogger

class EnhancedRSSReader:
    """
    Enhanced RSS Reader with additional features:
    - Improved clustering
    - Batch processing
    - Advanced summarization
    - Performance optimizations
    """
    
    def __init__(
        self,
        feeds=None,
        batch_size=25,
        batch_delay=15,
        max_workers=3,
        per_feed_limit=25
    ):
        """
        Initialize the enhanced RSS reader.
        
        Args:
            feeds: List of feed URLs
            batch_size: Number of items to process per batch
            batch_delay: Delay between batches
            max_workers: Maximum number of concurrent workers
            per_feed_limit: Maximum number of articles to process per feed
        """
        self.logger = logging.getLogger(__name__)
        
        # Create base reader
        self.reader = RSSReader(
            feeds=feeds,
            batch_size=batch_size,
            batch_delay=batch_delay,
            per_feed_limit=per_feed_limit
        )
        
        # Initialize summarizer
        self.summarizer = ArticleSummarizer()
        
        # Create fast summarizer
        self.fast_summarizer = FastSummarizer(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            rpm_limit=50,
            cache_size=256,
            max_batch_workers=max_workers
        )
        
        # Track processed clusters
        self.last_processed_clusters = []
    
    async def process_feeds(self):
        """
        Process all feeds and generate summaries and clusters.
        
        Returns:
            str: Path to output HTML file or None if processing failed
        """
        try:
            # Process feeds using base reader
            output_file = await self.reader.process_feeds()
            
            # Get the processed clusters
            self.last_processed_clusters = self.reader.last_processed_clusters
            
            return output_file
        except Exception as e:
            self.logger.error(f"Error in enhanced processing: {str(e)}")
            return None