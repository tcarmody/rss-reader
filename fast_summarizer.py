"""
FastArticleSummarizer with integrated parallel batch processing using enhanced isolation patterns.
"""

import os
# Disable tokenizers parallelism at the module level
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import asyncio
import concurrent.futures
import time
import traceback
from typing import Dict, List, Optional, Union, Any

from text_chunking import chunk_text, summarize_long_article
from models.selection import estimate_complexity, auto_select_model as select_model_func  # Fixed: Renamed import
from cache.tiered_cache import TieredSummaryCache
from api.rate_limiter import RateLimiter, adaptive_retry

class FastArticleSummarizer:
    """
    Optimized version of ArticleSummarizer with enhanced performance features
    and integrated parallel batch processing using improved isolation.
    """
    
    def __init__(
        self, 
        original_summarizer,
        rpm_limit=50, 
        cache_size=256, 
        cache_dir="./summary_cache", 
        ttl_days=30,
        enable_batch_processor=False
    ):
        """
        Initialize the optimized summarizer as a wrapper around the original.
        
        Args:
            original_summarizer: Original ArticleSummarizer instance
            rpm_limit: Requests per minute limit
            cache_size: Size of in-memory cache
            cache_dir: Directory for persistent cache
            ttl_days: TTL for cache entries in days
            enable_batch_processor: Whether to enable batch processing immediately
        """
        self.original = original_summarizer
        self.logger = original_summarizer.logger
        
        # Replace standard cache with tiered cache
        self.original.summary_cache = TieredSummaryCache(
            cache_dir=cache_dir,
            memory_size=cache_size,
            ttl_days=ttl_days
        )
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=rpm_limit)
        
        # Apply enhanced retry decorator to API call method
        self._apply_retry_decorator()
        
        # Log initialization
        self.logger.info(
            f"FastArticleSummarizer initialized with {rpm_limit} RPM limit, "
            f"{cache_size} cache entries, {ttl_days} days TTL"
        )
        
        # Add enhanced batch processing capability if requested
        if enable_batch_processor:
            self.add_batch_processor()
    
    def add_batch_processor(self, max_workers=3):
        """Add enhanced batch processing capability."""
        # Import here instead of at the module level to avoid circular imports
        from common.batch_processing import add_enhanced_batch_to_fast_summarizer
        add_enhanced_batch_to_fast_summarizer(self, max_workers=max_workers)
        return self
    
    def _apply_retry_decorator(self):
        """Apply the enhanced retry decorator to the API call method."""
        try:
            # Store original method reference
            original_call_method = self.original._call_claude_api
            
            # Create enhanced retry decorator
            enhanced_retry = adaptive_retry(
                max_retries=3,
                initial_backoff=2,
                max_backoff=60,
                rate_limiter=lambda _: self.rate_limiter
            )
            
            # Apply decorator to method
            self.original._call_claude_api = enhanced_retry(original_call_method)
            
            self.logger.debug("Applied enhanced retry decorator to API call method")
        except AttributeError:
            self.logger.warning(
                "Could not find _call_claude_api method in original summarizer. "
                "The FastArticleSummarizer will function without enhanced retries."
            )
    
    def summarize(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        force_refresh: bool = False,
        auto_select_model: bool = True,
        temperature: float = 0.3,
    ) -> Dict[str, str]:
        """
        Smart summarization that automatically selects best approach.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            model: Claude model to use (shorthand name or full identifier)
            force_refresh: Whether to force a new summary
            auto_select_model: Whether to automatically select model based on content
            temperature: Temperature setting for generation
            
        Returns:
            dict: The summary with headline and text
        """
        # Clean text
        text = self.original.clean_text(text)
        
        # Auto-select model if requested
        if auto_select_model and not model:
            # Fixed: Use renamed function to avoid name collision
            model = select_model_func(
                text, 
                self.original.AVAILABLE_MODELS,
                self.original.DEFAULT_MODEL,
                self.logger
            )
            
        # Choose approach based on text length
        if len(text) > 12000:
            self.logger.info(f"Using long article approach for {url} ({len(text)} chars)")
            return summarize_long_article(
                summarizer=self.original,
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature
            )
        else:
            self.logger.info(f"Using standard approach for {url} ({len(text)} chars)")
            return self.original.summarize_article(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature
            )
    
    # Note: The batch_summarize method is now added by the enhanced_batch_processor
    # and does not need to be defined directly in this class
    
    async def batch_summarize_legacy(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        model: Optional[str] = None,
        auto_select_model: bool = True,
        temperature: float = 0.3,
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Legacy batch processing method - kept for backward compatibility.
        The enhanced version is added by add_enhanced_batch_to_fast_summarizer.
        
        This method is deprecated and will be removed in a future version.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent processes
            model: Claude model to use (shorthand name or full identifier)
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting for generation
            
        Returns:
            List of article summaries with original metadata
        """
        self.logger.warning(
            "Using legacy batch processing method - this is deprecated. "
            "The enhanced batch processing is now used by default."
        )
        
        from common.batch_processing import SpawnBatchProcessor
        
        # Create the batch processor
        batch_processor = SpawnBatchProcessor()
        
        # Process the batch
        results = await batch_processor.batch_process_async(
            summarizer=self.original,
            articles=articles,
            model=model,
            max_concurrent=max_concurrent,
            auto_select_model=auto_select_model,
            temperature=temperature
        )
        
        # Format results
        formatted_results = []
        for result in results:
            if result['success']:
                formatted_results.append({
                    'original': result['original'],
                    'summary': result['summary']
                })
            else:
                formatted_results.append({
                    'original': result['original'],
                    'error': result.get('error', 'Unknown error'),
                    'summary': {
                        'headline': result['original']['title'],
                        'summary': f"Summary generation failed: {result.get('error', 'Unknown error')}. Please try again later."
                    }
                })
        
        return formatted_results


def create_fast_summarizer(
    original_summarizer, 
    rpm_limit=50, 
    cache_size=256, 
    cache_dir="./summary_cache", 
    ttl_days=30,
    max_batch_workers=3
):
    """
    Factory function to create a FastArticleSummarizer with enhanced batch processing.
    
    Args:
        original_summarizer: Original ArticleSummarizer instance
        rpm_limit: Requests per minute limit
        cache_size: Size of in-memory cache
        cache_dir: Directory for persistent cache
        ttl_days: TTL for cache entries in days
        max_batch_workers: Maximum number of worker processes for batch processing
        
    Returns:
        FastArticleSummarizer: The configured summarizer instance
    """
    # Create without batch processor initially
    summarizer = FastArticleSummarizer(
        original_summarizer=original_summarizer,
        rpm_limit=rpm_limit,
        cache_size=cache_size,
        cache_dir=cache_dir,
        ttl_days=ttl_days,
        enable_batch_processor=False
    )
    
    # Add batch processor after creation
    summarizer.add_batch_processor(max_workers=max_batch_workers)
    
    return summarizer


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create original summarizer
    from summarization.article_summarizer import ArticleSummarizer
    original = ArticleSummarizer()
    
    # Create fast summarizer with enhanced batch processing
    fast = create_fast_summarizer(
        original_summarizer=original,
        rpm_limit=60,
        max_batch_workers=3
    )
    
    # Example articles
    articles = [
        {
            'text': 'Sample article text about artificial intelligence.',
            'title': 'AI Developments',
            'url': 'https://example.com/ai'
        },
        {
            'text': 'Sample article text about machine learning.',
            'title': 'ML Advancements',
            'url': 'https://example.com/ml'
        }
    ]
    
    # Run asynchronous batch processing
    async def test_batch():
        results = await fast.batch_summarize(
            articles=articles,
            auto_select_model=True
        )
        
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Headline: {result['summary']['headline']}")
                print(f"Summary: {result['summary']['summary'][:100]}...")
    
    # Run the test
    import asyncio
    asyncio.run(test_batch())