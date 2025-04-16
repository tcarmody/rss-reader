import logging
import asyncio
from typing import Dict, List, Optional, Union, Any

from text_chunking import chunk_text, summarize_long_article
from model_selection import estimate_complexity, auto_select_model
from async_batch import summarize_articles_batch
from tiered_cache import TieredSummaryCache
from rate_limiter import RateLimiter, adaptive_retry

class FastArticleSummarizer:
    """
    Optimized version of ArticleSummarizer with enhanced performance features.
    """
    
    def __init__(
        self, 
        original_summarizer,
        rpm_limit=50, 
        cache_size=256, 
        cache_dir="./summary_cache", 
        ttl_days=30
    ):
        """
        Initialize the optimized summarizer as a wrapper around the original.
        
        Args:
            original_summarizer: Original ArticleSummarizer instance
            rpm_limit: Requests per minute limit
            cache_size: Size of in-memory cache
            cache_dir: Directory for persistent cache
            ttl_days: TTL for cache entries in days
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
    
    def _apply_retry_decorator(self):
        """Apply the enhanced retry decorator to the API call method."""
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
            # Use the auto_select_model function imported from model_selection
            model = auto_select_model(
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
    
    async def batch_summarize(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        model: Optional[str] = None,
        auto_select_model: bool = True,
        temperature: float = 0.3,
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Process multiple articles in parallel batches.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent API calls
            model: Claude model to use (shorthand name or full identifier)
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting for generation
            
        Returns:
            List of article summaries with original metadata
        """
        return await summarize_articles_batch(
            summarizer=self.original,
            articles=articles,
            model=model,
            max_concurrent=max_concurrent,
            temperature=temperature,
            auto_select_model=auto_select_model
        )