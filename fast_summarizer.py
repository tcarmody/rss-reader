"""
FastArticleSummarizer with integrated parallel batch processing using the spawn method.
"""

import os
# Disable tokenizers parallelism at the module level
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import asyncio
import multiprocessing
import concurrent.futures
import time
import traceback
from typing import Dict, List, Optional, Union, Any

from text_chunking import chunk_text, summarize_long_article
from model_selection import estimate_complexity, auto_select_model
from tiered_cache import TieredSummaryCache
from rate_limiter import RateLimiter, adaptive_retry

class FastArticleSummarizer:
    """
    Optimized version of ArticleSummarizer with enhanced performance features
    and integrated parallel batch processing using spawn method.
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
    
    @staticmethod
    def _process_article_worker(worker_data):
        """
        Worker function that runs in a separate process.
        This isolates each process to avoid tokenizer conflicts.
        
        Args:
            worker_data: Dictionary with all necessary data
            
        Returns:
            Processing result
        """
        article = worker_data['article']
        model = worker_data['model']
        temperature = worker_data['temperature']
        
        try:
            # Import inside worker to ensure clean environment
            from summarizer import ArticleSummarizer
            
            # Create a fresh summarizer instance
            summarizer = ArticleSummarizer()
            
            # Process the article
            start_time = time.time()
            summary = summarizer.summarize_article(
                text=article['text'],
                title=article['title'],
                url=article['url'],
                model=model,
                temperature=temperature
            )
            elapsed = time.time() - start_time
            
            return {
                'success': True,
                'original': article,
                'summary': summary,
                'elapsed': elapsed
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                'success': False,
                'original': article,
                'error': str(e),
                'traceback': tb
            }
    
    def batch_summarize_sync(
        self,
        articles: List[Dict[str, str]],
        model: Optional[str] = None,
        max_workers: int = 3,
        auto_select_model: bool = False,
        temperature: float = 0.3
    ) -> List[Dict]:
        """
        Process multiple articles in parallel using multiprocessing with spawn method.
        This is a synchronous version.
        
        Args:
            articles: List of articles to process
            model: Model to use
            max_workers: Maximum number of parallel workers
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Processing batch of {len(articles)} articles with {max_workers} workers")
        start_time = time.time()
        
        # Prepare worker data with models selected if needed
        worker_data_list = []
        for article in articles:
            if auto_select_model and not model:
                # Import here to avoid circular imports
                from model_selection import auto_select_model as select_model
                selected_model = select_model(
                    article['text'],
                    self.original.AVAILABLE_MODELS,
                    self.original.DEFAULT_MODEL,
                    self.logger
                )
            else:
                selected_model = self.original._get_model(model)
            
            worker_data = {
                'article': article,
                'model': selected_model,
                'temperature': temperature
            }
            worker_data_list.append(worker_data)
        
        # Initialize multiprocessing with spawn context
        ctx = multiprocessing.get_context('spawn')
        
        # Process articles in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx
        ) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self._process_article_worker, data): i
                for i, data in enumerate(worker_data_list)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    if result['success']:
                        self.logger.info(f"Processed article {idx+1}/{len(articles)}: {articles[idx]['url']}")
                    else:
                        self.logger.error(
                            f"Error processing article {idx+1}/{len(articles)}: "
                            f"{articles[idx]['url']} - {result['error']}"
                        )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Worker exception for article {idx+1}: {str(e)}")
                    results.append({
                        'success': False,
                        'original': articles[idx],
                        'error': str(e)
                    })
        
        # Sort results to match input order
        sorted_results = []
        for i in range(len(articles)):
            for result in results:
                if result['original'] == articles[i]:
                    sorted_results.append(result)
                    break
        
        elapsed = time.time() - start_time
        self.logger.info(f"Batch processing completed in {elapsed:.2f}s")
        
        # Format results for consistent API
        formatted_results = []
        for result in sorted_results:
            if result['success']:
                formatted_results.append({
                    'original': result['original'],
                    'summary': result['summary'],
                    'elapsed': result.get('elapsed')
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
    
    async def batch_summarize(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        model: Optional[str] = None,
        auto_select_model: bool = True,
        temperature: float = 0.3,
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Process multiple articles in parallel batches using spawn method.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent processes
            model: Claude model to use (shorthand name or full identifier)
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting for generation
            
        Returns:
            List of article summaries with original metadata
        """
        # Run the synchronous batch processing in a thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: self.batch_summarize_sync(
                    articles=articles,
                    model=model,
                    max_workers=max_concurrent,
                    auto_select_model=auto_select_model,
                    temperature=temperature
                )
            )
        
        return result