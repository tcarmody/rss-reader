"""
Fast and optimized summarizer implementation with batch processing.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Generator, Callable

from common.batch_processing import BatchProcessor
from common.errors import retry_with_backoff  # Add this line
from models.selection import get_model_identifier, estimate_complexity, select_model_by_complexity
from summarization.article_summarizer import ArticleSummarizer
from summarization.base import BaseSummarizer  # Import BaseSummarizer
from cache.tiered_cache import TieredCache
from api.rate_limiter import RateLimiter
from common.logging import StructuredLogger
from anthropic import Anthropic
import time
from summarization.text_processing import (
    clean_text, extract_source_from_url, create_summary_prompt, 
    get_system_prompt, parse_summary_response
)
from common.errors import APIError, RateLimitError, AuthenticationError, ConnectionError

class FastSummarizer(BaseSummarizer):  # Inherit from BaseSummarizer
    """
    Optimized summarizer with batch processing capabilities.
    
    Features:
    - Batch processing with parallelism
    - Automatic model selection based on content complexity
    - Long article handling with chunking
    - Tiered caching for performance
    - Rate limiting for API stability
    """
    
    def __init__(
        self, 
        api_key=None, 
        rpm_limit=50, 
        cache_size=256, 
        cache_dir="./summary_cache", 
        ttl_days=30,
        max_batch_workers=3,
        rate_limit_delay=0.5
    ):
        """
        Initialize the fast summarizer.
        
        Args:
            api_key: Anthropic API key (defaults to env var)
            rpm_limit: Requests per minute limit
            cache_size: In-memory cache size
            cache_dir: Directory for disk cache
            ttl_days: Cache TTL in days
            max_batch_workers: Maximum concurrent workers
            rate_limit_delay: Delay between API calls
        """
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise APIError("Anthropic API key not found")
        
        # Initialize the base class
        super().__init__(api_key=api_key, cache=None)
        
        self.logger = logging.getLogger(__name__)
        
        # Create base summarizer
        self.summarizer = ArticleSummarizer(api_key=api_key)
        
        # Set up advanced cache
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache = TieredCache(
            memory_size=cache_size,
            disk_path=cache_dir,
            ttl_days=ttl_days
        )
        self.summarizer.cache = self.cache
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=rpm_limit)
        
        # Create batch processor
        self.batch_processor = BatchProcessor(
            max_workers=max_batch_workers,
            rate_limit_delay=rate_limit_delay
        )
        
        self.logger.info(
            f"FastSummarizer initialized with {rpm_limit} RPM limit, "
            f"{cache_size} cache entries, {ttl_days} days TTL, "
            f"{max_batch_workers} batch workers"
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
            auto_select_model: Whether to select model based on content
            temperature: Temperature setting
            
        Returns:
            dict: The summary with headline and text
        """
        # Auto-select model if requested
        if auto_select_model and not model:
            clean_text_content = self.clean_text(text)
            complexity = estimate_complexity(clean_text_content)
            model = select_model_by_complexity(complexity)
            
        # Choose approach based on text length
        if len(text) > 12000:
            self.logger.info(f"Using long article approach for {url} ({len(text)} chars)")
            return self.summarize_long_article(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature
            )
        else:
            self.logger.info(f"Using standard approach for {url} ({len(text)} chars)")
            return self.summarizer.summarize_article(
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
        timeout: Optional[float] = None
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Process a batch of articles in parallel.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent processes
            model: Claude model to use (shorthand name or full identifier)
            auto_select_model: Whether to select model based on content
            temperature: Temperature setting
            timeout: Optional timeout in seconds
            
        Returns:
            List of article summaries with original metadata
        """
        if not articles:
            return []
            
        self.logger.info(f"Processing batch of {len(articles)} articles")
        
        # Group articles by model if auto-selecting
        if auto_select_model and not model:
            # Group by complexity
            article_groups = {}
            
            for article in articles:
                # Get text and clean it
                text = article.get('text', article.get('content', ''))
                text = self.clean_text(text)
                
                # Estimate complexity and select model
                complexity = estimate_complexity(text)
                selected_model = select_model_by_complexity(complexity)
                
                # Add to group
                if selected_model not in article_groups:
                    article_groups[selected_model] = []
                article_groups[selected_model].append(article)
                
            # Process each group in parallel
            tasks = []
            for model_name, group_articles in article_groups.items():
                task = self._process_article_group(
                    articles=group_articles,
                    model=model_name,
                    max_concurrent=max_concurrent,
                    temperature=temperature,
                    timeout=timeout
                )
                tasks.append(task)
                
            # Wait for all groups to complete
            results = []
            for completed_task in await asyncio.gather(*tasks):
                results.extend(completed_task)
                
            return results
        else:
            # Process all articles with the same model
            return await self._process_article_group(
                articles=articles,
                model=model,
                max_concurrent=max_concurrent,
                temperature=temperature,
                timeout=timeout
            )
    
    async def _process_article_group(
        self,
        articles: List[Dict[str, str]],
        model: Optional[str] = None,
        max_concurrent: int = 3,
        temperature: float = 0.3,
        timeout: Optional[float] = None
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Process a group of articles with the same model.
        
        Args:
            articles: List of article dicts
            model: Claude model to use
            max_concurrent: Maximum concurrent processes
            temperature: Temperature setting
            timeout: Optional timeout in seconds
            
        Returns:
            List of processed article summaries
        """
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for each article
        tasks = []
        for article in articles:
            task = self._process_single_article(
                article=article,
                model=model,
                semaphore=semaphore,
                temperature=temperature
            )
            tasks.append(task)
        
        # Wait for all tasks to complete with optional timeout
        if timeout:
            try:
                return await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Batch processing timed out after {timeout} seconds")
                # Return any completed results
                completed_results = []
                for task in tasks:
                    if task.done() and not task.exception():
                        completed_results.append(task.result())
                return completed_results
        else:
            # No timeout
            return await asyncio.gather(*tasks)
    
    async def _process_single_article(
        self,
        article: Dict[str, str],
        model: Optional[str],
        semaphore: asyncio.Semaphore,
        temperature: float = 0.3
    ) -> Dict[str, Dict[str, str]]:
        """
        Process a single article with rate limiting.
        
        Args:
            article: Article dict
            model: Claude model to use
            semaphore: Semaphore for concurrency control
            temperature: Temperature setting
            
        Returns:
            Article summary with original metadata
        """
        async with semaphore:
            # Apply rate limiting using the async method if available
            try:
                # First try the async acquire method
                if hasattr(self.rate_limiter, 'acquire_async'):
                    await self.rate_limiter.acquire_async()
                else:
                    # Fall back to running the synchronous method in a thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.rate_limiter.acquire)
            except Exception as e:
                self.logger.warning(f"Rate limiting error (proceeding anyway): {str(e)}")
            
            try:
                # Get article data
                text = article.get('text', article.get('content', ''))
                title = article.get('title', 'Untitled')
                url = article.get('url', article.get('link', '#'))
                
                # Run in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                summary = await loop.run_in_executor(
                    None,
                    lambda: self.summarize(
                        text=text,
                        title=title,
                        url=url,
                        model=model,
                        temperature=temperature
                    )
                )
                
                return {
                    'original': article,
                    'summary': summary
                }
                
            except Exception as e:
                self.logger.error(f"Error processing article: {str(e)}")
                return {
                    'original': article,
                    'summary': {
                        'headline': article.get('title', 'Error'),
                        'summary': f"Error generating summary: {str(e)}"
                    }
                }

    # Note: We're keeping these methods as they override BaseSummarizer methods
    def summarize_article(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        force_refresh: bool = False,
        temperature: float = 0.3,
    ) -> Dict[str, str]:
        """
        Generate a concise summary of the article text.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            model: Claude model to use (shorthand name or full identifier)
            force_refresh: Whether to force a new summary instead of using cache
            temperature: Temperature setting for generation (0.0-1.0)
            
        Returns:
            dict: The summary with headline and text
        """
        # Set up request-specific context for structured logging
        self.logger.add_context(
            operation="summarize_article",
            url=url,
            title=title,
            text_length=len(text),
            requested_model=model,
            temperature=temperature
        )
        
        try:
            # Check cache first if not forcing refresh
            if not force_refresh and self.cache:
                cache_key = f"{text}:{model or 'default'}:{temperature}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Retrieved summary from cache", cache_hit=True)
                    return cached_result
            
            # Clean the text first
            text = self.clean_text(text)
            
            # Extract source from URL for attribution
            source_name = self.extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = get_model_identifier(model)
            
            # Create the prompt
            prompt = create_summary_prompt(text, url, source_name)
            
            # Generate summary using Claude
            summary_text = self.call_api(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=400
            )
            
            # Parse the response
            result = parse_summary_response(summary_text, title, url, source_name)
            
            # Cache the result
            if self.cache:
                cache_key = f"{text}:{model_id}:{temperature}"
                self.cache.set(cache_key, result)
            
            self.logger.info(
                "Summary generated successfully", 
                headline_length=len(result['headline']),
                summary_length=len(result['summary'])
            )
            
            # Return the complete summary result
            return result
        except Exception as e:
            self.logger.exception(f"Error in summarize_article: {str(e)}")
            # Return a fallback summary
            source_name = self.extract_source_from_url(url)
            return {
                'headline': title,
                'summary': f"Failed to generate summary: {str(e)}\n\nSource: {source_name}\n{url}"
            }
        finally:
            # Clear context after the operation
            self.logger.clear_context()
    
    def summarize_article_streaming(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, Dict[str, str]]:
        """
        Generate a summary of the article with streaming response.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            model: Claude model to use (shorthand name or full identifier)
            callback: Optional callback function to process streamed chunks
            temperature: Temperature setting for generation (0.0-1.0)
            
        Yields:
            str: Chunks of the summary as they are generated
            
        Returns:
            dict: The complete summary with headline and text when finished
        """
        # Set up request-specific context for structured logging
        self.logger.add_context(
            operation="summarize_article_streaming",
            url=url,
            title=title,
            text_length=len(text),
            requested_model=model,
            temperature=temperature
        )
        
        try:
            # Clean the text first
            text = self.clean_text(text)

            # Extract source from URL for attribution
            source_name = self.extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = get_model_identifier(model)
            
            # Create the prompt
            prompt = create_summary_prompt(text, url, source_name)

            # Collect the full text as we stream
            full_text = ""
            
            # Generate summary using Claude with streaming
            for text_chunk in self.call_api_streaming(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=400
            ):
                # Append to full text
                full_text += text_chunk
                
                # Yield the chunk
                yield text_chunk
                
                # Call the callback if provided
                if callback:
                    try:
                        callback(text_chunk)
                    except Exception as callback_error:
                        self.logger.warning(f"Callback error (continuing streaming): {str(callback_error)}")
            
            # Parse the complete response
            result = parse_summary_response(full_text, title, url, source_name)
            
            # Cache the result
            if self.cache:
                cache_key = f"{text}:{model_id}:{temperature}"
                self.cache.set(cache_key, result)
            
            self.logger.info(
                "Streaming summary completed", 
                headline_length=len(result['headline']),
                summary_length=len(result['summary'])
            )
            
            return result
        except Exception as e:
            self.logger.exception(f"Error in summarize_article_streaming: {str(e)}")
            # Return a fallback summary
            source_name = self.extract_source_from_url(url)
            return {
                'headline': title,
                'summary': f"Failed to generate summary: {str(e)}\n\nSource: {source_name}\n{url}"
            }
        finally:
            # Clear context after the operation
            self.logger.clear_context()
    
    def summarize_long_article(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        force_refresh: bool = False,
        temperature: float = 0.3,
    ) -> Dict[str, str]:
        """
        Generate summary for long articles by chunking and meta-summarization.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            model: Claude model to use (shorthand name or full identifier)
            force_refresh: Whether to force a new summary instead of using cache
            temperature: Temperature setting for generation
            
        Returns:
            dict: The summary with headline and text
        """
        from summarization.text_processing import chunk_text
        
        # Clean the text first
        text = self.clean_text(text)
        
        # If text is short enough, use regular summarization
        if len(text) < 12000:
            return self.summarize_article(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature
            )
            
        # Split into chunks
        chunks = chunk_text(text)
        
        # Summarize each chunk
        chunk_summaries = []
        model_id = get_model_identifier(model)
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Summarizing chunk {i+1}/{len(chunks)} for {url}")
            
            # Create a chunk-specific prompt
            chunk_prompt = (
                f"Summarize this section (section {i+1} of {len(chunks)}) of an article in 2-3 sentences, "
                "capturing the key points and facts:\n\n"
                f"{chunk}"
            )
            
            # Get summary for this chunk
            chunk_summary = self.call_api(
                model_id=model_id,
                prompt=chunk_prompt,
                temperature=temperature,
                max_tokens=150
            )
            
            chunk_summaries.append(chunk_summary)
            
        # Create a meta-summary from the chunk summaries
        combined_chunks = "\n\n".join([
            f"Section {i+1} summary: {summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        source_name = self.extract_source_from_url(url)
        
        meta_prompt = (
            "Based on these section summaries, create a coherent overall summary "
            "of the complete article following these guidelines:\n\n"
            "1. First line: Create a headline in sentence case\n"
            "2. Then a blank line\n"
            "3. Then a summary of three to five sentences that captures the key points\n"
            "4. Then a blank line\n"
            "5. Then add 'Source: [publication name]' followed by the URL\n\n"
            f"Article title: {title}\n\n"
            f"Section summaries:\n{combined_chunks}\n\n"
            f"URL: {url}\n"
            f"Publication: {source_name}"
        )
        
        # Generate the final meta-summary
        final_summary = self.call_api(
            model_id=model_id,
            prompt=meta_prompt,
            temperature=temperature,
            max_tokens=400
        )
        
        # Parse the final summary
        result = parse_summary_response(
            final_summary, 
            title, 
            url, 
            source_name
        )
        
        return result

def create_fast_summarizer(
    original_summarizer=None,
    api_key=None,
    rpm_limit=50,
    cache_size=256,
    max_batch_workers=3
):
    """
    Factory function to create a configured FastSummarizer instance.
    
    Args:
        original_summarizer: An existing summarizer to build upon
        api_key: API key for Anthropic
        rpm_limit: Rate limit in requests per minute
        cache_size: Cache size
        max_batch_workers: Max concurrent workers
        
    Returns:
        FastSummarizer instance
    """
    # Create the summarizer
    summarizer = FastSummarizer(
        api_key=api_key,
        rpm_limit=rpm_limit,
        cache_size=cache_size,
        max_batch_workers=max_batch_workers
    )
    
    # If an original summarizer is provided, copy its settings
    if original_summarizer:
        # Copy other attributes if they exist
        if hasattr(original_summarizer, 'cache'):
            summarizer.cache = original_summarizer.cache
    
    return summarizer