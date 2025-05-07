"""
Fast and optimized summarizer implementation with batch processing.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any

from common.batch_processing import BatchProcessor
from common.errors import retry_with_backoff  # Add this line
from models.selection import get_model_identifier, estimate_complexity, select_model_by_complexity
from summarization.article_summarizer import ArticleSummarizer
from cache.tiered_cache import TieredCache
from api.rate_limiter import RateLimiter

class FastSummarizer:
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
            clean_text = self.summarizer.clean_text(text)
            complexity = estimate_complexity(clean_text)
            model = select_model_by_complexity(complexity)
            
        # Choose approach based on text length
        if len(text) > 12000:
            self.logger.info(f"Using long article approach for {url} ({len(text)} chars)")
            return self.summarizer.summarize_long_article(
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
                text = self.summarizer.clean_text(text)
                
                # Estimate complexity and select model
                complexity = estimate_complexity(text)
                selected_model = select_model_by_complexity(complexity)
                
                # Add to group
    """
    Base class for article summarization with core functionality.
    """
    
    def __init__(self, api_key, cache=None):
        """
        Initialize the base summarizer.
        
        Args:
            api_key: Anthropic API key
            cache: Optional cache implementation
        """
        self.logger = StructuredLogger(__name__)
        self.client = Anthropic(api_key=api_key)
        self.cache = cache
    
    def clean_text(self, text: str) -> str:
        """Clean HTML and normalize text for summarization."""
        return clean_text(text)
    
    def extract_source_from_url(self, url: str) -> str:
        """Extract publication name from URL."""
        return extract_source_from_url(url)
    
    @retry_with_backoff(max_retries=3, initial_backoff=2)
    def call_api(self, model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Call the Claude API with retry logic.
        
        Args:
            model_id: Claude model identifier
            prompt: The prompt to send
            temperature: Temperature setting
            max_tokens: Maximum tokens for the response
            
        Returns:
            The response text from Claude
        """
        try:
            self.logger.info(f"Calling Claude API with model {model_id}")
            start_time = time.time()
            
            response = self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"API call completed in {elapsed_time:.2f}s")
            
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}")
            # Convert to our custom exceptions
            if "rate limit" in str(e).lower():
                raise RateLimitError(str(e))
            elif "auth" in str(e).lower():
                raise AuthenticationError(str(e))
            elif "connect" in str(e).lower():
                raise ConnectionError(str(e))
            else:
                raise APIError(str(e))
    
    @retry_with_backoff(max_retries=2, initial_backoff=1)
    def call_api_streaming(self, model_id: str, prompt: str, 
                          temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """
        Call the Claude API with streaming.
        
        Args:
            model_id: Claude model identifier
            prompt: The prompt to send
            temperature: Temperature setting
            max_tokens: Maximum tokens for the response
            
        Yields:
            Text chunks from the Claude API
        """
        try:
            self.logger.info(f"Starting streaming API call with model {model_id}")
            start_time = time.time()
            
            stream = self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            # Process each chunk
            for chunk in stream:
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    yield chunk.delta.text
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Streaming API call completed in {elapsed_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Streaming API call failed: {str(e)}")
            # Convert to our custom exceptions
            if "rate limit" in str(e).lower():
                raise RateLimitError(str(e))
            elif "auth" in str(e).lower():
                raise AuthenticationError(str(e))
            elif "connect" in str(e).lower():
                raise ConnectionError(str(e))
            else:
                raise APIError(str(e))
    
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