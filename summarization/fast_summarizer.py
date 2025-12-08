"""
Fast and optimized summarizer implementation with batch processing.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Generator, Callable

from common.batch_processing import BatchProcessor
from common.errors import retry_with_backoff
from models.selection import get_model_identifier, estimate_complexity, select_model_by_complexity
from summarization.article_summarizer import ArticleSummarizer
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
        rate_limit_delay=0.5,
        original_summarizer=None
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
            original_summarizer: Existing summarizer to build upon
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise APIError("Anthropic API key not found")
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # Create or store the article summarizer
        if original_summarizer:
            self.summarizer = original_summarizer
        else:
            self.summarizer = ArticleSummarizer(api_key=api_key)
        
        # Set up advanced cache
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache = TieredCache(
            memory_size=cache_size,
            disk_path=cache_dir,
            ttl_days=ttl_days
        )
        
        # Update the summarizer's cache
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
        style: str = "default",  # Add style parameter
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
            style: Summary style ('default', 'bullet', 'newswire')
            
        Returns:
            dict: The summary with headline and text
        """
        # Handle None or empty parameters early
        if not text:
            self.logger.warning("Empty text provided for summarization")
            text = ""
        
        if not title:
            self.logger.warning("Empty title provided for summarization")
            title = "Untitled Article"
        
        if not url:
            self.logger.warning("Empty URL provided for summarization")
            url = "#"
        
        # Auto-select model if requested
        if auto_select_model and not model:
            clean_text_content = clean_text(text)
            complexity = estimate_complexity(clean_text_content)
            model = select_model_by_complexity(complexity)
            
        # Choose approach based on text length
        if len(text) > 12000:
            self.logger.info(f"Using long article approach for {url} ({len(text)} chars)")
            return self._summarize_long_article(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature,
                style=style  # Pass style parameter
            )
        else:
            self.logger.info(f"Using standard approach for {url} ({len(text)} chars)")
            # Now use our own implementation instead of delegating to summarizer
            return self._summarize_article(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature,
                style=style  # Pass style parameter
            )
    
    async def batch_summarize(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        model: Optional[str] = None,
        auto_select_model: bool = True,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
        style: str = "default",  # Add style parameter
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
            style: Summary style ('default', 'bullet', 'newswire')
            
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
                text = clean_text(text or "")
                
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
                    timeout=timeout,
                    style=style  # Pass style parameter
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
                timeout=timeout,
                style=style  # Pass style parameter
            )
    
    async def _process_article_group(
        self,
        articles: List[Dict[str, str]],
        model: Optional[str] = None,
        max_concurrent: int = 3,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
        style: str = "default",  # Add style parameter
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Process a group of articles with the same model.
        
        Args:
            articles: List of article dicts
            model: Claude model to use
            max_concurrent: Maximum concurrent processes
            temperature: Temperature setting
            timeout: Optional timeout in seconds
            style: Summary style ('default', 'bullet', 'newswire')
            
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
                temperature=temperature,
                style=style  # Pass style parameter
            )
            tasks.append(task)
        
        # Wait for all tasks to complete with optional timeout and error handling
        if timeout:
            try:
                # Use return_exceptions=True to get partial results even if some fail
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Batch processing timed out after {timeout} seconds")
                # Return any completed results
                completed_results = []
                for task in tasks:
                    if task.done():
                        try:
                            result = task.result()
                            if not isinstance(result, Exception):
                                completed_results.append(result)
                        except Exception:
                            pass  # Skip failed tasks
                return completed_results
        else:
            # No timeout - use return_exceptions to get partial results
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        successful_results = []
        failed_count = 0
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                self.logger.error(f"Task {idx} failed with exception: {str(result)}")
            else:
                successful_results.append(result)

        if failed_count > 0:
            self.logger.warning(
                f"Batch completed with {failed_count} failures, "
                f"{len(successful_results)} successes"
            )

        return successful_results
    
    async def _process_single_article(
        self,
        article: Dict[str, str],
        model: Optional[str],
        semaphore: asyncio.Semaphore,
        temperature: float = 0.3,
        style: str = "default",  # Add style parameter
    ) -> Dict[str, Dict[str, str]]:
        """
        Process a single article with rate limiting.
        
        Args:
            article: Article dict
            model: Claude model to use
            semaphore: Semaphore for concurrency control
            temperature: Temperature setting
            style: Summary style ('default', 'bullet', 'newswire')
            
        Returns:
            Article summary with original metadata
        """
        async with semaphore:
            # Apply rate limiting
            try:
                await self.rate_limiter.acquire()
            except Exception as e:
                self.logger.warning(f"Rate limiting error (proceeding anyway): {str(e)}")
            
            try:
                # Get article data with safe defaults
                text = article.get('text', article.get('content', '')) or ""
                title = article.get('title', '') or "Untitled Article"
                url = article.get('url', article.get('link', '')) or "#"
                
                # Run in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                summary = await loop.run_in_executor(
                    None,
                    lambda: self.summarize(
                        text=text,
                        title=title,
                        url=url,
                        model=model,
                        temperature=temperature,
                        style=style  # Pass style parameter
                    )
                )
                
                return {
                    'original': article,
                    'summary': summary
                }
                
            except Exception as e:
                self.logger.error(f"Error processing article {article.get('url', 'unknown')}: {str(e)}")
                # Return error summary instead of raising
                source_name = extract_source_from_url(article.get('url', '#'))
                return {
                    'original': article,
                    'summary': {
                        'headline': article.get('title', 'Error'),
                        'summary': f"Error generating summary: {str(e)}\n\nSource: {source_name}\n{article.get('url', '#')}",
                        'style': style,
                        'error': True  # Mark as error for filtering if needed
                    }
                }
    
    def _summarize_article(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        force_refresh: bool = False,
        temperature: float = 0.3,
        style: str = "default",  # Add style parameter
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
            style: Summary style ('default', 'bullet', 'newswire')
            
        Returns:
            dict: The summary with headline and text
        """
        # Handle None or empty parameters
        if not text:
            self.logger.warning("Empty text provided for summarization")
            text = ""
        
        if not title:
            self.logger.warning("Empty title provided for summarization")
            title = "Untitled Article"
        
        if not url:
            self.logger.warning("Empty URL provided for summarization")
            url = "#"
        
        # Set up request-specific context for structured logging
        if hasattr(self.logger, 'add_context'):
            self.logger.add_context(
                operation="summarize_article",
                url=url,
                title=title,
                text_length=len(text),
                requested_model=model,
                temperature=temperature,
                style=style  # Add style to logging
            )
        
        try:
            # Check cache first if not forcing refresh
            if not force_refresh and self.cache:
                cache_key = f"{text}:{model or 'default'}:{temperature}:{style}"  # Update cache key
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Retrieved summary from cache")
                    return cached_result
            
            # Clean the text first
            text = clean_text(text)
            
            # Extract source from URL for attribution
            source_name = extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = get_model_identifier(model)
            
            # Create the prompt with style
            prompt = create_summary_prompt(text, url, source_name, style)
            
            # Generate summary using Claude
            summary_text = self._call_api(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=400
            )
            
            # Parse the response
            result = parse_summary_response(summary_text, title, url, source_name, style)
            
            # Cache the result
            if self.cache:
                cache_key = f"{text}:{model_id}:{temperature}:{style}"
                self.cache.set(cache_key, result)
            
            # Modified: Use standard string formatting for logging
            self.logger.info(
                f"Summary generated successfully - Style: {style}, Headline length: {len(result['headline'])}, "
                f"Summary length: {len(result['summary'])}"
            )
            
            # Return the complete summary result
            return result
        except Exception as e:
            self.logger.exception(f"Error in summarize_article: {str(e)}")
            # Return a fallback summary
            source_name = extract_source_from_url(url)
            return {
                'headline': title,
                'summary': f"Failed to generate summary: {str(e)}\n\nSource: {source_name}\n{url}",
                'style': style
            }
        finally:
            # Clear context after the operation
            if hasattr(self.logger, 'clear_context'):
                self.logger.clear_context()
    
    def _summarize_article_streaming(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        temperature: float = 0.3,
        style: str = "default",  # Add style parameter
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
            style: Summary style ('default', 'bullet', 'newswire')
            
        Yields:
            str: Chunks of the summary as they are generated
            
        Returns:
            dict: The complete summary with headline and text when finished
        """
        # Handle None or empty parameters
        if not text:
            self.logger.warning("Empty text provided for streaming summarization")
            text = ""
        
        if not title:
            self.logger.warning("Empty title provided for streaming summarization")
            title = "Untitled Article"
        
        if not url:
            self.logger.warning("Empty URL provided for streaming summarization")
            url = "#"
        
        # Set up request-specific context for structured logging
        if hasattr(self.logger, 'add_context'):
            self.logger.add_context(
                operation="summarize_article_streaming",
                url=url,
                title=title,
                text_length=len(text),
                requested_model=model,
                temperature=temperature,
                style=style  # Add style to logging
            )
        
        try:
            # Clean the text first
            text = clean_text(text)

            # Extract source from URL for attribution
            source_name = extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = get_model_identifier(model)
            
            # Create the prompt with style
            prompt = create_summary_prompt(text, url, source_name, style)

            # Collect the full text as we stream
            full_text = ""
            
            # Generate summary using Claude with streaming
            for text_chunk in self._call_api_streaming(
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
            result = parse_summary_response(full_text, title, url, source_name, style)
            
            # Cache the result
            if self.cache:
                cache_key = f"{text}:{model_id}:{temperature}:{style}"
                self.cache.set(cache_key, result)
            
            # Modified: Use standard string formatting for logging
            self.logger.info(
                f"Streaming summary completed - Style: {style}, Headline length: {len(result['headline'])}, "
                f"Summary length: {len(result['summary'])}"
            )
            
            return result
        except Exception as e:
            self.logger.exception(f"Error in summarize_article_streaming: {str(e)}")
            # Return a fallback summary
            source_name = extract_source_from_url(url)
            return {
                'headline': title,
                'summary': f"Failed to generate summary: {str(e)}\n\nSource: {source_name}\n{url}",
                'style': style
            }
        finally:
            # Clear context after the operation
            if hasattr(self.logger, 'clear_context'):
                self.logger.clear_context()
    
    def _summarize_long_article(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        force_refresh: bool = False,
        temperature: float = 0.3,
        style: str = "default",  # Add style parameter
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
            style: Summary style ('default', 'bullet', 'newswire')
            
        Returns:
            dict: The summary with headline and text
        """
        from summarization.text_processing import chunk_text
        
        # Handle None or empty parameters
        if not text:
            text = ""
        
        if not title:
            title = "Untitled Article"
        
        if not url:
            url = "#"
        
        # Clean the text first
        text = clean_text(text)
        
        # If text is short enough, use regular summarization
        if len(text) < 12000:
            return self._summarize_article(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                temperature=temperature,
                style=style  # Pass style parameter
            )
            
        # Split into chunks
        chunks = chunk_text(text)

        # Summarize each chunk in parallel
        model_id = get_model_identifier(model)

        # Create an event loop for parallel processing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Use async to parallelize chunk summarization
        chunk_summaries = loop.run_until_complete(
            self._summarize_chunks_parallel(chunks, model_id, temperature, url)
        )
            
        # Create a meta-summary from the chunk summaries
        combined_chunks = "\n\n".join([
            f"Section {i+1} summary: {summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        source_name = extract_source_from_url(url)
        
        # Adjust meta prompt based on style
        if style == "bullet":
            meta_prompt = (
                "Based on these section summaries, create a coherent overall bullet-point summary "
                "of the complete article following these guidelines:\n\n"
                "1. First line: Create a headline in sentence case\n"
                "2. Then a blank line\n"
                "3. Then a bullet point summary with 3-5 key points (use • for bullets)\n"
                "4. Then a blank line\n"
                "5. Then add 'Source: [publication name]' followed by the URL\n\n"
                f"Article title: {title}\n\n"
                f"Section summaries:\n{combined_chunks}\n\n"
                f"URL: {url}\n"
                f"Publication: {source_name}"
            )
        elif style == "newswire":
            meta_prompt = (
                "Based on these section summaries, create a coherent overall newswire-style summary "
                "of the complete article following these guidelines:\n\n"
                "1. First line: Create a headline in sentence case\n"
                "2. Then a blank line\n"
                "3. First paragraph: A concise lead that answers who, what, when, where, and why\n"
                "4. Following paragraphs: 2-3 short paragraphs with supporting details\n"
                "5. Then a blank line\n"
                "6. Then add 'Source: [publication name]' followed by the URL\n\n"
                f"Article title: {title}\n\n"
                f"Section summaries:\n{combined_chunks}\n\n"
                f"URL: {url}\n"
                f"Publication: {source_name}"
            )
        else:  # default style
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
        final_summary = self._call_api(
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
            source_name,
            style  # Pass style parameter
        )
        
        return result

    async def _summarize_chunks_parallel(
        self,
        chunks: List[str],
        model_id: str,
        temperature: float,
        url: str
    ) -> List[str]:
        """
        Summarize multiple chunks in parallel with concurrency control.

        Args:
            chunks: List of text chunks to summarize
            model_id: Claude model identifier
            temperature: Temperature setting
            url: Article URL for logging

        Returns:
            List of chunk summaries
        """
        # Limit concurrent API calls to 3
        semaphore = asyncio.Semaphore(3)

        async def summarize_chunk(i: int, chunk: str) -> tuple:
            """Summarize a single chunk."""
            async with semaphore:
                self.logger.info(f"Summarizing chunk {i+1}/{len(chunks)} for {url}")

                # Create chunk-specific prompt
                chunk_prompt = (
                    f"Summarize this section (section {i+1} of {len(chunks)}) of an article in 2-3 sentences, "
                    "capturing the key points and facts:\n\n"
                    f"{chunk}"
                )

                # Run API call in thread executor to avoid blocking
                loop = asyncio.get_event_loop()
                chunk_summary = await loop.run_in_executor(
                    None,
                    lambda: self._call_api(
                        model_id=model_id,
                        prompt=chunk_prompt,
                        temperature=temperature,
                        max_tokens=150
                    )
                )

                return (i, chunk_summary)

        # Create tasks for all chunks
        tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Sort by index and extract summaries
        results.sort(key=lambda x: x[0])
        return [summary for _, summary in results]

    @retry_with_backoff(max_retries=3, initial_backoff=2)
    def _call_api(self, model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
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
    def _call_api_streaming(self, model_id: str, prompt: str, 
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
        max_batch_workers=max_batch_workers,
        original_summarizer=original_summarizer
    )
    
    return summarizer