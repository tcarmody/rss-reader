"""
Base class for article summarization.
"""

import logging
import time
from typing import Dict, Optional, Generator, Callable, List, Any
from anthropic import Anthropic

from common.errors import retry_with_backoff, APIError, RateLimitError, AuthenticationError, ConnectionError
from common.logging import StructuredLogger
from models.selection import get_model_identifier
from summarization.text_processing import (
    clean_text, extract_source_from_url, create_summary_prompt, 
    get_system_prompt, parse_summary_response
)

class BaseSummarizer:
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
    
    def summarize_article(
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
            summary_text = self.call_api(
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
    
    def summarize_article_streaming(
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
            result = parse_summary_response(full_text, title, url, source_name, style)
            
            # Cache the result
            if self.cache:
                cache_key = f"{text}:{model_id}:{temperature}:{style}"
                self.cache.set(cache_key, result)
            
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
    
    @retry_with_backoff(max_retries=3, initial_backoff=2)
    def call_api(self, model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call the Claude API with retry logic."""
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
        """Call the Claude API with streaming."""
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