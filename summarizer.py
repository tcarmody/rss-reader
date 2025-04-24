"""Article summarization with Anthropic Claude API with model selection and streaming."""

import re
import html
import logging
import time
import anthropic
import os
from typing import Dict, List, Optional, Union, Generator, Callable, Any
from datetime import datetime
from functools import wraps
from enum import Enum

from bs4 import BeautifulSoup

from utils.config import get_env_var
from cache import SummaryCache


# Custom exception classes for more granular error handling
class SummarizerError(Exception):
    """Base exception for ArticleSummarizer errors."""
    pass


class APIConnectionError(SummarizerError):
    """Raised when the API connection fails."""
    pass


class APIResponseError(SummarizerError):
    """Raised when the API returns an unexpected response."""
    pass


class APIRateLimitError(SummarizerError):
    """Raised when the API enforces rate limiting."""
    pass


class APIAuthError(SummarizerError):
    """Raised when API authentication fails."""
    pass


class TextProcessingError(SummarizerError):
    """Raised when text processing fails."""
    pass


class ModelSelectionError(SummarizerError):
    """Raised when an invalid model is specified."""
    pass


# Logger setup for structured logging
class StructuredLogger:
    """Logger that provides structured context with each log entry."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def add_context(self, **kwargs):
        """Add context to be included in all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context data."""
        self.context = {}
    
    def _format_context(self, extra_context=None):
        """Format context for logging."""
        context = self.context.copy()
        if extra_context:
            context.update(extra_context)
        
        timestamp = datetime.now().isoformat()
        context_str = ' '.join(f"{k}={v}" for k, v in context.items())
        return f"[{timestamp}] {context_str}"
    
    def debug(self, msg, **kwargs):
        """Log debug message with context."""
        self.logger.debug(f"{msg} {self._format_context(kwargs)}")
    
    def info(self, msg, **kwargs):
        """Log info message with context."""
        self.logger.info(f"{msg} {self._format_context(kwargs)}")
    
    def warning(self, msg, **kwargs):
        """Log warning message with context."""
        self.logger.warning(f"{msg} {self._format_context(kwargs)}")
    
    def error(self, msg, **kwargs):
        """Log error message with context."""
        self.logger.error(f"{msg} {self._format_context(kwargs)}")
    
    def exception(self, msg, exc_info=True, **kwargs):
        """Log exception with context."""
        self.logger.exception(f"{msg} {self._format_context(kwargs)}", exc_info=exc_info)


# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=3, initial_backoff=1, backoff_factor=2, 
                       retryable_exceptions=(APIConnectionError, APIRateLimitError)):
    """
    Decorator that implements retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries before giving up
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor by which the backoff time increases
        retryable_exceptions: Tuple of exceptions that should trigger retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from self if it exists, otherwise create one
            self = args[0] if args else None
            logger = getattr(self, 'logger', logging.getLogger(__name__))
            
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded", 
                                     error_type=type(e).__name__,
                                     final_attempt=True)
                        raise
                    
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}",
                                  error_type=type(e).__name__,
                                  backoff_time=backoff,
                                  retry_count=retries)
                    
                    time.sleep(backoff)
                    backoff *= backoff_factor
                    
                    # Add jitter to avoid thundering herd
                    backoff *= (0.8 + 0.4 * time.time() % 1)
        return wrapper
    return decorator


class ArticleSummarizer:
    """
    Summarizes articles using the Anthropic Claude API.
    
    This class handles:
    - Text cleaning and normalization
    - API communication with Claude
    - Caching of results to avoid redundant API calls
    - Tag generation for articles
    - Model selection capability
    - Streaming response support
    - Robust error handling with retries
    
    Example:
        summarizer = ArticleSummarizer()
        summary = summarizer.summarize_article(
            "Article text here...",
            "Article Title",
            "https://example.com/article",
            model="claude-3-sonnet-20240229"
        )
        
        # Or with streaming:
        for chunk in summarizer.summarize_article_streaming(
            "Article text here...",
            "Article Title",
            "https://example.com/article"
        ):
            print(chunk, end="", flush=True)
    """

    # Available Claude models
    AVAILABLE_MODELS = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229",
        "opus": "claude-3-opus-20240229",
        "haiku-legacy": "claude-2.0",
        "sonnet-legacy": "claude-2.1",
        "sonnet-3.5": "claude-3-5-sonnet-20240620",
        "haiku-3.5": "claude-3-5-haiku-20240307",
        "sonnet-3.7": "claude-3-7-sonnet-20250219"
    }
    
    # Default model to use (using the latest model as default)
    DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

    def __init__(self):
        """Initialize the summarizer with Claude API client."""
        try:
            api_key = get_env_var('ANTHROPIC_API_KEY')
            if not api_key:
                raise APIAuthError("Anthropic API key not found")
                
            self.client = anthropic.Anthropic(api_key=api_key)
            cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
            self.summary_cache = SummaryCache(cache_dir=cache_dir)
            self.logger = StructuredLogger("ArticleSummarizer")
        except Exception as e:
            # Convert to our custom exception type
            if "API key" in str(e):
                raise APIAuthError(f"Failed to initialize API client: {str(e)}")
            else:
                raise SummarizerError(f"Failed to initialize summarizer: {str(e)}")

    def clean_text(self, text: str) -> str:
        """
        Clean HTML and normalize text for summarization.
        
        Args:
            text: Raw text that may contain HTML
            
        Returns:
            Cleaned and normalized text
        """
        try:
            # Record text length for logging
            original_length = len(text)
            
            # Remove HTML tags
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()

            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Decode HTML entities
            text = html.unescape(text)
            
            # Log the transformation
            cleaned_length = len(text)
            self.logger.debug(
                "Text cleaned", 
                original_length=original_length,
                cleaned_length=cleaned_length,
                reduction_percent=round((original_length - cleaned_length) / original_length * 100 if original_length > 0 else 0)
            )

            return text
        except Exception as e:
            self.logger.error("Failed to clean text", error=str(e))
            raise TextProcessingError(f"Failed to clean text: {str(e)}")
    
    def _get_model(self, model: Optional[str] = None) -> str:
        """
        Get the Claude model identifier.
        
        Args:
            model: Model name or identifier
            
        Returns:
            Model identifier string to use with the API
        """
        try:
            if not model:
                return self.DEFAULT_MODEL
                
            # If a full model identifier is provided, use it directly
            if model.startswith("claude-"):
                return model
                
            # If a shorthand name is provided, look it up
            if model in self.AVAILABLE_MODELS:
                return self.AVAILABLE_MODELS[model]
                
            # If not found, log a warning and return the default
            available_models = ", ".join(self.AVAILABLE_MODELS.keys())
            self.logger.warning(
                f"Model '{model}' not found. Using default model.", 
                requested_model=model,
                default_model=self.DEFAULT_MODEL,
                available_models=available_models
            )
            return self.DEFAULT_MODEL
        except Exception as e:
            self.logger.error("Failed to select model", error=str(e), requested_model=model)
            raise ModelSelectionError(f"Failed to select model '{model}': {str(e)}")
    
    def _extract_source_from_url(self, url: str) -> str:
        """
        Extract publication name from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Publication name
        """
        try:
            source_name = url.split('//')[1].split('/')[0] if '//' in url else url
            source_name = source_name.replace('www.', '')
            self.logger.debug("Extracted source", url=url, source=source_name)
            return source_name
        except Exception as e:
            self.logger.warning(
                "Failed to extract source from URL, using fallback", 
                url=url, 
                error=str(e)
            )
            return url.replace('https://', '').replace('http://', '')

    def _create_summary_prompt(self, text: str, url: str, source_name: str) -> str:
        """
        Create the prompt for article summarization.
        
        Args:
            text: Cleaned article text
            url: Article URL
            source_name: Publication source name
            
        Returns:
            Formatted prompt for Claude
        """
        # Log token usage estimate (rough approximation)
        text_tokens = len(text.split())
        self.logger.debug(
            "Creating summary prompt", 
            article_tokens=text_tokens,
            url=url
        )
        
        return (
            "Summarize the article below following these guidelines:\n\n"
            "Structure:\n"
            "1. First line: Create a headline in sentence case\n"
            "2. Then a blank line\n"
            "3. Then a summary of three to five sentences that:\n"
            "   - Presents key information directly and factually\n"
            "   - Includes technical details relevant to AI developers\n"
            "   - Covers implications for the AI industry or technology landscape\n"
            "   - Mentions price and availability details for new models/tools (if applicable)\n"
            "4. Then a blank line\n"
            "5. Then add 'Source: [publication name]' followed by the URL\n\n"
            "Style guidelines:\n"
            "- Use active voice (e.g., 'Company released product' not 'Product was released by company')\n"
            "- Use non-compound verbs (e.g., 'banned' instead of 'has banned')\n"
            "- Avoid self-explanatory phrases like 'This article explains...', 'This is important because...', or 'The author discusses...'\n"
            "- Present information directly without meta-commentary\n"
            "- Avoid the words 'content' and 'creator'\n"
            "- Spell out numbers (e.g., '8 billion' not '8B', '100 million' not '100M')\n"
            "- Spell out 'percent' instead of using the '%' symbol\n"
            "- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods\n"
            "- Use smart quotes, not straight quotes\n"
            "- Ensure the headline doesn't repeat too many words from the summary\n\n"
            f"Article:\n{text}\n\n"
            f"URL: {url}\n"
            f"Publication: {source_name}"
        )
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for Claude.
        
        Returns:
            System prompt string
        """
        return (
            "You are an expert at creating summaries of articles. Your summaries should be "
            "factual, informative, concise, and written in a direct journalistic style. "
            "Avoid meta-language or self-explanatory phrases like 'This article explains...', "
            "'This is important for AI developers because...', or 'The author discusses...'. "
            "Instead, present information directly and factually. Write in a clear, "
            "straightforward manner without exaggeration, hype, or marketing speak. "
            "Focus on conveying the key points and implications without explicitly stating that you're doing so."
        )

    def _parse_summary_response(self, summary_text: str, title: str, url: str, source_name: str) -> Dict[str, str]:
        """
        Parse the summary response from Claude.
        
        Args:
            summary_text: Raw summary text from Claude
            title: Original article title
            url: Article URL
            source_name: Publication source name
            
        Returns:
            Dictionary with headline and summary
        """
        try:
            # Split into headline and summary with source attribution
            parts = summary_text.split('\n\n')
            
            # Handle different possible formats
            if len(parts) >= 3:  # Proper format with headline, summary, and source
                headline = parts[0].strip()
                summary = parts[1].strip()
                source_info = parts[2].strip()
                format_type = "standard"
            elif len(parts) == 2:  # Missing source or other format issue
                headline = parts[0].strip()
                summary = parts[1].strip()
                source_info = f"Source: {source_name}\n{url}"
                format_type = "missing_source"
            else:  # Fallback if formatting is completely off
                lines = summary_text.split('\n', 1)
                if len(lines) == 2:
                    headline = lines[0].strip()
                    summary = lines[1].strip()
                    format_type = "fallback_newline"
                else:
                    headline = title
                    summary = summary_text
                    format_type = "complete_fallback"
                source_info = f"Source: {source_name}\n{url}"
                
            # Ensure the summary has the source information
            if not summary.endswith(url):
                summary = f"{summary}\n\n{source_info}"

            self.logger.debug(
                "Parsed summary response", 
                format_type=format_type,
                headline_length=len(headline),
                summary_length=len(summary),
                url=url
            )

            return {
                'headline': headline,
                'summary': summary
            }
        except Exception as e:
            self.logger.error(
                "Failed to parse summary response", 
                error=str(e),
                response_length=len(summary_text) if summary_text else 0,
                url=url
            )
            # Return a fallback summary
            return {
                'headline': title,
                'summary': f"Failed to parse summary: {str(e)}\n\nSource: {source_name}\n{url}"
            }

    @retry_with_backoff(max_retries=3, initial_backoff=2)
    def _call_claude_api(self, model_id: str, prompt: str, temperature: float, max_tokens: int) -> str:
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
        # Set up context for structured logging
        self.logger.add_context(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_tokens=len(prompt.split())
        )
        
        try:
            self.logger.info("Calling Claude API")
            start_time = time.time()
            
            response = self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._get_system_prompt(),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                "Claude API call completed", 
                elapsed_time=round(elapsed_time, 2),
                response_tokens=len(response.content[0].text.split())
            )
            
            return response.content[0].text
        except anthropic.APIError as e:
            # Map Anthropic exception types to our custom exceptions
            error_type = str(e.__class__.__name__)
            status_code = getattr(e, 'status_code', None)
            
            self.logger.error(
                "Claude API error", 
                error_type=error_type,
                status_code=status_code,
                error=str(e)
            )
            
            if status_code == 429:
                raise APIRateLimitError(f"Rate limit exceeded: {str(e)}")
            elif status_code == 401:
                raise APIAuthError(f"Authentication failed: {str(e)}")
            elif status_code and 500 <= status_code < 600:
                raise APIConnectionError(f"Claude API server error: {str(e)}")
            else:
                raise APIResponseError(f"Claude API error: {str(e)}")
        except Exception as e:
            self.logger.error(
                "Unexpected error calling Claude API", 
                error_type=str(e.__class__.__name__),
                error=str(e)
            )
            raise APIConnectionError(f"Failed to call Claude API: {str(e)}")
        finally:
            # Clear context after the API call
            self.logger.clear_context()

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
            # Clean the text first
            text = self.clean_text(text)

            # Generate cache key that includes model information
            model_id = self._get_model(model)
            cache_key = f"{text}:{model_id}:{temperature}"

            # Check cache first
            if not force_refresh:
                cached_summary = self.summary_cache.get(cache_key)
                if cached_summary:
                    self.logger.info("Using cached summary")
                    return cached_summary['summary']

            # Extract source from URL for attribution
            source_name = self._extract_source_from_url(url)
            
            # Create the prompt
            prompt = self._create_summary_prompt(text, url, source_name)

            # Log the request (without full text for brevity)
            self.logger.info(
                "Requesting summary", 
                model=model_id,
                source=source_name
            )

            # Generate summary using Claude
            summary_text = self._call_claude_api(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=400
            )
            
            # Parse the response
            result = self._parse_summary_response(summary_text, title, url, source_name)
            
            # Cache the result
            self.summary_cache.set(cache_key, {'summary': result})
            
            self.logger.info(
                "Summary generated successfully", 
                headline_length=len(result['headline']),
                summary_length=len(result['summary'])
            )
            
            return result

        except SummarizerError as e:
            # Log and re-raise our custom exceptions
            self.logger.exception(
                f"Summarization error: {str(e)}", 
                error_type=type(e).__name__
            )
            raise
        except Exception as e:
            # Log and wrap unexpected exceptions
            self.logger.exception(
                f"Unexpected error in summarize_article: {str(e)}",
                error_type=type(e).__name__
            )
            return {
                'headline': title,
                'summary': f"Summary generation failed: {str(e)}. Please try again later."
            }
        finally:
            # Clear context after the operation
            self.logger.clear_context()

    @retry_with_backoff(max_retries=3, initial_backoff=2)
    def _call_claude_api_streaming(self, model_id: str, prompt: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
        """
        Call the Claude API with streaming and retry logic.
        
        Args:
            model_id: Claude model identifier
            prompt: The prompt to send
            temperature: Temperature setting
            max_tokens: Maximum tokens for the response
            
        Yields:
            Text chunks from the Claude API
        """
        # Set up context for structured logging
        self.logger.add_context(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_tokens=len(prompt.split()),
            stream_mode=True
        )
        
        try:
            self.logger.info("Starting Claude API streaming request")
            start_time = time.time()
            chunk_count = 0
            total_chars = 0
            
            # Start the streaming request
            with self.client.messages.stream(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._get_system_prompt(),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            ) as stream:
                # Process each chunk
                for chunk in stream:
                    if chunk.type == "content_block_delta" and chunk.delta.text:
                        # Get the text chunk
                        text_chunk = chunk.delta.text
                        chunk_count += 1
                        total_chars += len(text_chunk)
                        
                        # Yield the chunk
                        yield text_chunk
                        
                        # Log progress periodically
                        if chunk_count % 10 == 0:
                            elapsed = time.time() - start_time
                            self.logger.debug(
                                "Streaming progress", 
                                chunks=chunk_count, 
                                total_chars=total_chars,
                                elapsed_seconds=round(elapsed, 2)
                            )
            
            # Log completion
            elapsed_time = time.time() - start_time
            self.logger.info(
                "Claude API streaming completed", 
                elapsed_time=round(elapsed_time, 2),
                total_chunks=chunk_count,
                total_chars=total_chars,
                chars_per_second=round(total_chars/elapsed_time, 2) if elapsed_time > 0 else 0
            )
            
        except anthropic.APIError as e:
            # Map Anthropic exception types to our custom exceptions
            error_type = str(e.__class__.__name__)
            status_code = getattr(e, 'status_code', None)
            
            self.logger.error(
                "Claude API streaming error", 
                error_type=error_type,
                status_code=status_code,
                error=str(e)
            )
            
            if status_code == 429:
                raise APIRateLimitError(f"Rate limit exceeded: {str(e)}")
            elif status_code == 401:
                raise APIAuthError(f"Authentication failed: {str(e)}")
            elif status_code and 500 <= status_code < 600:
                raise APIConnectionError(f"Claude API server error: {str(e)}")
            else:
                raise APIResponseError(f"Claude API error: {str(e)}")
        except Exception as e:
            self.logger.error(
                "Unexpected error in Claude API streaming", 
                error_type=str(e.__class__.__name__),
                error=str(e)
            )
            raise APIConnectionError(f"Failed to stream from Claude API: {str(e)}")
        finally:
            # Clear context after the API call
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
            source_name = self._extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = self._get_model(model)
            
            # Create the prompt
            prompt = self._create_summary_prompt(text, url, source_name)

            # Log the request
            self.logger.info(
                "Requesting streaming summary", 
                model=model_id,
                source=source_name
            )

            # Collect the full text as we stream
            full_text = ""
            
            # Generate summary using Claude with streaming
            for text_chunk in self._call_claude_api_streaming(
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
                        self.logger.warning(
                            "Callback error (continuing streaming)", 
                            error=str(callback_error)
                        )
            
            # Parse the complete response
            result = self._parse_summary_response(full_text, title, url, source_name)
            
            # Cache the result (using the same cache key format as non-streaming)
            cache_key = f"{text}:{model_id}:{temperature}"
            self.summary_cache.set(cache_key, {'summary': result})
            
            self.logger.info(
                "Streaming summary completed", 
                headline_length=len(result['headline']),
                summary_length=len(result['summary'])
            )
            
            # Return the complete summary result
            return result

        except SummarizerError as e:
            # Log and handle our custom exceptions
            self.logger.exception(
                f"Streaming summarization error: {str(e)}", 
                error_type=type(e).__name__
            )
            error_message = f"Summary generation failed: {str(e)}. Please try again later."
            yield error_message
            if callback:
                try:
                    callback(error_message)
                except:
                    pass
                    
            return {
                'headline': title,
                'summary': error_message
            }
        except Exception as e:
            # Log and handle unexpected exceptions
            self.logger.exception(
                f"Unexpected error in summarize_article_streaming: {str(e)}",
                error_type=type(e).__name__
            )
            error_message = f"Summary generation failed: {str(e)}. Please try again later."
            yield error_message
            if callback:
                try:
                    callback(error_message)
                except:
                    pass
                    
            return {
                'headline': title,
                'summary': error_message
            }
        finally:
            # Clear context after the operation
            self.logger.clear_context()

    @retry_with_backoff(max_retries=2, initial_backoff=1)
    def generate_tags(
        self, 
        content: str,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate tags for an article using Claude.
        
        Args:
            content: Article content to extract tags from
            model: Claude model to use (shorthand name or full identifier)
            temperature: Temperature setting for generation (0.0-1.0)
            
        Returns:
            list: Generated tags as strings
        """
        # Set up request-specific context for structured logging
        self.logger.add_context(
            operation="generate_tags",
            content_length=len(content),
            requested_model=model,
            temperature=temperature
        )
        
        try:
            # Clean the text for better tag extraction
            content = self.clean_text(content)
            
            # Get the actual model identifier
            model_id = self._get_model(model)
            
            self.logger.info("Generating tags", model=model_id)
            
            response = self.client.messages.create(
                model=model_id,
                max_tokens=100,
                temperature=temperature,
                system="Extract specific entities from the text and return them as tags. Include:\n"
                       "- Company names (e.g., 'Apple', 'Microsoft')\n"
                       "- Technologies (e.g., 'ChatGPT', 'iOS 17')\n"
                       "- People (e.g., 'Tim Cook', 'Satya Nadella')\n"
                       "- Products (e.g., 'iPhone 15', 'Surface Pro')\n"
                       "Format: Return only the tags as a comma-separated list, with no categories or explanations.",
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            tags = [tag.strip() for tag in response.content[0].text.split(',')]
            
            self.logger.info(
                "Tags generated successfully", 
                tag_count=len(tags),
                tags=", ".join(tags[:5]) + ("..." if len(tags) > 5 else "")
            )
            
            return tags
        except SummarizerError as e:
            # Log and re-raise our custom exceptions
            self.logger.exception(
                f"Tag generation error: {str(e)}", 
                error_type=type(e).__name__
            )
            raise
        except Exception as e:
            self.logger.exception(
                f"Unexpected error in generate_tags: {str(e)}",
                error_type=type(e).__name__
            )
            return []
        finally:
            # Clear context after the operation
            self.logger.clear_context()
            
    async def batch_summarize(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        auto_select_model: bool = True,
        temperature: float = 0.3
    ) -> List[Dict]:
        """
        Summarize a batch of articles concurrently.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent API calls
            auto_select_model: Whether to automatically select the appropriate model
            temperature: Temperature setting for generation
            
        Returns:
            List of dicts with original article and summary
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        self.logger.info(f"Starting batch summarization of {len(articles)} articles")
        
        # Queue for managing concurrent API calls
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_article(article):
            """Process a single article with concurrency control."""
            async with semaphore:
                try:
                    # Select model based on content length if auto_select is enabled
                    model = None
                    if auto_select_model:
                        text_length = len(article.get('text', ''))
                        if text_length < 2000:
                            model = "haiku"  # Use fastest model for short articles
                        elif text_length > 10000:
                            model = "sonnet-3.7"  # Use most capable model for long articles
                        else:
                            model = "sonnet"  # Use balanced model for medium articles
                    
                    title = article.get('title', 'No Title')
                    url = article.get('url', '#')
                    text = article.get('text', '')
                    
                    self.logger.info(f"Processing article: {title} with model {model or 'default'}")
                    
                    # Use a thread to run the synchronous summarize_article method
                    with ThreadPoolExecutor() as executor:
                        summary = await asyncio.get_event_loop().run_in_executor(
                            executor, 
                            lambda: self.summarize_article(
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
                    self.logger.error(f"Error processing article {article.get('title')}: {str(e)}")
                    return {
                        'original': article,
                        'error': str(e)
                    }
        
        # Create tasks for all articles
        tasks = [process_article(article) for article in articles]
        
        # Process all tasks and collect results
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(completed_results)
        
        # Log completion
        success_count = sum(1 for r in results if 'summary' in r)
        error_count = sum(1 for r in results if 'error' in r)
        self.logger.info(f"Batch summarization completed: {success_count} successes, {error_count} errors")
        
        return results


# Usage examples

def example_basic_usage():
    """Example of basic usage."""
    summarizer = ArticleSummarizer()
    
    try:
        summary = summarizer.summarize_article(
            "Article text here...",
            "Article Title",
            "https://example.com/article"
        )
        print(f"Headline: {summary['headline']}")
        print(f"Summary: {summary['summary']}")
    except SummarizerError as e:
        print(f"Summarization failed: {e}")


def example_model_selection():
    """Example of using model selection."""
    summarizer = ArticleSummarizer()
    
    # Select different models based on needs
    models_to_try = [
        # Fast, efficient model for routine summaries
        "haiku-3.5",  
        # High-quality model for important articles
        "sonnet-3.7", 
        # Most capable model for complex technical content
        "opus"        
    ]
    
    for model_name in models_to_try:
        try:
            print(f"\nTrying model: {model_name}")
            summary = summarizer.summarize_article(
                "Article text here...",
                "Article Title",
                "https://example.com/article",
                model=model_name
            )
            print(f"Success with {model_name}!")
            break
        except APIRateLimitError:
            print(f"Rate limited on {model_name}, waiting before retry...")
            time.sleep(30)  # Wait before retry
        except (APIConnectionError, APIResponseError) as e:
            print(f"Error with {model_name}: {e}")
            continue  # Try next model
        except APIAuthError as e:
            print(f"Authentication error: {e}")
            break  # No point trying other models


def example_streaming():
    """Example of using streaming responses with error handling."""
    summarizer = ArticleSummarizer()
    
    print("Streaming summary:")
    
    try:
        # Progress tracking
        chunk_count = 0
        start_time = time.time()
        
        # Simple streaming with generator
        for chunk in summarizer.summarize_article_streaming(
            "Article text here...",
            "Article Title",
            "https://example.com/article",
            model="sonnet-3.7"  # Using the latest model
        ):
            chunk_count += 1
            if chunk_count % 5 == 0:
                elapsed = time.time() - start_time
                print(f"\n[Progress: {chunk_count} chunks, {elapsed:.1f}s]", end="")
            print(chunk, end="", flush=True)
        
        print("\n\nStreaming completed successfully!")
        
    except SummarizerError as e:
        print(f"\nStreaming failed: {e}")


def example_error_handling():
    """Example demonstrating error handling."""
    summarizer = ArticleSummarizer()
    
    # Deliberately cause an error with an invalid model
    try:
        summary = summarizer.summarize_article(
            "Article text here...",
            "Article Title",
            "https://example.com/article",
            model="nonexistent-model"  # This should trigger a ModelSelectionError
        )
        print("Summary successful despite invalid model (fallback used)")
    except ModelSelectionError as e:
        print(f"Expected error caught: {e}")
    
    # Test API error handling (simulation)
    try:
        # We can't easily trigger a real API error in an example,
        # but we can show how it would be handled
        print("If an API error occurred, it would be handled like this:")
        print("try:")
        print("    summary = summarizer.summarize_article(...)")
        print("except APIConnectionError as e:")
        print("    print(f'Connection error: {e}')")
        print("    # Implement fallback summarization or retry logic")
        print("except APIRateLimitError as e:")
        print("    print(f'Rate limited: {e}')")
        print("    # Implement backoff and retry")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("summarizer.log"),
            logging.StreamHandler()
        ]
    )
    
    # Run examples
    print("=== Basic Usage Example ===")
    example_basic_usage()
    
    print("\n=== Model Selection Example ===")
    example_model_selection()
    
    print("\n=== Streaming Example ===")
    example_streaming()
    
    print("\n=== Error Handling Example ===")
    example_error_handling()