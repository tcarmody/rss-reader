"""Article summarization with Anthropic Claude API with model selection and streaming."""

import re
import html
import logging
import time
from anthropic import Anthropic, APIError, APIConnectionError, APIResponseValidationError, RateLimitError, AuthenticationError
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
        "sonnet-3-5": "claude-3-5-sonnet-20240620",
        "haiku-3-5": "claude-3-5-haiku-20241022",
        "sonnet-3-5-new": "claude-3-5-sonnet-20241022",
        "sonnet-3-7": "claude-3-7-sonnet-20250219"
    }
    
    # Default model to use - using the latest working model
    DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

    def __init__(self):
        """Initialize the summarizer with Claude API client."""
        try:
            api_key = get_env_var('ANTHROPIC_API_KEY')
            if not api_key:
                raise APIAuthError("Anthropic API key not found")
            
            # Check Anthropic SDK version
            self._check_anthropic_version()
            
            # Initialize the client - SDK 0.50.0 simplified initialization
            self.client = Anthropic(api_key=api_key)
            
            # Initialize cache and logger
            cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
            # Create the cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            self.summary_cache = SummaryCache(cache_dir=cache_dir)
            self.logger = StructuredLogger("ArticleSummarizer")
            
        except Exception as e:
            # Convert to our custom exception type
            if "API key" in str(e):
                raise APIAuthError(f"Failed to initialize API client: {str(e)}")
            else:
                raise SummarizerError(f"Failed to initialize summarizer: {str(e)}")

    def _check_anthropic_version(self):
        """Check if the installed Anthropic SDK version is compatible."""
        try:
            import anthropic
            import pkg_resources
            
            version = pkg_resources.get_distribution("anthropic").version
            required_version = "0.50.0"
            
            if version < required_version:
                self.logger.warning(
                    f"Anthropic SDK version {version} is older than recommended version {required_version}. "
                    f"Some features may not work as expected."
                )
            elif version > required_version:
                # Add this to handle potential breaking changes in future versions
                major_version = version.split('.')[0]
                required_major = required_version.split('.')[0]
                if major_version > required_major:
                    self.logger.warning(
                        f"Using Anthropic SDK version {version}, which is newer than the tested version {required_version}. "
                        f"If you encounter issues, consider downgrading to version {required_version}."
                    )
            else:
                self.logger.info(f"Using compatible Anthropic SDK version: {version}")
                
            return version
        except Exception as e:
            self.logger.error(f"Failed to check Anthropic SDK version: {str(e)}")
            return None

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
            
            # Updated for SDK 0.50.0
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
        except RateLimitError as e:
            self.logger.error(
                "Claude API rate limit error", 
                error_type="RateLimitError",
                error=str(e)
            )
            raise APIRateLimitError(f"Rate limit exceeded: {str(e)}")
        except AuthenticationError as e:
            self.logger.error(
                "Claude API authentication error", 
                error_type="AuthenticationError",
                error=str(e)
            )
            raise APIAuthError(f"Authentication failed: {str(e)}")
        except APIConnectionError as e:
            self.logger.error(
                "Claude API connection error", 
                error_type="APIConnectionError",
                error=str(e)
            )
            raise APIConnectionError(f"Connection error: {str(e)}")
        except APIError as e:
            self.logger.error(
                "Claude API error", 
                error_type="APIError",
                error=str(e)
            )
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

    @retry_with_backoff(max_retries=2, initial_backoff=1)
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
            
            # Updated for SDK 0.50.0
            stream = self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self._get_system_prompt(),
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                stream=True
            )
            
            # Process each chunk
            for chunk in stream:
                # Check if the chunk contains text content
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
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
            
        except RateLimitError as e:
            self.logger.error(
                "Claude API streaming rate limit error", 
                error_type="RateLimitError",
                error=str(e)
            )
            raise APIRateLimitError(f"Rate limit exceeded: {str(e)}")
        except AuthenticationError as e:
            self.logger.error(
                "Claude API streaming authentication error", 
                error_type="AuthenticationError",
                error=str(e)
            )
            raise APIAuthError(f"Authentication failed: {str(e)}")
        except APIConnectionError as e:
            self.logger.error(
                "Claude API streaming connection error", 
                error_type="APIConnectionError",
                error=str(e)
            )
            raise APIConnectionError(f"Connection error: {str(e)}")
        except APIError as e:
            self.logger.error(
                "Claude API streaming error", 
                error_type="APIError",
                error=str(e)
            )
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
            if not force_refresh:
                cache_key = f"{text}:{model or self.DEFAULT_MODEL}:{temperature}"
                cached_result = self.summary_cache.get(cache_key)
                if cached_result:
                    self.logger.info("Retrieved summary from cache", cache_hit=True)
                    return cached_result['summary']
            
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
            cache_key = f"{text}:{model_id}:{temperature}"
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
            self.logger.exception(
                f"Unexpected error in summarize_article: {str(e)}",
                error_type=type(e).__name__
            )
            # Return a fallback summary
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
        self.logger.add_context(operation="generate_tags", content_length=len(content))
        
        try:
            # Check cache first
            cache_key = f"tags:{hash(content)}"
            cached_tags = self.summary_cache.get(cache_key)
            if cached_tags:
                self.logger.info("Retrieved tags from cache", cache_hit=True)
                return cached_tags
            
            # Prepare the prompt for tag generation
            prompt = (
                "Extract relevant tags from the following article content. "
                "Focus on key topics, entities, technologies, and themes. "
                "Return exactly 5-8 tags as a comma-separated list. "
                "Tags should be 1-3 words each, lowercase, and contain no special characters.\n\n"
                f"Article content:\n{content[:4000]}"  # Limit content length
            )
            
            # Get the model ID
            model_id = self._get_model(model)
            
            # Call the API - Updated for SDK 0.50.0
            self.logger.info("Calling Claude API for tag generation", model=model_id)
            response = self.client.messages.create(
                model=model_id,
                max_tokens=100,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Process the response
            tags_text = response.content[0].text
            
            # Extract tags (assuming comma-separated format)
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            
            # Clean tags (remove any special characters, ensure lowercase)
            tags = [re.sub(r'[^\w\s-]', '', tag).lower() for tag in tags]
            
            # Remove duplicates while preserving order
            unique_tags = []
            for tag in tags:
                if tag and tag not in unique_tags:
                    unique_tags.append(tag)
            
            # Cache the result
            self.summary_cache.set(cache_key, unique_tags)
            
            self.logger.info("Generated tags successfully", tag_count=len(unique_tags))
            return unique_tags
            
        except Exception as e:
            self.logger.error("Failed to generate tags", error=str(e))
            # Return empty list on error rather than raising
            return []