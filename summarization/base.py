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