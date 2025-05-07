"""
Standard article summarizer implementation.
"""

import logging
import os
from typing import Dict, Optional, Generator, Callable, Any

from common.errors import APIError
from models.selection import get_model_identifier
from summarization.base import BaseSummarizer
from cache.memory_cache import MemoryCache

class ArticleSummarizer(BaseSummarizer):
    """
    Standard implementation of article summarizer.
    
    Features:
    - Text cleaning and normalization
    - Summary generation using Claude API
    - Caching results to avoid redundant API calls
    - Customizable model selection
    - Streaming response support
    """
    
    def __init__(self, api_key=None, cache_dir=None, cache_size=256):
        """
        Initialize the article summarizer.
        
        Args:
            api_key: Anthropic API key (defaults to env var)
            cache_dir: Directory for cache storage
            cache_size: Maximum cache size
        """
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise APIError("Anthropic API key not found")
        
        # Initialize cache
        cache = MemoryCache(max_size=cache_size)
        
        # Initialize base class
        super().__init__(api_key=api_key, cache=cache)
        self.logger = logging.getLogger(__name__)
        
        # Set cache directory if provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
    
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
        # Use the base implementation
        return super().summarize_article(
            text=text,
            title=title,
            url=url,
            model=model,
            force_refresh=force_refresh,
            temperature=temperature
        )
    
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
        # Use the base implementation
        return super().summarize_article_streaming(
            text=text,
            title=title,
            url=url,
            model=model,
            callback=callback,
            temperature=temperature
        )