"""
Generic source extraction utilities and content processing.

This module provides utilities for extracting and processing content from various sources,
including content cleaning, validation, and caching functionality.
"""

import re
import hashlib
import logging
import os
import time
from typing import Optional, Dict, List, Set
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from .base import ContentCleaner, URLValidator

logger = logging.getLogger(__name__)


class HTMLContentCleaner(ContentCleaner):
    """Cleans and normalizes HTML content."""
    
    # Elements to remove from content
    UNWANTED_SELECTORS = [
        'script', 'style', 'nav', 'header', 'footer', 
        '.ads', '.advertisement', '.ad', '.promo', 
        '.comments', '.comment', '.related', '.sidebar',
        '.share', '.social', '.newsletter', '.subscription'
    ]
    
    # Content selectors in order of preference
    CONTENT_SELECTORS = [
        'article', '.article', '.post-content', '.entry-content', 
        '.content', '.story', '.story-body', '.article-body',
        'main', '#main', '.main', '.container'
    ]
    
    def clean_content(self, content: str, source_url: str = "") -> str:
        """
        Clean and normalize HTML content.
        
        Args:
            content: Raw HTML content to clean
            source_url: Optional source URL for context
            
        Returns:
            Cleaned text content
        """
        if not content:
            return ""
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted elements
            for selector in self.UNWANTED_SELECTORS:
                for element in soup.select(selector):
                    element.decompose()
            
            # Find main content area
            main_content = self._find_main_content(soup)
            
            if main_content:
                # Extract text from paragraphs
                paragraphs = main_content.find_all('p')
                if paragraphs:
                    text_content = '\n\n'.join(
                        p.get_text().strip() 
                        for p in paragraphs 
                        if len(p.get_text().strip()) > 20
                    )
                    
                    if text_content and len(text_content) > 100:
                        return self._post_process_text(text_content)
                
                # Fallback to all text if paragraphs don't work
                text_content = main_content.get_text()
                if text_content:
                    return self._post_process_text(text_content)
            
            # Last resort: get all text from body
            if soup.body:
                text_content = soup.body.get_text()
                return self._post_process_text(text_content)
            
            return self._post_process_text(soup.get_text())
            
        except Exception as e:
            logger.warning(f"Error cleaning content: {e}")
            # Return raw text if HTML parsing fails
            return self._post_process_text(content)
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional:
        """Find the main content area in the HTML."""
        for selector in self.CONTENT_SELECTORS:
            elements = soup.select(selector)
            if elements:
                # Return the largest element by text length
                return max(elements, key=lambda el: len(el.get_text()))
        return None
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Clean up common artifacts
        text = re.sub(r'Advertisement\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Subscribe.*?newsletter', '', text, flags=re.IGNORECASE)
        
        return text.strip()


class ComprehensiveURLValidator(URLValidator):
    """Comprehensive URL validation with multiple criteria."""
    
    # Known reliable news sources (expanded list)
    KNOWN_NEWS_SOURCES = {
        'techcrunch.com', 'theverge.com', 'wired.com', 'bloomberg.com', 'wsj.com',
        'nytimes.com', 'washingtonpost.com', 'cnn.com', 'bbc.com', 'reuters.com',
        'arstechnica.com', 'engadget.com', 'cnet.com', 'zdnet.com', 'venturebeat.com',
        'axios.com', 'politico.com', 'thehill.com', 'npr.org', 'apnews.com',
        'usatoday.com', 'latimes.com', 'chicagotribune.com', 'bostonglobe.com',
        'seattletimes.com', 'denverpost.com', 'sfgate.com', 'chron.com'
    }
    
    # Domains to exclude from source consideration
    EXCLUDED_DOMAINS = {
        'twitter.com', 'facebook.com', 'linkedin.com', 't.co', 'bit.ly',
        'instagram.com', 'youtube.com', 'reddit.com', 'techmeme.com',
        'news.google.com', 'tinyurl.com', 'goo.gl', 'ow.ly', 'buff.ly',
        'pinterest.com', 'tumblr.com', 'flickr.com', 'vimeo.com'
    }
    
    # Suspicious TLDs that might indicate spam/malicious sites
    SUSPICIOUS_TLDS = {
        '.tk', '.ml', '.ga', '.cf', '.click', '.download', '.zip'
    }
    
    def is_valid_source_url(self, url: str) -> bool:
        """Check if URL is a valid news source with comprehensive validation."""
        if not url:
            return False
        
        try:
            # Basic URL structure validation
            if not url.startswith(('http://', 'https://')):
                return False
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check if it's an excluded domain
            if self.should_exclude_domain(url):
                return False
            
            # Check for suspicious TLDs
            if any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS):
                return False
            
            # Must have a reasonable domain structure
            if '.' not in domain or len(domain) < 4:
                return False
            
            # Check for known good sources (high confidence)
            if any(source in domain for source in self.KNOWN_NEWS_SOURCES):
                return True
            
            # Additional validation for unknown domains
            return self._validate_unknown_domain(domain, parsed.path)
            
        except Exception:
            return False
    
    def should_exclude_domain(self, url: str) -> bool:
        """Check if domain should be excluded."""
        try:
            domain = urlparse(url).netloc.lower()
            return any(excluded in domain for excluded in self.EXCLUDED_DOMAINS)
        except Exception:
            return True
    
    def _validate_unknown_domain(self, domain: str, path: str) -> bool:
        """Additional validation for unknown domains."""
        # Remove www prefix for checking
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Domain should have at least 2 parts
        parts = domain.split('.')
        if len(parts) < 2:
            return False
        
        # Check for reasonable domain structure
        # Should not be just numbers or very short
        main_domain = parts[0]
        if main_domain.isdigit() or len(main_domain) < 3:
            return False
        
        # Path should look like an article (optional but helpful)
        if path and len(path) > 1:
            # Look for article-like patterns
            article_patterns = [
                r'/\d{4}/\d{2}/',  # Date-based URLs
                r'/article',        # Article in path
                r'/news',          # News in path
                r'/post',          # Post in path
                r'/blog',          # Blog in path
            ]
            
            for pattern in article_patterns:
                if re.search(pattern, path):
                    return True
        
        # Default to True for unknown but structurally valid domains
        return True


class ContentValidator:
    """Validates extracted content quality."""
    
    def is_valid_content(self, content: str) -> bool:
        """
        Check if extracted content is valid and substantial.
        
        Args:
            content: The content to validate
            
        Returns:
            True if content is valid and substantial
        """
        if not content:
            return False
        
        # Minimum length check
        if len(content) < 100:
            return False
        
        # Check word count
        words = content.split()
        if len(words) < 50:
            return False
        
        # Check for reasonable paragraph structure
        if content.count('\n\n') < 2:
            return False
        
        # Check for spam indicators
        spam_indicators = [
            'click here', 'buy now', 'limited time offer',
            'act now', 'call now', 'free trial'
        ]
        
        content_lower = content.lower()
        spam_count = sum(1 for indicator in spam_indicators if indicator in content_lower)
        
        # Too many spam indicators
        if spam_count > 2:
            return False
        
        return True


class ContentCache:
    """Simple file-based cache for extracted content."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self._ensure_cache_dir()
    
    def _get_default_cache_dir(self) -> str:
        """Get default cache directory."""
        return os.path.join(os.path.expanduser("~"), ".rss_reader_cache", "content")
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory: {e}")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}.txt")
    
    def get(self, url: str, max_age: int = 86400) -> Optional[str]:
        """
        Get cached content for URL.
        
        Args:
            url: The URL to get cached content for
            max_age: Maximum age in seconds (default 24 hours)
            
        Returns:
            Cached content or None if not available/expired
        """
        try:
            cache_key = self._get_cache_key(url)
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                # Check if cache is still valid
                if time.time() - os.path.getmtime(cache_path) < max_age:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content:
                            logger.debug(f"Using cached content for {url}")
                            return content
                else:
                    # Remove expired cache
                    os.remove(cache_path)
            
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
        
        return None
    
    def set(self, url: str, content: str) -> None:
        """
        Cache content for URL.
        
        Args:
            url: The URL to cache content for
            content: The content to cache
        """
        if not content:
            return
        
        try:
            cache_key = self._get_cache_key(url)
            cache_path = self._get_cache_path(cache_key)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.debug(f"Cached content for {url}")
            
        except Exception as e:
            logger.warning(f"Error caching content: {e}")
    
    def clear_expired(self, max_age: int = 86400) -> int:
        """
        Clear expired cache entries.
        
        Args:
            max_age: Maximum age in seconds
            
        Returns:
            Number of entries cleared
        """
        cleared = 0
        
        try:
            if not os.path.exists(self.cache_dir):
                return 0
            
            current_time = time.time()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.cache_dir, filename)
                    if current_time - os.path.getmtime(file_path) > max_age:
                        os.remove(file_path)
                        cleared += 1
            
            logger.info(f"Cleared {cleared} expired cache entries")
            
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
        
        return cleared


# Default instances
default_content_cleaner = HTMLContentCleaner()
default_url_validator = ComprehensiveURLValidator()
default_content_validator = ContentValidator()
default_content_cache = ContentCache()


def clean_html_content(content: str, source_url: str = "") -> str:
    """
    Convenience function to clean HTML content.
    
    Args:
        content: Raw HTML content
        source_url: Optional source URL for context
        
    Returns:
        Cleaned text content
    """
    return default_content_cleaner.clean_content(content, source_url)


def is_valid_source_url(url: str) -> bool:
    """
    Convenience function to validate source URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if URL is valid
    """
    return default_url_validator.is_valid_source_url(url)


def is_valid_content(content: str) -> bool:
    """
    Convenience function to validate content.
    
    Args:
        content: The content to validate
        
    Returns:
        True if content is valid
    """
    return default_content_validator.is_valid_content(content)