"""
Backward compatibility layer for common.archive module.

This module provides the same interface as the original archive.py but delegates
to the new content.archive modules. This allows existing code to work without
changes during the migration period.
"""

import logging

logger = logging.getLogger(__name__)

# Import from new modules
try:
    from content.archive.paywall import is_paywalled as _is_paywalled
    from content.archive.providers import default_provider_manager
    from content.archive.specialized.wsj import fetch_wsj_content
except ImportError as e:
    logger.warning(f"Could not import new archive modules: {e}")
    # Fallback to original implementations if new modules aren't available
    def _is_paywalled(url):
        return False
    
    class _MockProviderManager:
        def get_archived_content(self, url, **kwargs):
            return None
    
    default_provider_manager = _MockProviderManager()
    
    def fetch_wsj_content(url):
        return None


def is_paywalled(url):
    """
    Check if a URL is likely behind a paywall.
    
    Args:
        url: The article URL to check
        
    Returns:
        bool: True if the URL is likely paywalled
    """
    return _is_paywalled(url)


def fetch_article_content(url, session=None):
    """
    Fetch article content, bypassing paywalls if necessary.
    
    Args:
        url: The article URL to fetch
        session: Optional requests session
        
    Returns:
        str: Article content or empty string if failed
    """
    try:
        # Check if it's paywalled
        if is_paywalled(url):
            logger.info(f"Detected paywall for {url}, trying archive services")
            
            # Try archive services
            result = default_provider_manager.get_archived_content(url)
            if result.success and result.content:
                return result.content
            
            # Try WSJ-specific bypass if applicable
            if 'wsj.com' in url:
                wsj_content = fetch_wsj_content(url)
                if wsj_content:
                    return wsj_content
        
        # If not paywalled or archive failed, try direct access
        if not session:
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
        
        response = session.get(url, timeout=15)
        if response.status_code == 200:
            # Simple content extraction
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments'):
                unwanted.decompose()
            
            # Try to find main content
            for selector in ['article', '.article', '.content', '.post-content', 'main']:
                elements = soup.select(selector)
                if elements:
                    paragraphs = elements[0].find_all('p')
                    content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                    if content:
                        return content
            
            # Fallback to body text
            if soup.body:
                return soup.body.get_text()
    
    except Exception as e:
        logger.warning(f"Error fetching article content: {e}")
    
    return ""


# Re-export other commonly used functions for backward compatibility
try:
    # Import from new content modules instead of old common.archive
    from content.extractors.source import default_content_cache
    
    def get_cached_content(url):
        return default_content_cache.get(url)
    
    def cache_content(url, content):
        return default_content_cache.set(url, content)
    
    def get_or_create_cache_directory():
        return default_content_cache.cache_dir
    
    # These functions are now provided locally
    COMPAT_FUNCTIONS_AVAILABLE = True
except ImportError:
    # Provide minimal implementations if original module is not available
    def get_direct_access_headers():
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def get_or_create_cache_directory():
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".rss_reader_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def get_cached_content(url):
        return None
    
    def cache_content(url, content):
        pass
    
    def is_valid_content(content):
        return bool(content and len(content) > 100)