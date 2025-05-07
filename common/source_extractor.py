"""
Source extractor for article links in RSS feeds.

This module provides functions to detect and extract original article URLs
from aggregator sites like Techmeme, Google News, etc.
"""

import re
import urllib.parse
from typing import Optional, List, Dict, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of common news aggregator domains
AGGREGATOR_DOMAINS = [
    'news.google.com',
    'techmeme.com',
    'memeorandum.com',
    'reddit.com',
    'feedly.com',
    'news.ycombinator.com',
    'slashdot.org',
    'digg.com',
    'flipboard.com',
    'inoreader.com',
    'theoldreader.com',
    'nuzzel.com',
    't.co',
    'bit.ly',
    'tinyurl.com',
    'goo.gl',
    'feedly.com',
    'smartnews.com',
    'linkedin.com',
    'news.yahoo.com',
    'news.microsoft.com',
    'bing.com/news',
]

def is_aggregator_link(url: str) -> bool:
    """
    Check if the URL is from a known news aggregator.
    
    Args:
        url: The URL to check
        
    Returns:
        bool: True if the URL is from a known aggregator, False otherwise
    """
    if not url:
        return False
        
    try:
        # Parse the URL and extract domain
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if domain matches any known aggregator
        for aggregator in AGGREGATOR_DOMAINS:
            if domain == aggregator or domain.endswith('.' + aggregator):
                return True
                
        # Check for redirect parameters in query string
        query_params = urllib.parse.parse_qs(parsed_url.query)
        for param in query_params:
            param_lower = param.lower()
            # Common redirect parameter names
            if param_lower in ['url', 'u', 'link', 'redirect', 'target']:
                param_value = query_params[param][0]
                if param_value.startswith('http'):
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking if URL is from aggregator: {str(e)}")
        return False

def extract_original_source_url(url: str) -> str:
    """
    Extract the original source URL from an aggregator link.
    
    Args:
        url: The aggregator URL
        
    Returns:
        str: The original source URL if found, or the original URL if not
    """
    if not url:
        return url
        
    try:
        # Parse the URL
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Handle Google News links
        if 'news.google.com' in domain:
            # Google News format: .../articles/[source]/[article-id]
            if '/articles/' in url:
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if 'url' in query_params:
                    return query_params['url'][0]
                
                # Try to find URL in path
                path_parts = parsed_url.path.split('/')
                if len(path_parts) > 3 and path_parts[-3] == 'articles':
                    # Extract the article URL, which is often in the format:
                    # ./articles/CBMiXGh0dHBzOi8vdGhlY...
                    article_part = path_parts[-1]
                    if article_part.startswith('CBMi'):
                        try:
                            # Google uses a custom encoding, but often the URL is 
                            # included as a parameter after the ? character
                            return url.split('?')[1].split('&')[0]
                        except:
                            pass
            
        # Handle Techmeme links
        elif 'techmeme.com' in domain:
            query_params = urllib.parse.parse_qs(parsed_url.query)
            if 'u' in query_params:
                return query_params['u'][0]
            
        # Handle Reddit links
        elif 'reddit.com' in domain:
            if '/comments/' in url and 'url=' in url:
                try:
                    return url.split('url=')[1].split('&')[0]
                except:
                    pass
                    
        # Handle Twitter/X t.co links
        elif 't.co' in domain:
            # t.co links are redirects, we would need to follow the redirect
            # but we'll return the original for now
            return url
            
        # Handle other aggregators with common URL params
        query_params = urllib.parse.parse_qs(parsed_url.query)
        for param in ['url', 'u', 'link', 'target', 'redirect']:
            if param in query_params and query_params[param][0].startswith('http'):
                extracted_url = query_params[param][0]
                # URL decode if needed
                if '%' in extracted_url:
                    extracted_url = urllib.parse.unquote(extracted_url)
                return extracted_url
                
        return url
    except Exception as e:
        logger.error(f"Error extracting original source URL: {str(e)}")
        return url