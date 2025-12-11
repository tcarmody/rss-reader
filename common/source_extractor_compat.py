"""
Backward compatibility layer for common.source_extractor module.

This module provides the same interface as the original source_extractor.py but delegates
to the new content.extractors modules. This allows existing code to work without
changes during the migration period.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Import from new modules
try:
    from content.extractors.aggregator import (
        is_aggregator_link as _is_aggregator_link,
        extract_source_url as _extract_source_url
    )
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import new extractor modules: {e}")
    NEW_MODULES_AVAILABLE = False


def is_aggregator_link(url: str) -> bool:
    """
    Check if the URL is from a known news aggregator.
    
    Args:
        url: The URL to check
        
    Returns:
        bool: True if the URL is from a known aggregator, False otherwise
    """
    if NEW_MODULES_AVAILABLE:
        return _is_aggregator_link(url)
    
    # Fallback implementation
    if not url:
        return False
    
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        # Basic aggregator detection
        aggregator_domains = [
            'news.google.com', 'techmeme.com', 'reddit.com', 'news.ycombinator.com'
        ]
        
        return any(agg_domain in domain for agg_domain in aggregator_domains)
    except Exception:
        return False


def extract_original_source_url(url: str, session=None) -> Optional[str]:
    """
    Extract the original source URL from an aggregator link.
    
    Args:
        url: The aggregator URL
        session: Optional requests session
        
    Returns:
        Original source URL or None if extraction failed
    """
    if NEW_MODULES_AVAILABLE:
        result = _extract_source_url(url, session)
        if result.success and result.extracted_url:
            return result.extracted_url
        return None
    
    # Fallback implementation with basic URL parameter extraction
    if not url:
        return None
    
    try:
        from urllib.parse import urlparse, parse_qs, unquote
        
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Common redirect parameter names
        redirect_params = ['url', 'u', 'link', 'redirect', 'target']
        
        for param in redirect_params:
            if param in query_params:
                extracted_url = unquote(query_params[param][0])
                if extracted_url.startswith(('http://', 'https://')):
                    return extracted_url
        
        # If no URL parameters found, return None
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting source URL: {e}")
        return None


# Additional backward compatibility functions that might be used
def extract_source_info(url: str, session=None) -> dict:
    """
    Extract source information from aggregator URL.
    
    Args:
        url: The aggregator URL
        session: Optional requests session
        
    Returns:
        Dict with source information
    """
    if NEW_MODULES_AVAILABLE:
        result = _extract_source_url(url, session)
        return {
            'original_url': result.original_url,
            'extracted_url': result.extracted_url,
            'source_name': result.source_name,
            'success': result.success,
            'confidence': result.confidence,
            'error_message': result.error_message
        }
    
    # Fallback implementation
    extracted_url = extract_original_source_url(url, session)
    if extracted_url:
        return {
            'original_url': url,
            'extracted_url': extracted_url,
            'source_name': 'Unknown',
            'success': True,
            'confidence': 0.7,
            'error_message': None
        }
    else:
        return {
            'original_url': url,
            'extracted_url': None,
            'source_name': None,
            'success': False,
            'confidence': 0.0,
            'error_message': 'Extraction failed'
        }


# Re-export constants and other utilities for backward compatibility
try:
    # Import from new content modules instead of old common.source_extractor
    from content.extractors.aggregator import AggregatorPatternDetector
    from content.extractors.source import ComprehensiveURLValidator
    
    detector = AggregatorPatternDetector()
    validator = ComprehensiveURLValidator()
    
    AGGREGATOR_DOMAINS = detector.AGGREGATOR_DOMAINS
    KNOWN_NEWS_SOURCES = list(validator.KNOWN_NEWS_SOURCES)
    EXCLUDED_DOMAINS = list(validator.EXCLUDED_DOMAINS)
except ImportError:
    # Provide fallback constants
    AGGREGATOR_DOMAINS = [
        'news.google.com', 'techmeme.com', 'memeorandum.com', 'reddit.com',
        'feedly.com', 'news.ycombinator.com', 'slashdot.org', 'digg.com'
    ]
    
    KNOWN_NEWS_SOURCES = [
        'techcrunch.com', 'theverge.com', 'wired.com', 'bloomberg.com',
        'nytimes.com', 'washingtonpost.com', 'cnn.com', 'bbc.com'
    ]
    
    EXCLUDED_DOMAINS = [
        'twitter.com', 'facebook.com', 'linkedin.com', 't.co',
        'instagram.com', 'youtube.com', 'reddit.com'
    ]