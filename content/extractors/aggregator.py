"""
Consolidated aggregator source extraction with support for multiple aggregator types.

This module combines functionality from the original source_extractor.py and 
aggregator_extractor.py files, providing a unified interface for extracting
original source URLs from news aggregators.
"""

import re
import logging
import requests
from urllib.parse import urlparse, parse_qs, unquote, quote
from typing import Optional, Dict, List, Tuple, Set
from bs4 import BeautifulSoup

from .base import (
    SourceExtractor, BaseSourceExtractor, AggregatorDetector, 
    ExtractionResult, ExtractionMethod, AggregatorInfo, URLValidator
)

logger = logging.getLogger(__name__)


class AggregatorPatternDetector(AggregatorDetector):
    """Detects aggregator sites based on domain patterns."""
    
    # Common news aggregator domains
    AGGREGATOR_DOMAINS = [
        'news.google.com', 'techmeme.com', 'memeorandum.com', 'reddit.com',
        'feedly.com', 'news.ycombinator.com', 'slashdot.org', 'digg.com',
        'flipboard.com', 'inoreader.com', 'theoldreader.com', 'nuzzel.com',
        't.co', 'bit.ly', 'tinyurl.com', 'goo.gl', 'smartnews.com',
        'linkedin.com', 'news.yahoo.com', 'news.microsoft.com', 'bing.com/news'
    ]
    
    # Aggregator-specific information
    AGGREGATOR_INFO = {
        'techmeme.com': AggregatorInfo(
            name='Techmeme',
            domain='techmeme.com',
            patterns=[r'techmeme\.com'],
            extraction_method=ExtractionMethod.HTML_PARSING,
            requires_html_parsing=True
        ),
        'news.google.com': AggregatorInfo(
            name='Google News',
            domain='news.google.com',
            patterns=[r'news\.google\.com'],
            extraction_method=ExtractionMethod.URL_PARSING,
            requires_html_parsing=False
        ),
        'reddit.com': AggregatorInfo(
            name='Reddit',
            domain='reddit.com',
            patterns=[r'reddit\.com'],
            extraction_method=ExtractionMethod.URL_PARSING,
            requires_html_parsing=False
        ),
        'news.ycombinator.com': AggregatorInfo(
            name='Hacker News',
            domain='news.ycombinator.com',
            patterns=[r'news\.ycombinator\.com'],
            extraction_method=ExtractionMethod.HTML_PARSING,
            requires_html_parsing=True
        )
    }
    
    def is_aggregator(self, url: str) -> bool:
        """Check if URL is from a known aggregator."""
        if not url:
            return False
            
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Check direct domain matches
            for aggregator_domain in self.AGGREGATOR_DOMAINS:
                if domain == aggregator_domain or domain.endswith('.' + aggregator_domain):
                    return True
            
            # Check for redirect parameters (indicates aggregator)
            query_params = parse_qs(parsed_url.query)
            redirect_params = ['url', 'u', 'link', 'redirect', 'target', 'goto', 'redir']
            for param in query_params:
                if param.lower() in redirect_params:
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def get_aggregator_info(self, url: str) -> Optional[AggregatorInfo]:
        """Get aggregator information for the URL."""
        try:
            domain = urlparse(url).netloc.lower()
            for agg_domain, info in self.AGGREGATOR_INFO.items():
                if agg_domain in domain:
                    return info
        except Exception:
            pass
        return None


class NewsSourceValidator(URLValidator):
    """Validates extracted source URLs."""
    
    # Known reliable news sources
    KNOWN_NEWS_SOURCES = [
        'techcrunch.com', 'theverge.com', 'wired.com', 'bloomberg.com', 'wsj.com',
        'nytimes.com', 'washingtonpost.com', 'cnn.com', 'bbc.com', 'reuters.com',
        'arstechnica.com', 'engadget.com', 'cnet.com', 'zdnet.com', 'venturebeat.com'
    ]
    
    # Domains to exclude from source consideration
    EXCLUDED_DOMAINS = [
        'twitter.com', 'facebook.com', 'linkedin.com', 't.co',
        'instagram.com', 'youtube.com', 'reddit.com', 'techmeme.com',
        'news.google.com', 'bit.ly', 'tinyurl.com'
    ]
    
    def is_valid_source_url(self, url: str) -> bool:
        """Check if URL is a valid news source."""
        if not url:
            return False
        
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check if it's a known excluded domain
            if self.should_exclude_domain(url):
                return False
            
            # Check for valid URL structure
            if not url.startswith(('http://', 'https://')):
                return False
            
            # Must have a reasonable domain
            if '.' not in domain or len(domain) < 4:
                return False
                
            return True
            
        except Exception:
            return False
    
    def should_exclude_domain(self, url: str) -> bool:
        """Check if domain should be excluded."""
        try:
            domain = urlparse(url).netloc.lower()
            return any(excluded in domain for excluded in self.EXCLUDED_DOMAINS)
        except Exception:
            return True


class TechmemeExtractor(BaseSourceExtractor):
    """Extractor for Techmeme links."""
    
    def __init__(self):
        super().__init__("Techmeme", ["techmeme.com"])
    
    def extract(self, url: str, session=None) -> ExtractionResult:
        """Extract original source from Techmeme URL."""
        if not session:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        
        try:
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return self._create_result(
                    url, success=False, error_message=f"HTTP {response.status_code}"
                )
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the main article link in Techmeme's structure
            selectors = [
                'a.ourhead',  # Main headline link
                '.itemdesc a[href^="http"]',  # Description links
                '.itemhead a[href^="http"]',  # Header links
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    if href and 'techmeme.com' not in href:
                        return self._create_result(
                            url,
                            extracted_url=href,
                            title=link.get_text().strip(),
                            source_name=self._extract_domain_name(href),
                            success=True,
                            extraction_method=ExtractionMethod.HTML_PARSING,
                            confidence=0.9
                        )
            
            return self._create_result(
                url, success=False, error_message="No external links found"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"Extraction failed: {str(e)}"
            )
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract clean domain name for source identification."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].title()
        except Exception:
            return "Unknown"


class GoogleNewsExtractor(BaseSourceExtractor):
    """Extractor for Google News links."""
    
    def __init__(self):
        super().__init__("Google News", ["news.google.com"])
    
    def extract(self, url: str, session=None) -> ExtractionResult:
        """Extract original source from Google News URL."""
        try:
            # Try URL parameter extraction first
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # Google News specific parameters
            for param in ['url', 'link', 'q']:
                if param in query_params:
                    extracted_url = unquote(query_params[param][0])
                    if extracted_url.startswith(('http://', 'https://')):
                        return self._create_result(
                            url,
                            extracted_url=extracted_url,
                            source_name=self._extract_domain_name(extracted_url),
                            success=True,
                            extraction_method=ExtractionMethod.URL_PARSING,
                            confidence=0.95
                        )
            
            # If URL parsing fails, try HTML parsing
            if session:
                return self._html_extraction_fallback(url, session)
            
            return self._create_result(
                url, success=False, error_message="No extractable URL found"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"Extraction failed: {str(e)}"
            )
    
    def _html_extraction_fallback(self, url: str, session: requests.Session) -> ExtractionResult:
        """Fallback to HTML parsing for Google News."""
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for canonical URL or main article link
                for selector in ['link[rel="canonical"]', 'a[href^="http"]:not([href*="google"])']:
                    elements = soup.select(selector)
                    if elements:
                        href = elements[0].get('href')
                        if href and 'google.com' not in href:
                            return self._create_result(
                                url,
                                extracted_url=href,
                                source_name=self._extract_domain_name(href),
                                success=True,
                                extraction_method=ExtractionMethod.HTML_PARSING,
                                confidence=0.8
                            )
            
            return self._create_result(
                url, success=False, error_message="HTML extraction failed"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"HTML extraction error: {str(e)}"
            )
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract clean domain name for source identification."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].title()
        except Exception:
            return "Unknown"


class RedditExtractor(BaseSourceExtractor):
    """Extractor for Reddit links."""
    
    def __init__(self):
        super().__init__("Reddit", ["reddit.com"])
    
    def extract(self, url: str, session=None) -> ExtractionResult:
        """Extract original source from Reddit URL."""
        try:
            # Reddit URLs often contain the target URL in the path or as a parameter
            parsed_url = urlparse(url)
            
            # Check for direct link posts (reddit.com/r/subreddit/comments/id/title/url)
            path_parts = parsed_url.path.split('/')
            if len(path_parts) > 6 and 'comments' in path_parts:
                # This is likely a comments page, need to get the actual link
                if session:
                    return self._extract_from_reddit_page(url, session)
            
            # Check query parameters
            query_params = parse_qs(parsed_url.query)
            for param in ['url', 'link']:
                if param in query_params:
                    extracted_url = unquote(query_params[param][0])
                    if extracted_url.startswith(('http://', 'https://')):
                        return self._create_result(
                            url,
                            extracted_url=extracted_url,
                            source_name=self._extract_domain_name(extracted_url),
                            success=True,
                            extraction_method=ExtractionMethod.URL_PARSING,
                            confidence=0.9
                        )
            
            return self._create_result(
                url, success=False, error_message="No extractable URL found in Reddit link"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"Extraction failed: {str(e)}"
            )
    
    def _extract_from_reddit_page(self, url: str, session: requests.Session) -> ExtractionResult:
        """Extract link from Reddit comments page."""
        try:
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for the main post link
                selectors = [
                    'a[data-click-id="body"]',  # New Reddit
                    '.title a[href^="http"]',   # Old Reddit
                    '.linkflairlabel + a',      # Link posts
                ]
                
                for selector in selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        href = element.get('href', '')
                        if href and 'reddit.com' not in href and href.startswith(('http://', 'https://')):
                            return self._create_result(
                                url,
                                extracted_url=href,
                                title=element.get_text().strip(),
                                source_name=self._extract_domain_name(href),
                                success=True,
                                extraction_method=ExtractionMethod.HTML_PARSING,
                                confidence=0.85
                            )
            
            return self._create_result(
                url, success=False, error_message="Could not extract from Reddit page"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"Reddit page extraction failed: {str(e)}"
            )
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract clean domain name for source identification."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].title()
        except Exception:
            return "Unknown"


class HackerNewsExtractor(BaseSourceExtractor):
    """Extractor for Hacker News links."""
    
    def __init__(self):
        super().__init__("Hacker News", ["news.ycombinator.com"])
    
    def extract(self, url: str, session=None) -> ExtractionResult:
        """Extract original source from Hacker News URL."""
        if not session:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        
        try:
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return self._create_result(
                    url, success=False, error_message=f"HTTP {response.status_code}"
                )
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the main story link
            selectors = [
                '.titleline > a[href^="http"]',  # New HN structure
                '.title > a[href^="http"]',      # Old HN structure
                'a.storylink[href^="http"]',     # Direct story links
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href', '')
                    if href and 'ycombinator.com' not in href:
                        return self._create_result(
                            url,
                            extracted_url=href,
                            title=element.get_text().strip(),
                            source_name=self._extract_domain_name(href),
                            success=True,
                            extraction_method=ExtractionMethod.HTML_PARSING,
                            confidence=0.9
                        )
            
            return self._create_result(
                url, success=False, error_message="No external links found"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"Extraction failed: {str(e)}"
            )
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract clean domain name for source identification."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].title()
        except Exception:
            return "Unknown"


class GenericAggregatorExtractor(BaseSourceExtractor):
    """Generic extractor for unknown aggregators using common patterns."""
    
    def __init__(self):
        super().__init__("Generic", ["*"])
    
    def can_extract(self, url: str) -> bool:
        """Generic extractor can try to handle any URL."""
        return True
    
    def extract(self, url: str, session=None) -> ExtractionResult:
        """Try generic extraction methods."""
        try:
            # First try URL parameter extraction
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # Common redirect parameter names
            redirect_params = ['url', 'u', 'link', 'redirect', 'target', 'goto', 'redir', 'uri']
            
            for param in redirect_params:
                if param in query_params:
                    extracted_url = unquote(query_params[param][0])
                    if extracted_url.startswith(('http://', 'https://')):
                        return self._create_result(
                            url,
                            extracted_url=extracted_url,
                            source_name=self._extract_domain_name(extracted_url),
                            success=True,
                            extraction_method=ExtractionMethod.URL_PARSING,
                            confidence=0.7
                        )
            
            # If no URL parameters found, this might not be an aggregator
            return self._create_result(
                url, success=False, error_message="No redirect parameters found"
            )
            
        except Exception as e:
            return self._create_result(
                url, success=False, error_message=f"Generic extraction failed: {str(e)}"
            )
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract clean domain name for source identification."""
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].title()
        except Exception:
            return "Unknown"


# Factory function to create a configured extractor manager
def create_aggregator_extractor_manager() -> 'ExtractorManager':
    """Create a configured ExtractorManager with all aggregator extractors."""
    from .base import ExtractorManager
    
    manager = ExtractorManager()
    
    # Add specific extractors
    manager.add_extractor(TechmemeExtractor())
    manager.add_extractor(GoogleNewsExtractor())
    manager.add_extractor(RedditExtractor())
    manager.add_extractor(HackerNewsExtractor())
    
    # Add generic extractor last (fallback)
    manager.add_extractor(GenericAggregatorExtractor())
    
    # Set components
    manager.set_aggregator_detector(AggregatorPatternDetector())
    manager.set_url_validator(NewsSourceValidator())
    
    return manager


# Default global instance
default_aggregator_manager = create_aggregator_extractor_manager()


def extract_source_url(url: str, session=None) -> ExtractionResult:
    """
    Convenience function to extract source URL from aggregator.
    
    Args:
        url: The aggregator URL
        session: Optional requests session
        
    Returns:
        ExtractionResult with extracted information
    """
    return default_aggregator_manager.extract_source(url, session)


def is_aggregator_link(url: str) -> bool:
    """
    Convenience function to check if URL is from an aggregator.
    
    Args:
        url: The URL to check
        
    Returns:
        True if URL is from an aggregator
    """
    detector = AggregatorPatternDetector()
    return detector.is_aggregator(url)