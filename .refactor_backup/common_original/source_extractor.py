"""
Enhanced source extractor for article links in aggregator sites.

This module provides functions to detect and extract original article URLs
from aggregator sites like Techmeme, Google News, and other news aggregators.
It supports both direct URL parameter extraction and HTML parsing when needed.
"""

import re
import logging
import urllib.parse
from typing import Optional, List, Dict, Union, Set
from urllib.parse import urlparse, parse_qs

# Optional imports - module will work without these but with reduced functionality
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

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

# Known news sources for prioritization
KNOWN_NEWS_SOURCES = [
    'techcrunch.com', 'theverge.com', 'wired.com', 'bloomberg.com', 'wsj.com',
    'nytimes.com', 'washingtonpost.com', 'cnn.com', 'bbc.com', 'reuters.com',
    'arstechnica.com', 'engadget.com', 'cnet.com', 'zdnet.com', 'venturebeat.com'
]

# Domains to exclude from valid source consideration
EXCLUDED_DOMAINS = [
    'twitter.com', 'facebook.com', 'linkedin.com', 't.co',
    'instagram.com', 'youtube.com', 'reddit.com', 'techmeme.com'
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
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if domain matches any known aggregator
        for aggregator in AGGREGATOR_DOMAINS:
            if domain == aggregator or domain.endswith('.' + aggregator):
                return True
                
        # Check for redirect parameters in query string
        query_params = parse_qs(parsed_url.query)
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

def is_valid_source_url(url: str) -> bool:
    """
    Check if a URL is likely to be a valid news source.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if the URL appears to be a valid news source
    """
    # Skip if URL is None or empty
    if not url:
        return False
        
    # Skip common non-news domains
    for domain in EXCLUDED_DOMAINS:
        if domain in url:
            return False
    
    # Check if the URL has a proper protocol
    if not url.startswith('http'):
        return False
    
    # Make sure URL isn't too short
    if len(url) < 12:  # http://a.com is minimum valid URL (11 chars)
        return False
    
    return True

def prioritize_known_news_sources(urls: List[str]) -> Optional[str]:
    """
    Give priority to URLs from known news sources.
    
    Args:
        urls: List of URLs to prioritize
        
    Returns:
        str: The highest priority URL, or None if the list is empty
    """
    if not urls:
        return None
        
    for url in urls:
        for source in KNOWN_NEWS_SOURCES:
            if source in url:
                return url
    
    return urls[0]  # Return the first URL if no known sources found

def extract_original_source_url(url: str, session=None, use_html_parsing: bool = True, extract_all_homepage_stories: bool = False) -> Union[str, List[Dict[str, str]]]:
    """
    Extract the original source URL(s) from an aggregator link.
    
    Args:
        url: The aggregator URL
        session: Optional requests session to use for HTML parsing
        use_html_parsing: Whether to use HTML parsing (requires requests and BeautifulSoup)
        extract_all_homepage_stories: Whether to return all stories from homepage-like URLs (changes return type)
        
    Returns:
        str: The original source URL if found for a single story (default behavior)
        List[Dict[str, str]]: List of stories with source URLs if extract_all_homepage_stories=True and URL is a homepage
        
    Note:
        When extract_all_homepage_stories=True, the return type changes to List[Dict] for homepage URLs,
        which may break existing code expecting a string. Use with caution.
    """
    if not url:
        return url
        
    try:
        # Try URL parameter extraction first (doesn't require requests/BS4)
        extracted_url = _extract_from_url_params(url)
        if extracted_url != url:
            return extracted_url
            
        # If parameter extraction didn't work and HTML parsing is enabled
        if use_html_parsing and REQUESTS_AVAILABLE and BS4_AVAILABLE:
            # Parse the URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            if not session:
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
            
            # Handle domain-specific extraction methods
            if 'techmeme.com' in domain:
                # Special case for Techmeme homepage if extract_all_homepage_stories is True
                if extract_all_homepage_stories:
                    # Check if this is likely the homepage
                    is_homepage = url.rstrip('/') == 'https://techmeme.com' or url.rstrip('/') == 'https://www.techmeme.com'
                    
                    if is_homepage:
                        homepage_stories = extract_techmeme_homepage_stories(url, session)
                        if homepage_stories:
                            return homepage_stories
                
                # Otherwise, use the standard extraction for story pages (maintains backwards compatibility)
                extracted_url = _extract_techmeme_source(url, session)
                if extracted_url:
                    return extracted_url
            elif 'news.google.com' in domain:
                extracted_url = _extract_google_news_source(url, session)
                if extracted_url:
                    return extracted_url
        
        # If all else fails, return the original URL
        return url
    except Exception as e:
        logger.error(f"Error extracting original source URL: {str(e)}")
        return url

def _extract_from_url_params(url: str) -> str:
    """
    Extract source URL from parameters in the URL.
    
    Args:
        url: The URL to extract from
        
    Returns:
        str: The extracted URL or the original URL if extraction failed
    """
    try:
        # Parse the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        query_params = parse_qs(parsed_url.query)
        
        # Handle Google News links
        if 'news.google.com' in domain:
            # Check for URL parameter
            if 'url' in query_params and query_params['url']:
                return query_params['url'][0]
                
            # Try to extract from path for article format
            if '/articles/' in url:
                path_parts = parsed_url.path.split('/')
                if len(path_parts) > 3 and path_parts[-3] == 'articles':
                    article_part = path_parts[-1]
                    if article_part.startswith('CBMi'):
                        try:
                            return url.split('?')[1].split('&')[0]
                        except:
                            pass
            
        # Handle Techmeme links
        elif 'techmeme.com' in domain:
            if 'u' in query_params:
                return query_params['u'][0]
            
        # Handle Reddit links
        elif 'reddit.com' in domain:
            if '/comments/' in url and 'url=' in url:
                try:
                    extracted = url.split('url=')[1].split('&')[0]
                    return urllib.parse.unquote(extracted)
                except:
                    pass
                    
        # Handle common redirect parameters
        for param in ['url', 'u', 'link', 'target', 'redirect', 'to']:
            if param in query_params and query_params[param][0].startswith('http'):
                extracted_url = query_params[param][0]
                # URL decode if needed
                if '%' in extracted_url:
                    extracted_url = urllib.parse.unquote(extracted_url)
                return extracted_url
                
        return url
    except Exception as e:
        logger.error(f"Error extracting from URL parameters: {str(e)}")
        return url

def _extract_techmeme_source(url: str, session, debug: bool = False) -> Optional[str]:
    """
    Extract the original source URL from a Techmeme story page using HTML parsing.
    This improved version handles different page layouts and has better error recovery.
    
    Args:
        url: Techmeme URL (story page)
        session: Requests session to use
        debug: Whether to log additional debug information
        
    Returns:
        str: Original source URL or None if extraction failed
    """
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup is not available, cannot extract from Techmeme HTML")
        return None
        
    try:
        # Fetch the Techmeme page
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch Techmeme page: {url}, status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if debug:
            # Dump HTML structure to log for inspection
            logger.debug(f"HTML structure of Techmeme page: {soup.prettify()[:1000]}")
            
            # Log all potential source links found
            all_links = soup.find_all('a')
            logger.debug(f"All links on the page: {[link['href'] for link in all_links if link.has_attr('href')][:20]}")
        
        # Check if this is a story page or the homepage
        if soup.select('.clus') and not soup.select('.newsItem'):
            # This is likely the homepage, not a story page
            logger.warning(f"URL appears to be the Techmeme homepage, not a story page: {url}")
            # Try to extract sources using homepage extraction logic instead
            homepage_stories = extract_techmeme_homepage_stories(url, session, debug)
            if homepage_stories and len(homepage_stories) > 0:
                # Return the first story's source URL to maintain backward compatibility
                return homepage_stories[0]['source_url']
            return None
        
        candidate_urls = []
        
        # Enhanced Method 1: Look for the main headline's primary source
        # Try multiple selectors to handle different page layouts
        for news_selector in ['.newsItem', '.item']:
            news_items = soup.select(news_selector)
            for news_item in news_items:
                # Try to find the headline
                for headline_selector in ['strong.title', 'b.title', '.title', 'h2']:
                    headline = news_item.select_one(headline_selector)
                    if headline:
                        # Try to find the source link
                        # First check if there's a direct link in a sibling element
                        source_container = headline.find_next_sibling()
                        if source_container and source_container.find('a'):
                            source_url = source_container.find('a')['href']
                            if is_valid_source_url(source_url):
                                logger.info(f"Method 1: Found source URL from main headline: {source_url}")
                                candidate_urls.append(source_url)
                                
                        # Also check if the source is in a span inside the headline
                        source_spans = headline.select('span')
                        for span in source_spans:
                            source_link = span.find('a')
                            if source_link and source_link.has_attr('href'):
                                source_url = source_link['href']
                                if is_valid_source_url(source_url):
                                    logger.info(f"Method 1b: Found source URL in headline span: {source_url}")
                                    candidate_urls.append(source_url)
        
        # Method 2: Look for the "More:" section which contains additional sources
        for more_selector in ['.continuation', '.more', '.readmore']:
            more_sections = soup.select(more_selector)
            for more_section in more_sections:
                source_links = more_section.select('a')
                for source_link in source_links:
                    if source_link.has_attr('href'):
                        source_url = source_link['href']
                        if is_valid_source_url(source_url):
                            logger.info(f"Method 2: Found source URL from 'More:' section: {source_url}")
                            candidate_urls.append(source_url)
        
        # Method 3: Find any author attribution links
        for attr_selector in ['.itemsource', '.source', '.byline', '.attribution']:
            attributions = soup.select(attr_selector)
            for attribution in attributions:
                source_links = attribution.select('a')
                for source_link in source_links:
                    if source_link.has_attr('href'):
                        source_url = source_link['href']
                        if is_valid_source_url(source_url):
                            logger.info(f"Method 3: Found source URL from attribution: {source_url}")
                            candidate_urls.append(source_url)
        
        # Method 4: Try original selectors as fallback
        for selector in ['.ii a', '.ourh a', '.quote a', '.entry a']:
            source_links = soup.select(selector)
            for source_link in source_links:
                if source_link.has_attr('href'):
                    source_url = source_link['href']
                    if is_valid_source_url(source_url):
                        logger.info(f"Method 4: Found source URL using selector '{selector}': {source_url}")
                        candidate_urls.append(source_url)
        
        # Method 5: Look for any news source links (last resort fallback)
        if not candidate_urls:
            all_links = soup.select('a')
            for link in all_links:
                if link.has_attr('href'):
                    href = link['href']
                    if is_valid_source_url(href):
                        if any(domain in href for domain in ['.com', '.org', '.net', '.io']):
                            # Skip Techmeme's own links
                            if 'techmeme.com' not in href:
                                source_url = href
                                logger.info(f"Method 5: Found potential source URL: {source_url}")
                                candidate_urls.append(source_url)
                                if len(candidate_urls) >= 5:
                                    break
        
        # Choose the best URL from candidates
        if candidate_urls:
            best_url = prioritize_known_news_sources(candidate_urls)
            logger.info(f"Selected source URL: {best_url} from {len(candidate_urls)} candidates")
            return best_url
            
        logger.warning(f"Failed to extract source URL from Techmeme: {url}")
    
    except Exception as e:
        logger.warning(f"Error extracting source from Techmeme: {str(e)}")
    
    return None

def extract_techmeme_homepage_stories(url: str, session=None, debug: bool = False) -> List[Dict[str, str]]:
    """
    Extract all stories with their source URLs from the Techmeme homepage.
    
    Args:
        url: Techmeme homepage URL
        session: Optional requests session to use
        debug: Whether to log additional debug information
        
    Returns:
        List of dicts with 'headline', 'techmeme_link', 'source_url', and 'source_name' keys
    """
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup is not available, cannot extract from Techmeme HTML")
        return []
        
    try:
        if not session:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
        
        # Fetch the Techmeme page
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch Techmeme page: {url}, status code: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        
        # Find all story clusters
        clusters = soup.select('.clus')
        
        if debug:
            logger.debug(f"Found {len(clusters)} story clusters on Techmeme homepage")
        
        for cluster in clusters:
            # Find the main headline
            headline_elem = cluster.select_one('strong.title')
            if not headline_elem:
                continue
                
            headline_text = headline_elem.get_text().strip()
            
            # Find the Techmeme link for this story
            headline_link = headline_elem.find('a')
            if not headline_link or not headline_link.has_attr('href'):
                continue
                
            techmeme_link = headline_link['href']
            if not techmeme_link.startswith('http'):
                techmeme_link = f"https://techmeme.com{techmeme_link}"
            
            # Find the source link
            source_container = headline_elem.find_next_sibling()
            if not source_container or not source_container.find('a'):
                continue
                
            source_link = source_container.find('a')
            if not source_link.has_attr('href'):
                continue
                
            source_url = source_link['href']
            source_name = source_link.get_text().strip()
            
            if is_valid_source_url(source_url):
                results.append({
                    'headline': headline_text,
                    'techmeme_link': techmeme_link,
                    'source_url': source_url,
                    'source_name': source_name
                })
                
                if debug:
                    logger.debug(f"Extracted headline: {headline_text}, source: {source_name}, URL: {source_url}")
            
            # Also check for related items in this cluster
            related_items = cluster.select('.ii')
            for item in related_items:
                item_link = item.find('a')
                if not item_link or not item_link.has_attr('href'):
                    continue
                    
                item_url = item_link['href']
                item_text = item_link.get_text().strip()
                
                if is_valid_source_url(item_url):
                    results.append({
                        'headline': f"Related: {item_text}",
                        'techmeme_link': techmeme_link,  # Same Techmeme link as the main story
                        'source_url': item_url,
                        'source_name': item_link.get_text().strip()
                    })
                    
                    if debug:
                        logger.debug(f"Extracted related item: {item_text}, URL: {item_url}")
        
        logger.info(f"Extracted {len(results)} stories from Techmeme homepage")
        return results
        
    except Exception as e:
        logger.warning(f"Error extracting stories from Techmeme homepage: {str(e)}")
        return []

def _extract_google_news_source(url: str, session) -> Optional[str]:
    """
    Extract the original source URL from a Google News link using HTML parsing.
    
    Args:
        url: Google News URL
        session: Requests session to use
        
    Returns:
        str: Original source URL or None if extraction failed
    """
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup is not available, cannot extract from Google News HTML")
        return None
        
    try:
        # Check if the URL contains the source URL as a parameter
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Google News often includes the source URL in the 'url' parameter
        if 'url' in query_params and query_params['url']:
            source_url = query_params['url'][0]
            if is_valid_source_url(source_url):
                logger.info(f"Extracted original source URL from Google News URL parameter: {source_url}")
                return source_url
        
        # If not in the URL, fetch the page and look for redirects or source links
        response = session.get(url, timeout=10, allow_redirects=False)
        
        # Check if there's a redirect
        if response.status_code in (301, 302, 303, 307, 308) and 'Location' in response.headers:
            redirect_url = response.headers['Location']
            if is_valid_source_url(redirect_url):
                logger.info(f"Extracted original source URL from Google News redirect: {redirect_url}")
                return redirect_url
            
        # If no redirect, parse the page
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the main article link
            article_links = soup.select('a[data-n-au]')
            if article_links:
                source_url = article_links[0]['href']
                # If it's a relative URL, complete it
                if source_url.startswith('/'):
                    source_url = f"https://news.google.com{source_url}"
                if is_valid_source_url(source_url):
                    logger.info(f"Extracted original source URL from Google News HTML: {source_url}")
                    return source_url
                    
            # Try alternative selectors
            for selector in ['a.VDXfz', '.DY5T1d', 'a[target="_blank"]']:
                links = soup.select(selector)
                for link in links:
                    if link.has_attr('href'):
                        source_url = link['href']
                        if is_valid_source_url(source_url):
                            logger.info(f"Extracted original source URL using alternative selector '{selector}': {source_url}")
                            return source_url
    
    except Exception as e:
        logger.warning(f"Error extracting source from Google News: {str(e)}")
    
    return None

def batch_extract_sources(urls: List[str], use_html_parsing: bool = True) -> Dict[str, str]:
    """
    Process a batch of URLs to extract original sources.
    
    Args:
        urls: List of URLs to process
        use_html_parsing: Whether to use HTML parsing for extraction
        
    Returns:
        dict: Mapping from original URLs to extracted source URLs
    """
    results = {}
    
    # Create a single session for all requests (more efficient)
    session = None
    if use_html_parsing and REQUESTS_AVAILABLE:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    for url in urls:
        if is_aggregator_link(url):
            source_url = extract_original_source_url(url, session, use_html_parsing)
            results[url] = source_url
        else:
            results[url] = url  # Not an aggregator, keep original
            
    return results