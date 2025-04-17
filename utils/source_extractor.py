"""Utilities for extracting original source URLs from aggregator sites."""

import re
import logging
import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

def is_aggregator_link(url):
    """
    Check if a URL is from a news aggregator like Techmeme or Google News.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if the URL is from a known aggregator
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Check for Techmeme
    if 'techmeme.com' in domain:
        return True
    
    # Check for Google News
    if 'news.google.com' in domain:
        return True
        
    return False

def extract_original_source_url(url, session=None):
    """
    Extract the original source URL from an aggregator link.
    
    Args:
        url: Aggregator URL (Techmeme or Google News)
        session: Optional requests session to use
        
    Returns:
        str: Original source URL or None if extraction failed
    """
    if not session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    try:
        # Handle Techmeme links
        if 'techmeme.com' in domain:
            return _extract_techmeme_source(url, session)
        
        # Handle Google News links
        if 'news.google.com' in domain:
            return _extract_google_news_source(url, session)
            
    except Exception as e:
        logging.warning(f"Error extracting original source URL from {url}: {str(e)}")
    
    return None

def is_valid_source_url(url):
    """
    Check if a URL is likely to be a valid news source.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if the URL appears to be a valid news source
    """
    # Skip common non-news domains
    excluded_domains = ['twitter.com', 'facebook.com', 'linkedin.com', 't.co',
                       'instagram.com', 'youtube.com', 'reddit.com', 'techmeme.com']
    for domain in excluded_domains:
        if domain in url:
            return False
    
    # Check if the URL has a proper protocol
    if not url.startswith('http'):
        return False
    
    # Make sure URL isn't too short
    if len(url) < 12:  # http://a.com is minimum valid URL (11 chars)
        return False
    
    return True

def prioritize_known_news_sources(urls):
    """
    Give priority to URLs from known news sources.
    
    Args:
        urls: List of URLs to prioritize
        
    Returns:
        str: The highest priority URL, or None if the list is empty
    """
    if not urls:
        return None
        
    known_sources = [
        'techcrunch.com', 'theverge.com', 'wired.com', 'bloomberg.com', 'wsj.com',
        'nytimes.com', 'washingtonpost.com', 'cnn.com', 'bbc.com', 'reuters.com',
        'arstechnica.com', 'engadget.com', 'cnet.com', 'zdnet.com', 'venturebeat.com'
    ]
    
    for url in urls:
        for source in known_sources:
            if source in url:
                return url
    
    return urls[0]  # Return the first URL if no known sources found

def _extract_techmeme_source(url, session, debug=False):
    """
    Extract the original source URL from a Techmeme link.
    
    Args:
        url: Techmeme URL
        session: Requests session to use
        debug: Whether to log additional debug information
        
    Returns:
        str: Original source URL or None if extraction failed
    """
    try:
        # Fetch the Techmeme page
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch Techmeme page: {url}, status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if debug:
            # Dump HTML structure to log for inspection
            logging.debug(f"HTML structure of Techmeme page: {soup.prettify()[:1000]}")
            
            # Log all potential source links found
            all_links = soup.find_all('a')
            logging.debug(f"All links on the page: {[link['href'] for link in all_links if link.has_attr('href')][:20]}")
        
        candidate_urls = []
        
        # Method 1: Look for the main headline's primary source
        # Techmeme typically has news items in a block structure
        news_item = soup.select_one('.newsItem')
        if news_item:
            # Find the first link after the headline (typically the main source)
            headline = news_item.select_one('strong.title')
            if headline:
                # Get the next element after the headline that contains a link
                source_container = headline.find_next_sibling()
                if source_container and source_container.find('a'):
                    source_url = source_container.find('a')['href']
                    if is_valid_source_url(source_url):
                        logging.info(f"Method 1: Found source URL from main headline: {source_url}")
                        candidate_urls.append(source_url)
        
        # Method 2: Look for the "More:" section which contains additional sources
        more_section = soup.select_one('.continuation')
        if more_section:
            # The first link in this section is typically the primary source
            source_link = more_section.select_one('a')
            if source_link and source_link.has_attr('href'):
                source_url = source_link['href']
                if is_valid_source_url(source_url):
                    logging.info(f"Method 2: Found source URL from 'More:' section: {source_url}")
                    candidate_urls.append(source_url)
        
        # Method 3: Find any author attribution links
        # Techmeme often has author/publication names before sources
        attribution = soup.select_one('.itemsource')
        if attribution:
            source_link = attribution.select_one('a')
            if source_link and source_link.has_attr('href'):
                source_url = source_link['href']
                if is_valid_source_url(source_url):
                    logging.info(f"Method 3: Found source URL from attribution: {source_url}")
                    candidate_urls.append(source_url)
        
        # Method 4: Try original selectors as fallback (for backwards compatibility)
        for selector in ['.ii a', '.ourh a']:
            source_link = soup.select_one(selector)
            if source_link and source_link.has_attr('href'):
                source_url = source_link['href']
                if is_valid_source_url(source_url):
                    logging.info(f"Method 4: Found source URL using legacy selector '{selector}': {source_url}")
                    candidate_urls.append(source_url)
        
        # Method 5: Look for any news source links (last resort fallback)
        if not candidate_urls:
            all_links = soup.select('a')
            for link in all_links:
                # Skip internal Techmeme links and social media links
                if link.has_attr('href'):
                    href = link['href']
                    if is_valid_source_url(href):
                        # Check if this might be a news URL
                        if any(domain in href for domain in ['.com', '.org', '.net', '.io']):
                            source_url = href
                            logging.info(f"Method 5: Found potential source URL: {source_url}")
                            candidate_urls.append(source_url)
                            # Only get the first few candidates to avoid excessive logging
                            if len(candidate_urls) >= 5:
                                break
        
        # Choose the best URL from candidates
        if candidate_urls:
            best_url = prioritize_known_news_sources(candidate_urls)
            logging.info(f"Selected source URL: {best_url} from {len(candidate_urls)} candidates")
            return best_url
            
        # Log failure
        logging.warning(f"Failed to extract source URL from Techmeme: {url}")
    
    except Exception as e:
        logging.warning(f"Error extracting source from Techmeme: {str(e)}")
    
    return None

def _extract_google_news_source(url, session):
    """
    Extract the original source URL from a Google News link.
    
    Args:
        url: Google News URL
        session: Requests session to use
        
    Returns:
        str: Original source URL or None if extraction failed
    """
    try:
        # Check if the URL contains the source URL as a parameter
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Google News often includes the source URL in the 'url' parameter
        if 'url' in query_params and query_params['url']:
            source_url = query_params['url'][0]
            if is_valid_source_url(source_url):
                logging.info(f"Extracted original source URL from Google News URL parameter: {source_url}")
                return source_url
        
        # If not in the URL, fetch the page and look for redirects or source links
        response = session.get(url, timeout=10, allow_redirects=False)
        
        # Check if there's a redirect
        if response.status_code in (301, 302, 303, 307, 308) and 'Location' in response.headers:
            redirect_url = response.headers['Location']
            if is_valid_source_url(redirect_url):
                logging.info(f"Extracted original source URL from Google News redirect: {redirect_url}")
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
                    logging.info(f"Extracted original source URL from Google News HTML: {source_url}")
                    return source_url
                    
            # Try alternative selectors
            for selector in ['a.VDXfz', '.DY5T1d', 'a[target="_blank"]']:
                links = soup.select(selector)
                for link in links:
                    if link.has_attr('href'):
                        source_url = link['href']
                        if is_valid_source_url(source_url):
                            logging.info(f"Extracted original source URL using alternative selector '{selector}': {source_url}")
                            return source_url
    
    except Exception as e:
        logging.warning(f"Error extracting source from Google News: {str(e)}")
    
    return None