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

def _extract_techmeme_source(url, session):
    """Extract the original source URL from a Techmeme link."""
    try:
        # Fetch the Techmeme page
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Techmeme typically has the source link in the first item with class 'ii'
        source_link = soup.select_one('.ii a')
        if source_link and source_link.has_attr('href'):
            source_url = source_link['href']
            logging.info(f"Extracted original source URL from Techmeme: {source_url}")
            return source_url
            
        # Alternative method: look for the main story link
        main_link = soup.select_one('.ourh a')
        if main_link and main_link.has_attr('href'):
            source_url = main_link['href']
            logging.info(f"Extracted original source URL from Techmeme (alternative method): {source_url}")
            return source_url
    
    except Exception as e:
        logging.warning(f"Error extracting source from Techmeme: {str(e)}")
    
    return None

def _extract_google_news_source(url, session):
    """Extract the original source URL from a Google News link."""
    try:
        # Check if the URL contains the source URL as a parameter
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Google News often includes the source URL in the 'url' parameter
        if 'url' in query_params and query_params['url']:
            source_url = query_params['url'][0]
            logging.info(f"Extracted original source URL from Google News URL parameter: {source_url}")
            return source_url
        
        # If not in the URL, fetch the page and look for redirects or source links
        response = session.get(url, timeout=10, allow_redirects=False)
        
        # Check if there's a redirect
        if response.status_code in (301, 302, 303, 307, 308) and 'Location' in response.headers:
            redirect_url = response.headers['Location']
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
                logging.info(f"Extracted original source URL from Google News HTML: {source_url}")
                return source_url
    
    except Exception as e:
        logging.warning(f"Error extracting source from Google News: {str(e)}")
    
    return None
