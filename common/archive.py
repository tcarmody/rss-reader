"""Archive service utilities for bypassing paywalls."""

import re
import time
import logging
import random
from urllib.parse import urlparse, quote
import requests
from bs4 import BeautifulSoup

# Import source extractor utilities
from common.source_extractor import is_aggregator_link, extract_original_source_url

# List of known paywall domains
PAYWALL_DOMAINS = [
    'nytimes.com',
    'wsj.com',
    'washingtonpost.com',
    'bloomberg.com',
    'ft.com',
    'economist.com',
    'newyorker.com',
    'wired.com',
    'theatlantic.com',
    'technologyreview.com',
    'hbr.org',
    'forbes.com',
    'businessinsider.com',
    'medium.com',
]

# Archive services
ARCHIVE_SERVICES = [
    {
        'name': 'Archive.is',
        'url': 'https://archive.is/{url}',
        'search_url': 'https://archive.is/search/?q={url}',
        'submit_url': 'https://archive.is/submit/',
        'needs_submission': True,
    },
    {
        'name': 'Archive.org',
        'url': 'https://web.archive.org/web/{timestamp}/{url}',
        'search_url': 'https://web.archive.org/cdx/search/cdx?url={url}&output=json&limit=1',
        'submit_url': 'https://web.archive.org/save/{url}',
        'needs_submission': True,
    },
    {
        'name': 'Google Cache',
        'url': 'https://webcache.googleusercontent.com/search?q=cache:{url}',
        'search_url': None,
        'submit_url': None,
        'needs_submission': False,
    },
]


def is_paywalled(url):
    """
    Check if a URL is likely behind a paywall.
    
    Args:
        url: The article URL to check
        
    Returns:
        bool: True if the URL is likely paywalled
    """
    domain = urlparse(url).netloc.lower()
    
    # Check against known paywall domains
    for paywall_domain in PAYWALL_DOMAINS:
        if paywall_domain in domain:
            return True
            
    return False


def get_archive_url(url, session=None, force_new=False):
    """
    Get an archive URL for the given article URL.
    
    Args:
        url: The article URL to get an archive for
        session: Optional requests session to use
        force_new: Force creation of a new archive
        
    Returns:
        str: Archive URL or None if not available
    """
    if not session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    # Try each archive service
    random.shuffle(ARCHIVE_SERVICES)  # Randomize to distribute load
    
    for service in ARCHIVE_SERVICES:
        try:
            archive_url = None
            
            # If we need a new archive or the service requires submission
            if force_new and service['needs_submission']:
                archive_url = _submit_to_archive(url, service, session)
                if archive_url:
                    logging.info(f"Created new archive at {service['name']}: {archive_url}")
                    return archive_url
            
            # Try to find existing archive
            if not force_new and service['search_url']:
                archive_url = _find_existing_archive(url, service, session)
                if archive_url:
                    logging.info(f"Found existing archive at {service['name']}: {archive_url}")
                    return archive_url
            
            # Direct access for services like Google Cache
            if not service['needs_submission']:
                if service['name'] == 'Google Cache':
                    encoded_url = quote(url, safe='')
                    archive_url = service['url'].format(url=encoded_url)
                    logging.info(f"Using {service['name']}: {archive_url}")
                    return archive_url
        
        except Exception as e:
            logging.warning(f"Error with {service['name']}: {str(e)}")
            continue
    
    logging.warning(f"Could not find or create archive for: {url}")
    return None


def _find_existing_archive(url, service, session):
    """Find an existing archive for the URL."""
    try:
        if service['name'] == 'Archive.is':
            search_url = service['search_url'].format(url=quote(url, safe=''))
            response = session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.select('.results > .result-url')
            
            if results:
                # Get the first result
                archive_path = results[0].get('href')
                if archive_path:
                    return f"https://archive.is{archive_path}"
        
        elif service['name'] == 'Archive.org':
            search_url = service['search_url'].format(url=quote(url, safe=''))
            response = session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and len(data[1]) > 1:
                    # Extract the timestamp from the response
                    timestamp = data[1][1]
                    return service['url'].format(timestamp=timestamp, url=url)
    
    except Exception as e:
        logging.warning(f"Error finding existing archive: {str(e)}")
    
    return None


def _submit_to_archive(url, service, session):
    """Submit a URL to an archive service."""
    try:
        if service['name'] == 'Archive.is':
            data = {'url': url}
            response = session.post(service['submit_url'], data=data, timeout=30)
            
            # Check if we were redirected to an archive page
            if response.status_code == 200 and 'archive.is' in response.url:
                return response.url
        
        elif service['name'] == 'Archive.org':
            submit_url = service['submit_url'].format(url=url)
            response = session.get(submit_url, timeout=30)
            
            # Wait for archiving to complete
            time.sleep(5)
            
            # Try to get the archived version
            search_url = service['search_url'].format(url=quote(url, safe=''))
            search_response = session.get(search_url, timeout=10)
            
            if search_response.status_code == 200:
                data = search_response.json()
                if len(data) > 1 and len(data[1]) > 1:
                    timestamp = data[1][1]
                    return service['url'].format(timestamp=timestamp, url=url)
    
    except Exception as e:
        logging.warning(f"Error submitting to archive: {str(e)}")
    
    return None


def get_content_from_archive(archive_url, session=None):
    """
    Extract content from an archive URL.
    
    Args:
        archive_url: The archive URL to extract content from
        session: Optional requests session to use
        
    Returns:
        str: Extracted content or empty string if failed
    """
    if not session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    try:
        response = session.get(archive_url, timeout=15)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .related, .sidebar'):
            unwanted.decompose()
        
        # Try to find the main content
        main_content = None
        
        # Common content selectors
        content_selectors = [
            'article', '.article', '.post-content', '.entry-content', '.content', 
            'main', '#main', '.main', '.story', '.story-body'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0]
                break
        
        # If no main content found, use the body
        if not main_content:
            main_content = soup.body
        
        # Extract text
        if main_content:
            # Get all paragraphs
            paragraphs = main_content.find_all('p')
            content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
            
            # If no substantial paragraphs found, get all text
            if not content:
                content = main_content.get_text()
            
            return content.strip()
    
    except Exception as e:
        logging.warning(f"Error extracting content from archive: {str(e)}")
    
    return ""


def fetch_article_content(url, session=None):
    """
    Fetch article content, using archive services for paywalled content.
    
    Args:
        url: The article URL
        session: Optional requests session to use
        
    Returns:
        str: Article content or empty string if failed
    """
    if not session:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    content = ""
    original_url = url
    source_extracted = False
    
    # Check if this is a news aggregator link (Techmeme or Google News)
    if is_aggregator_link(url):
        logging.info(f"Detected aggregator link: {url}, extracting original source")
        source_url = extract_original_source_url(url, session)
        
        if source_url:
            logging.info(f"Successfully extracted original source URL: {source_url} from aggregator: {url}")
            original_url = source_url
            source_extracted = True
        else:
            logging.warning(f"Failed to extract original source URL from aggregator: {url}")
    
    # Check if the URL is paywalled
    if is_paywalled(original_url):
        logging.info(f"Detected paywall for {original_url}, trying archive services")
        
        # Try to get an archive URL
        archive_url = get_archive_url(original_url, session)
        
        if archive_url:
            # Extract content from the archive
            content = get_content_from_archive(archive_url, session)
            
            if content:
                logging.info(f"Successfully extracted content from archive for {original_url}")
                return content
            else:
                logging.warning(f"Failed to extract content from archive for {original_url}")
    
    # If not paywalled or archive failed, try direct access
    try:
        response = session.get(original_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .related, .sidebar'):
                unwanted.decompose()
            
            # Try to find the main content using common selectors
            content_selectors = [
                'article', '.article', '.post-content', '.entry-content', '.content', 
                'main', '#main', '.main', '.story', '.story-body'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get all paragraphs
                    paragraphs = elements[0].find_all('p')
                    content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                    
                    if content:
                        if source_extracted:
                            logging.info(f"Successfully extracted content from original source: {original_url} (via aggregator: {url})")
                        else:
                            logging.info(f"Successfully extracted content directly from {original_url}")
                        return content
            
            # If no content found with selectors, use readability as fallback
            if not content:
                # Simple extraction of paragraphs from the body
                paragraphs = soup.body.find_all('p')
                content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                
                if content:
                    if source_extracted:
                        logging.info(f"Extracted content using fallback method from original source: {original_url} (via aggregator: {url})")
                    else:
                        logging.info(f"Extracted content using fallback method from {original_url}")
                    return content
    
    except Exception as e:
        logging.warning(f"Error fetching content directly: {str(e)}")
    
    logging.warning(f"Failed to extract content from {original_url}")
    return content
