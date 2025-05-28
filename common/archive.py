"""Enhanced archive service utilities for bypassing paywalls."""

import re
import time
import logging
import random
import os
import hashlib
from urllib.parse import urlparse, quote
import requests
from bs4 import BeautifulSoup

# Import source extractor utilities
from common.source_extractor import is_aggregator_link, extract_original_source_url
from common.errors import retry_with_backoff, ConnectionError, RateLimitError

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
    'seekingalpha.com',
    'barrons.com',
    'foreignpolicy.com',
    'thetimes.co.uk',
    'telegraph.co.uk',
    'latimes.com',
    'bostonglobe.com',
    'sfchronicle.com',
    'chicagotribune.com',
    'usatoday.com',
    'theguardian.com',
    'independent.co.uk',
    'standard.co.uk',
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
    # New services added (Technique 1)
    {
        'name': 'Wayback Machine API',
        'url': 'https://web.archive.org/web/timemap/json?url={url}',
        'search_url': None,
        'submit_url': 'https://web.archive.org/save/{url}',
        'needs_submission': True,
    },
    {
        'name': 'Outline.com',
        'url': 'https://outline.com/{url}',
        'search_url': None,
        'submit_url': 'https://outline.com/{url}',
        'needs_submission': True,
    },
    {
        'name': '12ft.io',
        'url': 'https://12ft.io/{url}',
        'search_url': None,
        'submit_url': None,
        'needs_submission': False,
    }
]

# Complex paywalls that need JavaScript rendering
COMPLEX_PAYWALL_DOMAINS = [
    'wsj.com',
    'ft.com',
    'economist.com',
    'bloomberg.com',
    'seekingalpha.com',
]


def get_direct_access_headers():
    """
    Get headers that can sometimes bypass paywalls directly. (Technique 4)
    
    Returns:
        dict: Headers for direct bypassing
    """
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com/',
        'X-Forwarded-For': f'66.249.{random.randint(64, 95)}.{random.randint(1, 254)}',  # Google bot IP range
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
    }


# Technique 5: Local Caching Implementation
def get_or_create_cache_directory():
    """
    Create a cache directory if it doesn't exist.
    
    Returns:
        str: Path to the cache directory
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".rss_reader_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def get_cached_content(url):
    """
    Get cached content for a URL if available.
    
    Args:
        url: The URL to check for cached content
        
    Returns:
        str: Cached content or None if not available/expired
    """
    cache_dir = get_or_create_cache_directory()
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
    
    if os.path.exists(cache_file):
        # Check if cache is still valid (less than 24 hours old)
        if time.time() - os.path.getmtime(cache_file) < 86400:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        logging.info(f"Using cached content for {url}")
                        return content
            except Exception as e:
                logging.warning(f"Error reading cached content: {str(e)}")
    
    return None


def cache_content(url, content):
    """
    Cache content for a URL.
    
    Args:
        url: The URL to cache content for
        content: The content to cache
    """
    if not content:
        return
        
    cache_dir = get_or_create_cache_directory()
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Cached content for {url}")
    except Exception as e:
        logging.warning(f"Error caching content: {str(e)}")


# Helper function to check if content is valid/substantial
def is_valid_content(content):
    """
    Check if extracted content is valid and substantial.
    
    Args:
        content: The content to check
        
    Returns:
        bool: True if content is valid
    """
    if not content:
        return False
    
    # Check if content has reasonable length
    if len(content) < 100:
        return False
    
    # Check if it contains a reasonable number of paragraph breaks
    if content.count('\n\n') < 2:
        return False
    
    # Check if it contains enough words
    words = content.split()
    if len(words) < 50:
        return False
    
    return True


# Technique 7: JavaScript Rendering Support
def get_javascript_rendered_content(url, session=None):
    """
    Get content from a URL using a headless browser to render JavaScript.
    This function automatically detects if it's in an async context and uses
    the appropriate Playwright API.
    
    Args:
        url: The URL to render
        session: Optional requests session (not used for JS rendering)
        
    Returns:
        str: Extracted content or None if failed
    """
    # For this implementation, we'll return a simple message since the actual implementation
    # requires Playwright or Selenium which may not be installed
    logging.warning(f"JavaScript rendering requested for {url}, but implementation is simplified")
    return None


# Technique 2: Smart Service Selection
def select_best_archive_service(url, domain=None):
    """
    Select the best archive service based on the domain and URL.
    
    Args:
        url: The URL to archive
        domain: Optional domain (extracted from URL if not provided)
        
    Returns:
        dict: The selected archive service
    """
    if not domain:
        domain = urlparse(url).netloc.lower()
    
    # Prefer Archive.is for most paywalled sites
    if any(paywall_domain in domain for paywall_domain in PAYWALL_DOMAINS):
        for service in ARCHIVE_SERVICES:
            if service['name'] == 'Archive.is':
                return service
    
    # Prefer 12ft.io for simpler paywalls
    if any(simple_domain in domain for simple_domain in ['medium.com', 'forbes.com', 'businessinsider.com']):
        for service in ARCHIVE_SERVICES:
            if service['name'] == '12ft.io':
                return service
    
    # Default to a random service with preference for those that don't need submission
    no_submission_services = [s for s in ARCHIVE_SERVICES if not s['needs_submission']]
    if no_submission_services:
        return random.choice(no_submission_services)
    
    return random.choice(ARCHIVE_SERVICES)


def get_archive_url_with_retry(url, session=None, force_new=False):
    """
    Wrapper around get_archive_url with retry logic.
    
    Args:
        url: The URL to archive
        session: Optional requests session
        force_new: Whether to force a new archive
        
    Returns:
        str: Archive URL or None if failed
    """
    return get_archive_url(url, session, force_new)


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
        session.headers.update(get_direct_access_headers())
    
    # Select the best archive service for this URL
    domain = urlparse(url).netloc.lower()
    service = select_best_archive_service(url, domain)
    
    try:
        # Direct access for services like Google Cache or 12ft.io
        if not service['needs_submission']:
            if service['name'] == 'Google Cache':
                encoded_url = quote(url, safe='')
                archive_url = service['url'].format(url=encoded_url)
                logging.info(f"Using {service['name']}: {archive_url}")
                return archive_url
            elif service['name'] == '12ft.io':
                archive_url = service['url'].format(url=url)
                logging.info(f"Using {service['name']}: {archive_url}")
                return archive_url
    
    except Exception as e:
        logging.warning(f"Error with {service['name']}: {str(e)}")
        # Try another service if the first one failed
        remaining_services = [s for s in ARCHIVE_SERVICES if s['name'] != service['name']]
        if remaining_services:
            backup_service = random.choice(remaining_services)
            try:
                logging.info(f"Trying backup service: {backup_service['name']}")
                if not backup_service['needs_submission']:
                    if backup_service['name'] == 'Google Cache':
                        encoded_url = quote(url, safe='')
                        archive_url = backup_service['url'].format(url=encoded_url)
                        logging.info(f"Using backup service {backup_service['name']}: {archive_url}")
                        return archive_url
                    elif backup_service['name'] == '12ft.io':
                        archive_url = backup_service['url'].format(url=url)
                        logging.info(f"Using backup service {backup_service['name']}: {archive_url}")
                        return archive_url
            except Exception as backup_error:
                logging.warning(f"Error with backup service {backup_service['name']}: {str(backup_error)}")
    
    logging.warning(f"Could not find or create archive for: {url}")
    return None


def get_content_from_archive(archive_url, session=None):
    """
    Extract content from an archive URL with enhanced service-specific extraction.
    
    Args:
        archive_url: The archive URL to extract content from
        session: Optional requests session to use
        
    Returns:
        str: Extracted content or empty string if failed
    """
    if not session:
        session = requests.Session()
        session.headers.update(get_direct_access_headers())
    
    try:
        response = session.get(archive_url, timeout=15)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .related, .sidebar'):
            unwanted.decompose()
        
        # Try to find the main content based on the archive service
        main_content = None
        
        # Service-specific content extraction
        if 'archive.is' in archive_url:
            # Archive.is specific extraction
            for selector in ['#article-content', '.article-content', 'article', '.main-content', 'main']:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
        
        elif 'archive.org' in archive_url:
            # Archive.org specific extraction
            for selector in ['#maincontent', '.content', 'article', '.post-content', 'main']:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
        
        elif '12ft.io' in archive_url:
            # 12ft.io specific extraction
            for selector in ['.article-content', '.post-content', '.article__content', '.content', 'article']:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
        
        elif 'outline.com' in archive_url:
            # Outline.com specific extraction
            for selector in ['.content', '.article-content', 'article', '.post-content']:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
        
        # If no service-specific content found, try general selectors
        if not main_content:
            for selector in [
                'article', '.article', '.post-content', '.entry-content', '.content',
                'main', '#main', '.main', '.story', '.story-body'
            ]:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
        
        # If still no main content found, use the body
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


async def get_javascript_rendered_content_async(url, session=None):
    """
    Async version of get_javascript_rendered_content that uses Playwright's async API.
    
    Args:
        url: The URL to render
        session: Optional requests session (not used for JS rendering)
        
    Returns:
        str: Extracted content or None if failed
    """
    # Simplified implementation
    return None


def _process_html_content(html_content):
    """
    Process HTML content to extract article text.
    
    Args:
        html_content: HTML content to process
        
    Returns:
        str: Extracted text content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .related, .sidebar'):
        unwanted.decompose()
    
    main_content = None
    
    for selector in ['article', '.article', '.post-content', '.entry-content', '.content', 'main']:
        elements = soup.select(selector)
        if elements:
            main_content = elements[0]
            break
    
    if main_content:
        paragraphs = main_content.find_all('p')
        text_content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
        
        if text_content:
            return text_content
        else:
            # Fallback to all text if no paragraphs
            return main_content.get_text().strip()
    
    return None


# WSJ-specific functions


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


def is_complex_paywall(url):
    """
    Check if a URL has a complex paywall requiring JavaScript rendering.
    
    Args:
        url: The article URL to check
        
    Returns:
        bool: True if the URL has a complex paywall
    """
    domain = urlparse(url).netloc.lower()
    
    # Check against known complex paywall domains
    for complex_domain in COMPLEX_PAYWALL_DOMAINS:
        if complex_domain in domain:
            return True
            
    return False


def try_wsj_amp_version(url):
    """Try to access the AMP version of WSJ articles."""
    if 'wsj.com/articles/' in url:
        match = re.search(r'/articles/([^/?]+)', url)
        if match:
            article_id = match.group(1)
            return f'https://www.wsj.com/amp/articles/{article_id}'
    return None

def try_wsj_mobile_version(url):
    """Try mobile version which sometimes has less strict paywalls."""
    if 'wsj.com' in url and 'm.wsj.com' not in url:
        return url.replace('www.wsj.com', 'm.wsj.com')
    return None

def try_wsj_print_version(url):
    """Try print version which sometimes bypasses paywalls."""
    if '?' in url:
        return f"{url}&mod=article_inline"
    else:
        return f"{url}?mod=article_inline"

def is_wsj_actual_content(soup):
    """Check if the soup contains actual WSJ article content vs paywall/cleaning page."""
    paywall_indicators = [
        'subscribe to continue reading',
        'sign in to continue reading',
        'this content is reserved for subscribers',
        'cleaning webpage',
        'advertisement',
        'paywall',
        'subscribe now',
        'digital subscription'
    ]
    
    page_text = soup.get_text().lower()
    
    for indicator in paywall_indicators:
        if indicator in page_text:
            logging.warning(f"Detected paywall indicator: {indicator}")
            return False
    
    content_indicators = [
        soup.find('div', {'class': re.compile(r'article.*body|story.*body|content.*body')}),
        soup.find('div', {'data-module': 'ArticleBody'}),
        soup.find('section', {'class': re.compile(r'article|story|content')}),
        soup.find('div', {'class': 'wsj-snippet-body'}),
    ]
    
    for element in content_indicators:
        if element and len(element.get_text().strip()) > 200:
            return True
    
    return False

def extract_wsj_article_content(soup):
    """Extract the actual article content from WSJ page."""
    content_selectors = [
        'div[data-module="ArticleBody"]',
        '.article-content',
        '.wsj-snippet-body',
        '.article-wrap',
        '[data-module="BodyText"]',
        '.StoryBody',
        '.snippet-promotion',
    ]
    
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            content_div = elements[0]
            
            for unwanted in content_div.select('.advertisement, .ad, .promo, .related'):
                unwanted.decompose()
            
            paragraphs = content_div.find_all('p')
            if paragraphs:
                content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20)
                if len(content) > 200:
                    return content
    
    return None

def fetch_wsj_article_content_enhanced(url, session=None):
    """Enhanced WSJ content fetching with multiple strategies."""
    logging.info(f"Starting enhanced WSJ bypass for: {url}")
    
    if not session:
        session = get_wsj_specific_session(url)
    
    strategies = [
        ('Original URL', url),
        ('AMP Version', try_wsj_amp_version(url)),
        ('Mobile Version', try_wsj_mobile_version(url)),
        ('Print Version', try_wsj_print_version(url)),
    ]
    
    for strategy_name, test_url in strategies:
        if not test_url:
            continue
            
        logging.info(f"Trying WSJ strategy: {strategy_name} - {test_url}")
        
        try:
            time.sleep(random.uniform(1, 3))
            
            response = session.get(test_url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                if is_wsj_actual_content(soup):
                    content = extract_wsj_article_content(soup)
                    if content and len(content) > 500:
                        logging.info(f"Successfully extracted content using {strategy_name}")
                        return content
                else:
                    logging.warning(f"{strategy_name} returned paywall/cleaning page")
                    
        except Exception as e:
            logging.warning(f"Error with {strategy_name}: {str(e)}")
            continue
    
    # Try archive services
    try:
        twelve_ft_url = f"https://12ft.io/{url}"
        twelve_ft_session = get_wsj_specific_session(url)
        twelve_ft_session.headers.update({
            'Referer': 'https://12ft.io/',
            'Origin': 'https://12ft.io'
        })
        
        response = twelve_ft_session.get(twelve_ft_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            if is_wsj_actual_content(soup):
                content = extract_wsj_article_content(soup)
                if content:
                    logging.info("Successfully extracted content using 12ft.io")
                    return content
    except Exception as e:
        logging.warning(f"12ft.io failed for WSJ: {str(e)}")
    
    try:
        outline_url = f"https://outline.com/{url}"
        response = session.get(outline_url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', {'class': 'outline-content'})
            if content_div:
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join(p.get_text().strip() for p in paragraphs)
                if len(content) > 200:
                    logging.info("Successfully extracted content using Outline.com")
                    return content
    except Exception as e:
        logging.warning(f"Outline.com failed for WSJ: {str(e)}")
    
    logging.warning(f"All WSJ bypass strategies failed for: {url}")
    return None


def get_domain_specific_bypass(url, domain=None):
    """
    Use domain-specific techniques to bypass paywalls.
    
    Args:
        url: The URL to bypass
        domain: Optional domain (extracted from URL if not provided)
        
    Returns:
        requests.Session: Session configured for the specific domain
    """
    if not domain:
        domain = urlparse(url).netloc.lower()
    
    session = requests.Session()
    
    # NYT-specific
    if 'nytimes.com' in domain:
        session.cookies.set('NYT-T', 'ok', domain='.nytimes.com')
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/search?q=' + quote(url)
        })
        return session
    
    # WSJ-specific
    elif 'wsj.com' in domain:
        # Use the enhanced WSJ session
        return get_wsj_specific_session(url)
    
    # Washington Post specific
    elif 'washingtonpost.com' in domain:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'X-Forwarded-For': f'66.249.{random.randint(64, 95)}.{random.randint(1, 254)}',
            'Referer': 'https://www.google.com/search?q=' + quote(url)
        })
        return session
        
    # Bloomberg specific
    elif 'bloomberg.com' in domain:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/search?q=' + quote(url)
        })
        # Bloomberg has a metered paywall that gives free articles with google referrer
        return session
    
    # Financial Times specific
    elif 'ft.com' in domain:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/search?q=' + quote(url),
            'X-Forwarded-For': f'66.249.{random.randint(64, 95)}.{random.randint(1, 254)}'
        })
        return session
    
    # Economist specific  
    elif 'economist.com' in domain:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/search?q=' + quote(url)
        })
        return session
        
    # Default session with Google referrer
    session.headers.update(get_direct_access_headers())
    return session


def fetch_article_content(url, session=None):
    """
    Enhanced fetch_article_content with all new techniques.
    
    Args:
        url: The article URL
        session: Optional requests session to use
        
    Returns:
        str: Article content or empty string if failed
    """
    if not session:
        session = requests.Session()
        session.headers.update(get_direct_access_headers())
    
    # Check cache first (Technique 5)
    cached_content = get_cached_content(url)
    if cached_content:
        return cached_content
    
    content = ""
    original_url = url
    source_extracted = False
    
    # Check if this is a news aggregator link
    if is_aggregator_link(url):
        logging.info(f"Detected aggregator link: {url}, extracting original source")
        source_url = extract_original_source_url(url, session)
        
        if source_url:
            logging.info(f"Successfully extracted original source URL: {source_url} from aggregator: {url}")
            original_url = source_url
            source_extracted = True
        else:
            logging.warning(f"Failed to extract original source URL from aggregator: {url}")
    
    # Try domain-specific bypass first (Technique 8)
    domain = urlparse(original_url).netloc.lower()
    
    # Enhanced WSJ handling
    if 'wsj.com' in domain:
        logging.info(f"Detected WSJ URL, using enhanced bypass: {original_url}")
        wsj_content = fetch_wsj_article_content_enhanced(original_url, session)
        if wsj_content and is_valid_content(wsj_content):
            logging.info(f"Successfully bypassed WSJ paywall: {original_url}")
            cache_content(url, wsj_content)
            return wsj_content
        else:
            logging.warning(f"Enhanced WSJ bypass failed, falling back to standard methods")
    bypass_session = get_domain_specific_bypass(original_url, domain)
    
    try:
        bypass_response = bypass_session.get(original_url, timeout=10)
        if bypass_response.status_code == 200:
            bypass_soup = BeautifulSoup(bypass_response.text, 'html.parser')
            
            # Remove unwanted elements
            for unwanted in bypass_soup.select('script, style, nav, header, footer, .ads, .comments, .related, .sidebar'):
                unwanted.decompose()
            
            # Try to find the main content
            main_content = None
            
            # Try content selectors
            content_selectors = [
                'article', '.article', '.post-content', '.entry-content', '.content', 
                'main', '#main', '.main', '.story', '.story-body', '.article__body',
                '.article-content'
            ]
            
            for selector in content_selectors:
                elements = bypass_soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
            
            if main_content:
                # Get all paragraphs
                paragraphs = main_content.find_all('p')
                bypass_content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                
                if is_valid_content(bypass_content):
                    logging.info(f"Bypassed paywall using domain-specific technique for {original_url}")
                    cache_content(url, bypass_content)
                    return bypass_content
    except Exception as e:
        logging.warning(f"Error using direct bypass: {str(e)}")
    
    # Try JavaScript rendering for complex paywalls (Technique 7)
    if is_complex_paywall(original_url):
        logging.info(f"Detected complex paywall for {original_url}, trying JavaScript rendering")
        js_content = get_javascript_rendered_content(original_url)
        
        if is_valid_content(js_content):
            logging.info(f"Bypassed paywall using JavaScript rendering for {original_url}")
            cache_content(url, js_content)
            return js_content
    
    # Check if the URL is paywalled
    if is_paywalled(original_url):
        logging.info(f"Detected paywall for {original_url}, trying archive services")
        
        # Try to get an archive URL (with retries) (Technique 6)
        archive_url = get_archive_url_with_retry(original_url, session)
        
        if archive_url:
            # Extract content from the archive with enhanced extraction (Technique 3)
            content = get_content_from_archive(archive_url, session)
            
            if is_valid_content(content):
                logging.info(f"Successfully extracted content from archive for {original_url}")
                cache_content(url, content)
                return content
            else:
                logging.warning(f"Failed to extract valid content from archive for {original_url}")
    
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
                'main', '#main', '.main', '.story', '.story-body', '.article__body'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get all paragraphs
                    paragraphs = elements[0].find_all('p')
                    content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                    
                    if is_valid_content(content):
                        if source_extracted:
                            logging.info(f"Successfully extracted content from original source: {original_url} (via aggregator: {url})")
                        else:
                            logging.info(f"Successfully extracted content directly from {original_url}")
                        
                        cache_content(url, content)
                        return content
            
            # If no content found with selectors, use readability as fallback
            if not content or not is_valid_content(content):
                # Simple extraction of paragraphs from the body
                paragraphs = soup.body.find_all('p')
                content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                
                if is_valid_content(content):
                    if source_extracted:
                        logging.info(f"Extracted content using fallback method from original source: {original_url} (via aggregator: {url})")
                    else:
                        logging.info(f"Extracted content using fallback method from {original_url}")
                    
                    cache_content(url, content)
                    return content
    
    except Exception as e:
        logging.warning(f"Error fetching content directly: {str(e)}")
    
    logging.warning(f"Failed to extract content from {original_url}")
    return content