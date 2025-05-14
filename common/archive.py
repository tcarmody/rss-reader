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
    
    for complex_domain in COMPLEX_PAYWALL_DOMAINS:
        if complex_domain in domain:
            return True
            
    return False


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
    
    # Domain-specific optimizations
    if 'nytimes.com' in domain:
        # NYT works better with Archive.org
        return next((s for s in ARCHIVE_SERVICES if s['name'] == 'Archive.org'), ARCHIVE_SERVICES[0])
    elif 'wsj.com' in domain:
        # WSJ works better with 12ft.io
        return next((s for s in ARCHIVE_SERVICES if s['name'] == '12ft.io'), ARCHIVE_SERVICES[0])
    elif 'washingtonpost.com' in domain:
        # WaPo works better with Archive.is
        return next((s for s in ARCHIVE_SERVICES if s['name'] == 'Archive.is'), ARCHIVE_SERVICES[0])
    elif 'bloomberg.com' in domain:
        # Bloomberg works better with Outline.com
        return next((s for s in ARCHIVE_SERVICES if s['name'] == 'Outline.com'), ARCHIVE_SERVICES[0])
    elif 'ft.com' in domain:
        # Financial Times works better with 12ft.io
        return next((s for s in ARCHIVE_SERVICES if s['name'] == '12ft.io'), ARCHIVE_SERVICES[0])
    
    # Default to random selection with higher weight for more reliable services
    weights = {
        'Archive.is': 0.3,
        'Archive.org': 0.3,
        '12ft.io': 0.2,
        'Outline.com': 0.1,
        'Google Cache': 0.05,
        'Wayback Machine API': 0.05
    }
    
    services_with_weights = [(s, weights.get(s['name'], 0.1)) for s in ARCHIVE_SERVICES]
    total_weight = sum(weight for _, weight in services_with_weights)
    normalized_weights = [weight/total_weight for _, weight in services_with_weights]
    
    return random.choices(ARCHIVE_SERVICES, weights=normalized_weights, k=1)[0]


# Technique 6: Integrate Retry Logic
@retry_with_backoff(max_retries=3, initial_backoff=2.0, backoff_factor=2.0, retryable_exceptions=(ConnectionError, RateLimitError, requests.exceptions.RequestException))
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


def _find_existing_archive(url, service, session):
    """
    Find an existing archive for the URL.
    
    Args:
        url: The URL to find an archive for
        service: The archive service to use
        session: Requests session
        
    Returns:
        str: Archive URL or None if not found
    """
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
        
        elif service['name'] == 'Wayback Machine API':
            search_url = service['url'].format(url=quote(url, safe=''))
            response = session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data and len(data) > 1:
                        # Get the most recent capture
                        most_recent = data[-1]
                        if len(most_recent) >= 2:
                            timestamp = most_recent[1]
                            return f"https://web.archive.org/web/{timestamp}/{url}"
                except:
                    pass
    
    except Exception as e:
        logging.warning(f"Error finding existing archive: {str(e)}")
    
    return None


def _submit_to_archive(url, service, session):
    """
    Submit a URL to an archive service.
    
    Args:
        url: The URL to submit
        service: The archive service to use
        session: Requests session
        
    Returns:
        str: Archive URL or None if submission failed
    """
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
        
        elif service['name'] == 'Outline.com':
            submit_url = service['submit_url'].format(url=url)
            response = session.get(submit_url, timeout=30, allow_redirects=True)
            
            if response.status_code == 200:
                # Outline redirects to a new URL with the outline
                return response.url
    
    except Exception as e:
        logging.warning(f"Error submitting to archive: {str(e)}")
    
    return None


# Technique 3: Smarter Content Extraction
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
    # Check if we're in an async context
    try:
        import asyncio
        in_async_context = asyncio.get_event_loop().is_running()
    except (ImportError, RuntimeError):
        in_async_context = False
    
    # If we're in an async context, use the async version
    if in_async_context:
        try:
            import asyncio
            return asyncio.get_event_loop().run_until_complete(get_javascript_rendered_content_async(url))
        except Exception as e:
            logging.error(f"Error running async JavaScript rendering: {str(e)}")
            return None
    
    # Otherwise, use the sync version
    try:
        # Try to import playwright
        try:
            from playwright.sync_api import sync_playwright
            has_playwright = True
        except ImportError:
            logging.warning("Playwright not installed. Cannot render JavaScript.")
            has_playwright = False
            
            # Try selenium as fallback
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                has_selenium = True
            except ImportError:
                logging.warning("Neither Playwright nor Selenium is installed. Cannot render JavaScript.")
                return None
        
        # Use Playwright if available
        if has_playwright:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set Google referrer to help bypass some paywalls
                page.set_extra_http_headers({
                    'Referer': 'https://www.google.com/',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                })
                
                try:
                    page.goto(url, wait_until='networkidle', timeout=30000)
                    
                    # Wait for content to load
                    page.wait_for_selector('article, .article, .content, p', timeout=10000)
                    
                    # Sleep to allow JS to fully execute
                    time.sleep(2)
                    
                    # Extract content
                    content = page.content()
                    
                    browser.close()
                    
                    # Process with BeautifulSoup
                    return _process_html_content(content)
                except Exception as page_error:
                    logging.error(f"Error in Playwright rendering: {str(page_error)}")
                    browser.close()
        
        # Use Selenium as fallback
        elif has_selenium:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            chrome_options.add_argument("--referer=https://www.google.com/")
            
            driver = webdriver.Chrome(options=chrome_options)
            try:
                driver.get(url)
                
                # Wait for content to load
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "article"))
                    )
                except:
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "p"))
                        )
                    except:
                        pass
                
                # Sleep to allow JS to fully execute
                time.sleep(2)
                
                # Get the page content
                html_content = driver.page_source
                
                # Process with BeautifulSoup
                return _process_html_content(html_content)
            finally:
                driver.quit()
    
    except Exception as e:
        logging.error(f"Error rendering JavaScript content: {str(e)}")
    
    return None


async def get_javascript_rendered_content_async(url, session=None):
    """
    Async version of get_javascript_rendered_content that uses Playwright's async API.
    
    Args:
        url: The URL to render
        session: Optional requests session (not used for JS rendering)
        
    Returns:
        str: Extracted content or None if failed
    """
    try:
        # Try to import playwright async API
        try:
            from playwright.async_api import async_playwright
            has_playwright = True
        except ImportError:
            logging.warning("Playwright not installed. Cannot render JavaScript.")
            has_playwright = False
            return None
        
        # Use Playwright async API
        if has_playwright:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set Google referrer to help bypass some paywalls
                await page.set_extra_http_headers({
                    'Referer': 'https://www.google.com/',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                })
                
                try:
                    await page.goto(url, wait_until='networkidle', timeout=30000)
                    
                    # Wait for content to load
                    await page.wait_for_selector('article, .article, .content, p', timeout=10000)
                    
                    # Sleep to allow JS to fully execute
                    await asyncio.sleep(2)
                    
                    # Extract content
                    content = await page.content()
                    
                    await browser.close()
                    
                    # Process with BeautifulSoup
                    return _process_html_content(content)
                except Exception as page_error:
                    logging.error(f"Error in async Playwright rendering: {str(page_error)}")
                    await browser.close()
    
    except Exception as e:
        logging.error(f"Error in async JavaScript rendering: {str(e)}")
    
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


# Technique 8: Domain-Specific Bypass
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
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/search?q=' + quote(url)
        })
        return session
    
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