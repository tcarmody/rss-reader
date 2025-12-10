"""Main RSS Reader class for fetching and processing feeds with enhanced clustering and aggregator handling."""

import feedparser
import os
import time
import logging
import traceback
import re
import requests
from urllib.parse import urlparse, parse_qs, unquote

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple

from common.config import get_env_var
from common.http import create_http_session
from common.performance import track_performance
# Updated imports to use new content modules
from content.archive.paywall import is_paywalled
from content.archive.providers import default_provider_manager
from content.extractors.aggregator import is_aggregator_link, extract_source_url
from common.batch_processing import BatchProcessor
from summarization.article_summarizer import ArticleSummarizer
from clustering.base import ArticleClusterer

# Import BeautifulSoup if available
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available for advanced aggregator extraction")

# Import the simple clusterer for lightweight clustering
try:
    from clustering.simple import SimpleClustering
    SIMPLE_CLUSTERING_AVAILABLE = True
except ImportError:
    logging.warning("Simple clustering module not available. Using basic clustering.")
    SIMPLE_CLUSTERING_AVAILABLE = False

# Import the enhanced clusterer as fallback
try:
    from clustering.enhanced import create_enhanced_clusterer
    ENHANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    logging.warning("Enhanced clustering module not available.")
    ENHANCED_CLUSTERING_AVAILABLE = False

# Helper function to maintain compatibility with old fetch_article_content
def fetch_article_content(url, session=None):
    """
    Fetch article content using the new archive system.

    Args:
        url: The article URL to fetch
        session: Optional requests session

    Returns:
        str: Article content or empty string if failed
    """
    # Sites known to require JavaScript rendering
    js_rendering_sites = [
        'openai.com/blog',
        'openai.com/index',
        'openai.com/news',
        'notion.so',
        'medium.com/@',  # Some Medium posts
    ]

    try:
        # Check if it's paywalled
        if is_paywalled(url):
            logging.info(f"Detected paywall for {url}, trying archive services")

            # Try archive services
            result = default_provider_manager.get_archived_content(url)
            if result.success and result.content and len(result.content) > 200:
                return result.content

        # Check if site needs JavaScript rendering
        needs_js_rendering = any(js_site in url for js_site in js_rendering_sites)

        if needs_js_rendering:
            logging.info(f"Site requires JavaScript rendering: {url}")
            content = fetch_js_rendered_content(url)
            if content and len(content) > 200:
                return content

        # If not paywalled or archive failed, try direct access
        if not session:
            session = create_http_session()

        response = session.get(url, timeout=15)
        if response.status_code == 200:
            if BS4_AVAILABLE:
                from bs4 import BeautifulSoup
                # Detect content type and use appropriate parser
                content_type = response.headers.get('content-type', '').lower()
                parser = 'xml' if 'xml' in content_type or url.endswith('.xml') else 'html.parser'
                soup = BeautifulSoup(response.text, parser)

                # Remove unwanted elements
                for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments'):
                    unwanted.decompose()

                # Try to find main content
                for selector in ['article', '.article', '.content', '.post-content', 'main']:
                    elements = soup.select(selector)
                    if elements:
                        paragraphs = elements[0].find_all('p')
                        content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                        if content and len(content) > 200:
                            return content

                # If direct fetch failed to get good content, try JS rendering as fallback
                if soup.body:
                    body_text = soup.body.get_text()
                    # If body text is too short, might be JS-rendered - try Playwright
                    if len(body_text.strip()) < 500:
                        logging.info(f"Insufficient content from direct fetch, trying JS rendering for {url}")
                        js_content = fetch_js_rendered_content(url)
                        if js_content and len(js_content) > 200:
                            return js_content
                    return body_text
            else:
                # Basic text extraction without BeautifulSoup
                return response.text

    except Exception as e:
        logging.warning(f"Error fetching article content: {e}")

    return ""

def fetch_js_rendered_content(url):
    """
    Fetch content from JavaScript-rendered pages using Playwright.

    Args:
        url: The URL to fetch

    Returns:
        str: Rendered content or empty string if failed
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate and wait for network to be idle
            page.goto(url, wait_until='networkidle', timeout=30000)

            # Wait a bit for any dynamic content
            page.wait_for_timeout(2000)

            # Get the rendered HTML
            html = page.content()
            browser.close()

            # Parse the rendered HTML
            if BS4_AVAILABLE:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')

                # Remove unwanted elements
                for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .sidebar'):
                    unwanted.decompose()

                # Try to find main content
                for selector in ['article', '.article', '.content', '.post-content', '.entry-content', 'main']:
                    elements = soup.select(selector)
                    if elements:
                        paragraphs = elements[0].find_all('p')
                        content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                        if content and len(content) > 200:
                            return content

                # Fallback to body text
                if soup.body:
                    paragraphs = soup.body.find_all('p')
                    content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                    if content:
                        return content

    except ImportError:
        logging.warning("Playwright not available - install with: pip install playwright && playwright install chromium")
    except Exception as e:
        logging.warning(f"Error fetching JS-rendered content: {e}")

    return ""

# Helper function for source URL extraction
def extract_original_source_url(url, session=None):
    """
    Extract original source URL from aggregator link.
    
    Args:
        url: The aggregator URL
        session: Optional requests session
        
    Returns:
        str: Original source URL or None if extraction failed
    """
    result = extract_source_url(url, session)
    if result.success and result.extracted_url:
        return result.extracted_url
    return None


class AggregatorSourceExtractor:
    """
    Integrated source extractor for handling news aggregator links.
    """
    
    AGGREGATOR_RULES = {
        'techmeme.com': {
            'name': 'Techmeme',
            'patterns': [r'techmeme\.com'],
            'extraction_method': 'techmeme'
        },
        'news.google.com': {
            'name': 'Google News',
            'patterns': [r'news\.google\.com'],
            'extraction_method': 'google_news'
        },
        'reddit.com': {
            'name': 'Reddit',
            'patterns': [r'reddit\.com'],
            'extraction_method': 'reddit'
        },
        'news.ycombinator.com': {
            'name': 'Hacker News',
            'patterns': [r'news\.ycombinator\.com'],
            'extraction_method': 'hackernews'
        }
    }
    
    def __init__(self, session=None):
        self.session = session
    
    def detect_aggregator(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if URL is from a known aggregator.
        
        Returns:
            Tuple of (is_aggregator, aggregator_type, aggregator_name)
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        for agg_type, rules in self.AGGREGATOR_RULES.items():
            for pattern in rules['patterns']:
                if re.search(pattern, domain):
                    return True, agg_type, rules['name']
        
        return False, None, None
    
    def extract_source(self, url: str) -> Dict[str, any]:
        """
        Extract original source information from an aggregator URL.
        """
        is_agg, agg_type, agg_name = self.detect_aggregator(url)
        
        if not is_agg:
            return {
                'original_url': url,
                'source_name': self._extract_domain_name(url),
                'aggregator_name': None,
                'aggregator_url': None,
                'extraction_method': 'direct',
                'confidence': 1.0
            }
        
        # Try basic extraction first
        original_url = extract_original_source_url(url, self.session)
        
        # If basic extraction worked and URL changed, we're done
        if original_url != url:
            return {
                'original_url': original_url,
                'source_name': self._extract_domain_name(original_url),
                'aggregator_name': agg_name,
                'aggregator_url': url,
                'extraction_method': agg_type,
                'confidence': 0.9
            }
        
        # Try specific extraction methods
        if agg_type == 'techmeme':
            result = self._extract_techmeme_advanced(url)
        elif agg_type == 'google_news':
            result = self._extract_google_news_advanced(url)
        else:
            result = None
        
        if result:
            return result
        
        # Fallback
        return {
            'original_url': url,
            'source_name': agg_name,
            'aggregator_name': agg_name,
            'aggregator_url': url,
            'extraction_method': 'fallback',
            'confidence': 0.0
        }
    
    def _extract_techmeme_advanced(self, url: str) -> Optional[Dict[str, any]]:
        """Advanced Techmeme extraction with HTML parsing."""
        if not BS4_AVAILABLE or not self.session:
            return None
            
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for the main story link
                main_story = soup.find('strong', class_='L1')
                if main_story and main_story.find('a'):
                    link = main_story.find('a')
                    original_url = link.get('href', '')
                    source_name = link.text.strip()
                    
                    if original_url and 'techmeme.com' not in original_url:
                        return {
                            'original_url': original_url,
                            'source_name': source_name or self._extract_domain_name(original_url),
                            'aggregator_name': 'Techmeme',
                            'aggregator_url': url,
                            'extraction_method': 'techmeme_html',
                            'confidence': 0.95
                        }
        except Exception as e:
            logging.debug(f"Advanced Techmeme extraction failed: {str(e)}")
        
        return None
    
    def _extract_google_news_advanced(self, url: str) -> Optional[Dict[str, any]]:
        """Advanced Google News extraction."""
        # Try to get redirect URL
        if self.session:
            try:
                response = self.session.get(url, timeout=10, allow_redirects=False)
                if response.status_code in [301, 302, 303, 307, 308]:
                    redirect_url = response.headers.get('Location', '')
                    if redirect_url and 'google.com' not in redirect_url:
                        return {
                            'original_url': redirect_url,
                            'source_name': self._extract_domain_name(redirect_url),
                            'aggregator_name': 'Google News',
                            'aggregator_url': url,
                            'extraction_method': 'google_redirect',
                            'confidence': 0.95
                        }
            except Exception as e:
                logging.debug(f"Advanced Google News extraction failed: {str(e)}")
        
        return None
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract a readable domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            domain = re.sub(r'^www\.', '', domain)
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                return domain_parts[-2].title()
            return domain
        except:
            return "Unknown Source"


class RSSReader:
    """
    Main class that handles RSS feed processing, article summarization, clustering,
    and aggregator source extraction.

    This enhanced version includes:
    - Automatic detection and extraction of original sources from aggregator links
    - Fetching content from original sources for better summaries
    - Clear attribution showing both original source and aggregator
    - Statistics tracking for aggregator processing
    """

    def __init__(self, feeds=None, batch_size=25, batch_delay=15, per_feed_limit=25):
        """
        Initialize RSSReader with feeds and settings.
        
        Args:
            feeds: List of RSS feed URLs (None for default feeds, [] for no feeds)
            batch_size: Number of feeds to process in a batch
            batch_delay: Delay between batches in seconds
            per_feed_limit: Maximum number of articles to process per feed
        """
        # Explicitly handle None vs empty list
        if feeds is None:
            logging.info("No feeds provided, loading defaults")
            self.feeds = self._load_default_feeds()
        else:
            logging.info(f"Using {len(feeds)} provided feeds")
            self.feeds = feeds
            
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.per_feed_limit = per_feed_limit
        self.session = create_http_session()
        self.batch_processor = BatchProcessor(max_workers=5)
        self.summarizer = ArticleSummarizer()
        
        # Initialize aggregator extractor
        self.aggregator_extractor = AggregatorSourceExtractor(self.session)
        
        # Track aggregator statistics
        self.aggregator_stats = {
            'total_aggregator_links': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'by_aggregator': {}
        }
        
        # Use clustering based on user preferences
        use_simple = os.environ.get('USE_SIMPLE_CLUSTERING', 'true').lower() == 'true'
        use_enhanced = os.environ.get('USE_ENHANCED_CLUSTERING', 'false').lower() == 'true'
        
        if use_simple and SIMPLE_CLUSTERING_AVAILABLE:
            self.clusterer = SimpleClustering()
            logging.info("Using simple clustering for lightweight article grouping")
        elif use_enhanced and ENHANCED_CLUSTERING_AVAILABLE:
            self.clusterer = create_enhanced_clusterer(summarizer=self.summarizer)
            logging.info("Using enhanced clustering for better article grouping")
        elif SIMPLE_CLUSTERING_AVAILABLE:
            # Fallback to simple clustering if enhanced not available
            self.clusterer = SimpleClustering()
            logging.info("Falling back to simple clustering for article grouping")
        else:
            self.clusterer = ArticleClusterer()
            logging.info("Using basic clustering for article grouping")
        
        self.last_processed_clusters = []  # Store the last processed clusters for web server access

    def _load_default_feeds(self):
        """
        Load feed URLs from the default file.
        
        Returns:
            list: List of feed URLs
        """
        feeds = []
        try:
            # Check for the file in the package directory or in the current directory
            file_paths = [
                os.path.join(os.path.dirname(__file__), 'rss_feeds.txt'),
                'rss_feeds.txt'
            ]
            
            for path in file_paths:
                if os.path.exists(path):
                    logging.info(f"Loading default feeds from {path}")
                    with open(path, 'r') as f:
                        for line in f:
                            url = line.strip()
                            if url and not url.startswith('#'):
                                # Remove any inline comments and clean the URL
                                url = url.split('#')[0].strip()
                                url = ''.join(c for c in url if ord(c) >= 32)  # Remove control characters
                                if url:
                                    feeds.append(url)
                    logging.info(f"Loaded {len(feeds)} default feeds")
                    return feeds
            
            logging.error("No rss_feeds.txt file found")
            return []
        except Exception as e:
            logging.error(f"Error loading feed URLs: {str(e)}")
            return []

    def _parse_date(self, date_str):
        """
        Parse date string with multiple formats.

        Args:
            date_str: Date string to parse

        Returns:
            datetime object
        """
        if not date_str:
            return datetime.now()

        try:
            # First, try feedparser's date parsing
            parsed_date = feedparser._parse_date(date_str)
            if parsed_date:
                return datetime(*parsed_date[:6])
        except:
            pass

        # Handle ISO 8601 dates with colons in timezone (e.g., -07:00 -> -0700)
        # This is needed because strptime %z doesn't handle colons before Python 3.11
        date_str_normalized = date_str
        if date_str and len(date_str) > 6:
            # Check for timezone with colon pattern: +HH:MM or -HH:MM at the end
            import re
            date_str_normalized = re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', date_str)

        # Try common date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 822
            '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 with timezone name
            '%Y-%m-%dT%H:%M:%S.%f%z',    # ISO 8601 with milliseconds and timezone
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601 with timezone
            '%Y-%m-%dT%H:%M:%S.%fZ',     # ISO 8601 with milliseconds and Z
            '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 with Z
            '%Y-%m-%d %H:%M:%S',         # Common format
            '%Y-%m-%d',                  # Date only
            '%d %b %Y %H:%M:%S %z',      # Another common format
            '%d %b %Y %H:%M:%S %Z',      # Another common format
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str_normalized, fmt)
            except ValueError:
                continue

        # If all parsing attempts fail, return current time
        logging.warning(f"Could not parse date: {date_str}")
        return datetime.now()

    def _filter_articles_by_date(self, articles, hours):
        """
        Filter articles to include only those from the specified time range.

        Args:
            articles: List of articles to filter (each should be a dict)
            hours: Number of hours in the past to include

        Returns:
            List of filtered articles
        """
        if not articles or not isinstance(articles, list) or hours <= 0:
            return articles

        # Get current time as timezone-aware (UTC)
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered_articles = []

        for article in articles:
            # Skip if article is not a dictionary
            if not isinstance(article, dict):
                logging.debug(f"Skipping non-dict article: {article}")
                continue

            try:
                # Safely get published date with fallback to empty string
                published_date = article.get('published', '') if hasattr(article, 'get') else ''
                article_date = self._parse_date(published_date)

                # Convert to UTC for comparison if timezone-aware
                if article_date.tzinfo is not None:
                    article_date_utc = article_date.astimezone(timezone.utc)
                else:
                    # If naive, assume UTC
                    article_date_utc = article_date.replace(tzinfo=timezone.utc)

                if article_date_utc >= cutoff_date:
                    filtered_articles.append(article)
                else:
                    # Log filtered out articles for debugging
                    title = article.get('title', 'Untitled')
                    logging.debug(f"Filtered out old article '{title}' (published: {article_date_utc.isoformat()}, cutoff: {cutoff_date.isoformat()})")
            except Exception as e:
                # Log the error and include the article anyway
                title = article.get('title', 'Untitled')
                logging.debug(f"Could not parse date for article '{title}': {str(e)}. Including it anyway.")
                filtered_articles.append(article)

        return filtered_articles

    def _parse_entry(self, entry, feed_title):
        """
        Parse a feed entry into an article dictionary with aggregator detection.
        
        Args:
            entry: feedparser entry object
            feed_title: Title of the feed
            
        Returns:
            dict: Parsed article data or None if parsing failed
        """
        try:
            # Extract basic article information
            article = {
                'title': getattr(entry, 'title', 'No Title'),
                'link': getattr(entry, 'link', '#'),
                'published': getattr(entry, 'published', 'Unknown date'),
                'feed_source': feed_title,
                'content': ''  # Will be filled later
            }
            
            # Detect and extract aggregator source information
            source_info = self.aggregator_extractor.extract_source(article['link'])
            
            # Add source information to article
            article['original_url'] = source_info['original_url']
            article['source_name'] = source_info['source_name']
            article['is_aggregator'] = source_info['aggregator_name'] is not None
            article['aggregator_name'] = source_info['aggregator_name']
            article['aggregator_url'] = source_info['aggregator_url']
            article['source_extraction_confidence'] = source_info['confidence']
            
            # Update statistics if it's an aggregator
            if article['is_aggregator']:
                self.aggregator_stats['total_aggregator_links'] += 1
                agg_name = article.get('aggregator_name', 'Unknown')
                self.aggregator_stats['by_aggregator'][agg_name] = \
                    self.aggregator_stats['by_aggregator'].get(agg_name, 0) + 1
                
                if source_info['confidence'] > 0:
                    self.aggregator_stats['successful_extractions'] += 1
                    logging.info(f"Extracted source from {agg_name}: {article['original_url']}")
                else:
                    self.aggregator_stats['failed_extractions'] += 1
                    logging.warning(f"Failed to extract source from {agg_name}: {article['link']}")
            
            # Extract content - prefer original URL if available
            content = self._extract_content_from_entry(entry, article)
            article['content'] = content

            return article

        except Exception as e:
            logging.error(f"Error parsing entry: {str(e)}")
            return None
            
    def _extract_content_from_entry(self, entry, article_info):
        """
        Extract and clean content from a feed entry, fetching from original sources when needed.
        
        Args:
            entry: feedparser entry object
            article_info: Parsed article dictionary with source information
            
        Returns:
            str: Cleaned content text
        """
        content = ''
        # First try to get content from the feed entry
        if hasattr(entry, 'content'):
            raw_content = entry.content
            if isinstance(raw_content, list) and raw_content:
                content = raw_content[0].get('value', '')
            elif isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, (list, tuple)):
                content = ' '.join(str(item) for item in raw_content)
            else:
                content = str(raw_content)

        # Fallback to summary
        if not content and hasattr(entry, 'summary'):
            content = entry.summary

        # Final fallback to title
        if not content:
            content = getattr(entry, 'title', '')
            logging.warning("Using title as content fallback")

        # Clean content
        content = content.strip()
        
        # For aggregator links or short content, fetch from original source
        fetch_url = article_info.get('original_url', article_info['link'])
        is_aggregator = article_info.get('is_aggregator', False)
        paywall_bypass_enabled = get_env_var('ENABLE_PAYWALL_BYPASS', 'false').lower() == 'true'
        is_short_content = len(content) < 1000
        
        if is_aggregator or (paywall_bypass_enabled and (is_short_content or is_paywalled(fetch_url))):
            if is_aggregator:
                logging.info(f"Fetching full content from original source: {fetch_url}")
            else:
                logging.info(f"Attempting to fetch full content for: {fetch_url}")
            
            try:
                full_content = fetch_article_content(fetch_url, self.session)
                
                if full_content and len(full_content) > len(content):
                    logging.info(f"Successfully retrieved full content from: {fetch_url}")
                    return full_content
            except Exception as e:
                logging.warning(f"Error fetching full content: {str(e)}")
        
        return content

    @track_performance
    def process_cluster_summaries(self, clusters):
        """
        Process and generate summaries for article clusters with aggregator awareness.
        
        Args:
            clusters: List of article clusters
            
        Returns:
            list: Processed clusters with summaries
        """
        processed_clusters = []
        for i, cluster in enumerate(clusters, 1):
            try:
                if not cluster:
                    logging.warning(f"Empty cluster {i}, skipping")
                    continue

                logging.info(f"Processing cluster {i}/{len(clusters)} with {len(cluster)} articles")
                
                # Log aggregator information for this cluster
                # Add type check to handle cases where cluster contains strings instead of dicts
                aggregator_articles = [a for a in cluster if isinstance(a, dict) and a.get('is_aggregator')]
                if aggregator_articles:
                    logging.info(f"Cluster contains {len(aggregator_articles)} aggregator links:")
                    for article in aggregator_articles:
                        logging.info(f"  - {article.get('aggregator_name')}: {article['title']} -> {article.get('source_name')}")
                
                # First, try to fetch full content for articles that don't have it
                # Filter out any non-dict items in cluster
                valid_articles = [a for a in cluster if isinstance(a, dict)]
                if len(valid_articles) != len(cluster):
                    logging.warning(f"Cluster {i} contains {len(cluster) - len(valid_articles)} non-dict items, filtering them out")
                
                for article in valid_articles:
                    if not article.get('content') or len(article.get('content', '')) < 200:
                        # Use original URL if available
                        fetch_url = article.get('original_url', article['link'])
                        
                        try:
                            logging.info(f"Fetching full content for article: {article['title']}")
                            full_content = fetch_article_content(fetch_url, self.session)
                            if full_content and len(full_content) > 200:
                                article['content'] = full_content
                                logging.info(f"Successfully fetched full content from: {fetch_url}")
                            else:
                                logging.warning(f"Could not fetch meaningful content for {article['title']}")
                        except Exception as e:
                            logging.warning(f"Error fetching content for {article['title']}: {str(e)}")

                if len(valid_articles) > 1:
                    # For clusters with multiple articles, generate a combined summary
                    # Use the article with the most content for summarization
                    best_article = max(valid_articles, key=lambda a: len(a.get('content', '')))
                    
                    # If the best article still has limited content, combine all articles
                    if len(best_article.get('content', '')) < 500:
                        combined_text = "\n\n".join([
                            f"Title: {article['title']}\n{article.get('content', '')}"
                            for article in valid_articles
                        ])
                    else:
                        # Use the best article for summarization
                        combined_text = f"Title: {best_article['title']}\n{best_article.get('content', '')}"

                    logging.info(f"Generating summary for cluster {i} with {len(valid_articles)} articles")
                    cluster_summary = self._generate_summary(
                        combined_text,
                        f"Combined summary of {len(valid_articles)} related articles",
                        valid_articles[0].get('original_url', valid_articles[0]['link'])
                    )

                    # Add the cluster summary to each article
                    for article in valid_articles:
                        article['summary'] = cluster_summary
                        article['cluster_size'] = len(valid_articles)
                else:
                    # Single article
                    if not valid_articles:
                        logging.warning(f"Cluster {i} has no valid articles, skipping")
                        continue
                    article = valid_articles[0]
                    if not article.get('summary'):
                        logging.info(f"Generating summary for single article: {article['title']}")
                        article['summary'] = self._generate_summary(
                            article.get('content', ''),
                            article['title'],
                            article.get('original_url', article['link'])
                        )
                    article['cluster_size'] = 1
                    
                    # Ensure summary has the correct structure
                    if not isinstance(article['summary'], dict):
                        article['summary'] = {
                            'headline': article['title'],
                            'summary': str(article['summary'])
                        }
                    elif 'headline' not in article['summary'] or 'summary' not in article['summary']:
                        # Fix incomplete summary structure
                        summary_text = article['summary'].get('summary', '')
                        if not summary_text and isinstance(article['summary'], str):
                            summary_text = article['summary']
                        
                        article['summary'] = {
                            'headline': article['summary'].get('headline', article['title']),
                            'summary': summary_text or 'No summary available.'
                        }

                processed_clusters.append(valid_articles)
                logging.info(f"Successfully processed cluster {i}")

            except Exception as cluster_error:
                logging.error(f"Error processing cluster {i}: {str(cluster_error)}", exc_info=True)
                continue

        # Log aggregator statistics
        self._log_aggregator_stats()

        return processed_clusters

    def _log_aggregator_stats(self):
        """Log statistics about aggregator link processing."""
        stats = self.aggregator_stats
        
        if stats['total_aggregator_links'] == 0:
            return
        
        logging.info("=" * 60)
        logging.info("AGGREGATOR LINK STATISTICS:")
        logging.info(f"Total aggregator links found: {stats['total_aggregator_links']}")
        logging.info(f"Successful source extractions: {stats['successful_extractions']}")
        logging.info(f"Failed extractions: {stats['failed_extractions']}")
        
        success_rate = (stats['successful_extractions'] / stats['total_aggregator_links']) * 100
        logging.info(f"Success rate: {success_rate:.1f}%")
        
        if stats['by_aggregator']:
            logging.info("\nBreakdown by aggregator:")
            for agg, count in sorted(stats['by_aggregator'].items()):
                logging.info(f"  - {agg}: {count} links")
        logging.info("=" * 60)

    def _generate_summary(self, article_text, title, url):
        """
        Generate a summary for an article using the ArticleSummarizer.
        
        Args:
            article_text: Text of the article to summarize
            title: Title of the article
            url: URL of the article (original source URL)
            
        Returns:
            dict: Summary with headline and content
        """
        try:
            # Handle None or empty parameters
            if not article_text:
                article_text = ""
            
            if not title:
                title = "Untitled Article"
            
            if not url:
                url = "#"
            
            summary = self.summarizer.summarize_article(article_text, title, url)
            
            # Ensure the summary has the correct structure
            if not isinstance(summary, dict):
                summary = {
                    'headline': title,
                    'summary': str(summary)
                }
            elif 'headline' not in summary or 'summary' not in summary:
                # Fix incomplete summary structure
                summary_text = summary.get('summary', '')
                if not summary_text and isinstance(summary, str):
                    summary_text = summary
                    
                summary = {
                    'headline': summary.get('headline', title),
                    'summary': summary_text or 'No summary available.'
                }
                
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {
                'headline': title or "Error",
                'summary': "Summary generation failed. Please try again later."
            }

    @track_performance
    async def process_feeds(self):
        """
        Process all RSS feeds and generate summaries.
        
        This is the main method that orchestrates the full process:
        1. Fetch and parse feeds
        2. Extract original sources from aggregator links
        3. Cluster articles using enhanced clustering
        4. Generate summaries from original content
        5. Create HTML output with proper source attribution
        
        Returns:
            str: Path to the generated HTML file or None if processing failed
        """
        try:
            all_articles = []

            # Process feeds in batches
            for batch in self._get_feed_batches():
                logging.info(f"\n🔄 Processing Batch {batch['current']}/{batch['total']}: "
                           f"Feeds {batch['start']} to {batch['end']}")

                # Process each feed in the batch in parallel
                with ThreadPoolExecutor(max_workers=min(len(batch['feeds']), 10)) as executor:
                    futures = [executor.submit(self._process_feed, feed) for feed in batch['feeds']]
                    batch_articles = []
                    for future in as_completed(futures):
                        articles = future.result()
                        if articles:
                            batch_articles.extend(articles)
                            logging.info(f"Added {len(articles)} articles to batch")

                all_articles.extend(batch_articles)
                logging.info(f"Batch complete. Total articles so far: {len(all_articles)}")

                # Add delay between batches if there are more
                if batch['current'] < batch['total']:
                    time.sleep(self.batch_delay)

            logging.info(f"Total articles collected: {len(all_articles)}")

            # Apply time range filtering if enabled
            time_range_hours = int(os.environ.get('TIME_RANGE_HOURS', '0'))
            if time_range_hours > 0:
                filtered_articles = self._filter_articles_by_date(all_articles, time_range_hours)
                logging.info(f"Filtered articles from {len(all_articles)} to {len(filtered_articles)} using {time_range_hours} hour time range")
                all_articles = filtered_articles
                
                if not all_articles:
                    logging.warning("No articles remaining after time range filtering")
                    return None

            if not all_articles:
                logging.error("No articles collected from any feeds")
                return None

            # Cluster the articles using the improved clusterer
            logging.info("Clustering similar articles...")
            
            # Use a different clustering method depending on which clusterer we have
            if SIMPLE_CLUSTERING_AVAILABLE and hasattr(self.clusterer, 'cluster_with_topics'):
                # Use the simple clustering with topics
                cluster_results = self.clusterer.cluster_with_topics(all_articles)
                # Extract articles from cluster objects - cluster_with_topics returns dicts with 'articles' key
                clusters = [cluster_result['articles'] for cluster_result in cluster_results]
                logging.info(f"Created {len(clusters)} clusters with simple clustering")
            elif ENHANCED_CLUSTERING_AVAILABLE and hasattr(self.clusterer, 'cluster_with_summaries'):
                # Use the enhanced clustering with summaries if available
                clusters = self.clusterer.cluster_with_summaries(all_articles)
                logging.info(f"Created {len(clusters)} clusters with enhanced clustering")
            else:
                # Fall back to basic clustering
                clusters = self.clusterer.cluster_articles(all_articles)
                logging.info(f"Created {len(clusters)} clusters with basic clustering")

            if not clusters:
                logging.error("No clusters created")
                return None

            # Generate summaries for each cluster
            logging.info("Generating summaries for article clusters...")
            processed_clusters = self.process_cluster_summaries(clusters)

            if not processed_clusters:
                logging.error("No clusters were successfully processed")
                return None

            logging.info(f"Successfully processed {len(processed_clusters)} clusters")

            # Store the processed clusters for web server access
            self.last_processed_clusters = processed_clusters
            
            # Generate HTML output
            output_file = self.generate_html_output(processed_clusters)
            if output_file:
                logging.info(f"Successfully generated HTML output: {output_file}")
            else:
                logging.error("Failed to generate HTML output")

            return output_file

        except Exception as e:
            logging.error(f"Error processing feeds: {str(e)}", exc_info=True)
            return None

    @track_performance
    def _process_feed(self, feed_url):
        """
        Process a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            list: Processed articles from the feed
        """
        try:
            logging.info(f"Processing feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            articles = []

            if feed.entries:
                feed_title = feed.feed.get('title', feed_url)
                logging.info(f"📰 Found {len(feed.entries)} articles in feed: {feed_url}")

                for entry in feed.entries[:self.per_feed_limit]:
                    article = self._parse_entry(entry, feed_title)
                    if article:
                        articles.append(article)

            logging.info(f"Successfully processed {len(articles)} articles from {feed_url}")
            return articles

        except Exception as e:
            logging.error(f"Error processing feed {feed_url}: {str(e)}")
            return []

    def _get_feed_batches(self):
        """
        Generate batches of feeds to process.
        
        Yields:
            dict: Batch information containing feeds and metadata
        """
        logging.info("🚀 Initializing RSS Reader...")
        logging.info(f"📊 Total Feeds: {len(self.feeds)}")
        logging.info(f"📦 Batch Size: {self.batch_size}")
        logging.info(f"⏱️  Batch Delay: {self.batch_delay} seconds")

        total_batches = (len(self.feeds) + self.batch_size - 1) // self.batch_size
        logging.info(f"🔄 Total Batches: {total_batches}")

        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(self.feeds))
            yield {
                'current': batch_num + 1,
                'total': total_batches,
                'start': start_idx + 1,
                'end': end_idx,
                'feeds': self.feeds[start_idx:end_idx]
            }
            
    @track_performance
    def generate_html_output(self, clusters):
        """
        Generate HTML output from the processed clusters with aggregator awareness.
        
        Args:
            clusters: List of article clusters with summaries
            
        Returns:
            str: Path to generated HTML file or None if generation failed
        """
        try:
            # Create output directory if it doesn't exist (using absolute path)
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')
            
            # Check if the template file exists
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
            template_file = os.path.join(template_dir, 'feed-summary.html')
            
            if not os.path.exists(template_file):
                logging.error(f"Template file not found: {template_file}")
                # Create a aggregator-aware fallback template if the template is missing
                return self._generate_aggregator_aware_html(clusters, output_file)
            
            try:
                # Use Jinja2 directly instead of Flask
                from jinja2 import Environment, FileSystemLoader, select_autoescape
                
                # Create Jinja2 environment
                env = Environment(
                    loader=FileSystemLoader(template_dir),
                    autoescape=select_autoescape(['html', 'xml'])
                )
                
                # Add custom filters for aggregator display
                env.filters['aggregator_info'] = self._aggregator_info_filter
                
                # Load the template
                template = env.get_template('feed-summary.html')
                
                # Render the template with aggregator stats
                html_content = template.render(
                    clusters=clusters,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    aggregator_stats=self.aggregator_stats
                )
                
                # Write to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logging.info(f"Successfully wrote HTML output to {output_file}")
                return output_file
                
            except Exception as render_error:
                logging.error(f"Error rendering template: {str(render_error)}")
                # Try fallback method if template rendering fails
                return self._generate_aggregator_aware_html(clusters, output_file)
                
        except Exception as e:
            logging.error(f"Error generating HTML output: {str(e)}", exc_info=True)
            return None
            
    def _aggregator_info_filter(self, article):
        """Jinja2 filter to format aggregator information."""
        if article.get('is_aggregator') and article.get('aggregator_name'):
            return f' (via <a href="{article["aggregator_url"]}" target="_blank">{article["aggregator_name"]}</a>)'
        return ''
            
    def _generate_aggregator_aware_html(self, clusters, output_file):
        """
        Generate HTML output with aggregator awareness as a fallback.
        
        Args:
            clusters: List of article clusters
            output_file: Path to output file
            
        Returns:
            str: Path to generated HTML file or None if generation failed
        """
        try:
            html_parts = [
                '<!DOCTYPE html>',
                '<html><head>',
                '<meta charset="UTF-8">',
                '<title>RSS Summary with Source Attribution</title>',
                '<style>',
                self._get_enhanced_css(),
                '</style>',
                '</head><body>',
                '<div class="container">',
                '<h1>RSS Summary</h1>',
                f'<p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
                self._get_aggregator_stats_html(),
            ]
            
            for cluster_idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                    
                html_parts.append('<div class="cluster">')
                
                # Get cluster headline
                headline = cluster[0].get('summary', {}).get('headline', cluster[0].get('title', 'Untitled'))
                html_parts.append(f'<h2 class="cluster-headline">{headline}</h2>')
                
                # Add summary
                summary = cluster[0].get('summary', {}).get('summary', '')
                if summary:
                    html_parts.append(f'<div class="cluster-summary">{summary}</div>')
                
                # Add articles with source attribution
                html_parts.append('<div class="cluster-articles">')
                for article in cluster:
                    html_parts.append(self._generate_article_html(article))
                html_parts.append('</div>')  # cluster-articles
                
                html_parts.append('</div>')  # cluster
            
            html_parts.extend([
                '</div>',  # container
                '</body></html>'
            ])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))
            
            logging.info(f"Successfully wrote aggregator-aware HTML output to {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"Error generating aggregator-aware HTML: {str(e)}", exc_info=True)
            return None
    
    def _generate_article_html(self, article):
        """Generate HTML for a single article with source attribution."""
        # Determine which URL to use for the article link
        display_url = article.get('original_url', article['link'])
        is_aggregated = article.get('is_aggregator', False)
        
        html = ['<div class="article">']
        
        # Article title with link
        html.append(f'<h3 class="article-title">')
        html.append(f'<a href="{display_url}" target="_blank">{article["title"]}</a>')
        html.append('</h3>')
        
        # Article metadata
        html.append('<div class="article-meta">')
        
        # Source information
        source_parts = []
        
        # Original source
        source_name = article.get('source_name', 'Unknown')
        source_parts.append(f'<span class="source">Source: <strong>{source_name}</strong></span>')
        
        # If it was aggregated, show via information
        if is_aggregated and article.get('aggregator_name'):
            agg_url = article.get('aggregator_url', article['link'])
            agg_name = article['aggregator_name']
            source_parts.append(f'<span class="via">via <a href="{agg_url}" target="_blank">{agg_name}</a></span>')
        
        # Feed source
        source_parts.append(f'<span class="feed">Feed: {article.get("feed_source", "Unknown")}</span>')
        
        # Published date
        source_parts.append(f'<span class="date">Published: {article.get("published", "Unknown date")}</span>')
        
        html.append(' | '.join(source_parts))
        html.append('</div>')  # article-meta
        
        html.append('</div>')  # article
        
        return '\n'.join(html)
    
    def _get_enhanced_css(self):
        """Get enhanced CSS for aggregator-aware display."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        .timestamp {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 20px;
        }
        
        .aggregator-stats {
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 4px;
        }
        
        .aggregator-stats h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .cluster {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .cluster-headline {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .cluster-summary {
            color: #555;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
            border-radius: 4px;
            line-height: 1.8;
        }
        
        .cluster-articles {
            border-top: 1px solid #eee;
            padding-top: 15px;
        }
        
        .article {
            padding: 15px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .article:last-child {
            border-bottom: none;
        }
        
        .article-title {
            margin: 0 0 8px 0;
            font-size: 1.1em;
        }
        
        .article-title a {
            color: #2980b9;
            text-decoration: none;
        }
        
        .article-title a:hover {
            text-decoration: underline;
            color: #21618c;
        }
        
        .article-meta {
            font-size: 0.9em;
            color: #7f8c8d;
            line-height: 1.5;
        }
        
        .article-meta .source strong {
            color: #27ae60;
        }
        
        .article-meta .via {
            color: #e74c3c;
            font-style: italic;
        }
        
        .article-meta .via a {
            color: #c0392b;
            text-decoration: none;
        }
        
        .article-meta .via a:hover {
            text-decoration: underline;
        }
        
        .article-meta .feed {
            color: #8e44ad;
        }
        
        .article-meta .date {
            color: #95a5a6;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .cluster {
                padding: 15px;
            }
            
            .article-meta {
                font-size: 0.8em;
                line-height: 1.8;
            }
            
            .article-meta span {
                display: block;
                margin-bottom: 2px;
            }
        }
        """
    
    def _get_aggregator_stats_html(self):
        """Generate HTML for aggregator statistics."""
        if self.aggregator_stats['total_aggregator_links'] == 0:
            return ''
        
        stats = self.aggregator_stats
        success_rate = 0
        if stats['total_aggregator_links'] > 0:
            success_rate = (stats['successful_extractions'] / stats['total_aggregator_links']) * 100
        
        html = [
            '<div class="aggregator-stats">',
            '<h3>📊 Source Extraction Statistics</h3>',
            '<ul style="margin: 0; padding-left: 20px;">',
            f'<li>Total aggregator links processed: <strong>{stats["total_aggregator_links"]}</strong></li>',
            f'<li>Successful source extractions: <strong>{stats["successful_extractions"]}</strong> ({success_rate:.1f}%)</li>',
        ]
        
        if stats['by_aggregator']:
            html.append('<li>By aggregator:')
            html.append('<ul>')
            for agg, count in sorted(stats['by_aggregator'].items()):
                html.append(f'<li>{agg}: {count} links</li>')
            html.append('</ul>')
            html.append('</li>')
        
        html.extend(['</ul>', '</div>'])
        
        return '\n'.join(html)