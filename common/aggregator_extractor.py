"""
Enhanced source extraction logic for aggregator links with special rules
for Techmeme and Google News to ensure original sources are displayed
and summaries are generated from the actual source content.
"""

import re
import logging
import requests
from urllib.parse import urlparse, parse_qs, unquote
from typing import Optional, Dict, List, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class AggregatorSourceExtractor:
    """
    Enhanced extractor for handling news aggregator links with special rules
    for different aggregator types.
    """
    
    # Aggregator patterns with their extraction rules
    AGGREGATOR_RULES = {
        'techmeme.com': {
            'name': 'Techmeme',
            'patterns': [r'techmeme\.com'],
            'extraction_method': 'techmeme',
            'requires_html_parsing': True
        },
        'news.google.com': {
            'name': 'Google News',
            'patterns': [r'news\.google\.com'],
            'extraction_method': 'google_news',
            'requires_html_parsing': False  # Usually can extract from URL
        },
        'reddit.com': {
            'name': 'Reddit',
            'patterns': [r'reddit\.com'],
            'extraction_method': 'reddit',
            'requires_html_parsing': False
        },
        'news.ycombinator.com': {
            'name': 'Hacker News',
            'patterns': [r'news\.ycombinator\.com'],
            'extraction_method': 'hackernews',
            'requires_html_parsing': True
        }
    }
    
    def __init__(self, session=None):
        """
        Initialize the extractor.
        
        Args:
            session: Optional requests session for HTTP requests
        """
        self.session = session or requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_aggregator(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Check if URL is from a known aggregator and return the aggregator type.
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_aggregator, aggregator_type)
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        for agg_type, rules in self.AGGREGATOR_RULES.items():
            for pattern in rules['patterns']:
                if re.search(pattern, domain):
                    return True, agg_type
        
        return False, None
    
    def extract_source(self, url: str) -> Dict[str, any]:
        """
        Extract original source information from an aggregator URL.
        
        Args:
            url: Aggregator URL
            
        Returns:
            Dict with extracted information:
            {
                'original_url': str,        # The actual source URL
                'source_name': str,         # Name of the source publication
                'aggregator_name': str,     # Name of the aggregator
                'aggregator_url': str,      # Original aggregator URL
                'extraction_method': str,   # Method used for extraction
                'confidence': float         # Confidence score (0-1)
            }
        """
        is_agg, agg_type = self.is_aggregator(url)
        
        if not is_agg:
            return {
                'original_url': url,
                'source_name': self._extract_domain_name(url),
                'aggregator_name': None,
                'aggregator_url': None,
                'extraction_method': 'direct',
                'confidence': 1.0
            }
        
        # Get extraction rules for this aggregator
        rules = self.AGGREGATOR_RULES[agg_type]
        method_name = f"_extract_{rules['extraction_method']}"
        
        if hasattr(self, method_name):
            extraction_method = getattr(self, method_name)
            result = extraction_method(url)
            
            # Add metadata
            result['aggregator_name'] = rules['name']
            result['aggregator_url'] = url
            result['extraction_method'] = rules['extraction_method']
            
            return result
        else:
            logger.warning(f"No extraction method found for {agg_type}")
            return {
                'original_url': url,
                'source_name': self._extract_domain_name(url),
                'aggregator_name': rules['name'],
                'aggregator_url': url,
                'extraction_method': 'fallback',
                'confidence': 0.0
            }
    
    def _extract_techmeme(self, url: str) -> Dict[str, any]:
        """
        Extract original source from Techmeme URL.
        
        Techmeme has different URL patterns:
        - Story page: https://www.techmeme.com/YYMMDD/pNNNN
        - Direct link with tracking: https://www.techmeme.com/outbound?u=SOURCE_URL
        """
        # First check if it's an outbound link
        parsed_url = urlparse(url)
        if 'outbound' in parsed_url.path:
            query_params = parse_qs(parsed_url.query)
            if 'u' in query_params:
                original_url = unquote(query_params['u'][0])
                return {
                    'original_url': original_url,
                    'source_name': self._extract_domain_name(original_url),
                    'confidence': 1.0
                }
        
        # Otherwise, it's a story page - need to fetch and parse HTML
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for the main story link
                # Techmeme structure: main story link is in <strong class="L1">/strong>
                main_story = soup.find('strong', class_='L1')
                if main_story and main_story.find('a'):
                    link = main_story.find('a')
                    original_url = link.get('href', '')
                    source_name = link.text.strip()
                    
                    if original_url:
                        return {
                            'original_url': original_url,
                            'source_name': source_name or self._extract_domain_name(original_url),
                            'confidence': 0.9
                        }
                
                # Fallback: look for any external link that's not Techmeme
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('http') and 'techmeme.com' not in href:
                        return {
                            'original_url': href,
                            'source_name': self._extract_domain_name(href),
                            'confidence': 0.5
                        }
        except Exception as e:
            logger.error(f"Error extracting Techmeme source: {str(e)}")
        
        return {
            'original_url': url,
            'source_name': 'Techmeme',
            'confidence': 0.0
        }
    
    def _extract_google_news(self, url: str) -> Dict[str, any]:
        """
        Extract original source from Google News URL.
        
        Google News URL patterns:
        - https://news.google.com/articles/...
        - https://news.google.com/rss/articles/...
        - URL parameter based: ?...&url=ENCODED_URL
        """
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        # Method 1: Check URL parameter
        if 'url' in query_params:
            original_url = unquote(query_params['url'][0])
            return {
                'original_url': original_url,
                'source_name': self._extract_domain_name(original_url),
                'confidence': 1.0
            }
        
        # Method 2: For RSS/articles format, try to decode from the path
        if '/articles/' in url or '/rss/articles/' in url:
            # Google News sometimes encodes the URL in the article ID
            try:
                # Try to fetch the page and look for redirects
                response = self.session.get(url, timeout=10, allow_redirects=False)
                if response.status_code in [301, 302, 303, 307, 308]:
                    redirect_url = response.headers.get('Location', '')
                    if redirect_url and 'google.com' not in redirect_url:
                        return {
                            'original_url': redirect_url,
                            'source_name': self._extract_domain_name(redirect_url),
                            'confidence': 0.9
                        }
            except Exception as e:
                logger.warning(f"Error following Google News redirect: {str(e)}")
        
        return {
            'original_url': url,
            'source_name': 'Google News',
            'confidence': 0.0
        }
    
    def _extract_reddit(self, url: str) -> Dict[str, any]:
        """
        Extract original source from Reddit URL.
        
        Reddit patterns:
        - Link posts have the actual URL in the 'url' parameter
        - Text posts don't have an external source
        """
        if '/comments/' in url:
            # It's a Reddit post - try to get the actual linked URL
            try:
                # Append .json to get JSON data
                json_url = url.rstrip('/') + '.json'
                response = self.session.get(json_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        post_data = data[0]['data']['children'][0]['data']
                        if not post_data.get('is_self', True):
                            # It's a link post
                            original_url = post_data.get('url', '')
                            if original_url and 'reddit.com' not in original_url:
                                return {
                                    'original_url': original_url,
                                    'source_name': self._extract_domain_name(original_url),
                                    'confidence': 1.0
                                }
            except Exception as e:
                logger.warning(f"Error extracting Reddit source: {str(e)}")
        
        return {
            'original_url': url,
            'source_name': 'Reddit',
            'confidence': 0.0
        }
    
    def _extract_hackernews(self, url: str) -> Dict[str, any]:
        """
        Extract original source from Hacker News URL.
        """
        # Similar pattern to Reddit - HN links to external articles
        if 'item?id=' in url:
            try:
                # Use HN API
                item_id = url.split('id=')[1].split('&')[0]
                api_url = f'https://hacker-news.firebaseio.com/v0/item/{item_id}.json'
                response = self.session.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and 'url' in data:
                        original_url = data['url']
                        return {
                            'original_url': original_url,
                            'source_name': self._extract_domain_name(original_url),
                            'confidence': 1.0
                        }
            except Exception as e:
                logger.warning(f"Error extracting HN source: {str(e)}")
        
        return {
            'original_url': url,
            'source_name': 'Hacker News',
            'confidence': 0.0
        }
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract a readable domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            domain = re.sub(r'^www\.', '', domain)
            # Remove TLD for readability
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                return domain_parts[-2].title()
            return domain
        except:
            return "Unknown Source"


def enhance_article_with_source_info(article: Dict[str, any], extractor: AggregatorSourceExtractor) -> Dict[str, any]:
    """
    Enhance an article dictionary with original source information.
    
    Args:
        article: Article dictionary with at least 'link' field
        extractor: AggregatorSourceExtractor instance
        
    Returns:
        Enhanced article dictionary with source information
    """
    if 'link' not in article:
        return article
    
    # Extract source information
    source_info = extractor.extract_source(article['link'])
    
    # Add source information to article
    article['original_url'] = source_info['original_url']
    article['source_name'] = source_info['source_name']
    article['is_aggregator'] = source_info['aggregator_name'] is not None
    article['aggregator_name'] = source_info['aggregator_name']
    article['aggregator_url'] = source_info['aggregator_url']
    article['source_extraction_confidence'] = source_info['confidence']
    
    # Log extraction result
    if article['is_aggregator'] and source_info['confidence'] > 0:
        logger.info(f"Extracted source from {source_info['aggregator_name']}: "
                   f"{article['link']} -> {article['original_url']}")
    
    return article