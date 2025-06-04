"""
WSJ-specific paywall bypass and content extraction utilities.
"""

import re
import time
import logging
import random
import requests
from typing import Optional, List, Tuple
from urllib.parse import quote
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WSJBypassSession:
    """Specialized session for WSJ with advanced evasion techniques."""
    
    # WSJ-specific user agents that often bypass paywalls
    WSJ_USER_AGENTS = [
        'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)',
        'Twitterbot/1.0',
        'LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)',
        'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/600.2.5 (KHTML, like Gecko) Version/8.0.2 Safari/600.2.5 (Applebot/0.1; +http://www.apple.com/go/applebot)',
    ]
    
    def __init__(self, url: str):
        self.url = url
        self.session = self._create_optimized_session()
    
    def _create_optimized_session(self) -> requests.Session:
        """Create a session specifically optimized for WSJ."""
        session = requests.Session()
        
        # Random user agent selection
        user_agent = random.choice(self.WSJ_USER_AGENTS)
        
        # Advanced headers that mimic legitimate traffic
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        # Referrer strategies that often work
        referrer_strategies = [
            f'https://www.google.com/search?q={quote(self.url)}',
            'https://www.facebook.com/',
            'https://twitter.com/',
            'https://www.linkedin.com/',
            'https://news.google.com/',
            'https://t.co/',
            '',  # No referrer
        ]
        
        headers['Referer'] = random.choice(referrer_strategies)
        
        # WSJ-specific cookies that can help
        session.cookies.update({
            'wsjregion': 'na,us',
            'gdprApplies': 'false',
            'ccpaApplies': 'false',
            'usr_bkt': 'C0H2lPLMlD',
        })
        
        session.headers.update(headers)
        return session
    
    def get_session(self) -> requests.Session:
        """Get the configured session."""
        return self.session


class WSJURLVariants:
    """Generate different URL variants for WSJ articles."""
    
    @staticmethod
    def get_amp_version(url: str) -> Optional[str]:
        """Try to access the AMP version of WSJ articles."""
        if 'wsj.com/articles/' in url:
            match = re.search(r'/articles/([^/?]+)', url)
            if match:
                article_id = match.group(1)
                return f'https://www.wsj.com/amp/articles/{article_id}'
        return None
    
    @staticmethod
    def get_mobile_version(url: str) -> Optional[str]:
        """Try mobile version which sometimes has less strict paywalls."""
        if 'wsj.com' in url and 'm.wsj.com' not in url:
            return url.replace('www.wsj.com', 'm.wsj.com')
        return None
    
    @staticmethod
    def get_print_version(url: str) -> Optional[str]:
        """Try print version which sometimes bypasses paywalls."""
        if '?' in url:
            return f"{url}&mod=article_inline"
        else:
            return f"{url}?mod=article_inline"


class WSJContentValidator:
    """Validate WSJ content to ensure it's not a paywall page."""
    
    PAYWALL_INDICATORS = [
        'subscribe to continue reading',
        'sign in to continue reading',
        'this content is reserved for subscribers',
        'cleaning webpage',
        'advertisement',
        'paywall',
        'subscribe now',
        'digital subscription'
    ]
    
    @classmethod
    def is_actual_content(cls, soup: BeautifulSoup) -> bool:
        """Check if the soup contains actual WSJ article content vs paywall/cleaning page."""
        page_text = soup.get_text().lower()
        
        # Check for paywall indicators
        for indicator in cls.PAYWALL_INDICATORS:
            if indicator in page_text:
                logger.warning(f"Detected paywall indicator: {indicator}")
                return False
        
        # Check for content indicators
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


class WSJContentExtractor:
    """Extract article content from WSJ pages."""
    
    CONTENT_SELECTORS = [
        'div[data-module="ArticleBody"]',
        '.article-content',
        '.wsj-snippet-body',
        '.article-wrap',
        '[data-module="BodyText"]',
        '.StoryBody',
        '.snippet-promotion',
    ]
    
    @classmethod
    def extract_content(cls, soup: BeautifulSoup) -> Optional[str]:
        """Extract the actual article content from WSJ page."""
        for selector in cls.CONTENT_SELECTORS:
            elements = soup.select(selector)
            if elements:
                content_div = elements[0]
                
                # Remove unwanted elements
                for unwanted in content_div.select('.advertisement, .ad, .promo, .related'):
                    unwanted.decompose()
                
                # Extract paragraphs
                paragraphs = content_div.find_all('p')
                if paragraphs:
                    content = '\n\n'.join(
                        p.get_text().strip() 
                        for p in paragraphs 
                        if len(p.get_text().strip()) > 20
                    )
                    if len(content) > 200:
                        return content
        
        return None


class WSJArchiveBypass:
    """Use archive services specifically for WSJ content."""
    
    @staticmethod
    def try_12ft_io(url: str, session: requests.Session) -> Optional[str]:
        """Try to bypass using 12ft.io."""
        try:
            twelve_ft_url = f"https://12ft.io/{url}"
            twelve_ft_session = requests.Session()
            twelve_ft_session.headers.update(session.headers)
            twelve_ft_session.headers.update({
                'Referer': 'https://12ft.io/',
                'Origin': 'https://12ft.io'
            })
            
            response = twelve_ft_session.get(twelve_ft_url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                if WSJContentValidator.is_actual_content(soup):
                    content = WSJContentExtractor.extract_content(soup)
                    if content:
                        logger.info("Successfully extracted content using 12ft.io")
                        return content
        except Exception as e:
            logger.warning(f"12ft.io failed for WSJ: {str(e)}")
        
        return None
    
    @staticmethod
    def try_outline_com(url: str, session: requests.Session) -> Optional[str]:
        """Try to bypass using Outline.com."""
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
                        logger.info("Successfully extracted content using Outline.com")
                        return content
        except Exception as e:
            logger.warning(f"Outline.com failed for WSJ: {str(e)}")
        
        return None


class WSJBypassService:
    """Main service for bypassing WSJ paywalls with multiple strategies."""
    
    def __init__(self):
        self.url_variants = WSJURLVariants()
        self.content_validator = WSJContentValidator()
        self.content_extractor = WSJContentExtractor()
        self.archive_bypass = WSJArchiveBypass()
    
    def fetch_content(self, url: str) -> Optional[str]:
        """
        Enhanced WSJ content fetching with multiple strategies.
        
        Args:
            url: The WSJ article URL
            
        Returns:
            Article content or None if all strategies fail
        """
        logger.info(f"Starting enhanced WSJ bypass for: {url}")
        
        # Create optimized session
        bypass_session = WSJBypassSession(url)
        session = bypass_session.get_session()
        
        # Try different URL variants
        strategies = [
            ('Original URL', url),
            ('AMP Version', self.url_variants.get_amp_version(url)),
            ('Mobile Version', self.url_variants.get_mobile_version(url)),
            ('Print Version', self.url_variants.get_print_version(url)),
        ]
        
        for strategy_name, test_url in strategies:
            if not test_url:
                continue
                
            logger.info(f"Trying WSJ strategy: {strategy_name} - {test_url}")
            
            try:
                # Add random delay to avoid detection
                time.sleep(random.uniform(1, 3))
                
                response = session.get(test_url, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    if self.content_validator.is_actual_content(soup):
                        content = self.content_extractor.extract_content(soup)
                        if content and len(content) > 500:
                            logger.info(f"Successfully extracted content using {strategy_name}")
                            return content
                    else:
                        logger.warning(f"{strategy_name} returned paywall/cleaning page")
                        
            except Exception as e:
                logger.warning(f"Error with {strategy_name}: {str(e)}")
                continue
        
        # Try archive services as fallback
        logger.info("Direct strategies failed, trying archive services")
        
        content = self.archive_bypass.try_12ft_io(url, session)
        if content:
            return content
        
        content = self.archive_bypass.try_outline_com(url, session)
        if content:
            return content
        
        logger.warning(f"All WSJ bypass strategies failed for: {url}")
        return None


# Default service instance
wsj_bypass_service = WSJBypassService()


def fetch_wsj_content(url: str) -> Optional[str]:
    """
    Convenience function to fetch WSJ content.
    
    Args:
        url: The WSJ article URL
        
    Returns:
        Article content or None if failed
    """
    return wsj_bypass_service.fetch_content(url)