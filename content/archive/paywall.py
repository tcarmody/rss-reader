"""
Paywall detection and bypass utilities.
"""

import logging
import random
import requests
from typing import Optional, List, Dict, Set
from urllib.parse import urlparse, quote

from .base import PaywallDetector, PaywallDetectionResult, PaywallType

logger = logging.getLogger(__name__)


class DomainBasedPaywallDetector(PaywallDetector):
    """Paywall detector based on known domain lists."""
    
    def __init__(self):
        # Known paywall domains
        self.paywall_domains: Set[str] = {
            'nytimes.com', 'wsj.com', 'washingtonpost.com', 'bloomberg.com',
            'ft.com', 'economist.com', 'newyorker.com', 'wired.com',
            'theatlantic.com', 'technologyreview.com', 'hbr.org', 'forbes.com',
            'businessinsider.com', 'medium.com', 'seekingalpha.com', 'barrons.com',
            'foreignpolicy.com', 'thetimes.co.uk', 'telegraph.co.uk', 'latimes.com',
            'bostonglobe.com', 'sfchronicle.com', 'chicagotribune.com', 'usatoday.com',
            'theguardian.com', 'independent.co.uk', 'standard.co.uk'
        }
        
        # Complex paywall domains that need JavaScript rendering
        self.complex_paywall_domains: Set[str] = {
            'wsj.com', 'ft.com', 'economist.com', 'bloomberg.com', 'seekingalpha.com'
        }
        
        # Simple paywall domains that work well with basic bypass
        self.simple_paywall_domains: Set[str] = {
            'medium.com', 'forbes.com', 'businessinsider.com'
        }
    
    def detect_paywall(self, url: str, content: Optional[str] = None) -> PaywallDetectionResult:
        """
        Detect if URL has a paywall based on domain and content analysis.
        
        Args:
            url: The URL to check
            content: Optional page content to analyze
            
        Returns:
            PaywallDetectionResult with detection details
        """
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return PaywallDetectionResult(domain=url)
        
        result = PaywallDetectionResult(domain=domain)
        
        # Check against known paywall domains
        for paywall_domain in self.paywall_domains:
            if paywall_domain in domain:
                result.is_paywalled = True
                result.confidence = 0.9
                
                # Determine paywall type
                if paywall_domain in self.complex_paywall_domains:
                    result.paywall_type = PaywallType.COMPLEX
                elif paywall_domain in self.simple_paywall_domains:
                    result.paywall_type = PaywallType.SIMPLE
                else:
                    result.paywall_type = PaywallType.SIMPLE
                
                break
        
        # Content-based detection if content is provided
        if content and result.is_paywalled:
            paywall_indicators = [
                'subscribe', 'subscription', 'paywall', 'premium',
                'sign up', 'register', 'free trial', 'members only',
                'unlock this article', 'become a member'
            ]
            
            content_lower = content.lower()
            indicator_count = sum(1 for indicator in paywall_indicators 
                                if indicator in content_lower)
            
            if indicator_count >= 2:
                result.confidence = min(0.95, result.confidence + 0.05)
            
        return result


class ContentBasedPaywallDetector(PaywallDetector):
    """Paywall detector based on content analysis."""
    
    def __init__(self):
        self.paywall_indicators = [
            'subscribe', 'subscription', 'paywall', 'premium content',
            'sign up', 'register', 'free trial', 'members only',
            'unlock this article', 'become a member', 'limited access',
            'read more with subscription', 'subscriber exclusive',
            'this article is for subscribers', 'continue reading'
        ]
        
        self.javascript_indicators = [
            'please enable javascript', 'javascript required',
            'browser compatibility', 'loading...', 'please wait'
        ]
    
    def detect_paywall(self, url: str, content: Optional[str] = None) -> PaywallDetectionResult:
        """
        Detect paywall based on content analysis.
        
        Args:
            url: The URL to check
            content: Page content to analyze
            
        Returns:
            PaywallDetectionResult with detection details
        """
        domain = urlparse(url).netloc.lower() if url else ""
        result = PaywallDetectionResult(domain=domain)
        
        if not content:
            return result
        
        content_lower = content.lower()
        
        # Check for paywall indicators
        paywall_score = 0
        for indicator in self.paywall_indicators:
            if indicator in content_lower:
                paywall_score += 1
        
        # Check for JavaScript requirements
        js_score = 0
        for indicator in self.javascript_indicators:
            if indicator in content_lower:
                js_score += 1
        
        # Determine if paywalled
        if paywall_score >= 2:
            result.is_paywalled = True
            result.confidence = min(0.8, paywall_score * 0.15)
            
            if js_score >= 1:
                result.paywall_type = PaywallType.JAVASCRIPT
            else:
                result.paywall_type = PaywallType.SIMPLE
        elif js_score >= 2:
            result.is_paywalled = True
            result.paywall_type = PaywallType.JAVASCRIPT
            result.confidence = min(0.7, js_score * 0.2)
        
        return result


class HybridPaywallDetector(PaywallDetector):
    """Combines domain-based and content-based detection."""
    
    def __init__(self):
        self.domain_detector = DomainBasedPaywallDetector()
        self.content_detector = ContentBasedPaywallDetector()
    
    def detect_paywall(self, url: str, content: Optional[str] = None) -> PaywallDetectionResult:
        """
        Detect paywall using both domain and content analysis.
        
        Args:
            url: The URL to check
            content: Optional page content to analyze
            
        Returns:
            PaywallDetectionResult with combined analysis
        """
        domain_result = self.domain_detector.detect_paywall(url, content)
        content_result = self.content_detector.detect_paywall(url, content)
        
        # Combine results
        result = PaywallDetectionResult(domain=domain_result.domain)
        
        # If either detector says it's paywalled, it probably is
        if domain_result.is_paywalled or content_result.is_paywalled:
            result.is_paywalled = True
            
            # Use the higher confidence score
            result.confidence = max(domain_result.confidence, content_result.confidence)
            
            # Prefer domain-based type determination as it's more reliable
            if domain_result.paywall_type != PaywallType.NONE:
                result.paywall_type = domain_result.paywall_type
            else:
                result.paywall_type = content_result.paywall_type
        
        return result


class PaywallBypassHeaders:
    """Utility class for generating headers that can bypass paywalls."""
    
    @staticmethod
    def get_direct_access_headers() -> Dict[str, str]:
        """
        Get headers that can sometimes bypass paywalls directly.
        
        Returns:
            Dict of headers for direct bypassing
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
    
    @staticmethod
    def get_social_media_headers(url: str) -> Dict[str, str]:
        """
        Get headers that mimic social media crawlers.
        
        Args:
            url: The URL being accessed
            
        Returns:
            Dict of social media crawler headers
        """
        user_agents = [
            'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)',
            'Twitterbot/1.0',
            'LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)',
            'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
        ]
        
        referrer_strategies = [
            f'https://www.google.com/search?q={quote(url)}',
            'https://www.facebook.com/',
            'https://twitter.com/',
            'https://www.linkedin.com/',
            'https://news.google.com/',
            'https://t.co/',
        ]
        
        return {
            'User-Agent': random.choice(user_agents),
            'Referer': random.choice(referrer_strategies),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    @staticmethod
    def get_mobile_headers() -> Dict[str, str]:
        """
        Get headers that mimic mobile browsers.
        
        Returns:
            Dict of mobile browser headers
        """
        mobile_user_agents = [
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0',
            'Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36',
        ]
        
        return {
            'User-Agent': random.choice(mobile_user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }


# Default detector instance
default_paywall_detector = HybridPaywallDetector()


def is_paywalled(url: str, content: Optional[str] = None) -> bool:
    """
    Convenience function to check if a URL is paywalled.
    
    Args:
        url: The URL to check
        content: Optional page content to analyze
        
    Returns:
        True if the URL is likely paywalled
    """
    result = default_paywall_detector.detect_paywall(url, content)
    return result.is_paywalled


def get_paywall_type(url: str, content: Optional[str] = None) -> PaywallType:
    """
    Convenience function to get the paywall type.
    
    Args:
        url: The URL to check
        content: Optional page content to analyze
        
    Returns:
        PaywallType enum value
    """
    result = default_paywall_detector.detect_paywall(url, content)
    return result.paywall_type