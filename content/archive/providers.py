"""
Concrete implementations of archive service providers.
"""

import logging
import random
import requests
from typing import Optional, List, Dict
from urllib.parse import urlparse, quote
import json

from .base import ArchiveProvider, ArchiveResult
from common.errors import retry_with_backoff, ConnectionError, RateLimitError

logger = logging.getLogger(__name__)


class ArchiveIsProvider(ArchiveProvider):
    """Archive.is (archive.today) provider."""
    
    @property
    def name(self) -> str:
        return "Archive.is"
    
    @property
    def needs_submission(self) -> bool:
        return True
    
    def get_archived_url(self, url: str) -> Optional[str]:
        """Get archived URL from Archive.is."""
        return f"https://archive.is/{url}"
    
    def submit_url(self, url: str) -> bool:
        """Submit URL to Archive.is for archiving."""
        try:
            response = requests.post(
                "https://archive.is/submit/",
                data={"url": url},
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to submit to Archive.is: {e}")
            return False
    
    def search_archives(self, url: str) -> Optional[str]:
        """Search for existing archives on Archive.is."""
        try:
            search_url = f"https://archive.is/search/?q={quote(url)}"
            response = requests.get(search_url, timeout=15)
            if response.status_code == 200:
                # Simple check for archived content
                if "https://archive.is/" in response.text:
                    return search_url
        except Exception as e:
            logger.error(f"Archive.is search failed: {e}")
        return None


class WaybackMachineProvider(ArchiveProvider):
    """Internet Archive Wayback Machine provider."""
    
    @property
    def name(self) -> str:
        return "Wayback Machine"
    
    @property
    def needs_submission(self) -> bool:
        return True
    
    def get_archived_url(self, url: str, timestamp: str = "") -> Optional[str]:
        """Get archived URL from Wayback Machine."""
        if timestamp:
            return f"https://web.archive.org/web/{timestamp}/{url}"
        return f"https://web.archive.org/web/*/{url}"
    
    def submit_url(self, url: str) -> bool:
        """Submit URL to Wayback Machine for archiving."""
        try:
            response = requests.post(
                f"https://web.archive.org/save/{url}",
                timeout=60
            )
            return response.status_code in [200, 302]
        except Exception as e:
            logger.error(f"Failed to submit to Wayback Machine: {e}")
            return False
    
    def search_archives(self, url: str) -> Optional[str]:
        """Search for existing archives on Wayback Machine."""
        try:
            search_url = f"https://web.archive.org/cdx/search/cdx?url={quote(url)}&output=json&limit=1"
            response = requests.get(search_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:  # First row is headers
                    timestamp = data[1][1]  # timestamp column
                    return self.get_archived_url(url, timestamp)
        except Exception as e:
            logger.error(f"Wayback Machine search failed: {e}")
        return None


class GoogleCacheProvider(ArchiveProvider):
    """Google Cache provider."""
    
    @property
    def name(self) -> str:
        return "Google Cache"
    
    @property
    def needs_submission(self) -> bool:
        return False
    
    def get_archived_url(self, url: str) -> Optional[str]:
        """Get cached URL from Google Cache."""
        return f"https://webcache.googleusercontent.com/search?q=cache:{url}"
    
    def submit_url(self, url: str) -> bool:
        """Google Cache doesn't support manual submission."""
        return False
    
    def search_archives(self, url: str) -> Optional[str]:
        """Google Cache doesn't have a search API."""
        return self.get_archived_url(url)


class TwelveftProvider(ArchiveProvider):
    """12ft.io paywall bypass provider."""
    
    @property
    def name(self) -> str:
        return "12ft.io"
    
    @property
    def needs_submission(self) -> bool:
        return False
    
    def get_archived_url(self, url: str) -> Optional[str]:
        """Get bypassed URL from 12ft.io."""
        return f"https://12ft.io/{url}"
    
    def submit_url(self, url: str) -> bool:
        """12ft.io doesn't require submission."""
        return True
    
    def search_archives(self, url: str) -> Optional[str]:
        """12ft.io doesn't have archives, just direct bypass."""
        return self.get_archived_url(url)


class OutlineProvider(ArchiveProvider):
    """Outline.com provider."""
    
    @property
    def name(self) -> str:
        return "Outline.com"
    
    @property
    def needs_submission(self) -> bool:
        return True
    
    def get_archived_url(self, url: str) -> Optional[str]:
        """Get outline URL."""
        return f"https://outline.com/{url}"
    
    def submit_url(self, url: str) -> bool:
        """Submit URL to Outline.com."""
        try:
            response = requests.get(f"https://outline.com/{url}", timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to submit to Outline.com: {e}")
            return False
    
    def search_archives(self, url: str) -> Optional[str]:
        """Outline.com doesn't have search, just direct access."""
        return self.get_archived_url(url)


class ArchiveProviderManager:
    """Manages multiple archive providers and selects the best one."""
    
    def __init__(self):
        self.providers: List[ArchiveProvider] = [
            ArchiveIsProvider(),
            WaybackMachineProvider(),
            GoogleCacheProvider(),
            TwelveftProvider(),
            OutlineProvider(),
        ]
        
        # Known paywall domains for provider selection
        self.paywall_domains = [
            'nytimes.com', 'wsj.com', 'washingtonpost.com', 'bloomberg.com',
            'ft.com', 'economist.com', 'newyorker.com', 'wired.com',
            'theatlantic.com', 'technologyreview.com', 'hbr.org', 'forbes.com',
            'businessinsider.com', 'medium.com', 'seekingalpha.com'
        ]
        
        # Simple paywall domains (work well with 12ft.io)
        self.simple_paywall_domains = [
            'medium.com', 'forbes.com', 'businessinsider.com'
        ]
    
    def get_provider_by_name(self, name: str) -> Optional[ArchiveProvider]:
        """Get provider by name."""
        for provider in self.providers:
            if provider.name.lower() == name.lower():
                return provider
        return None
    
    def select_best_provider(self, url: str) -> ArchiveProvider:
        """
        Select the best provider based on the URL domain.
        
        Args:
            url: The URL to archive
            
        Returns:
            Best archive provider for the URL
        """
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            # If URL parsing fails, use a default provider
            return self.providers[0]
        
        # For simple paywall sites, prefer 12ft.io
        if any(simple_domain in domain for simple_domain in self.simple_paywall_domains):
            provider = self.get_provider_by_name("12ft.io")
            if provider:
                return provider
        
        # For complex paywalls, prefer Archive.is
        if any(paywall_domain in domain for paywall_domain in self.paywall_domains):
            provider = self.get_provider_by_name("Archive.is")
            if provider:
                return provider
        
        # Prefer providers that don't need submission for unknown domains
        no_submission_providers = [p for p in self.providers if not p.needs_submission]
        if no_submission_providers:
            return random.choice(no_submission_providers)
        
        # Fallback to random provider
        return random.choice(self.providers)
    
    def get_archived_content(self, url: str, preferred_provider: str = None) -> ArchiveResult:
        """
        Get archived content using the best available provider.
        
        Args:
            url: The URL to archive
            preferred_provider: Optional preferred provider name
            
        Returns:
            ArchiveResult with archived content
        """
        # Select provider
        if preferred_provider:
            provider = self.get_provider_by_name(preferred_provider)
            if not provider:
                logger.warning(f"Provider {preferred_provider} not found, using auto-selection")
                provider = self.select_best_provider(url)
        else:
            provider = self.select_best_provider(url)
        
        result = ArchiveResult(service_name=provider.name)
        
        try:
            # First try to find existing archive
            archived_url = provider.search_archives(url)
            
            # If no existing archive and provider supports submission, submit URL
            if not archived_url and provider.needs_submission:
                if provider.submit_url(url):
                    # Wait a moment for processing
                    import time
                    time.sleep(2)
                    archived_url = provider.get_archived_url(url)
                else:
                    result.error_message = f"Failed to submit URL to {provider.name}"
                    return result
            elif not archived_url:
                archived_url = provider.get_archived_url(url)
            
            if archived_url:
                result.url = archived_url
                result.success = True
                
                # Try to fetch content from archived URL
                try:
                    response = requests.get(archived_url, timeout=30)
                    if response.status_code == 200:
                        result.content = response.text
                except Exception as e:
                    logger.warning(f"Failed to fetch content from {archived_url}: {e}")
                    # Still consider it successful if we have the URL
            else:
                result.error_message = f"No archived URL available from {provider.name}"
                
        except Exception as e:
            result.error_message = f"Error using {provider.name}: {str(e)}"
            logger.error(f"Archive provider error: {e}")
        
        return result


# Default global instance
default_provider_manager = ArchiveProviderManager()