"""
Abstract base classes and interfaces for archive services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class PaywallType(Enum):
    """Types of paywalls detected."""
    NONE = "none"
    SIMPLE = "simple"
    COMPLEX = "complex"
    JAVASCRIPT = "javascript"


@dataclass
class ArchiveResult:
    """Result from an archive service operation."""
    url: Optional[str] = None
    content: Optional[str] = None
    success: bool = False
    service_name: str = ""
    error_message: Optional[str] = None
    cached: bool = False


@dataclass
class PaywallDetectionResult:
    """Result from paywall detection."""
    is_paywalled: bool = False
    paywall_type: PaywallType = PaywallType.NONE
    confidence: float = 0.0
    domain: str = ""


class ArchiveProvider(ABC):
    """Abstract base class for archive service providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the archive provider."""
        pass
    
    @property
    @abstractmethod
    def needs_submission(self) -> bool:
        """Whether this provider requires URL submission."""
        pass
    
    @abstractmethod
    def get_archived_url(self, url: str) -> Optional[str]:
        """
        Get archived URL for the given URL.
        
        Args:
            url: The URL to get archived version of
            
        Returns:
            Archived URL or None if not available
        """
        pass
    
    @abstractmethod
    def submit_url(self, url: str) -> bool:
        """
        Submit URL for archiving.
        
        Args:
            url: The URL to submit for archiving
            
        Returns:
            True if submission was successful
        """
        pass
    
    @abstractmethod
    def search_archives(self, url: str) -> Optional[str]:
        """
        Search for existing archives of the URL.
        
        Args:
            url: The URL to search for
            
        Returns:
            Archived URL or None if not found
        """
        pass


class PaywallDetector(ABC):
    """Abstract base class for paywall detection."""
    
    @abstractmethod
    def detect_paywall(self, url: str, content: Optional[str] = None) -> PaywallDetectionResult:
        """
        Detect if URL has a paywall.
        
        Args:
            url: The URL to check
            content: Optional page content to analyze
            
        Returns:
            PaywallDetectionResult with detection details
        """
        pass


class ContentExtractor(ABC):
    """Abstract base class for content extraction."""
    
    @abstractmethod
    def extract_content(self, url: str, session=None) -> Optional[str]:
        """
        Extract content from URL.
        
        Args:
            url: The URL to extract content from
            session: Optional requests session
            
        Returns:
            Extracted content or None if failed
        """
        pass


class ArchiveService:
    """
    Main archive service that coordinates different providers and strategies.
    """
    
    def __init__(self):
        self.providers: list[ArchiveProvider] = []
        self.paywall_detector: Optional[PaywallDetector] = None
        self.content_extractor: Optional[ContentExtractor] = None
    
    def add_provider(self, provider: ArchiveProvider) -> None:
        """Add an archive provider."""
        self.providers.append(provider)
    
    def set_paywall_detector(self, detector: PaywallDetector) -> None:
        """Set the paywall detector."""
        self.paywall_detector = detector
    
    def set_content_extractor(self, extractor: ContentExtractor) -> None:
        """Set the content extractor."""
        self.content_extractor = extractor
    
    def get_archived_content(self, url: str, force_new: bool = False) -> ArchiveResult:
        """
        Get archived content for a URL.
        
        Args:
            url: The URL to get archived content for
            force_new: Whether to force creation of new archive
            
        Returns:
            ArchiveResult with the archived content
        """
        # This will be implemented in the concrete service
        return ArchiveResult()