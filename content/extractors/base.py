"""
Abstract base classes and interfaces for content extractors.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ExtractionMethod(Enum):
    """Methods for extracting content."""
    URL_PARSING = "url_parsing"
    HTML_PARSING = "html_parsing"
    API_CALL = "api_call"
    HYBRID = "hybrid"


@dataclass
class ExtractionResult:
    """Result from a content extraction operation."""
    original_url: Optional[str] = None
    extracted_url: Optional[str] = None
    content: Optional[str] = None
    title: Optional[str] = None
    source_name: Optional[str] = None
    success: bool = False
    extraction_method: ExtractionMethod = ExtractionMethod.URL_PARSING
    confidence: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AggregatorInfo:
    """Information about an aggregator site."""
    name: str
    domain: str
    patterns: List[str]
    extraction_method: ExtractionMethod
    requires_html_parsing: bool = False
    supports_direct_links: bool = True


class SourceExtractor(ABC):
    """Abstract base class for source extractors."""
    
    @abstractmethod
    def can_extract(self, url: str) -> bool:
        """
        Check if this extractor can handle the given URL.
        
        Args:
            url: The URL to check
            
        Returns:
            True if this extractor can handle the URL
        """
        pass
    
    @abstractmethod
    def extract(self, url: str, session=None) -> ExtractionResult:
        """
        Extract the original source URL and content.
        
        Args:
            url: The aggregator URL to extract from
            session: Optional requests session
            
        Returns:
            ExtractionResult with extracted information
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the extractor."""
        pass
    
    @property
    @abstractmethod
    def supported_domains(self) -> List[str]:
        """List of domains this extractor supports."""
        pass


class AggregatorDetector(ABC):
    """Abstract base class for detecting aggregator sites."""
    
    @abstractmethod
    def is_aggregator(self, url: str) -> bool:
        """
        Check if the URL is from an aggregator site.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is from an aggregator
        """
        pass
    
    @abstractmethod
    def get_aggregator_info(self, url: str) -> Optional[AggregatorInfo]:
        """
        Get information about the aggregator.
        
        Args:
            url: The aggregator URL
            
        Returns:
            AggregatorInfo or None if not an aggregator
        """
        pass


class ContentCleaner(ABC):
    """Abstract base class for cleaning extracted content."""
    
    @abstractmethod
    def clean_content(self, content: str, source_url: str = "") -> str:
        """
        Clean and normalize extracted content.
        
        Args:
            content: Raw content to clean
            source_url: Optional source URL for context
            
        Returns:
            Cleaned content
        """
        pass


class URLValidator(ABC):
    """Abstract base class for URL validation."""
    
    @abstractmethod
    def is_valid_source_url(self, url: str) -> bool:
        """
        Check if the extracted URL is a valid source.
        
        Args:
            url: The URL to validate
            
        Returns:
            True if the URL is valid
        """
        pass
    
    @abstractmethod
    def should_exclude_domain(self, url: str) -> bool:
        """
        Check if the domain should be excluded.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the domain should be excluded
        """
        pass


class ExtractorManager:
    """Manages multiple extractors and routes requests to appropriate ones."""
    
    def __init__(self):
        self.extractors: List[SourceExtractor] = []
        self.aggregator_detector: Optional[AggregatorDetector] = None
        self.content_cleaner: Optional[ContentCleaner] = None
        self.url_validator: Optional[URLValidator] = None
    
    def add_extractor(self, extractor: SourceExtractor) -> None:
        """Add an extractor to the manager."""
        self.extractors.append(extractor)
    
    def set_aggregator_detector(self, detector: AggregatorDetector) -> None:
        """Set the aggregator detector."""
        self.aggregator_detector = detector
    
    def set_content_cleaner(self, cleaner: ContentCleaner) -> None:
        """Set the content cleaner."""
        self.content_cleaner = cleaner
    
    def set_url_validator(self, validator: URLValidator) -> None:
        """Set the URL validator."""
        self.url_validator = validator
    
    def find_extractor(self, url: str) -> Optional[SourceExtractor]:
        """
        Find the best extractor for a given URL.
        
        Args:
            url: The URL to find an extractor for
            
        Returns:
            SourceExtractor that can handle the URL or None
        """
        for extractor in self.extractors:
            if extractor.can_extract(url):
                return extractor
        return None
    
    def extract_source(self, url: str, session=None) -> ExtractionResult:
        """
        Extract source information from a URL.
        
        Args:
            url: The URL to extract from
            session: Optional requests session
            
        Returns:
            ExtractionResult with extracted information
        """
        # Check if it's an aggregator
        if self.aggregator_detector and not self.aggregator_detector.is_aggregator(url):
            # If not an aggregator, return the original URL
            return ExtractionResult(
                original_url=url,
                extracted_url=url,
                success=True,
                confidence=1.0,
                extraction_method=ExtractionMethod.URL_PARSING
            )
        
        # Find appropriate extractor
        extractor = self.find_extractor(url)
        if not extractor:
            return ExtractionResult(
                original_url=url,
                success=False,
                error_message="No suitable extractor found",
                confidence=0.0
            )
        
        # Extract using the found extractor
        result = extractor.extract(url, session)
        
        # Validate extracted URL
        if result.success and result.extracted_url and self.url_validator:
            if self.url_validator.should_exclude_domain(result.extracted_url):
                result.success = False
                result.error_message = "Extracted domain is excluded"
            elif not self.url_validator.is_valid_source_url(result.extracted_url):
                result.success = False
                result.error_message = "Extracted URL is not valid"
        
        # Clean content if available
        if result.success and result.content and self.content_cleaner:
            result.content = self.content_cleaner.clean_content(
                result.content, 
                result.extracted_url or ""
            )
        
        return result


class BaseSourceExtractor(SourceExtractor):
    """Base implementation with common functionality."""
    
    def __init__(self, name: str, supported_domains: List[str]):
        self._name = name
        self._supported_domains = supported_domains
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def supported_domains(self) -> List[str]:
        return self._supported_domains
    
    def can_extract(self, url: str) -> bool:
        """Default implementation checks if URL domain is in supported domains."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            return any(supported_domain in domain for supported_domain in self.supported_domains)
        except Exception:
            return False
    
    def _create_result(self, url: str, **kwargs) -> ExtractionResult:
        """Helper to create ExtractionResult with common fields."""
        return ExtractionResult(
            original_url=url,
            **kwargs
        )