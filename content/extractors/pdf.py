"""
PDF Content Extractor

Handles extraction of text content from PDF files for processing by the RSS reader.
"""

import logging
import io
import requests
from typing import Optional, Dict, Any
from urllib.parse import urlparse

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not available. PDF processing disabled.")

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text content from PDF files."""

    def __init__(self, max_pages: int = 50, timeout: int = 30):
        """
        Initialize PDF extractor.

        Args:
            max_pages: Maximum number of pages to process (to avoid huge files)
            timeout: Request timeout in seconds
        """
        self.max_pages = max_pages
        self.timeout = timeout

        if not PDF_SUPPORT:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

    def is_pdf_url(self, url: str) -> bool:
        """
        Check if a URL points to a PDF file.

        Args:
            url: The URL to check

        Returns:
            bool: True if the URL appears to be a PDF
        """
        if not url:
            return False

        # Check file extension
        parsed = urlparse(url.lower())
        if parsed.path.endswith('.pdf'):
            return True

        # Check for PDF in URL path
        if '/pdf/' in parsed.path.lower() or 'pdf' in parsed.path.lower():
            return True

        return False

    def extract_from_url(self, url: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
        """
        Extract text content from a PDF URL.

        Args:
            url: URL pointing to a PDF file
            session: Optional requests session

        Returns:
            Dict containing success status, content, and metadata
        """
        try:
            if not session:
                session = requests.Session()

            logger.info(f"Fetching PDF from URL: {url}")

            # Download PDF content
            response = session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not self.is_pdf_url(url):
                return {
                    'success': False,
                    'content': '',
                    'error': f"Content type '{content_type}' does not appear to be PDF"
                }

            # Read PDF content into memory
            pdf_content = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                pdf_content.write(chunk)
            pdf_content.seek(0)

            # Extract text from PDF
            result = self.extract_from_bytes(pdf_content.getvalue())

            # Add URL metadata
            result['source_url'] = url
            result['content_type'] = content_type

            return result

        except requests.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return {
                'success': False,
                'content': '',
                'error': f"Download failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error processing PDF from {url}: {str(e)}")
            return {
                'success': False,
                'content': '',
                'error': f"PDF processing failed: {str(e)}"
            }

    def extract_from_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text content from PDF bytes.

        Args:
            pdf_bytes: Raw PDF file bytes

        Returns:
            Dict containing success status, content, and metadata
        """
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Get PDF metadata
            num_pages = len(pdf_reader.pages)
            metadata = pdf_reader.metadata or {}

            logger.info(f"Processing PDF with {num_pages} pages")

            # Limit pages to avoid processing huge documents
            pages_to_process = min(num_pages, self.max_pages)
            if pages_to_process < num_pages:
                logger.warning(f"PDF has {num_pages} pages, processing only first {pages_to_process}")

            # Extract text from each page
            text_content = []
            for page_num in range(pages_to_process):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text.strip():
                        text_content.append(page_text.strip())

                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue

            # Combine all text
            full_text = '\n\n'.join(text_content)

            # Clean up the text
            cleaned_text = self._clean_text(full_text)

            return {
                'success': True,
                'content': cleaned_text,
                'metadata': {
                    'total_pages': num_pages,
                    'processed_pages': pages_to_process,
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'creation_date': str(metadata.get('/CreationDate', '')),
                    'modification_date': str(metadata.get('/ModDate', '')),
                    'word_count': len(cleaned_text.split()) if cleaned_text else 0,
                    'character_count': len(cleaned_text) if cleaned_text else 0
                }
            }

        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"PDF read error: {str(e)}")
            return {
                'success': False,
                'content': '',
                'error': f"Invalid or corrupted PDF: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error processing PDF: {str(e)}")
            return {
                'success': False,
                'content': '',
                'error': f"PDF processing failed: {str(e)}"
            }

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text.

        Args:
            text: Raw extracted text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Basic text cleaning
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip very short lines (likely formatting artifacts)
            if len(line) < 3:
                continue

            # Skip lines that are mostly numbers or symbols
            if len([c for c in line if c.isalpha()]) < len(line) * 0.5:
                continue

            cleaned_lines.append(line)

        # Join lines and normalize whitespace
        cleaned_text = ' '.join(cleaned_lines)

        # Remove excessive whitespace
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        return cleaned_text.strip()


# Global instance for use across the application
_pdf_extractor = None

def get_pdf_extractor() -> PDFExtractor:
    """Get or create the global PDF extractor instance."""
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor