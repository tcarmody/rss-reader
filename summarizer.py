"""Article summarization with Anthropic Claude API."""

import re
import html
import logging
import anthropic

from bs4 import BeautifulSoup

from rss_reader.utils.config import get_env_var
from rss_reader.cache import SummaryCache


class ArticleSummarizer:
    """
    Summarizes articles using the Anthropic Claude API.
    
    This class handles:
    - Text cleaning and normalization
    - API communication with Claude
    - Caching of results to avoid redundant API calls
    - Tag generation for articles
    
    Example:
        summarizer = ArticleSummarizer()
        summary = summarizer.summarize_article(
            "Article text here...",
            "Article Title",
            "https://example.com/article"
        )
    """

    def __init__(self):
        """Initialize the summarizer with Claude API client."""
        self.client = anthropic.Anthropic(api_key=get_env_var('ANTHROPIC_API_KEY'))
        self.summary_cache = SummaryCache()

    def clean_text(self, text):
        """
        Clean HTML and normalize text for summarization.
        
        Args:
            text: Raw text that may contain HTML
            
        Returns:
            Cleaned and normalized text
        """
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Decode HTML entities
        text = html.unescape(text)

        return text

    def summarize_article(self, text, title, url, force_refresh=False):
        """
        Generate a concise summary of the article text.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            force_refresh: Whether to force a new summary instead of using cache
            
        Returns:
            dict: The summary with headline and text
        """
        # Clean the text first
        text = self.clean_text(text)

        # Check cache first
        cached_summary = self.summary_cache.get(text)
        if cached_summary and not force_refresh:
            return cached_summary['summary']

        try:
            # Generate summary using Claude
            prompt = (
                "Summarize this article in 3-4 sentences using active voice and factual tone. "
                "Follow this structure:\n"
                "1. First line: Create a headline in sentence case\n"
                "2. Then a blank line\n"
                "3. Then the summary that:\n"
                "- Explains what happened in simple language\n"
                "- Identifies key details for AI developers\n"
                "- Explains why it matters to AI industry followers\n"
                "- Spells out numbers and uses U.S./U.K. with periods\n\n"
                f"Article:\n{text}\n\n"
                f"URL: {url}"
            )

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                temperature=0.3,
                system="You are an expert AI technology journalist. Be concise and factual.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            summary_text = response.content[0].text

            # Split into headline and summary
            lines = summary_text.split('\n', 1)
            if len(lines) == 2:
                headline = lines[0].strip()
                summary = lines[1].strip()
            else:
                headline = title
                summary = summary_text

            result = {
                'headline': headline,
                'summary': summary
            }
            
            # Cache the result
            self.summary_cache.set(text, result)
            
            return result

        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {
                'headline': title,
                'summary': "Summary generation failed. Please try again later."
            }

    def generate_tags(self, content):
        """
        Generate tags for an article using Claude.
        
        Args:
            content: Article content to extract tags from
            
        Returns:
            list: Generated tags as strings
        """
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.7,
                system="Extract specific entities from the text and return them as tags. Include:\n"
                       "- Company names (e.g., 'Apple', 'Microsoft')\n"
                       "- Technologies (e.g., 'ChatGPT', 'iOS 17')\n"
                       "- People (e.g., 'Tim Cook', 'Satya Nadella')\n"
                       "- Products (e.g., 'iPhone 15', 'Surface Pro')\n"
                       "Format: Return only the tags as a comma-separated list, with no categories or explanations.",
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            tags = [tag.strip() for tag in response.content[0].text.split(',')]
            return tags
        except Exception as e:
            logging.error(f"Error generating tags: {str(e)}")
            return []
