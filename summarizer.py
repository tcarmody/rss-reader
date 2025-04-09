"""Article summarization with Anthropic Claude API."""

import re
import html
import logging
import anthropic

from bs4 import BeautifulSoup

from utils.config import get_env_var
from cache import SummaryCache


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
            # Extract source from URL for attribution
            source_name = url.split('//')[1].split('/')[0] if '//' in url else url
            source_name = source_name.replace('www.', '')
            
            # Generate summary using Claude with improved prompt
            prompt = (
                "Summarize the article below following these guidelines:\n\n"
                "Structure:\n"
                "1. First line: Create a headline in sentence case\n"
                "2. Then a blank line\n"
                "3. Then a summary of three to five sentences that:\n"
                "   - Presents key information directly and factually\n"
                "   - Includes technical details relevant to AI developers\n"
                "   - Covers implications for the AI industry or technology landscape\n"
                "   - Mentions price and availability details for new models/tools (if applicable)\n"
                "4. Then a blank line\n"
                "5. Then add 'Source: [publication name]' followed by the URL\n\n"
                "Style guidelines:\n"
                "- Use active voice (e.g., 'Company released product' not 'Product was released by company')\n"
                "- Use non-compound verbs (e.g., 'banned' instead of 'has banned')\n"
                "- Avoid self-explanatory phrases like 'This article explains...', 'This is important because...', or 'The author discusses...'\n"
                "- Present information directly without meta-commentary\n"
                "- Avoid the words 'content' and 'creator'\n"
                "- Spell out numbers (e.g., '8 billion' not '8B', '100 million' not '100M')\n"
                "- Spell out 'percent' instead of using the '%' symbol\n"
                "- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods\n"
                "- Use smart quotes, not straight quotes\n"
                "- Ensure the headline doesn't repeat too many words from the summary\n\n"
                f"Article:\n{text}\n\n"
                f"URL: {url}\n"
                f"Publication: {source_name}"
            )

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                temperature=0.3,
                system="You are an expert at creating summaries of articles. Your summaries should be factual, informative, concise, and written in a direct journalistic style. Avoid meta-language or self-explanatory phrases like 'This article explains...', 'This is important for AI developers because...', or 'The author discusses...'. Instead, present information directly and factually. Write in a clear, straightforward manner without exaggeration, hype, or marketing speak. Focus on conveying the key points and implications without explicitly stating that you're doing so.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            summary_text = response.content[0].text

            # Split into headline and summary with source attribution
            parts = summary_text.split('\n\n')
            
            # Handle different possible formats
            if len(parts) >= 3:  # Proper format with headline, summary, and source
                headline = parts[0].strip()
                summary = parts[1].strip()
                source_info = parts[2].strip()
            elif len(parts) == 2:  # Missing source or other format issue
                headline = parts[0].strip()
                summary = parts[1].strip()
                source_info = f"Source: {source_name}\n{url}"
            else:  # Fallback if formatting is completely off
                lines = summary_text.split('\n', 1)
                if len(lines) == 2:
                    headline = lines[0].strip()
                    summary = lines[1].strip()
                else:
                    headline = title
                    summary = summary_text
                source_info = f"Source: {source_name}\n{url}"
                
            # Ensure the summary has the source information
            if not summary.endswith(url):
                summary = f"{summary}\n\n{source_info}"

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
