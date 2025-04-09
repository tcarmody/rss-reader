"""Article summarization with Anthropic Claude API with model selection and streaming."""

import re
import html
import logging
import anthropic
from typing import Dict, List, Optional, Union, Generator, Callable

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
    - Model selection capability
    - Streaming response support
    
    Example:
        summarizer = ArticleSummarizer()
        summary = summarizer.summarize_article(
            "Article text here...",
            "Article Title",
            "https://example.com/article",
            model="claude-3-sonnet-20240229"
        )
        
        # Or with streaming:
        for chunk in summarizer.summarize_article_streaming(
            "Article text here...",
            "Article Title",
            "https://example.com/article"
        ):
            print(chunk, end="", flush=True)
    """

    # Available Claude models
    AVAILABLE_MODELS = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229",
        "opus": "claude-3-opus-20240229",
        "haiku-legacy": "claude-2.0",
        "sonnet-legacy": "claude-2.1",
        "sonnet-3.5": "claude-3-5-sonnet-20240620",
        "haiku-3.5": "claude-3-5-haiku-20240307",
        "sonnet-3.7": "claude-3-7-sonnet-20250219"
    }
    
    # Default model to use (using the latest model as default)
    DEFAULT_MODEL = "claude-3-7-sonnet-20250219"

    def __init__(self):
        """Initialize the summarizer with Claude API client."""
        self.client = anthropic.Anthropic(api_key=get_env_var('ANTHROPIC_API_KEY'))
        self.summary_cache = SummaryCache()
        self.logger = logging.getLogger("ArticleSummarizer")

    def clean_text(self, text: str) -> str:
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
    
    def _get_model(self, model: Optional[str] = None) -> str:
        """
        Get the Claude model identifier.
        
        Args:
            model: Model name or identifier
            
        Returns:
            Model identifier string to use with the API
        """
        if not model:
            return self.DEFAULT_MODEL
            
        # If a full model identifier is provided, use it directly
        if model.startswith("claude-"):
            return model
            
        # If a shorthand name is provided, look it up
        if model in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model]
            
        # If not found, log a warning and return the default
        self.logger.warning(f"Model '{model}' not found. Using default model.")
        return self.DEFAULT_MODEL
    
    def _extract_source_from_url(self, url: str) -> str:
        """
        Extract publication name from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Publication name
        """
        source_name = url.split('//')[1].split('/')[0] if '//' in url else url
        return source_name.replace('www.', '')

    def _create_summary_prompt(self, text: str, url: str, source_name: str) -> str:
        """
        Create the prompt for article summarization.
        
        Args:
            text: Cleaned article text
            url: Article URL
            source_name: Publication source name
            
        Returns:
            Formatted prompt for Claude
        """
        return (
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
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for Claude.
        
        Returns:
            System prompt string
        """
        return (
            "You are an expert at creating summaries of articles. Your summaries should be "
            "factual, informative, concise, and written in a direct journalistic style. "
            "Avoid meta-language or self-explanatory phrases like 'This article explains...', "
            "'This is important for AI developers because...', or 'The author discusses...'. "
            "Instead, present information directly and factually. Write in a clear, "
            "straightforward manner without exaggeration, hype, or marketing speak. "
            "Focus on conveying the key points and implications without explicitly stating that you're doing so."
        )

    def _parse_summary_response(self, summary_text: str, title: str, url: str, source_name: str) -> Dict[str, str]:
        """
        Parse the summary response from Claude.
        
        Args:
            summary_text: Raw summary text from Claude
            title: Original article title
            url: Article URL
            source_name: Publication source name
            
        Returns:
            Dictionary with headline and summary
        """
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

        return {
            'headline': headline,
            'summary': summary
        }

    def summarize_article(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        force_refresh: bool = False,
        temperature: float = 0.3,
    ) -> Dict[str, str]:
        """
        Generate a concise summary of the article text.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            model: Claude model to use (shorthand name or full identifier)
            force_refresh: Whether to force a new summary instead of using cache
            temperature: Temperature setting for generation (0.0-1.0)
            
        Returns:
            dict: The summary with headline and text
        """
        # Clean the text first
        text = self.clean_text(text)

        # Generate cache key that includes model information
        cache_key = f"{text}:{model or 'default'}"

        # Check cache first
        if not force_refresh:
            cached_summary = self.summary_cache.get(cache_key)
            if cached_summary:
                self.logger.info(f"Using cached summary for {url}")
                return cached_summary['summary']

        try:
            # Extract source from URL for attribution
            source_name = self._extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = self._get_model(model)
            
            # Create the prompt
            prompt = self._create_summary_prompt(text, url, source_name)

            # Log the request (without full text for brevity)
            self.logger.info(f"Requesting summary for {url} using model {model_id}")

            # Generate summary using Claude
            response = self.client.messages.create(
                model=model_id,
                max_tokens=400,
                temperature=temperature,
                system=self._get_system_prompt(),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            summary_text = response.content[0].text
            
            # Parse the response
            result = self._parse_summary_response(summary_text, title, url, source_name)
            
            # Cache the result
            self.summary_cache.set(cache_key, result)
            
            return result

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return {
                'headline': title,
                'summary': "Summary generation failed. Please try again later."
            }

    def summarize_article_streaming(
        self, 
        text: str, 
        title: str, 
        url: str, 
        model: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, Dict[str, str]]:
        """
        Generate a summary of the article with streaming response.
        
        Args:
            text: The article text to summarize
            title: The article title
            url: The article URL
            model: Claude model to use (shorthand name or full identifier)
            callback: Optional callback function to process streamed chunks
            temperature: Temperature setting for generation (0.0-1.0)
            
        Yields:
            str: Chunks of the summary as they are generated
            
        Returns:
            dict: The complete summary with headline and text when finished
        """
        # Clean the text first
        text = self.clean_text(text)

        try:
            # Extract source from URL for attribution
            source_name = self._extract_source_from_url(url)
            
            # Get the actual model identifier
            model_id = self._get_model(model)
            
            # Create the prompt
            prompt = self._create_summary_prompt(text, url, source_name)

            # Log the request
            self.logger.info(f"Requesting streaming summary for {url} using model {model_id}")

            # Generate summary using Claude with streaming
            with self.client.messages.stream(
                model=model_id,
                max_tokens=400,
                temperature=temperature,
                system=self._get_system_prompt(),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            ) as stream:
                # Collect the full text as we stream
                full_text = ""
                
                # Process each chunk
                for chunk in stream:
                    if chunk.type == "content_block_delta" and chunk.delta.text:
                        # Get the text chunk
                        text_chunk = chunk.delta.text
                        
                        # Append to full text
                        full_text += text_chunk
                        
                        # Yield the chunk
                        yield text_chunk
                        
                        # Call the callback if provided
                        if callback:
                            callback(text_chunk)
            
            # Parse the complete response
            result = self._parse_summary_response(full_text, title, url, source_name)
            
            # Cache the result (using the same cache key format as non-streaming)
            cache_key = f"{text}:{model or 'default'}"
            self.summary_cache.set(cache_key, result)
            
            # Return the complete summary result
            return result

        except Exception as e:
            self.logger.error(f"Error generating streaming summary: {str(e)}", exc_info=True)
            error_result = {
                'headline': title,
                'summary': "Summary generation failed. Please try again later."
            }
            
            # Yield the error message
            error_message = "Summary generation failed. Please try again later."
            yield error_message
            
            # Call the callback if provided
            if callback:
                callback(error_message)
                
            return error_result

    def generate_tags(
        self, 
        content: str,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate tags for an article using Claude.
        
        Args:
            content: Article content to extract tags from
            model: Claude model to use (shorthand name or full identifier)
            temperature: Temperature setting for generation (0.0-1.0)
            
        Returns:
            list: Generated tags as strings
        """
        try:
            # Get the actual model identifier
            model_id = self._get_model(model)
            
            response = self.client.messages.create(
                model=model_id,
                max_tokens=100,
                temperature=temperature,
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
            self.logger.error(f"Error generating tags: {str(e)}", exc_info=True)
            return []


# Usage examples

def example_basic_usage():
    """Example of basic usage."""
    summarizer = ArticleSummarizer()
    summary = summarizer.summarize_article(
        "Article text here...",
        "Article Title",
        "https://example.com/article"
    )
    print(f"Headline: {summary['headline']}")
    print(f"Summary: {summary['summary']}")


def example_model_selection():
    """Example of using model selection."""
    summarizer = ArticleSummarizer()
    
    # Using shorthand model names
    summary_haiku = summarizer.summarize_article(
        "Article text here...",
        "Article Title",
        "https://example.com/article",
        model="haiku"
    )
    
    # Using newer models
    summary_sonnet_3_5 = summarizer.summarize_article(
        "Article text here...",
        "Article Title",
        "https://example.com/article",
        model="sonnet-3.5"
    )
    
    summary_haiku_3_5 = summarizer.summarize_article(
        "Article text here...",
        "Article Title",
        "https://example.com/article",
        model="haiku-3.5"
    )
    
    summary_sonnet_3_7 = summarizer.summarize_article(
        "Article text here...",
        "Article Title",
        "https://example.com/article",
        model="sonnet-3.7"
    )
    
    # Using full model identifier
    summary_opus = summarizer.summarize_article(
        "Article text here...",
        "Article Title",
        "https://example.com/article",
        model="claude-3-opus-20240229"
    )


def example_streaming():
    """Example of using streaming responses."""
    summarizer = ArticleSummarizer()
    
    print("Streaming summary:")
    
    # Simple streaming with generator
    for chunk in summarizer.summarize_article_streaming(
        "Article text here...",
        "Article Title",
        "https://example.com/article"
    ):
        print(chunk, end="", flush=True)
    
    print("\n\nStreaming with callback:")
    
    # Using a callback function
    def process_chunk(chunk):
        # Process each chunk as it arrives
        print(f"Got chunk: {chunk}")
    
    result = summarizer.summarize_article_streaming(
        "Article text here...",
        "Article Title",
        "https://example.com/article",
        callback=process_chunk
    )
    
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    example_basic_usage()
    # example_model_selection()
    # example_streaming()