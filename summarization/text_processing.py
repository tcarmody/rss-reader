"""
Text processing utilities for summarization.
"""

import re
import html
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean HTML and normalize text for summarization.
    
    Args:
        text: Raw text that may contain HTML
        
    Returns:
        Cleaned and normalized text
    """
    # Record text length for logging
    original_length = len(text)
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Decode HTML entities
    text = html.unescape(text)
    
    # Log the transformation
    cleaned_length = len(text)
    logger.debug(
        f"Text cleaned: {original_length} -> {cleaned_length} chars "
        f"({round((original_length - cleaned_length) / original_length * 100 if original_length > 0 else 0)}% reduction)"
    )

    return text

def extract_source_from_url(url: str) -> str:
    """
    Extract publication name from URL.
    
    Args:
        url: Article URL
        
    Returns:
        Publication name
    """
    try:
        source_name = url.split('//')[1].split('/')[0] if '//' in url else url
        source_name = source_name.replace('www.', '')
        return source_name
    except Exception as e:
        logger.warning(f"Failed to extract source from URL: {str(e)}")
        return url.replace('https://', '').replace('http://', '')

def chunk_text(text: str, max_chunk_size: int = 8000) -> List[str]:
    """
    Split text into manageable chunks for processing.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
        
    chunks = []
    # Try to split on paragraph boundaries
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        # If adding this paragraph exceeds the limit, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If single paragraph is too large, split on sentence boundaries
            if len(paragraph) > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = sentence + " "
                    else:
                        current_chunk += sentence + " "
            else:
                current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")
    return chunks

def create_summary_prompt(text: str, url: str, source_name: str, style: str = "default") -> str:
    """
    Create the prompt for article summarization.
    
    Args:
        text: Cleaned article text
        url: Article URL
        source_name: Publication source name
        style: Summary style ('default', 'bullet', 'newswire')
        
    Returns:
        Formatted prompt for Claude
    """
    if style == "bullet":
        return (
            "Summarize the article below following these guidelines:\n\n"
            "Structure:\n"
            "1. First line: Create a headline in sentence case\n"
            "2. Then a blank line\n"
            "3. Then a bullet point summary with 3-5 key points that:\n"
            "   - Uses bullet points (•) for each main point\n"
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
    elif style == "newswire":
        return (
            "Summarize the article below in a concise newswire style following these guidelines:\n\n"
            "Structure:\n"
            "1. First line: Create a headline in sentence case that's clear and direct\n"
            "2. Then a blank line\n"
            "3. First paragraph: A concise lead that answers who, what, when, where, and why\n"
            "4. Following paragraphs: 2-3 short paragraphs with supporting details in descending order of importance\n"
            "5. Then a blank line\n"
            "6. Then add 'Source: [publication name]' followed by the URL\n\n"
            "Style guidelines:\n"
            "- Use the inverted pyramid structure (most important info first)\n"
            "- Keep sentences short and direct (15-20 words max)\n"
            "- Use active voice exclusively\n"
            "- Avoid adjectives and adverbs when possible\n"
            "- Include specific numbers, statistics and quotes when relevant\n"
            "- Use present tense for immediate events, past tense for completed actions\n"
            "- Spell out numbers (e.g., '8 billion' not '8B', '100 million' not '100M')\n"
            "- Spell out 'percent' instead of using the '%' symbol\n"
            "- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods\n\n"
            f"Article:\n{text}\n\n"
            f"URL: {url}\n"
            f"Publication: {source_name}"
        )
    else:  # default style
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

def get_system_prompt() -> str:
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

def parse_summary_response(summary_text: str, title: str, url: str, source_name: str, style: str = "default") -> Dict[str, str]:
    """
    Parse the summary response from Claude.
    
    Args:
        summary_text: Raw summary text from Claude
        title: Original article title
        url: Article URL
        source_name: Publication source name
        style: Summary style used ('default', 'bullet', 'newswire')
        
    Returns:
        Dictionary with headline and summary
    """
    try:
        # Split into headline and summary with source attribution
        parts = summary_text.split('\n\n')
        
        # Handle different possible formats
        if len(parts) >= 3:  # Proper format with headline, summary, and source
            headline = parts[0].strip()
            summary = parts[1].strip()
            source_info = parts[2].strip()
            format_type = "standard"
        elif len(parts) == 2:  # Missing source or other format issue
            headline = parts[0].strip()
            summary = parts[1].strip()
            source_info = f"Source: {source_name}\n{url}"
            format_type = "missing_source"
        else:  # Fallback if formatting is completely off
            lines = summary_text.split('\n', 1)
            if len(lines) == 2:
                headline = lines[0].strip()
                summary = lines[1].strip()
                format_type = "fallback_newline"
            else:
                headline = title
                summary = summary_text
                format_type = "complete_fallback"
            source_info = f"Source: {source_name}\n{url}"
            
        # Ensure the summary has the source information
        if not summary.endswith(url):
            summary = f"{summary}\n\n{source_info}"

        logger.debug(
            f"Parsed summary response: format_type={format_type}, "
            f"style={style}, headline_length={len(headline)}, summary_length={len(summary)}"
        )

        return {
            'headline': headline,
            'summary': summary,
            'style': style  # Add style to the result
        }
    except Exception as e:
        logger.error(f"Failed to parse summary response: {str(e)}")
        # Return a fallback summary
        return {
            'headline': title,
            'summary': f"Failed to parse summary: {str(e)}\n\nSource: {source_name}\n{url}",
            'style': style  # Add style to the fallback result
        }