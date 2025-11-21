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
    # Handle None or empty text
    if not text:
        return ""
    
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
        url: Article URL (can be None or empty)
        
    Returns:
        Publication name or default value
    """
    # Handle None or empty URL
    if not url:
        logger.warning("URL is None or empty, returning default source name")
        return "Unknown Source"
    
    try:
        # Extract domain from URL
        if '//' in url:
            # Handle full URLs like https://example.com/path
            source_name = url.split('//')[1].split('/')[0]
        else:
            # Handle URLs without protocol
            source_name = url.split('/')[0]
        
        # Remove www. prefix
        source_name = source_name.replace('www.', '')
        
        # Return the cleaned domain name
        return source_name if source_name else "Unknown Source"
        
    except Exception as e:
        logger.warning(f"Failed to extract source from URL '{url}': {str(e)}")
        
        # Fallback: try to clean the URL safely
        try:
            if isinstance(url, str):
                cleaned = url.replace('https://', '').replace('http://', '')
                # Take just the domain part if there's a path
                if '/' in cleaned:
                    cleaned = cleaned.split('/')[0]
                return cleaned if cleaned else "Unknown Source"
            else:
                return "Unknown Source"
        except Exception:
            return "Unknown Source"

def chunk_text(text: str, max_chunk_size: int = 8000) -> List[str]:
    """
    Split text into manageable chunks for processing.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chunk_size:
        return [text or ""]
        
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
    Create the prompt for article summarization with different style options.
    
    Args:
        text: Cleaned article text
        url: Article URL
        source_name: Publication source name
        style: Summary style ('default', 'bullet', 'newswire')
        
    Returns:
        Formatted prompt for Claude
    """
    # Handle None or empty parameters
    if not text:
        text = ""
    if not url:
        url = "#"
    if not source_name:
        source_name = "Unknown Source"
    
    # Common instructions for all styles
    common_instructions = (
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
    )
    
    # Default/Current 5-sentence summary style
    if style == "default":
        style_instructions = (
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
            f"{common_instructions}\n"
            "- Ensure the headline doesn't repeat too many words from the summary\n\n"
        )
    
    # Axios-style bullet points
    elif style == "bullet":
        style_instructions = (
            "Create an Axios-style summary of the article following these guidelines:\n\n"
            "Structure:\n"
            "1. First line: Create a bold, catchy headline in sentence case\n"
            "2. Then a blank line\n"
            "3. Then a brief 1-2 sentence overview of what the article is about\n"
            "4. Then a blank line\n"
            "5. Then a section called 'The big picture:' with 1-2 sentences of context\n"
            "6. Then a section called 'Key points:' with 4-6 bullet points that:\n"
            "   - Start each bullet with '•' followed by a bold statement or statistic\n"
            "   - Follow each bold statement with 1-2 explanatory sentences\n"
            "   - Include surprising details, not just the obvious points\n"
            "   - Mix essential facts with interesting implications\n"
            "7. If applicable, a section called 'What's next:' with 1-2 bullets about future implications\n"
            "8. Then a blank line\n"
            "9. Then add 'Source: [publication name]' followed by the URL\n\n"
            f"{common_instructions}\n"
            "- Make bullet points conversational but insightful\n"
            "- Ensure some bullets contain surprising or counterintuitive information\n\n"
        )
    
    # Traditional newswire style
    elif style == "newswire":
        style_instructions = (
            "Create a traditional newswire-style article summary following these guidelines:\n\n"
            "Structure:\n"
            "1. First line: Create a concise, factual headline in title case (AP style)\n"
            "2. Then a blank line\n"
            "3. Then a dateline in all caps (e.g., 'SAN FRANCISCO —')\n"
            "4. Then a first paragraph (lead) that covers the 5 Ws (who, what, when, where, why) in a single sentence\n"
            "5. Then 3-5 additional paragraphs that:\n"
            "   - Follow the inverted pyramid structure (most important to least important)\n"
            "   - Include at least one direct quote if present in the source material\n"
            "   - Provide context and background in later paragraphs\n"
            "   - Maintain a formal, objective tone throughout\n"
            "6. Then a blank line\n"
            "7. Then add 'Source: [publication name]' followed by the URL\n\n"
            f"{common_instructions}\n"
            "- Use short paragraphs (1-2 sentences each)\n"
            "- Focus on facts over analysis\n"
            "- Avoid subjective language or speculation\n\n"
        )
    
    else:
        # Default to default style if style parameter is not recognized
        # Fixed to avoid infinite recursion
        return create_summary_prompt(text, url, source_name, "default")
    
    # Combine style instructions with the article content
    full_prompt = (
        f"{style_instructions}"
        f"Article:\n{text}\n\n"
        f"URL: {url}\n"
        f"Publication: {source_name}"
    )
    
    return full_prompt

def get_system_prompt() -> str:
    """
    Get the system prompt for Claude.

    Returns:
        System prompt string
    """
    return (
        "You are an expert technical journalist specializing in AI and technology news. "
        "Your summaries are written for AI developers, researchers, and technology professionals "
        "who value precision, technical depth, and direct communication.\n\n"

        "Core principles:\n"
        "- Present information directly and factually in active voice\n"
        "- Avoid meta-language like 'This article explains...', 'This is important because...', "
        "or 'The author discusses...'\n"
        "- Include technical details, specifications, and industry implications\n"
        "- Use clear, straightforward language without hype, exaggeration, or marketing speak\n"
        "- Focus on what matters to technical practitioners: capabilities, limitations, pricing, availability\n\n"

        "Style conventions:\n"
        "- Use active voice and non-compound verbs (e.g., 'banned' not 'has banned')\n"
        "- Spell out numbers and 'percent' (e.g., '8 billion', not '8B' or '%')\n"
        "- Use smart quotes, not straight quotes\n"
        "- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods\n"
        "- Avoid the words 'content' and 'creator' when possible"
    )

def parse_summary_response(summary_text: str, title: str, url: str, source_name: str, style: str = "default") -> Dict[str, str]:
    """
    Parse the summary response from Claude based on style.
    
    Args:
        summary_text: Raw summary text from Claude
        title: Original article title
        url: Article URL
        source_name: Publication source name
        style: Summary style used ('default', 'bullet', 'newswire')
        
    Returns:
        Dictionary with headline and summary
    """
    # Handle None or empty parameters
    if not summary_text:
        summary_text = ""
    if not title:
        title = "Untitled Article"
    if not url:
        url = "#"
    if not source_name:
        source_name = "Unknown Source"
    
    try:
        # Style-specific parsing
        if style == "bullet":
            # For bullet points, try to extract headline, overview, sections
            parts = summary_text.split('\n\n', 2)
            if len(parts) >= 2:
                headline = parts[0].strip()
                summary = parts[1] if len(parts) > 1 else ""
                # Add remaining parts if they exist
                if len(parts) > 2:
                    summary += "\n\n" + parts[2]
            else:
                headline = title
                summary = summary_text
                
        elif style == "newswire":
            # For newswire, extract headline and rest of the article
            parts = summary_text.split('\n\n', 1)
            headline = parts[0].strip()
            summary = parts[1] if len(parts) > 1 else summary_text
            
        else:
            # Standard/default format - keep current logic
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
            if not "Source:" in summary:
                source_info = f"Source: {source_name}\n{url}"
                summary = f"{summary}\n\n{source_info}"
        
        logger.debug(
            f"Parsed summary response: style={style}, "
            f"headline_length={len(headline)}, summary_length={len(summary)}"
        )
        
        return {
            'headline': headline,
            'summary': summary,
            'style': style
        }
    except Exception as e:
        logger.error(f"Failed to parse summary response: {str(e)}")
        # Return a fallback summary
        return {
            'headline': title,
            'summary': f"Failed to parse summary: {str(e)}\n\nSource: {source_name}\n{url}",
            'style': style
        }