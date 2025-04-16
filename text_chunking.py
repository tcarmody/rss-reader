import re
from typing import List

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
        
    return chunks


def summarize_long_article(
    summarizer,
    text: str, 
    title: str, 
    url: str, 
    model: str = None,
    force_refresh: bool = False,
    temperature: float = 0.3,
) -> dict:
    """
    Generate summary for long articles by chunking and meta-summarization.
    
    Args:
        summarizer: ArticleSummarizer instance
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
    text = summarizer.clean_text(text)
    
    # If text is short enough, use regular summarization
    if len(text) < 12000:
        return summarizer.summarize_article(
            text=text,
            title=title,
            url=url,
            model=model,
            force_refresh=force_refresh,
            temperature=temperature
        )
        
    # Split into chunks
    chunks = chunk_text(text)
    
    # Summarize each chunk
    chunk_summaries = []
    model_id = summarizer._get_model(model)
    
    for i, chunk in enumerate(chunks):
        summarizer.logger.info(f"Summarizing chunk {i+1}/{len(chunks)} for {url}")
        
        # Create a chunk-specific prompt
        chunk_prompt = (
            f"Summarize this section (section {i+1} of {len(chunks)}) of an article in 2-3 sentences, "
            "capturing the key points and facts:\n\n"
            f"{chunk}"
        )
        
        # Get summary for this chunk
        chunk_summary = summarizer._call_claude_api(
            model_id=model_id,
            prompt=chunk_prompt,
            temperature=temperature,
            max_tokens=150
        )
        
        chunk_summaries.append(chunk_summary)
        
    # Create a meta-summary from the chunk summaries
    combined_chunks = "\n\n".join([
        f"Section {i+1} summary: {summary}" 
        for i, summary in enumerate(chunk_summaries)
    ])
    
    source_name = summarizer._extract_source_from_url(url)
    
    meta_prompt = (
        "Based on these section summaries, create a coherent overall summary "
        "of the complete article following these guidelines:\n\n"
        "1. First line: Create a headline in sentence case\n"
        "2. Then a blank line\n"
        "3. Then a summary of three to five sentences that captures the key points\n"
        "4. Then a blank line\n"
        "5. Then add 'Source: [publication name]' followed by the URL\n\n"
        f"Article title: {title}\n\n"
        f"Section summaries:\n{combined_chunks}\n\n"
        f"URL: {url}\n"
        f"Publication: {source_name}"
    )
    
    # Generate the final meta-summary
    final_summary = summarizer._call_claude_api(
        model_id=model_id,
        prompt=meta_prompt,
        temperature=temperature,
        max_tokens=400
    )
    
    # Parse the final summary
    result = summarizer._parse_summary_response(
        final_summary, 
        title, 
        url, 
        source_name
    )
    
    return result