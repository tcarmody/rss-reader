"""
Semantic cache key generation for improved hit rates.

Generates both exact and semantic cache keys to allow fuzzy matching
while maintaining exact match capability.
"""

import hashlib
import re
from typing import Tuple, List
from collections import Counter


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for better matching."""
    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()


def extract_title_keywords(text: str, n: int = 5) -> str:
    """
    Extract top N keywords from article beginning (likely the title/intro).

    Args:
        text: Article text
        n: Number of keywords to extract

    Returns:
        Pipe-separated keywords in sorted order
    """
    # Use first 500 chars (title + intro)
    snippet = text[:500].lower()

    # Remove punctuation and split
    words = re.findall(r'\b[a-z]{4,}\b', snippet)  # Words 4+ chars only

    # Common stop words to exclude
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they',
        'their', 'what', 'when', 'where', 'which', 'will', 'would', 'could',
        'about', 'after', 'before', 'these', 'those', 'there'
    }

    # Filter and count
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words)

    # Get top N keywords, sorted alphabetically for consistency
    top_words = sorted([word for word, _ in word_counts.most_common(n)])

    return '|'.join(top_words) if top_words else ''


def generate_cache_keys(
    text: str,
    model: str = 'default',
    temperature: float = 0.3,
    style: str = 'default'
) -> Tuple[str, str]:
    """
    Generate both exact and semantic cache keys.

    Args:
        text: Article text to cache
        model: Model identifier
        temperature: Generation temperature
        style: Summary style

    Returns:
        Tuple of (exact_key, semantic_key)
    """
    # Exact key (current approach)
    exact_key = f"{text}:{model}:{temperature}:{style}"

    # Semantic key components
    normalized_text = normalize_whitespace(text.lower())

    # Use first 2000 chars for semantic matching (title + core content)
    # This is enough to identify similar articles without being too specific
    text_fingerprint = normalized_text[:2000]

    # Extract keywords for additional matching signal
    keywords = extract_title_keywords(text)

    # Build semantic key
    semantic_components = f"{text_fingerprint}:{keywords}:{model}:{style}"
    semantic_key = hashlib.md5(semantic_components.encode('utf-8')).hexdigest()

    return exact_key, semantic_key


def get_all_cache_keys(
    text: str,
    model: str = 'default',
    temperature: float = 0.3,
    style: str = 'default'
) -> List[str]:
    """
    Get all cache keys to check (exact first, then semantic).

    Args:
        text: Article text
        model: Model identifier
        temperature: Generation temperature
        style: Summary style

    Returns:
        List of cache keys to check in priority order
    """
    exact_key, semantic_key = generate_cache_keys(text, model, temperature, style)

    # Return in priority order: exact match first, semantic second
    return [exact_key, semantic_key]


def create_cache_key_for_storage(
    text: str,
    model: str = 'default',
    temperature: float = 0.3,
    style: str = 'default',
    use_semantic: bool = False
) -> str:
    """
    Create the cache key to use for storage.

    Args:
        text: Article text
        model: Model identifier
        temperature: Generation temperature
        style: Summary style
        use_semantic: Whether to use semantic key (default: exact)

    Returns:
        Cache key string
    """
    exact_key, semantic_key = generate_cache_keys(text, model, temperature, style)

    if use_semantic:
        return semantic_key
    else:
        return exact_key
