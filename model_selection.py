import re
import logging
from typing import Optional

def estimate_complexity(text: str) -> float:
    """
    Estimate article complexity to determine appropriate model.
    
    Args:
        text: Article text
        
    Returns:
        Complexity score (0.0-1.0)
    """
    # Simple complexity estimator based on several factors
    word_count = len(text.split())
    
    # Avoid division by zero
    if word_count == 0:
        return 0.0
    
    # Average word length as proxy for vocabulary complexity
    avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
    
    # Sentence length as proxy for syntactic complexity
    sentences = re.split(r'[.!?]', text)
    sentence_count = max(1, len(sentences))
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / sentence_count
    
    # Presence of technical terms (simplified example)
    technical_terms = [
        'algorithm', 'neural', 'network', 'quantum', 'blockchain', 
        'cryptocurrency', 'artificial', 'intelligence', 'machine', 'learning',
        'framework', 'infrastructure', 'microservice', 'architecture', 'database',
        'encryption', 'protocol', 'biotechnology', 'genomics', 'pharmaceutical'
    ]
    tech_term_count = sum(1 for term in technical_terms if term.lower() in text.lower())
    
    # Calculate normalized complexity score (0-1)
    length_factor = min(1.0, word_count / 2000)  # Normalize by expected max length
    word_complexity = min(1.0, avg_word_length / 8)  # Normalize by expected max avg word length
    sentence_complexity = min(1.0, avg_sentence_length / 30)  # Normalize by expected max avg sentence length
    technical_factor = min(1.0, tech_term_count / 10)  # Normalize by expected max technical terms
    
    # Combined score with weights
    complexity = (
        0.3 * length_factor + 
        0.25 * word_complexity + 
        0.25 * sentence_complexity + 
        0.2 * technical_factor
    )
    
    return complexity


def auto_select_model(text: str, available_models: dict, default_model: str, logger=None) -> str:
    """
    Automatically select the appropriate model based on content complexity.
    
    Args:
        text: Article text
        available_models: Dictionary of available models
        default_model: Default model ID
        
    Returns:
        Model identifier
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    complexity = estimate_complexity(text)
    
    # Choose model based on complexity thresholds
    if complexity < 0.3:
        # Simple content - use fastest model
        selected_model = available_models.get("haiku-3.5", default_model)
    elif complexity < 0.7:
        # Moderate complexity - use balanced model
        selected_model = available_models.get("sonnet-3.5", default_model)
    else:
        # Complex content - use most capable model
        selected_model = available_models.get("sonnet-3.7", default_model)
    
    logger.info(f"Auto-selected model based on complexity score {complexity:.2f}: {selected_model}")
    return selected_model