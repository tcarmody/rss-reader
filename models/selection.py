"""
Model selection logic based on content characteristics.
"""

import re
import logging
from typing import Optional, Dict, List, Any
from .config import MODEL_IDENTIFIERS, MODEL_PROPERTIES, DEFAULT_MODEL, SHORTHAND_MAPPING

logger = logging.getLogger(__name__)

def get_model_identifier(model_name: Optional[str] = None) -> str:
    """
    Get the full API model identifier from a model name or shorthand.
    
    Args:
        model_name: Model name, shorthand, or full identifier
        
    Returns:
        Full API model identifier
    """
    if not model_name:
        return MODEL_IDENTIFIERS[DEFAULT_MODEL]
        
    # Check if it's already a full identifier
    if model_name in MODEL_IDENTIFIERS.values():
        return model_name
    
    # Check if it's a known model name
    if model_name in MODEL_IDENTIFIERS:
        return MODEL_IDENTIFIERS[model_name]
    
    # Check if it's a shorthand
    if model_name in SHORTHAND_MAPPING:
        return MODEL_IDENTIFIERS[SHORTHAND_MAPPING[model_name]]
    
    # If not found, log warning and return default
    logger.warning(f"Unknown model '{model_name}', using default: {DEFAULT_MODEL}")
    return MODEL_IDENTIFIERS[DEFAULT_MODEL]

def get_model_properties(model_name: str) -> dict:
    """
    Get properties for a given model.
    
    Args:
        model_name: Model name, shorthand, or identifier
        
    Returns:
        Dictionary of model properties
    """
    # Normalize to standard model name
    if model_name in SHORTHAND_MAPPING:
        model_name = SHORTHAND_MAPPING[model_name]
    elif model_name in MODEL_IDENTIFIERS.values():
        # Find the key for this identifier
        for key, value in MODEL_IDENTIFIERS.items():
            if value == model_name:
                model_name = key
                break
    
    return MODEL_PROPERTIES.get(model_name, {})

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

def select_model_by_complexity(complexity_score: float) -> str:
    """
    Select the appropriate model based on content complexity.
    
    Args:
        complexity_score: Content complexity score (0.0-1.0)
        
    Returns:
        Model identifier string
    """
    # Sort models by complexity threshold
    sorted_models = sorted(
        MODEL_PROPERTIES.items(),
        key=lambda x: x[1].get('complexity_threshold', 0)
    )
    
    # Find the most appropriate model
    selected_model = DEFAULT_MODEL
    for model_name, properties in sorted_models:
        if complexity_score >= properties.get('complexity_threshold', 0):
            selected_model = model_name
    
    logger.info(f"Selected {selected_model} for content with complexity score {complexity_score:.2f}")
    return MODEL_IDENTIFIERS.get(selected_model, MODEL_IDENTIFIERS[DEFAULT_MODEL])