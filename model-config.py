# model_config.py

"""
Model configuration for Claude API as of May 2025.

This module defines the available Claude models and their properties.
Current as of: May 3, 2025
"""

# Complete model identifiers for API calls
MODEL_IDENTIFIERS = {
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",  # Using placeholder date - update with actual release date
    "claude-3-opus": "claude-3-opus-20240229",  # Using placeholder date - update with actual release date
    "claude-3.5-haiku": "claude-3-5-haiku-20240307"  # Using placeholder date - update with actual release date
}

# Model properties and characteristics
MODEL_PROPERTIES = {
    "claude-3.7-sonnet": {
        "name": "Claude 3.7 Sonnet",
        "description": "Most intelligent model, released February 2025",
        "strengths": ["Complex analysis", "Advanced reasoning", "Technical content"],
        "speed": "medium",
        "cost": "high",
        "max_tokens": 200000,
        "recommended_for": ["Complex technical content", "Deep analysis", "Difficult summarization tasks"]
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "description": "Balanced model for general tasks",
        "strengths": ["General purpose", "Good balance of speed and quality"],
        "speed": "medium",
        "cost": "medium",
        "max_tokens": 200000,
        "recommended_for": ["General content", "Standard complexity articles", "Typical news summarization"]
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "description": "Excels at writing and complex tasks",
        "strengths": ["Creative writing", "Complex tasks", "High-quality output"],
        "speed": "slow",
        "cost": "high",
        "max_tokens": 200000,
        "recommended_for": ["Long-form content", "Articles requiring nuanced writing", "Complex narratives"]
    },
    "claude-3.5-haiku": {
        "name": "Claude 3.5 Haiku",
        "description": "Fastest model for daily tasks",
        "strengths": ["Speed", "Efficiency", "Quick responses"],
        "speed": "fast",
        "cost": "low",
        "max_tokens": 200000,
        "recommended_for": ["Simple content", "Quick summaries", "High-volume processing"]
    }
}

# Default model for fallback
DEFAULT_MODEL = "claude-3.5-sonnet"

# Mapping from shorthand names to full model names
SHORTHAND_MAPPING = {
    "sonnet-3.7": "claude-3.7-sonnet",
    "sonnet-3.5": "claude-3.5-sonnet",
    "opus": "claude-3-opus",
    "haiku": "claude-3.5-haiku",
    "haiku-3.5": "claude-3.5-haiku",
    "sonnet": "claude-3.5-sonnet",  # Default to 3.5 for backward compatibility
}

def get_model_identifier(model_name: str) -> str:
    """
    Get the full API model identifier from a model name or shorthand.
    
    Args:
        model_name: Model name, shorthand, or full identifier
        
    Returns:
        Full API model identifier
    """
    # Check if it's already a full identifier
    if model_name in MODEL_IDENTIFIERS.values():
        return model_name
    
    # Check if it's a known model name
    if model_name in MODEL_IDENTIFIERS:
        return MODEL_IDENTIFIERS[model_name]
    
    # Check if it's a shorthand
    if model_name in SHORTHAND_MAPPING:
        return MODEL_IDENTIFIERS[SHORTHAND_MAPPING[model_name]]
    
    # If not found, assume it's a custom identifier and return as-is
    return model_name

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

# Export available models for use in other modules
AVAILABLE_MODELS = MODEL_IDENTIFIERS
