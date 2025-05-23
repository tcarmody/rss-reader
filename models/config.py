"""
Centralized configuration for Claude models.
"""

from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Complete model identifiers for API calls
MODEL_IDENTIFIERS = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",  # Latest Sonnet version
    "claude-3.5-haiku": "claude-3-5-haiku-latest"     # Latest Haiku version
}

# Model properties and characteristics
MODEL_PROPERTIES = {
    "claude-sonnet-4": {
        "name": "Claude Sonnet 4",
        "description": "Most intelligent model, released May 2025",
        "strengths": ["Complex analysis", "Advanced reasoning", "Technical content"],
        "speed": "medium",
        "cost": "high",
        "max_tokens": 200000,
        "complexity_threshold": 0.6,
        "recommended_for": ["Complex technical content", "Deep analysis", "Difficult summarization tasks"]
    },
    "claude-3.5-haiku": {
        "name": "Claude 3.5 Haiku",
        "description": "Fastest model for daily tasks",
        "strengths": ["Speed", "Efficiency", "Quick responses"],
        "speed": "fast",
        "cost": "low",
        "max_tokens": 200000,
        "complexity_threshold": 0.0,
        "recommended_for": ["Simple content", "Quick summaries", "High-volume processing"]
    }
}

# Default model for fallback
DEFAULT_MODEL = "claude-sonnet-4"

# Mapping from shorthand names to full model names
SHORTHAND_MAPPING = {
    "sonnet-4": "claude-sonnet-4",
    "haiku-3.5": "claude-3.5-haiku",
    "haiku": "claude-3.5-haiku",    # All Haiku references point to 3.5
    "sonnet": "claude-sonnet-4"   # Default Sonnet is 4
}

def select_model_by_complexity(complexity_score: float) -> str:
    """
    Select the appropriate model based on content complexity.
    
    Args:
        complexity_score: Content complexity score (0.0-1.0)
        
    Returns:
        Model identifier string
    """
    # Simplified model selection
    if complexity_score >= 0.6:
        # High complexity content goes to the most capable model
        selected_model = "claude-sonnet-4"
    else:
        # Low to medium complexity content goes to the faster model
        selected_model = "claude-3.5-haiku"
    
    logger.info(f"Selected {selected_model} for content with complexity score {complexity_score:.2f}")
    return MODEL_IDENTIFIERS.get(selected_model, MODEL_IDENTIFIERS[DEFAULT_MODEL])