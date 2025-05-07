"""
Centralized configuration for Claude models.
"""

from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Complete model identifiers for API calls
MODEL_IDENTIFIERS = {
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3.5-haiku": "claude-3-5-haiku-20240307"
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
        "complexity_threshold": 0.6,
        "recommended_for": ["Complex technical content", "Deep analysis", "Difficult summarization tasks"]
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "description": "Balanced model for general tasks",
        "strengths": ["General purpose", "Good balance of speed and quality"],
        "speed": "medium",
        "cost": "medium",
        "max_tokens": 200000,
        "complexity_threshold": 0.3,
        "recommended_for": ["General content", "Standard complexity articles", "Typical news summarization"]
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "description": "Excels at writing and complex tasks",
        "strengths": ["Creative writing", "Complex tasks", "High-quality output"],
        "speed": "slow",
        "cost": "high",
        "max_tokens": 200000,
        "complexity_threshold": 0.8,
        "recommended_for": ["Long-form content", "Articles requiring nuanced writing", "Complex narratives"]
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