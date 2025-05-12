"""
Centralized configuration for Claude models.
"""

from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Complete model identifiers for API calls
MODEL_IDENTIFIERS = {
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3.5-haiku": "claude-3-5-haiku-20240307",
    "claude-3-haiku": "claude-3-haiku-20240307"  # Added for backward compatibility
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
    "claude-3.5-haiku": {
        "name": "Claude 3.5 Haiku",
        "description": "Fastest model for daily tasks",
        "strengths": ["Speed", "Efficiency", "Quick responses"],
        "speed": "fast",
        "cost": "low",
        "max_tokens": 200000,
        "complexity_threshold": 0.0,
        "recommended_for": ["Simple content", "Quick summaries", "High-volume processing"]
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku",
        "description": "Original fast model for basic tasks",
        "strengths": ["Speed", "Efficiency", "Basic summarization"],
        "speed": "fast",
        "cost": "low",
        "max_tokens": 200000,
        "complexity_threshold": 0.0,
        "recommended_for": ["Simple content", "Basic summaries", "High-volume processing"]
    }
}

# Default model for fallback
DEFAULT_MODEL = "claude-3.7-sonnet"

# Mapping from shorthand names to full model names
SHORTHAND_MAPPING = {
    "sonnet-3.7": "claude-3.7-sonnet",
    "haiku-3.5": "claude-3.5-haiku",
    "haiku-3": "claude-3-haiku",
    "haiku": "claude-3.5-haiku",  # Map generic "haiku" to 3.5 version
    "sonnet": "claude-3.7-sonnet"  # Default for backward compatibility
}