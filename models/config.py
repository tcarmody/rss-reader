"""
Centralized configuration for Claude models.
"""

from typing import Dict, Optional, List
import logging
import threading

logger = logging.getLogger(__name__)

# Thread-safe model usage tracking
_model_usage_lock = threading.Lock()
_model_usage_stats = {
    "claude-sonnet-4.5": 0,
    "claude-haiku-4.5": 0,
    "total_selections": 0,
    "complexity_scores": []  # Store for analysis
}


# Complete model identifiers for API calls
MODEL_IDENTIFIERS = {
    "claude-sonnet-4.5": "claude-sonnet-4-5",  # Claude 4.5 Sonnet
    "claude-haiku-4.5": "claude-haiku-4-5"     # Claude 4.5 Haiku
}

# Model properties and characteristics
MODEL_PROPERTIES = {
    "claude-sonnet-4.5": {
        "name": "Claude 4.5 Sonnet",
        "description": "Most intelligent model with enhanced capabilities",
        "strengths": ["Complex analysis", "Advanced reasoning", "Technical content"],
        "speed": "medium",
        "cost": "high",
        "max_tokens": 200000,
        "complexity_threshold": 0.6,
        "recommended_for": ["Complex technical content", "Deep analysis", "Difficult summarization tasks"]
    },
    "claude-haiku-4.5": {
        "name": "Claude 4.5 Haiku",
        "description": "Fast and efficient model with improved performance",
        "strengths": ["Speed", "Efficiency", "Quick responses"],
        "speed": "fast",
        "cost": "low",
        "max_tokens": 200000,
        "complexity_threshold": 0.0,
        "recommended_for": ["Simple content", "Quick summaries", "High-volume processing"]
    }
}

# Default model for fallback
DEFAULT_MODEL = "claude-sonnet-4.5"

# Mapping from shorthand names to full model names
SHORTHAND_MAPPING = {
    "sonnet-4.5": "claude-sonnet-4.5",
    "sonnet": "claude-sonnet-4.5",     # Default Sonnet is 4.5
    "haiku-4.5": "claude-haiku-4.5",
    "haiku": "claude-haiku-4.5"        # Default Haiku is 4.5
}

def select_model_by_complexity(complexity_score: float, url: Optional[str] = None) -> str:
    """
    Select the appropriate model based on content complexity.

    Args:
        complexity_score: Content complexity score (0.0-1.0)
        url: Optional URL for logging context

    Returns:
        Model identifier string
    """
    # Simplified model selection
    if complexity_score >= 0.6:
        # High complexity content goes to the most capable model
        selected_model = "claude-sonnet-4.5"
    else:
        # Low to medium complexity content goes to the faster model
        selected_model = "claude-haiku-4.5"

    # Track model usage statistics
    with _model_usage_lock:
        _model_usage_stats[selected_model] += 1
        _model_usage_stats["total_selections"] += 1
        _model_usage_stats["complexity_scores"].append(complexity_score)

        # Keep only last 1000 complexity scores to avoid unbounded memory
        if len(_model_usage_stats["complexity_scores"]) > 1000:
            _model_usage_stats["complexity_scores"] = _model_usage_stats["complexity_scores"][-1000:]

    # Enhanced logging with URL context
    url_context = f" for {url}" if url else ""
    logger.info(f"Selected {selected_model} (complexity={complexity_score:.2f}){url_context}")

    return MODEL_IDENTIFIERS.get(selected_model, MODEL_IDENTIFIERS[DEFAULT_MODEL])


def get_model_usage_stats() -> Dict:
    """
    Get statistics on model usage distribution.

    Returns:
        Dict with model usage counts, percentages, and complexity distribution
    """
    with _model_usage_lock:
        total = _model_usage_stats["total_selections"]
        if total == 0:
            return {
                "sonnet_count": 0,
                "haiku_count": 0,
                "total_selections": 0,
                "sonnet_percentage": 0.0,
                "haiku_percentage": 0.0,
                "avg_complexity": 0.0,
                "min_complexity": 0.0,
                "max_complexity": 0.0
            }

        sonnet_count = _model_usage_stats["claude-sonnet-4.5"]
        haiku_count = _model_usage_stats["claude-haiku-4.5"]
        complexity_scores = _model_usage_stats["complexity_scores"]

        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0

        return {
            "sonnet_count": sonnet_count,
            "haiku_count": haiku_count,
            "total_selections": total,
            "sonnet_percentage": round(sonnet_count / total * 100, 1),
            "haiku_percentage": round(haiku_count / total * 100, 1),
            "avg_complexity": round(avg_complexity, 3),
            "min_complexity": round(min(complexity_scores), 3) if complexity_scores else 0.0,
            "max_complexity": round(max(complexity_scores), 3) if complexity_scores else 0.0
        }


def log_model_usage_stats() -> None:
    """Log current model usage statistics at INFO level."""
    stats = get_model_usage_stats()
    if stats["total_selections"] == 0:
        logger.info("No model selections recorded yet")
        return

    logger.info(
        f"Model Usage Stats: "
        f"Haiku={stats['haiku_percentage']:.1f}% ({stats['haiku_count']}), "
        f"Sonnet={stats['sonnet_percentage']:.1f}% ({stats['sonnet_count']}), "
        f"Total={stats['total_selections']}, "
        f"Avg Complexity={stats['avg_complexity']:.2f} "
        f"(range: {stats['min_complexity']:.2f}-{stats['max_complexity']:.2f})"
    )


def reset_model_usage_stats() -> None:
    """Reset model usage statistics (for testing or new monitoring periods)."""
    with _model_usage_lock:
        _model_usage_stats["claude-sonnet-4.5"] = 0
        _model_usage_stats["claude-haiku-4.5"] = 0
        _model_usage_stats["total_selections"] = 0
        _model_usage_stats["complexity_scores"] = []
    logger.info("Model usage statistics reset")
