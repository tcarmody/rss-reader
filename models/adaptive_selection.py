"""
Adaptive model selection based on multiple content factors.

Improves upon simple complexity-based selection by considering:
- Content complexity (existing)
- Article length
- Technical domain
- Source quality tier
- User preferences (optional)
"""

import logging
import re
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from models.selection import estimate_complexity

logger = logging.getLogger(__name__)


# Source quality tiers (based on editorial standards and reliability)
SOURCE_TIERS = {
    # Tier 1: Premium sources (always use Sonnet)
    'tier1': {
        'arxiv.org', 'nature.com', 'science.org', 'acm.org', 'ieee.org',
        'nytimes.com', 'wsj.com', 'ft.com', 'economist.com', 'reuters.com',
        'bloomberg.com', 'apnews.com'
    },
    # Tier 2: High-quality sources (prefer Sonnet for technical content)
    'tier2': {
        'techcrunch.com', 'theverge.com', 'arstechnica.com', 'wired.com',
        'technologyreview.com', 'theguardian.com', 'bbc.com', 'cnn.com',
        'theinformation.com', 'stratechery.com'
    },
    # Tier 3: Standard sources (use complexity-based selection)
    'tier3': {
        'medium.com', 'substack.com', 'hackernews.com', 'reddit.com',
        'github.com', 'dev.to', 'stackoverflow.com'
    }
}


# Domain keyword patterns
DOMAIN_PATTERNS = {
    'research': [
        'arxiv', 'paper', 'study', 'research', 'findings', 'methodology',
        'experiment', 'hypothesis', 'peer-reviewed', 'journal', 'publication'
    ],
    'finance': [
        'earnings', 'revenue', 'valuation', 'ipo', 'quarterly', 'profit',
        'market', 'stock', 'investment', 'fiscal', 'financial'
    ],
    'policy': [
        'regulation', 'legislation', 'compliance', 'mandate', 'law',
        'policy', 'government', 'congress', 'senate', 'executive order'
    ],
    'product': [
        'launch', 'feature', 'update', 'release', 'available', 'users',
        'beta', 'version', 'app', 'platform', 'service'
    ],
    'technical': [
        'algorithm', 'architecture', 'implementation', 'framework', 'api',
        'database', 'infrastructure', 'code', 'development', 'engineering'
    ]
}


class AdaptiveModelSelector:
    """
    Multi-factor model selection for optimal quality/cost balance.

    Selection factors:
    1. Content complexity (35% weight)
    2. Technical domain (25% weight)
    3. Article length (15% weight)
    4. Source quality (15% weight)
    5. User preference (10% weight)
    """

    def __init__(self):
        """Initialize the adaptive model selector."""
        self.selection_history = []
        logger.info("Initialized AdaptiveModelSelector")

    def select_model(
        self,
        article: Dict[str, str],
        user_prefs: Optional[Dict[str, str]] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select the most appropriate model for an article.

        Args:
            article: Article dictionary with 'text', 'title', 'url'
            user_prefs: Optional user preferences dict

        Returns:
            Tuple of (model_id, factor_scores)
        """
        text = article.get('text', article.get('content', ''))
        url = article.get('url', '')

        # Calculate all factors
        factors = {
            'complexity': self._calculate_complexity(text),
            'length': self._calculate_length_factor(text),
            'technical_domain': self._detect_domain(text),
            'source_quality': self._get_source_tier_score(url),
            'user_preference': self._get_user_preference_bonus(user_prefs)
        }

        # Weighted scoring
        score = (
            factors['complexity'] * 0.35 +
            factors['technical_domain'] * 0.25 +
            factors['length'] * 0.15 +
            factors['source_quality'] * 0.15 +
            factors['user_preference'] * 0.10
        )

        # Select model based on score
        # Lower threshold (0.50) than original (0.55) to better utilize Sonnet
        if score >= 0.50:
            model = 'claude-sonnet-4-5-latest'
        else:
            model = 'claude-haiku-4-5-latest'

        # Log selection for analysis
        logger.debug(
            f"Model selection: {model} (score={score:.2f}, "
            f"complexity={factors['complexity']:.2f}, "
            f"domain={factors['technical_domain']:.2f})"
        )

        # Store in history (for potential future learning)
        self.selection_history.append({
            'url': url,
            'model': model,
            'score': score,
            'factors': factors
        })

        return model, factors

    def _calculate_complexity(self, text: str) -> float:
        """
        Calculate content complexity score (0-1).

        Uses existing complexity estimation logic.

        Args:
            text: Article text

        Returns:
            Complexity score 0-1
        """
        if not text:
            return 0.0

        return estimate_complexity(text)

    def _calculate_length_factor(self, text: str) -> float:
        """
        Calculate length factor score (0-1).

        Longer articles benefit from better model quality.

        Args:
            text: Article text

        Returns:
            Length score 0-1
        """
        if not text:
            return 0.0

        length = len(text)

        # Normalize to 0-1 (10,000 chars = 1.0)
        score = min(length / 10000, 1.0)

        return score

    def _detect_domain(self, text: str) -> float:
        """
        Detect technical domain and return score (0-1).

        Research and policy domains benefit most from Sonnet.

        Args:
            text: Article text

        Returns:
            Domain score 0-1
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Calculate scores for each domain
        domain_scores = {}
        for domain, keywords in DOMAIN_PATTERNS.items():
            matches = sum(1 for k in keywords if k in text_lower)
            domain_scores[domain] = matches / len(keywords)

        # Research and policy get highest weight
        research_score = domain_scores.get('research', 0.0)
        policy_score = domain_scores.get('policy', 0.0)
        technical_score = domain_scores.get('technical', 0.0)
        finance_score = domain_scores.get('finance', 0.0)

        # Weight: research=1.0, policy=0.9, technical=0.7, finance=0.6
        weighted_score = max(
            research_score * 1.0,
            policy_score * 0.9,
            technical_score * 0.7,
            finance_score * 0.6,
            domain_scores.get('product', 0.0) * 0.3
        )

        return min(weighted_score, 1.0)

    def _get_source_tier_score(self, url: str) -> float:
        """
        Get source quality tier score (0-1).

        Args:
            url: Article URL

        Returns:
            Source tier score 0-1
        """
        if not url:
            return 0.5  # Unknown sources get middle score

        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace('www.', '')

            # Check tiers
            if any(tier_domain in domain for tier_domain in SOURCE_TIERS['tier1']):
                return 1.0  # Premium source
            elif any(tier_domain in domain for tier_domain in SOURCE_TIERS['tier2']):
                return 0.7  # High-quality source
            elif any(tier_domain in domain for tier_domain in SOURCE_TIERS['tier3']):
                return 0.4  # Standard source
            else:
                return 0.5  # Unknown source

        except Exception as e:
            logger.warning(f"Error parsing URL {url}: {e}")
            return 0.5

    def _get_user_preference_bonus(
        self,
        user_prefs: Optional[Dict[str, str]]
    ) -> float:
        """
        Get user preference bonus (0-1).

        Args:
            user_prefs: User preferences dict

        Returns:
            Preference bonus 0-1
        """
        if not user_prefs:
            return 0.5  # Neutral

        pref = user_prefs.get('model_preference', 'auto')

        if pref == 'quality':
            return 1.0  # Always prefer Sonnet
        elif pref == 'speed':
            return 0.0  # Always prefer Haiku
        else:  # 'auto' or 'balanced'
            return 0.5  # No preference

    def get_selection_stats(self) -> Dict[str, any]:
        """
        Get statistics on model selection history.

        Returns:
            Dict with selection statistics
        """
        if not self.selection_history:
            return {
                'total_selections': 0,
                'sonnet_count': 0,
                'haiku_count': 0
            }

        sonnet_count = sum(
            1 for s in self.selection_history
            if 'sonnet' in s['model'].lower()
        )
        haiku_count = len(self.selection_history) - sonnet_count

        avg_score = sum(s['score'] for s in self.selection_history) / len(self.selection_history)

        return {
            'total_selections': len(self.selection_history),
            'sonnet_count': sonnet_count,
            'haiku_count': haiku_count,
            'sonnet_percentage': (sonnet_count / len(self.selection_history)) * 100,
            'average_score': avg_score
        }


# Global instance for reuse
_adaptive_selector = None


def get_adaptive_selector() -> AdaptiveModelSelector:
    """
    Get or create the global adaptive selector instance.

    Returns:
        AdaptiveModelSelector instance
    """
    global _adaptive_selector
    if _adaptive_selector is None:
        _adaptive_selector = AdaptiveModelSelector()
    return _adaptive_selector


def select_model_adaptive(
    article: Dict[str, str],
    user_prefs: Optional[Dict[str, str]] = None
) -> str:
    """
    Convenience function for adaptive model selection.

    Args:
        article: Article dictionary
        user_prefs: Optional user preferences

    Returns:
        Model ID string
    """
    selector = get_adaptive_selector()
    model, _ = selector.select_model(article, user_prefs)
    return model
