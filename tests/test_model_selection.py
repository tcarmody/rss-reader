"""
Tests for model selection functionality.
"""

import unittest
from unittest.mock import patch

from models.selection import (
    estimate_complexity,
    select_model_by_complexity,
    get_model_identifier,
    get_model_properties
)
from models.config import MODEL_PROPERTIES

class TestModelSelection(unittest.TestCase):
    """Test the model selection functions."""

    def test_estimate_complexity(self):
        """Test complexity estimation."""
        # Empty text
        self.assertEqual(estimate_complexity(""), 0.0)

        # Simple text
        simple_text = "This is a simple test. It has short words and sentences."
        simple_score = estimate_complexity(simple_text)
        self.assertLess(simple_score, 0.3)

        # Complex text
        complex_text = """
        The implementation of artificial intelligence algorithms requires careful consideration
        of computational complexity, network architecture, and parameter optimization.
        Quantum computing may provide significant advantages for certain machine learning
        frameworks by leveraging superposition and entanglement for parallel processing.
        """
        complex_score = estimate_complexity(complex_text)
        self.assertGreater(complex_score, 0.5)

        # Very complex text should have higher score
        self.assertGreater(complex_score, simple_score)

    def test_select_model_by_complexity(self):
        """Test model selection based on complexity."""
        # Very simple content (< 0.6) should use Haiku
        haiku_model = select_model_by_complexity(0.1)
        self.assertTrue("haiku" in haiku_model.lower())

        # Medium complexity (< 0.6) should still use Haiku
        medium_model = select_model_by_complexity(0.4)
        self.assertTrue("haiku" in medium_model.lower())

        # High complexity (>= 0.6) should use Sonnet
        sonnet_model = select_model_by_complexity(0.7)
        self.assertTrue("sonnet" in sonnet_model.lower())

        # Boundary case: exactly 0.6 should use Sonnet
        boundary_model = select_model_by_complexity(0.6)
        self.assertTrue("sonnet" in boundary_model.lower())

    def test_get_model_identifier(self):
        """Test model identifier resolution."""
        # Default model when None
        self.assertTrue(get_model_identifier(None).startswith("claude-"))

        # Shorthand
        self.assertTrue("haiku" in get_model_identifier("haiku").lower())
        self.assertTrue("sonnet" in get_model_identifier("sonnet").lower())

        # Full name
        self.assertEqual(
            get_model_identifier("claude-haiku-4.5"),
            get_model_identifier("haiku-4.5")
        )

        # Known full identifier should be returned as-is
        full_id = "claude-sonnet-4-5"
        self.assertEqual(get_model_identifier(full_id), full_id)

        # Unknown identifier should return default
        unknown_id = "claude-unknown-model"
        result = get_model_identifier(unknown_id)
        self.assertTrue(result.startswith("claude-"))

    def test_get_model_properties(self):
        """Test getting model properties."""
        # Get properties by shorthand
        haiku_props = get_model_properties("haiku")
        self.assertIsInstance(haiku_props, dict)
        self.assertIn("name", haiku_props)
        self.assertIn("speed", haiku_props)

        # Get properties by full name
        sonnet_props = get_model_properties("claude-sonnet-4.5")
        self.assertIsInstance(sonnet_props, dict)
        self.assertIn("name", sonnet_props)

        # Unknown model should return empty dict
        unknown_props = get_model_properties("unknown-model")
        self.assertEqual(unknown_props, {})


if __name__ == "__main__":
    unittest.main()
