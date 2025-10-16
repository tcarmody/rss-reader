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
        # Very simple content should use Haiku
        haiku_model = select_model_by_complexity(0.1)
        self.assertTrue("haiku" in haiku_model.lower())
        
        # Medium complexity should use Sonnet
        sonnet_model = select_model_by_complexity(0.4)
        self.assertTrue("sonnet" in sonnet_model.lower())
        
        # High complexity should use Sonnet 4.5
        complex_model = select_model_by_complexity(0.7)
        self.assertTrue("sonnet" in complex_model.lower() and "4" in complex_model.lower())
    
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

        # Full identifier should be returned as-is
        full_id = "claude-sonnet-4-5-latest"
        self.assertEqual(get_model_identifier(full_id), full_id)
    
    def test_get_model_properties(self):
        """Test getting model properties."""
        # Get properties by shorthand
        haiku_props = get_model_properties("haiku")
        self.assertEqual(haiku_props, )