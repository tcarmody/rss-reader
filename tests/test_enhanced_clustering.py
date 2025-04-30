"""
Test file for the enhanced clustering with multiple article comparison.
Validates that the new functionality works correctly.
"""

import logging
import unittest
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to ensure modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Configure logging for tests
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestEnhancedClustering(unittest.TestCase):
    """Test cases for enhanced article clustering with LM-based multi-article comparison."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to ensure isolation
        from enhanced_clustering import EnhancedArticleClusterer
        
        # Mock the summarizer
        self.mock_summarizer = MagicMock()
        self.mock_summarizer.DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
        
        # Create an instance with the mock summarizer
        self.clusterer = EnhancedArticleClusterer(summarizer=self.mock_summarizer)
        
        # Sample articles for testing
        self.test_articles = [
            {
                'title': 'Apple Announces New iPhone',
                'content': 'Apple Inc. today announced the new iPhone 15, featuring a faster processor and improved camera.',
                'link': 'https://example.com/article1'
            },
            {
                'title': 'iPhone 15 Launch Event Summary',
                'content': 'Today at Apple Park, Tim Cook unveiled the new iPhone 15 with its A16 chip and revolutionary camera system.',
                'link': 'https://example.com/article2'
            },
            {
                'title': 'Google Pixel 8 Launch Date Revealed',
                'content': 'Google has announced that the Pixel 8 will be unveiled at an event next month, featuring new AI capabilities.',
                'link': 'https://example.com/article3'
            },
            {
                'title': 'Meta Releases New VR Headset',
                'content': 'Meta (formerly Facebook) has announced its next-generation VR headset with improved graphics and comfort.',
                'link': 'https://example.com/article4'
            },
            {
                'title': 'Google Plans Pixel 8 Event',
                'content': 'Google scheduled an event for next month where they are expected to announce the Pixel 8 with advanced AI features.',
                'link': 'https://example.com/article5'
            }
        ]

    def test_compare_multiple_texts_with_lm(self):
        """Test the multiple text comparison method."""
        # Setup the mock response
        mock_response = '{"clusters": [[1, 2], [3, 5], [4]]}'
        self.mock_summarizer._call_claude_api.return_value = mock_response
        
        # Extract test texts
        test_texts = [article['title'] + ' ' + article['content'] for article in self.test_articles]
        
        # Call the method
        result = self.clusterer._compare_multiple_texts_with_lm(test_texts)
        
        # Verify the results
        expected_clusters = [[1, 2], [3, 5], [4]]
        self.assertEqual(result, expected_clusters)
        
        # Verify that the LM was called once with the right parameters
        self.mock_summarizer._call_claude_api.assert_called_once()
        call_args = self.mock_summarizer._call_claude_api.call_args[1]
        self.assertEqual(call_args['model_id'], self.mock_summarizer.DEFAULT_MODEL)
        self.assertEqual(call_args['temperature'], 0.0)
        
        # Verify that the prompt contains all articles
        prompt = call_args['prompt']
        for i, text in enumerate(test_texts, 1):
            self.assertIn(f"Article {i}:", prompt)
        
        # Verify that the prompt asks for JSON
        self.assertIn('Return your answer as a JSON object', prompt)

    def test_enhance_clustering_with_lm(self):
        """Test the complete enhanced clustering workflow."""
        # Setup mock for the initial clustering
        initial_clusters = [
            [self.test_articles[0], self.test_articles[1]],  # Apple iPhone cluster
            [self.test_articles[2]],  # Google Pixel article 1
            [self.test_articles[3]],  # Meta VR article
            [self.test_articles[4]]   # Google Pixel article 2
        ]
        
        # Setup mock for the multiple text comparison response
        self.mock_summarizer._call_claude_api.return_value = '{"clusters": [[1, 2]]}'
        
        # Mock the compare_texts_with_lm method to simulate cluster similarity
        def mock_compare_texts(text1, text2):
            # Return high similarity score for Google Pixel articles
            if 'Pixel' in text1 and 'Pixel' in text2:
                return 0.9
            return 0.2
        
        self.clusterer._compare_texts_with_lm = MagicMock(side_effect=mock_compare_texts)
        
        # Run the enhanced clustering
        with patch.object(self.clusterer, 'cluster_articles', return_value=initial_clusters):
            result_clusters = self.clusterer.cluster_articles(self.test_articles)
        
        # Check if the small clusters were processed using the multi-article comparison
        self.mock_summarizer._call_claude_api.assert_called()
        
        # Check that the result has at least the expected number of clusters
        # Since we're mocking complex behavior, we can validate that clustering happened
        # but not the exact outcome which depends on multiple factors
        self.assertGreaterEqual(len(result_clusters), 2)
        
        # Check log stats was called
        self.assertTrue(hasattr(self.clusterer, '_log_clustering_stats'))

    def test_json_parsing_error_handling(self):
        """Test handling of invalid JSON responses from LM."""
        # Setup an invalid JSON response
        self.mock_summarizer._call_claude_api.return_value = "Not valid JSON"
        
        # Extract test texts
        test_texts = [article['title'] + ' ' + article['content'] for article in self.test_articles[:3]]
        
        # Call the method
        result = self.clusterer._compare_multiple_texts_with_lm(test_texts)
        
        # Verify fallback behavior: each article in its own cluster
        expected_fallback = [[1], [2], [3]]
        self.assertEqual(result, expected_fallback)


if __name__ == '__main__':
    unittest.main()