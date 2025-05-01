"""
Optimized Language Model based Cluster Analysis module.

This module provides utilities for analyzing and refining article clusters
using language models. It contains helper functions for both pairwise and
multi-article clustering comparisons with performance optimizations.
"""

import logging
import re
import json
import os
import sys
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# Add the parent directory to sys.path to ensure modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class LMClusterAnalyzer:
    """
    Performs cluster analysis and refinement using language models with optimizations.
    
    This class provides methods to:
    1. Compare pairs of articles for similarity
    2. Cluster multiple articles based on their content
    3. Evaluate existing clusters for potential merging
    4. Extract key topic information from clusters
    """
    
    def __init__(self, summarizer=None, logger=None):
        """
        Initialize the cluster analyzer.
        
        Args:
            summarizer: An instance that provides access to the language model
            logger: Optional logger instance
        """
        self.summarizer = summarizer
        self.logger = logger or logging.getLogger("lm_cluster_analyzer")
        
        # Add result caching to reduce API calls
        self.comparison_cache = {}
        self.cluster_cache = {}
        
        # Set API call limit
        self.api_call_count = 0
        self.max_api_calls = int(os.environ.get('MAX_LM_API_CALLS', '50'))
        
    def _get_api_caller(self):
        """
        Get the appropriate method to call the Claude API based on summarizer type.
        
        Returns:
            tuple: (api_method, model_id) or (None, None) if no suitable method found
        """
        if not self.summarizer:
            return None, None
            
        # Direct API caller for ArticleSummarizer
        if hasattr(self.summarizer, '_call_claude_api'):
            return (self.summarizer._call_claude_api, 
                    getattr(self.summarizer, 'DEFAULT_MODEL', 'claude-3-7-sonnet-20250219'))
        
        # Access through original attribute for FastArticleSummarizer
        elif hasattr(self.summarizer, 'original') and hasattr(self.summarizer.original, '_call_claude_api'):
            return (self.summarizer.original._call_claude_api,
                    getattr(self.summarizer.original, 'DEFAULT_MODEL', 'claude-3-7-sonnet-20250219'))
                    
        # No suitable API caller found
        return None, None

    def _handle_rate_limits(self):
        """
        Handle rate limiting by implementing a simple delay
        rather than using the rate_limiter object which might not be available
        """
        # Check if we've hit the API call limit
        if self.api_call_count >= self.max_api_calls:
            self.logger.warning(f"Reached API call limit of {self.max_api_calls}. Using fast fallback methods.")
            return False
            
        # Increment API call counter
        self.api_call_count += 1
        
        # Simple delay to avoid rate limits
        time.sleep(0.5)
        return True
        
    def _get_comparison_cache_key(self, text1, text2):
        """Generate a cache key for text comparison."""
        # Use a consistent order to ensure same key regardless of argument order
        if hash(text1) < hash(text2):
            combined = text1[:300] + "||" + text2[:300]
        else:
            combined = text2[:300] + "||" + text1[:300]
            
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
        
    def _fast_text_similarity(self, text1, text2):
        """
        Calculate text similarity using a fast lexical method as fallback.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Extract significant words (4+ chars), ignoring common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        def tokenize(text):
            words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
            return [w for w in words if w not in stopwords]
            
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        # Handle empty token lists
        if not tokens1 or not tokens2:
            return 0.0
            
        # Calculate Jaccard similarity for speed
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union > 0:
            return intersection / union
        else:
            return 0.0

    def compare_article_pair(self, text1: str, text2: str) -> float:
        """
        Compare two article texts and return a similarity score.
        Uses caching to avoid redundant API calls.
        
        Args:
            text1: First article text
            text2: Second article text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Check cache first
        cache_key = self._get_comparison_cache_key(text1, text2)
        if cache_key in self.comparison_cache:
            self.logger.debug(f"Using cached similarity result for {cache_key[:8]}")
            return self.comparison_cache[cache_key]
        
        api_caller, model_id = self._get_api_caller()
        
        if not api_caller or not self._handle_rate_limits():
            # Use fast lexical similarity as fallback
            similarity = self._fast_text_similarity(text1, text2)
            
            # Cache the result
            self.comparison_cache[cache_key] = similarity
            return similarity
            
        try:
            # Limit text length for API efficiency
            text1 = text1[:800]
            text2 = text2[:800]
            
            # Create comparison prompt - optimized for shorter prompt
            prompt = (
                "Task: Rate the similarity of these two texts on a scale from 0 to 1.\n\n"
                f"Text 1: {text1}\n\n"
                f"Text 2: {text2}\n\n"
                "Return only the similarity score as a number between 0 and 1."
            )
            
            # Call the LM API
            response = api_caller(
                model_id=model_id,
                prompt=prompt,
                temperature=0.0,
                max_tokens=5  # Only need a number
            )
            
            # Extract the numerical score from the response
            score_match = re.search(r'([0-9]\.[0-9]|[01])', response)
            if score_match:
                score = float(score_match.group(1))
                
                # Cache the result
                self.comparison_cache[cache_key] = score
                return score
            else:
                self.logger.warning(f"Could not extract similarity score from response: {response}")
                
                # Fallback to fast method
                similarity = self._fast_text_similarity(text1, text2)
                self.comparison_cache[cache_key] = similarity
                return similarity
        except Exception as e:
            self.logger.error(f"Error comparing article pair: {str(e)}")
            
            # Fallback to fast method on error
            similarity = self._fast_text_similarity(text1, text2)
            self.comparison_cache[cache_key] = similarity
            return similarity