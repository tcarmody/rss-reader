"""
Language Model based Cluster Analysis module with performance optimizations.

This module provides utilities for analyzing and refining article clusters
using language models. It contains helper functions for both pairwise and
multi-article clustering comparisons.
"""

import logging
import re
import json
import os
import sys
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

# Add the parent directory to sys.path to ensure modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


class LMClusterAnalyzer:
    """
    Performs cluster analysis and refinement using language models.
    
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
                    getattr(self.summarizer, 'DEFAULT_MODEL', 'claude-3-7-sonnet-latest'))
        
        # Access through original attribute for FastArticleSummarizer
        elif hasattr(self.summarizer, 'original') and hasattr(self.summarizer.original, '_call_claude_api'):
            return (self.summarizer.original._call_claude_api,
                    getattr(self.summarizer.original, 'DEFAULT_MODEL', 'claude-3-7-sonnet-latest'))
                    
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
    
    def cluster_articles(self, articles: List[Dict[str, str]], 
                        text_extractor=None, 
                        similarity_threshold: float = 0.7,
                        max_comparisons: int = 50) -> List[List[int]]:
        """
        Cluster a list of articles based on their content similarity.
        Implements optimizations to reduce API calls and improve performance.
        
        Args:
            articles: List of article dictionaries
            text_extractor: Function to extract text from articles (defaults to title+content)
            similarity_threshold: Minimum similarity score to consider articles related
            max_comparisons: Maximum number of comparisons to perform
            
        Returns:
            List of clusters, each containing indices of related articles
        """
        # Check if it's a tiny list - no need for complex clustering
        if len(articles) <= 2:
            if len(articles) == 2:
                # Check if these two articles should be clustered
                text1 = text_extractor(articles[0]) if text_extractor else articles[0].get('content', '')
                text2 = text_extractor(articles[1]) if text_extractor else articles[1].get('content', '')
                
                similarity = self.compare_article_pair(text1, text2)
                if similarity >= similarity_threshold:
                    return [[1, 2]]  # One cluster with both articles
            
            # Default to separate clusters
            return [[i+1] for i in range(len(articles))]
        
        # Check if we have too many articles for LM-based clustering
        if len(articles) > max_comparisons:
            # Use pairwise clustering for efficiency
            return self._pairwise_clustering(articles, text_extractor, similarity_threshold, max_comparisons)
        
        # For smaller sets, we can use the LM to cluster them all at once
        api_caller, model_id = self._get_api_caller()
        
        if not api_caller or not self._handle_rate_limits():
            self.logger.warning("No suitable API call method available or limit reached. Using pairwise clustering.")
            return self._pairwise_clustering(articles, text_extractor, similarity_threshold, max_comparisons)
        
        # Default text extractor
        if text_extractor is None:
            text_extractor = lambda a: f"{a.get('title', '')} {a.get('content', '')}"
        
        # Check cache first
        cache_key = hash(str([text_extractor(a)[:100] for a in articles]))
        if cache_key in self.cluster_cache:
            self.logger.info(f"Using cached clustering result for {len(articles)} articles")
            return self.cluster_cache[cache_key]
        
        try:
            # Extract and prepare text for each article
            article_texts = []
            for article in articles:
                text = text_extractor(article)
                # Truncate for API efficiency - even shorter now
                article_texts.append(text[:600] if len(text) > 600 else text)
            
            # Create prompt for multi-article clustering - optimized for brevity
            articles_text = ""
            for i, text in enumerate(article_texts, 1):
                # Use just first part of each article for faster processing
                preview = text[:300] + "..." if len(text) > 300 else text
                articles_text += f"Article {i}: {preview}\n\n"
                
            prompt = (
                "Group these articles into clusters based on their topics:\n\n"
                f"{articles_text}\n"
                "Return your answer as a JSON object with this exact format:\n"
                "{\"clusters\": [[1, 3, 5], [2, 4], [6]]}\n"
                "Where each inner array represents a cluster with article numbers.\n"
                "Articles should be clustered if they cover the same story or highly related topics.\n"
                "Return ONLY the JSON object, no other text."
            )
            
            # Call the LM API with lower token limit
            response = api_caller(
                model_id=model_id,
                prompt=prompt,
                temperature=0.0,
                max_tokens=100
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    self.logger.info(f"Successfully parsed LM clustering result: {result}")
                    clusters = result.get('clusters', [[i+1] for i in range(len(articles))])
                    
                    # Cache the result
                    self.cluster_cache[cache_key] = clusters
                    return clusters
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")
            else:
                self.logger.warning(f"No JSON found in LM response: {response}")
                
            # Fallback to pairwise clustering
            return self._pairwise_clustering(articles, text_extractor, similarity_threshold, max_comparisons)
                
        except Exception as e:
            self.logger.error(f"Error in multi-article clustering: {str(e)}")
            # Fallback to pairwise clustering
            return self._pairwise_clustering(articles, text_extractor, similarity_threshold, max_comparisons)
            
    def _pairwise_clustering(self, articles, text_extractor, similarity_threshold, max_comparisons):
        """
        Perform pairwise clustering using optimized comparison strategy.
        
        Args:
            articles: List of article dictionaries
            text_extractor: Function to extract text from articles
            similarity_threshold: Minimum similarity score for clustering
            max_comparisons: Maximum number of comparisons to perform
            
        Returns:
            List of clusters, each containing indices of related articles
        """
        self.logger.info(f"Using pairwise clustering for {len(articles)} articles")
        
        # Default text extractor
        if text_extractor is None:
            text_extractor = lambda a: f"{a.get('title', '')} {a.get('content', '')}"
            
        # Extract texts first
        texts = [text_extractor(a) for a in articles]
        
        # Initial clusters - each article in its own cluster
        clusters = {i: [i+1] for i in range(len(articles))}
        
        # Use a Union-Find data structure for efficient clustering
        parent = list(range(len(articles)))
        
        def find(x):
            # Find root with path compression
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
            
        def union(x, y):
            # Union by rank
            parent[find(x)] = find(y)
        
        # Prioritize comparisons by focusing on titles first
        title_matches = []
        
        for i in range(len(articles)):
            title_i = articles[i].get('title', '').lower()
            title_words_i = set(re.findall(r'\b[A-Za-z]{4,}\b', title_i))
            
            for j in range(i+1, len(articles)):
                title_j = articles[j].get('title', '').lower()
                title_words_j = set(re.findall(r'\b[A-Za-z]{4,}\b', title_j))
                
                # Calculate Jaccard similarity of title words
                intersection = len(title_words_i.intersection(title_words_j))
                union_size = len(title_words_i.union(title_words_j))
                
                if union_size > 0:
                    title_sim = intersection / union_size
                    if title_sim > 0.3:  # Lower threshold for prioritization
                        title_matches.append((i, j, title_sim))
        
        # Sort by title similarity (highest first)
        title_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Limit number of comparisons
        comparisons_done = 0
        max_total_comparisons = min(max_comparisons, len(articles) * 3)  # Reasonable upper bound
        
        # First process title matches
        for i, j, title_sim in title_matches:
            if comparisons_done >= max_total_comparisons:
                break
                
            # If already in same cluster, skip
            if find(i) == find(j):
                continue
                
            # Compare full content
            similarity = self.compare_article_pair(texts[i], texts[j])
            comparisons_done += 1
            
            if similarity >= similarity_threshold:
                union(i, j)
                self.logger.debug(f"Clustered articles {i+1} and {j+1} with similarity {similarity:.2f}")
        
        # Create final clusters based on Union-Find results
        final_clusters = {}
        for i in range(len(articles)):
            root = find(i)
            if root not in final_clusters:
                final_clusters[root] = []
            final_clusters[root].append(i+1)  # Add 1 for 1-based indexing
            
        return list(final_clusters.values())
    
    def analyze_cluster_similarity(self, cluster1: List[Dict[str, str]], 
                                 cluster2: List[Dict[str, str]],
                                 text_extractor=None) -> float:
        """
        Analyze the similarity between two clusters with optimized performance.
        
        Args:
            cluster1: First cluster of articles
            cluster2: Second cluster of articles
            text_extractor: Function to extract text from articles
            
        Returns:
            float: Similarity score between the clusters (0-1)
        """
        if not cluster1 or not cluster2:
            return 0.0
            
        # Check if we've reached the API call limit
        if self.api_call_count >= self.max_api_calls:
            # Use fast title-based similarity
            return self._fast_cluster_similarity(cluster1, cluster2)
            
        if text_extractor is None:
            text_extractor = lambda a: f"{a.get('title', '')} {a.get('content', '')}"
            
        # Create representative text for each cluster (up to 3 articles)
        # Only use titles for faster processing
        cluster1_titles = [a.get('title', '') for a in cluster1[:3]]
        cluster2_titles = [a.get('title', '') for a in cluster2[:3]]
        
        cluster1_text = " | ".join(cluster1_titles)
        cluster2_text = " | ".join(cluster2_titles)
        
        # Compare the cluster representations
        return self.compare_article_pair(cluster1_text, cluster2_text)
    
    def _fast_cluster_similarity(self, cluster1, cluster2):
        """
        Calculate cluster similarity using a fast title-based approach.
        
        Args:
            cluster1: First cluster of articles
            cluster2: Second cluster of articles
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Extract all titles
        titles1 = [a.get('title', '').lower() for a in cluster1]
        titles2 = [a.get('title', '').lower() for a in cluster2]
        
        # Extract significant words from all titles
        words1 = set()
        words2 = set()
        
        for title in titles1:
            words1.update(re.findall(r'\b[A-Za-z]{4,}\b', title))
            
        for title in titles2:
            words2.update(re.findall(r'\b[A-Za-z]{4,}\b', title))
            
        # Calculate Jaccard similarity
        if words1 and words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union)
        else:
            return 0.0
    
    def extract_cluster_topics(self, cluster: List[Dict[str, str]], max_topics: int = 5) -> List[str]:
        """
        Extract main topics from a cluster of articles with optimizations.
        
        Args:
            cluster: Cluster of related articles
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of topic strings
        """
        api_caller, model_id = self._get_api_caller()
        
        if not api_caller or not cluster or not self._handle_rate_limits():
            # Use keyword extraction as fallback
            return self._extract_keywords_fast(cluster, max_topics)
            
        try:
            # Prepare representative text from the cluster
            # Use titles for faster processing and reduced token usage
            titles = [article.get('title', '') for article in cluster[:5]]
            combined_titles = " | ".join(titles)
            
            # Create simplified prompt for topic extraction
            prompt = (
                "Extract 5 key topics from these related news headlines as a JSON array:\n\n"
                f"{combined_titles}\n\n"
                "Format: ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']\n"
                "Return ONLY the JSON array."
            )
            
            # Call the LM API with reduced token limit
            response = api_caller(
                model_id=model_id,
                prompt=prompt,
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=50
            )
            
            # Extract JSON array from response
            array_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                try:
                    topics = json.loads(json_str)
                    return topics[:max_topics]
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse topics JSON")
            else:
                self.logger.warning(f"No JSON array found in topics response")
                
            # Fallback: extract keywords with simple text processing
            return self._extract_keywords_fast(cluster, max_topics)
                
        except Exception as e:
            self.logger.error(f"Error extracting cluster topics: {str(e)}")
            return self._extract_keywords_fast(cluster, max_topics)
            
    def _extract_keywords_fast(self, cluster: List[Dict[str, str]], max_keywords: int = 5) -> List[str]:
        """
        Extract keywords from a cluster using a fast algorithm optimized for performance.
        
        Args:
            cluster: Cluster of articles
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Collect all titles and first paragraphs
        text = ""
        for article in cluster:
            title = article.get('title', '')
            content = article.get('content', '')
            
            # Add title with more weight (repeat it)
            text += title + " " + title + " "
            
            # Add first paragraph only
            if content:
                paragraphs = content.split('\n\n')
                if paragraphs:
                    text += paragraphs[0] + " "
        
        # Extract significant words (4+ chars), ignoring common stopwords
        stopwords = {
            'this', 'that', 'with', 'from', 'what', 'when', 'where', 'which', 'about', 
            'have', 'will', 'your', 'their', 'there', 'they', 'these', 'those', 'some', 
            'were', 'after', 'before', 'could', 'should', 'would', 'says', 'said', 'year',
            'years', 'month', 'months', 'week', 'weeks', 'day', 'days', 'time', 'times'
        }
        
        words = re.findall(r'\b[A-Za-z][A-Za-z]{3,}\b', text.lower())
        word_counts = {}
        
        for word in words:
            if word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:max_keywords]]
    
    def reset_api_counter(self):
        """Reset the API call counter for a new batch of clustering."""
        self.api_call_count = 0
        self.logger.info("Reset API call counter")


# Helper function to create a cluster analyzer
def create_cluster_analyzer(summarizer=None):
    """
    Create a language model based cluster analyzer with optimized performance.
    
    Args:
        summarizer: Summarizer instance with LM access
        
    Returns:
        LMClusterAnalyzer instance
    """
    return LMClusterAnalyzer(summarizer=summarizer)