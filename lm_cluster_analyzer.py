"""
Language Model based Cluster Analysis module.

This module provides utilities for analyzing and refining article clusters
using language models. It contains helper functions for both pairwise and
multi-article clustering comparisons.
"""

import logging
import re
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Union

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
        self.logger = logger or logging.getLogger(__name__)
        
    def compare_article_pair(self, text1: str, text2: str) -> float:
        """
        Compare two article texts and return a similarity score.
        
        Args:
            text1: First article text
            text2: Second article text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not self.summarizer:
            self.logger.warning("No summarizer available for LM comparisons")
            return 0.0
            
        try:
            # Limit text length for API efficiency
            text1 = text1[:1000]
            text2 = text2[:1000]
            
            # Create comparison prompt
            prompt = (
                "Task: Evaluate the similarity of the following two news articles based on their topic and content.\n\n"
                f"Article 1: {text1}\n\n"
                f"Article 2: {text2}\n\n"
                "Evaluate the topical similarity on a scale from 0 to 1, where:\n"
                "- 0: Completely different topics\n"
                "- 0.5: Somewhat related topics\n"
                "- 1: Same topic and focus\n\n"
                "Return only a number between 0 and 1 representing the similarity score."
            )
            
            # Call the LM
            if hasattr(self.summarizer, '_call_claude_api'):
                response = self.summarizer._call_claude_api(
                    model_id=self.summarizer.DEFAULT_MODEL,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=10
                )
                
                # Extract the numerical score
                score_match = re.search(r'([0-9]\.[0-9]|[01])', response)
                if score_match:
                    return float(score_match.group(1))
                else:
                    self.logger.warning(f"Could not extract similarity score from response: {response}")
                    return 0.0
            else:
                self.logger.warning("API call method not available")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error comparing article pair: {str(e)}")
            return 0.0
    
    def cluster_articles(self, articles: List[Dict[str, str]], 
                        text_extractor=None, 
                        similarity_threshold: float = 0.7) -> List[List[int]]:
        """
        Cluster a list of articles based on their content similarity.
        
        Args:
            articles: List of article dictionaries
            text_extractor: Function to extract text from articles (defaults to title+content)
            similarity_threshold: Minimum similarity score to consider articles related
            
        Returns:
            List of clusters, each containing indices of related articles
        """
        if not self.summarizer:
            self.logger.warning("No summarizer available for LM clustering")
            return [[i] for i in range(len(articles))]
        
        # Default text extractor
        if text_extractor is None:
            text_extractor = lambda a: f"{a.get('title', '')} {a.get('content', '')}"
        
        try:
            # Extract and prepare text for each article
            article_texts = []
            for article in articles:
                text = text_extractor(article)
                # Truncate for API efficiency
                article_texts.append(text[:800] if len(text) > 800 else text)
            
            # Create prompt for multi-article clustering
            articles_text = ""
            for i, text in enumerate(article_texts, 1):
                articles_text += f"Article {i}: {text}\n\n"
                
            prompt = (
                "Task: Group the following news articles into clusters based on their topics and content.\n\n"
                f"{articles_text}\n"
                "Instructions:\n"
                "1. Group articles that cover the same news story or highly related topics (similarity > 0.7).\n"
                "2. Articles covering different aspects of the same general topic should be in separate clusters.\n"
                "3. Return your answer as a JSON object with cluster assignments:\n"
                "   {\"clusters\": [[1, 3, 5], [2, 4], [6]]}\n"
                "   Where each inner array represents a cluster, and the numbers are article indices.\n"
                "4. Articles that don't belong to any cluster should be in their own single-item cluster.\n"
                "5. IMPORTANT: Return ONLY the JSON object, nothing else.\n"
            )
            
            # Call the LM API
            response = self.summarizer._call_claude_api(
                model_id=self.summarizer.DEFAULT_MODEL,
                prompt=prompt,
                temperature=0.0,
                max_tokens=200
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    self.logger.info(f"Successfully parsed LM clustering result: {result}")
                    return result.get('clusters', [[i+1] for i in range(len(articles))])
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")
            else:
                self.logger.warning(f"No JSON found in LM response: {response}")
                
            # Return fallback: each article in its own cluster
            return [[i+1] for i in range(len(articles))]
                
        except Exception as e:
            self.logger.error(f"Error in multi-article clustering: {str(e)}")
            return [[i+1] for i in range(len(articles))]
    
    def analyze_cluster_similarity(self, cluster1: List[Dict[str, str]], 
                                 cluster2: List[Dict[str, str]],
                                 text_extractor=None) -> float:
        """
        Analyze the similarity between two clusters.
        
        Args:
            cluster1: First cluster of articles
            cluster2: Second cluster of articles
            text_extractor: Function to extract text from articles
            
        Returns:
            float: Similarity score between the clusters (0-1)
        """
        if not cluster1 or not cluster2:
            return 0.0
            
        if text_extractor is None:
            text_extractor = lambda a: f"{a.get('title', '')} {a.get('content', '')}"
            
        # Create representative text for each cluster (up to 3 articles)
        cluster1_texts = [text_extractor(a)[:300] for a in cluster1[:3]]
        cluster2_texts = [text_extractor(a)[:300] for a in cluster2[:3]]
        
        cluster1_text = " | ".join(cluster1_texts)
        cluster2_text = " | ".join(cluster2_texts)
        
        # Compare the cluster representations
        return self.compare_article_pair(cluster1_text, cluster2_text)
    
    def extract_cluster_topics(self, cluster: List[Dict[str, str]], max_topics: int = 5) -> List[str]:
        """
        Extract main topics from a cluster of articles.
        
        Args:
            cluster: Cluster of related articles
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of topic strings
        """
        if not cluster or not self.summarizer:
            return []
            
        try:
            # Prepare representative text from the cluster
            article_texts = []
            for article in cluster[:5]:  # Use up to 5 articles
                title = article.get('title', '')
                content = article.get('content', '')
                snippet = f"{title}\n{content[:500]}" if content else title
                article_texts.append(snippet)
                
            combined_text = "\n\n".join(article_texts)
            
            # Create prompt for topic extraction
            prompt = (
                "Extract the main topics from these related news articles. "
                "Return exactly 5 key topics as a JSON array of strings:\n\n"
                f"{combined_text}\n\n"
                "Example format:\n"
                "['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']\n\n"
                "IMPORTANT: Return ONLY the JSON array, nothing else."
            )
            
            # Call the LM API
            response = self.summarizer._call_claude_api(
                model_id=self.summarizer.DEFAULT_MODEL,
                prompt=prompt,
                temperature=0.2,  # Small amount of temperature for more natural topics
                max_tokens=100
            )
            
            # Extract JSON array from response
            array_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                try:
                    topics = json.loads(json_str)
                    return topics[:max_topics]
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse topics JSON: {json_str}")
            else:
                self.logger.warning(f"No JSON array found in topics response")
                
            # Fallback: extract keywords with simple text processing
            return self._extract_keywords(combined_text, max_topics)
                
        except Exception as e:
            self.logger.error(f"Error extracting cluster topics: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Simple keyword extraction fallback when LM fails.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keyword strings
        """
        # Simple word frequency approach
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9]{2,}\b', text)
        
        # Filter common stop words
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'are', 'was', 'not',
                      'has', 'have', 'had', 'will', 'would', 'could', 'should'}
        filtered_words = [w.lower() for w in words if w.lower() not in stop_words]
        
        # Count frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Get top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]


# Helper function to create a cluster analyzer
def create_cluster_analyzer(summarizer=None):
    """
    Create a language model based cluster analyzer.
    
    Args:
        summarizer: Summarizer instance with LM access
        
    Returns:
        LMClusterAnalyzer instance
    """
    return LMClusterAnalyzer(summarizer=summarizer)