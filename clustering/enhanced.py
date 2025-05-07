"""
Enhanced article clustering with multi-article language model comparison.

This module provides advanced clustering capabilities that use a language model
to compare multiple articles simultaneously for better cluster accuracy.
"""

import logging
import time
import json
import asyncio
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import defaultdict
from datetime import datetime

# Import base clustering functionality
from clustering import ArticleClusterer, CONFIG
from utils.performance import track_performance

# Try to import the language model-based cluster analyzer
try:
    from lm_cluster_analyzer import create_cluster_analyzer
    HAS_LM_ANALYZER = True
except ImportError:
    HAS_LM_ANALYZER = False
    logging.warning("LM cluster analyzer not available - using standard clustering")


class EnhancedArticleClusterer(ArticleClusterer):
    """
    Enhanced article clusterer with multi-article language model comparison.
    
    This clusterer can use a language model to compare multiple articles at once,
    providing better clustering accuracy by understanding semantic relationships
    across multiple articles simultaneously.
    """
    
    def __init__(self, summarizer=None):
        """
        Initialize the enhanced clusterer.
        
        Args:
            summarizer: A summarizer instance that provides access to the language model
        """
        super().__init__()
        self.summarizer = summarizer
        self.logger = logging.getLogger("EnhancedArticleClusterer")
        
        # Create a cluster analyzer for language model based operations
        if HAS_LM_ANALYZER and summarizer:
            self.analyzer = create_cluster_analyzer(summarizer=summarizer)
            self.logger.info("Initialized with language model cluster analyzer")
        else:
            self.analyzer = None
            if not HAS_LM_ANALYZER:
                self.logger.warning("LM cluster analyzer not available")
            if not summarizer:
                self.logger.warning("No summarizer provided - basic clustering only")
    
    def _prepare_cluster_candidates(self, articles: List[Dict], preliminary_clusters: List[List[Dict]]) -> Dict:
        """
        Prepare candidate groups for multi-article language model comparison.
        
        Args:
            articles: List of all articles
            preliminary_clusters: Initial clusters from embedding-based clustering
            
        Returns:
            Dict with candidate cluster information for LM verification
        """
        # Identify clusters that might need refinement
        candidates = {
            'uncertain_clusters': [],
            'boundary_articles': [],
            'potential_merges': []
        }
        
        # Find clusters with ambiguous boundaries
        for i, cluster in enumerate(preliminary_clusters):
            if len(cluster) > 1:
                # Calculate internal cluster similarity
                internal_sim = self._calculate_internal_similarity(cluster)
                
                if internal_sim < 0.7:  # Threshold for uncertain clusters
                    candidates['uncertain_clusters'].append({
                        'cluster_id': i,
                        'articles': cluster,
                        'similarity': internal_sim
                    })
        
        # Find potential articles that might belong in multiple clusters
        for article in articles:
            similarities = []
            for cluster_id, cluster in enumerate(preliminary_clusters):
                if article not in cluster:
                    # Calculate similarity to cluster centroid
                    sim = self._calculate_article_cluster_similarity(article, cluster)
                    similarities.append((cluster_id, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # If there are multiple clusters with high similarity
            if len(similarities) > 1 and similarities[0][1] - similarities[1][1] < 0.1:
                candidates['boundary_articles'].append({
                    'article': article,
                    'potential_clusters': similarities[:3]
                })
        
        return candidates
    
    def _calculate_internal_similarity(self, cluster: List[Dict]) -> float:
        """
        Calculate the internal similarity of a cluster.
        
        Args:
            cluster: List of articles in the cluster
            
        Returns:
            Average internal similarity score
        """
        if len(cluster) < 2:
            return 1.0
            
        similarities = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                # Use title and first paragraph for quick similarity check
                text1 = f"{cluster[i].get('title', '')} {cluster[i].get('content', '')[:200]}"
                text2 = f"{cluster[j].get('title', '')} {cluster[j].get('content', '')[:200]}"
                
                # Use either LM or embedding similarity
                if self.analyzer:
                    sim = self.analyzer.compare_article_pair(text1, text2)
                else:
                    sim = self._embedding_similarity(text1, text2)
                    
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_article_cluster_similarity(self, article: Dict, cluster: List[Dict]) -> float:
        """
        Calculate similarity between an article and a cluster.
        
        Args:
            article: Single article to compare
            cluster: Cluster of articles
            
        Returns:
            Similarity score
        """
        if not cluster:
            return 0.0
            
        # Compare against each article in the cluster
        similarities = []
        article_text = f"{article.get('title', '')} {article.get('content', '')[:200]}"
        
        for cluster_article in cluster:
            cluster_text = f"{cluster_article.get('title', '')} {cluster_article.get('content', '')[:200]}"
            
            if self.analyzer:
                sim = self.analyzer.compare_article_pair(article_text, cluster_text)
            else:
                sim = self._embedding_similarity(article_text, cluster_text)
                
            similarities.append(sim)
        
        # Return average similarity to cluster
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback method to calculate embedding similarity if LM analyzer is not available.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Embedding similarity score
        """
        import numpy as np
        
        try:
            # Get embeddings
            emb1 = self.model.encode(text1)
            emb2 = self.model.encode(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    @track_performance
    def cluster_with_summaries(self, articles: List[Dict]) -> List[List[Dict]]:
        """
        Enhanced clustering method that combines embedding-based clustering with
        language model verification and generates summaries for clusters.
        
        Args:
            articles: List of articles to cluster
            
        Returns:
            List of article clusters with summaries and topics
        """
        self.logger.info(f"Starting enhanced clustering for {len(articles)} articles")
        
        try:
            # First, perform standard embedding-based clustering
            preliminary_clusters = super().cluster_articles(articles)
            self.logger.info(f"Initial clustering created {len(preliminary_clusters)} clusters")
            
            # If LM analyzer is available, refine clusters using it
            if self.analyzer and len(articles) > 0:
                # Prepare candidates for multi-article comparison
                candidates = self._prepare_cluster_candidates(articles, preliminary_clusters)
                
                # Refine uncertain clusters
                refined_clusters = []
                for cluster_info in candidates['uncertain_clusters']:
                    cluster_id = cluster_info['cluster_id']
                    cluster = cluster_info['articles']
                    
                    # Use LM to analyze cluster coherence
                    if len(cluster) > 1 and len(cluster) <= 5:
                        try:
                            # Get topics for this cluster
                            topics = self.analyzer.extract_cluster_topics(cluster)
                            
                            # Verify cluster coherence using LM
                            cluster_articles = [{'content': a.get('content', '')} for a in cluster]
                            analyzed_clusters = self.analyzer.cluster_articles(
                                cluster_articles,
                                text_extractor=lambda a: a.get('content', ''),
                                similarity_threshold=0.6
                            )
                            
                            # If LM suggests different clustering, apply it
                            if len(analyzed_clusters) > 1:
                                self.logger.info(f"LM suggests splitting cluster {cluster_id}")
                                # Apply LM-suggested clustering
                                for sub_cluster_indices in analyzed_clusters:
                                    sub_cluster = [cluster[idx-1] for idx in sub_cluster_indices]
                                    refined_clusters.append(sub_cluster)
                            else:
                                refined_clusters.append(cluster)
                                
                        except Exception as e:
                            self.logger.error(f"Error refining cluster {cluster_id}: {e}")
                            refined_clusters.append(cluster)  # Use original cluster on error
                    else:
                        refined_clusters.append(cluster)
                
                # Add clusters that didn't need refinement
                for i, cluster in enumerate(preliminary_clusters):
                    if not any(i == c['cluster_id'] for c in candidates['uncertain_clusters']):
                        refined_clusters.append(cluster)
                        
                final_clusters = refined_clusters
            else:
                final_clusters = preliminary_clusters
            
            # Extract topics for each cluster
            for cluster in final_clusters:
                try:
                    if self.analyzer:
                        topics = self.analyzer.extract_cluster_topics(cluster)
                    else:
                        # Fallback: use simple keyword extraction
                        topics = self._extract_keywords_simple(cluster)
                        
                    # Add topics to each article in the cluster
                    for article in cluster:
                        article['cluster_topics'] = topics
                except Exception as e:
                    self.logger.error(f"Error extracting topics for cluster: {e}")
            
            self.logger.info(f"Enhanced clustering completed: {len(final_clusters)} final clusters")
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in enhanced clustering: {e}")
            # Fallback to basic clustering
            return super().cluster_articles(articles)
    
    def _extract_keywords_simple(self, cluster: List[Dict], max_keywords: int = 5) -> List[str]:
        """
        Simple keyword extraction fallback when LM analyzer is not available.
        
        Args:
            cluster: Cluster of articles
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        from collections import Counter
        import re
        
        # Combine all text from cluster
        all_text = ""
        for article in cluster:
            all_text += f"{article.get('title', '')} {article.get('content', '')} "
        
        # Simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+\b', all_text)  # Find capitalized words
        word_counts = Counter(words)
        
        # Get top keywords
        keywords = [word for word, _ in word_counts.most_common(max_keywords)]
        return keywords
    
    async def cluster_articles_async(self, articles: List[Dict]) -> List[List[Dict]]:
        """
        Asynchronous version of enhanced clustering.
        
        Args:
            articles: List of articles to cluster
            
        Returns:
            List of article clusters
        """
        self.logger.info(f"Starting async enhanced clustering for {len(articles)} articles")
        
        try:
            # Use regular clustering for now (can be enhanced with async operations later)
            clusters = self.cluster_with_summaries(articles)
            return clusters
        except Exception as e:
            self.logger.error(f"Error in async clustering: {e}")
            return [[article] for article in articles]  # Fallback to individual clusters


# Factory function to create the enhanced clusterer
def create_enhanced_clusterer(summarizer=None):
    """
    Create an enhanced article clusterer instance.
    
    Args:
        summarizer: A summarizer instance that provides access to the language model
        
    Returns:
        EnhancedArticleClusterer instance
    """
    return EnhancedArticleClusterer(summarizer=summarizer)