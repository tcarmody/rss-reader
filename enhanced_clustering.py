"""Enhanced clustering module for RSS reader that adds LM-based verification.

This module extends the standard clustering functionality by adding a second phase
that uses a language model to verify and refine the initial clustering results.
"""

import logging
import torch
import numpy as np
import re
from typing import List, Dict, Optional, Union, Any
from collections import defaultdict

# Import from the original clustering file to avoid code duplication
from clustering import ArticleClusterer, CONFIG


class EnhancedArticleClusterer(ArticleClusterer):
    """
    Enhanced article clusterer with language model verification.
    
    This class extends the basic ArticleClusterer with a second phase
    of clustering that uses a language model to verify and refine
    the initial clustering results.
    """
    
    def __init__(self, summarizer=None):
        """
        Initialize the enhanced article clusterer.
        
        Args:
            summarizer: Optional ArticleSummarizer instance to use for LM-based clustering
        """
        super().__init__()
        self.summarizer = summarizer
        self.logger = logging.getLogger("EnhancedArticleClusterer")
        
    def cluster_articles(self, articles):
        """
        Cluster articles with two-phase approach: embedding-based and LM-based.
        
        Args:
            articles: List of article dicts to cluster
            
        Returns:
            List of clusters, each a list of article dicts
        """
        # Phase 1: Standard embedding-based clustering from parent class
        initial_clusters = super().cluster_articles(articles)
        
        # Phase 2: Enhance clustering with language model verification
        enhanced_clusters = self.enhance_clustering_with_lm(initial_clusters)
        
        # Log clustering statistics
        self._log_clustering_stats(initial_clusters, enhanced_clusters)
        
        return enhanced_clusters
    
    def enhance_clustering_with_lm(self, initial_clusters):
        """
        Enhance clustering by using the language model to validate and refine clusters.
        
        Args:
            initial_clusters: List of clusters created by the embedding-based method
            
        Returns:
            List of refined clusters
        """
        if not initial_clusters:
            return []
            
        if not self.summarizer:
            self.logger.warning("No summarizer available for LM-based clustering enhancement")
            return initial_clusters
            
        self.logger.info(f"Starting LM-based enhancement of {len(initial_clusters)} initial clusters")
        
        refined_clusters = []
        candidates_for_merging = []
        
        # First, identify very small clusters (1-2 articles) that might need to be merged
        for i, cluster in enumerate(initial_clusters):
            if len(cluster) <= 2:
                candidates_for_merging.append((i, cluster))
            else:
                refined_clusters.append(cluster)
        
        self.logger.info(f"Found {len(candidates_for_merging)} small clusters as candidates for merging")
                
        # If we have candidates for merging, use LM to check if they should be merged with existing clusters
        if candidates_for_merging and self.summarizer:
            merged_count = 0
            
            for idx, candidate_cluster in candidates_for_merging:
                best_match = None
                best_score = 0
                
                # For each article in the candidate cluster
                for candidate_article in candidate_cluster:
                    candidate_text = f"{candidate_article.get('title', '')} {candidate_article.get('content', '')}"
                    
                    # Compare with each existing refined cluster
                    for i, refined_cluster in enumerate(refined_clusters):
                        # Skip empty clusters (shouldn't happen, but safety check)
                        if not refined_cluster:
                            continue
                            
                        # Create a representative text from the refined cluster
                        refined_texts = []
                        for article in refined_cluster[:3]:  # Use at most 3 articles to avoid too much text
                            refined_texts.append(f"{article.get('title', '')}")
                        refined_text = " | ".join(refined_texts)
                        
                        # Use the LM to compare the texts
                        similarity_score = self._compare_texts_with_lm(candidate_text, refined_text)
                        
                        # Update best match if this score is higher
                        if similarity_score > best_score and similarity_score > 0.7:  # 0.7 threshold
                            best_score = similarity_score
                            best_match = i
                
                # If a good match is found, merge the candidate with that cluster
                if best_match is not None:
                    refined_clusters[best_match].extend(candidate_cluster)
                    merged_count += 1
                    self.logger.info(f"Merged small cluster with cluster {best_match} (similarity: {best_score:.2f})")
                else:
                    # Otherwise, keep it as a separate cluster
                    refined_clusters.append(candidate_cluster)
            
            self.logger.info(f"LM enhancement complete: merged {merged_count} clusters")
        else:
            # If no summarizer available or no candidates, just use the initial clusters
            refined_clusters = initial_clusters
        
        return refined_clusters
    
    def _compare_texts_with_lm(self, text1, text2):
        """
        Compare two texts using the language model to determine similarity.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Limit text length to avoid excessive API usage
            text1 = text1[:1000]
            text2 = text2[:1000]
            
            # Create a prompt for the LM to evaluate similarity
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
            
            # Call the LM API
            if hasattr(self.summarizer, '_call_claude_api'):
                response = self.summarizer._call_claude_api(
                    model_id=self.summarizer.DEFAULT_MODEL,
                    prompt=prompt,
                    temperature=0.0,  # Use 0 temperature for deterministic response
                    max_tokens=10
                )
                
                # Extract the numerical score from the response
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
            self.logger.error(f"Error comparing texts with LM: {str(e)}")
            return 0.0
            
    def _detect_duplicate_summaries(self, clusters, similarity_threshold=0.7):
        """
        Detect and merge clusters with duplicate or very similar summaries.
        This helps catch cases where the embedding-based clustering missed
        some similar articles.
        
        Args:
            clusters: List of initial clusters
            similarity_threshold: Threshold for merging (0-1)
            
        Returns:
            List of refined clusters with duplicates merged
        """
        if not clusters or len(clusters) <= 1:
            return clusters
            
        # First, generate summaries for clusters that don't have them
        for cluster in clusters:
            if not cluster:
                continue
                
            has_summary = False
            for article in cluster:
                if article.get('summary') and isinstance(article['summary'], dict) and 'summary' in article['summary']:
                    has_summary = True
                    break
                    
            if not has_summary and self.summarizer and cluster:
                try:
                    # Use the first article's text for a quick summary
                    article = cluster[0]
                    summary = self.summarizer.summarize_article(
                        text=article.get('content', ''),
                        title=article.get('title', ''),
                        url=article.get('link', '#')
                    )
                    # Add summary to all articles in the cluster
                    for article in cluster:
                        article['summary'] = summary
                except Exception as e:
                    self.logger.warning(f"Failed to generate summary for cluster: {e}")
        
        # Now check for similar summaries
        merged_clusters = []
        skip_indices = set()
        
        for i, cluster1 in enumerate(clusters):
            if i in skip_indices or not cluster1:
                continue
                
            current_merged = list(cluster1)  # Start with the current cluster
            
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in skip_indices or not cluster2:
                    continue
                    
                # Get summaries for comparison
                summary1 = ""
                if cluster1[0].get('summary') and isinstance(cluster1[0]['summary'], dict):
                    summary1 = cluster1[0]['summary'].get('summary', '')
                
                summary2 = ""
                if cluster2[0].get('summary') and isinstance(cluster2[0]['summary'], dict):
                    summary2 = cluster2[0]['summary'].get('summary', '')
                
                # If we have summaries, compare them
                if summary1 and summary2:
                    similarity = self._compare_texts_with_lm(summary1, summary2)
                    
                    if similarity >= similarity_threshold:
                        # Merge the clusters
                        current_merged.extend(cluster2)
                        skip_indices.add(j)
                        self.logger.info(f"Merged clusters {i} and {j} due to similar summaries (score: {similarity:.2f})")
            
            merged_clusters.append(current_merged)
        
        return merged_clusters
        
    def cluster_with_summaries(self, articles):
        """
        Enhanced clustering that also uses summary content to further refine clusters.
        This is a complete pipeline that:
        1. Does initial embedding-based clustering
        2. Enhances with LM-based pairwise comparison
        3. Performs final refinement based on generated summaries
        
        Args:
            articles: List of article dicts to cluster
            
        Returns:
            List of clusters with similar content
        """
        # Phase 1: Standard embedding-based clustering
        initial_clusters = super().cluster_articles(articles)
        
        # Phase 2: Enhance clustering with language model verification
        lm_enhanced_clusters = self.enhance_clustering_with_lm(initial_clusters)
        
        # Phase 3: Final refinement based on summaries
        final_clusters = self._detect_duplicate_summaries(lm_enhanced_clusters)
        
        # Log clustering statistics
        self._log_clustering_stats(initial_clusters, final_clusters)
        
        return final_clusters
    
    def _log_clustering_stats(self, initial_clusters, final_clusters):
        """Log statistics about the clustering process."""
        initial_count = len(initial_clusters)
        final_count = len(final_clusters)
        
        # Calculate distribution of cluster sizes
        initial_sizes = [len(c) for c in initial_clusters]
        final_sizes = [len(c) for c in final_clusters]
        
        avg_initial_size = sum(initial_sizes) / max(1, len(initial_sizes))
        avg_final_size = sum(final_sizes) / max(1, len(final_sizes))
        
        self.logger.info(f"Clustering stats: Initial clusters: {initial_count}, Final clusters: {final_count}")
        self.logger.info(f"Average cluster size: Initial: {avg_initial_size:.2f}, Final: {avg_final_size:.2f}")
        
        # Calculate how many single-article clusters we have
        initial_singles = sum(1 for s in initial_sizes if s == 1)
        final_singles = sum(1 for s in final_sizes if s == 1)
        
        self.logger.info(f"Single-article clusters: Initial: {initial_singles}, Final: {final_singles}")


# Helper function to create an enhanced clusterer with the summarizer
def create_enhanced_clusterer(summarizer=None):
    """
    Create an enhanced article clusterer instance.
    
    Args:
        summarizer: Optional ArticleSummarizer instance
        
    Returns:
        EnhancedArticleClusterer instance
    """
    return EnhancedArticleClusterer(summarizer=summarizer)