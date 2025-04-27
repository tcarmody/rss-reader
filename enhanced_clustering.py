"""Enhanced clustering module for RSS reader that adds LM-based verification.

This module extends the standard clustering functionality by adding a second phase
that uses a language model to verify and refine the initial clustering results.
"""

import logging
import torch
import numpy as np
import re
import json
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
        
        # Group small clusters (1-2 articles) for potential merging
        small_clusters = []
        refined_clusters = []
        
        # First, identify very small clusters (1-2 articles) that might need to be merged
        for i, cluster in enumerate(initial_clusters):
            if len(cluster) <= 2:
                small_clusters.extend(cluster)
            else:
                refined_clusters.append(cluster)
        
        self.logger.info(f"Found {len(small_clusters)} articles in small clusters as candidates for merging")
                
        # If we have articles from small clusters, use the multi-article clustering
        if small_clusters and self.summarizer and len(small_clusters) >= 2:
            # Maximum batch size for LM processing
            max_batch_size = 5
            merged_count = 0
            
            # Process small clusters in batches
            for i in range(0, len(small_clusters), max_batch_size):
                batch = small_clusters[i:i+max_batch_size]
                
                # Get text representation for each article
                article_texts = []
                for article in batch:
                    text = f"{article.get('title', '')} {article.get('content', '')}"
                    # Truncate to reasonable length
                    article_texts.append(text[:1000])
                
                # Use the LM to cluster these articles
                try:
                    cluster_assignments = self._compare_multiple_texts_with_lm(article_texts)
                    
                    # Create new clusters based on LM assignments
                    batch_clusters = [[] for _ in range(len(cluster_assignments))]
                    for cluster_idx, article_indices in enumerate(cluster_assignments):
                        for idx in article_indices:
                            # Adjust for 0-based indexing
                            article_idx = idx - 1
                            if 0 <= article_idx < len(batch):
                                batch_clusters[cluster_idx].append(batch[article_idx])
                    
                    # Remove empty clusters
                    batch_clusters = [c for c in batch_clusters if c]
                    
                    # For each LM-created cluster, find the best matching existing cluster to merge with
                    for lm_cluster in batch_clusters:
                        # If the cluster has multiple articles, it's a valid cluster on its own
                        if len(lm_cluster) > 1:
                            refined_clusters.append(lm_cluster)
                            merged_count += len(lm_cluster)
                        else:
                            # For single articles, try to find the best existing cluster
                            best_match = None
                            best_score = 0
                            
                            article = lm_cluster[0]
                            article_text = f"{article.get('title', '')} {article.get('content', '')}"
                            article_text = article_text[:1000]  # Truncate for API
                            
                            # Compare with existing clusters
                            for cluster_idx, cluster in enumerate(refined_clusters):
                                if not cluster:
                                    continue
                                    
                                # Create representative text from cluster
                                cluster_texts = []
                                for cluster_article in cluster[:3]:  # Use up to 3 articles
                                    cluster_texts.append(f"{cluster_article.get('title', '')}")
                                cluster_text = " | ".join(cluster_texts)
                                
                                # Compare the single article with this cluster
                                similarity = self._compare_texts_with_lm(article_text, cluster_text)
                                
                                if similarity > best_score and similarity > 0.7:
                                    best_score = similarity
                                    best_match = cluster_idx
                            
                            # Merge or create new cluster
                            if best_match is not None:
                                refined_clusters[best_match].append(article)
                                merged_count += 1
                                self.logger.info(f"Merged single article with cluster {best_match} (similarity: {best_score:.2f})")
                            else:
                                refined_clusters.append(lm_cluster)
                
                except Exception as e:
                    self.logger.error(f"Error in LM-based clustering batch: {str(e)}")
                    # Add all articles as individual clusters as fallback
                    for article in batch:
                        refined_clusters.append([article])
            
            self.logger.info(f"LM enhancement complete: processed {merged_count} articles from small clusters")
        else:
            # If no small clusters or summarizer, add all small clusters individually
            for cluster in initial_clusters:
                if len(cluster) <= 2:
                    refined_clusters.append(cluster)
        
        return refined_clusters
    
    def _compare_multiple_texts_with_lm(self, texts_list):
        """
        Compare multiple texts using the language model to determine clustering.
        
        Args:
            texts_list: List of article texts to compare
            
        Returns:
            List of cluster assignments (indices of clusters for each article)
        """
        # Create a numbered list of texts for the prompt
        articles_text = ""
        for i, text in enumerate(texts_list, 1):
            # Truncate each text to a reasonable length
            truncated = text[:800] + "..." if len(text) > 800 else text
            articles_text += f"Article {i}: {truncated}\n\n"
        
        # Create the prompt for multi-article clustering
        prompt = (
            "Task: Group the following news articles into clusters based on their topics and content.\n\n"
            f"{articles_text}\n"
            "Instructions:\n"
            "1. Analyze the topical similarity and content of all articles.\n"
            "2. Group articles that cover the same news story or highly related topics.\n"
            "3. Return your answer as a JSON object with cluster assignments:\n"
            "   {\"clusters\": [[1, 3, 5], [2, 4], [6]]}\n"
            "   Where each inner array represents a cluster, and the numbers are article indices.\n"
            "4. Articles that don't belong to any cluster should be in their own single-item cluster.\n"
            "5. IMPORTANT: Return ONLY the JSON object, nothing else.\n"
        )
        
        # Call the LM API
        try:
            response = self.summarizer._call_claude_api(
                model_id=self.summarizer.DEFAULT_MODEL,
                prompt=prompt,
                temperature=0.0,  # Use 0 temperature for deterministic response
                max_tokens=200
            )
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    self.logger.info(f"Successfully parsed LM clustering result: {result}")
                    return result.get('clusters', [])
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")
            else:
                self.logger.warning(f"No JSON found in LM response: {response}")
            
            # Fallback: each article in its own cluster
            return [[i] for i in range(1, len(texts_list) + 1)]
                
        except Exception as e:
            self.logger.error(f"Error comparing multiple texts with LM: {str(e)}")
            # Return each article as its own cluster as fallback
            return [[i] for i in range(1, len(texts_list) + 1)]
    
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