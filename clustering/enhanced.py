"""
Enhanced article clustering with multi-article language model comparison.

This module provides advanced clustering capabilities that use a language model
to compare multiple articles simultaneously for better cluster accuracy.
Includes asynchronous processing capabilities for better performance.
"""

import logging
import time
import asyncio
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import defaultdict
from datetime import datetime
import numpy as np

# Import base clustering functionality
from clustering.base import ArticleClusterer, CONFIG
from common.performance import track_performance

# Global flags
HAS_LM_ANALYZER = False

# Try to import the language model-based cluster analyzer
try:
    from models.lm_analyzer import create_cluster_analyzer
    HAS_LM_ANALYZER = True
except ImportError:
    logging.warning("LM cluster analyzer not available - using standard clustering")


class LMClusterRefiner:
    """A simplified LM-based cluster refinement utility."""
    
    def __init__(self, summarizer=None):
        """Initialize with an optional summarizer/LM model."""
        self.model = None
        self.logger = logging.getLogger("LMClusterRefiner")
        
        if HAS_LM_ANALYZER and summarizer:
            try:
                self.model = create_cluster_analyzer(summarizer=summarizer)
                self.logger.info("Initialized with language model analyzer")
            except Exception as e:
                self.logger.warning(f"Failed to create LM analyzer: {e}")
    
    def can_refine(self):
        """Check if this refiner can actually perform refinement."""
        return self.model is not None
    
    def check_cluster_coherence(self, cluster):
        """Check if a cluster is coherent using the LM."""
        if not self.model or len(cluster) < 2:
            return 1.0  # Default: assume coherent
            
        try:
            # Get short text representations
            texts = [f"{a.get('title', '')} {a.get('content', '')[:200]}" for a in cluster]
            
            # Use the LM analyzer to check coherence
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    sim = self.model.compare_article_pair(texts[i], texts[j])
                    similarities.append(sim)
                    
            # Return average similarity
            return sum(similarities) / max(1, len(similarities))
        except Exception as e:
            self.logger.error(f"Error checking coherence: {e}")
            return 1.0  # Assume coherent on error
    
    async def check_cluster_coherence_async(self, cluster):
        """Async version of coherence check."""
        if not self.model or len(cluster) < 2:
            return 1.0  # Default: assume coherent
            
        try:
            # Get short text representations
            texts = [f"{a.get('title', '')} {a.get('content', '')[:200]}" for a in cluster]
            
            # Use the LM analyzer to check coherence
            similarities = []
            
            # Create comparison pairs
            pairs = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    pairs.append((texts[i], texts[j]))
            
            # Run comparisons in batches
            batch_size = 4  # Adjust based on your model's capabilities
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Create tasks for each pair
                tasks = [
                    asyncio.create_task(
                        asyncio.to_thread(self.model.compare_article_pair, t1, t2)
                    )
                    for t1, t2 in batch_pairs
                ]
                
                # Gather results
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"Comparison error: {result}")
                        similarities.append(0.5)  # Neutral score on error
                    else:
                        similarities.append(result)
                
                # Short sleep to allow other tasks to run
                await asyncio.sleep(0)
                
            # Return average similarity
            return sum(similarities) / max(1, len(similarities))
            
        except Exception as e:
            self.logger.error(f"Error in async coherence check: {e}")
            return 1.0  # Assume coherent on error
    
    def split_cluster(self, cluster):
        """Try to split a cluster into subclusters if needed."""
        if not self.model or len(cluster) < 3:
            return [cluster]
            
        try:
            # Check coherence first
            coherence = self.check_cluster_coherence(cluster)
            
            # If reasonably coherent, don't split
            if coherence >= 0.6:
                return [cluster]
                
            # Extract text for clustering
            cluster_articles = []
            for article in cluster:
                content = article.get('content', '')
                if len(content) > 500:
                    content = content[:500]  # Truncate long content
                cluster_articles.append({
                    'content': f"{article.get('title', '')} {content}"
                })
                
            # Use LM to suggest subclusters
            subclusters = self.model.cluster_articles(
                cluster_articles,
                text_extractor=lambda a: a.get('content', ''),
                similarity_threshold=0.6
            )
            
            # If no meaningful subclusters found, return original
            if len(subclusters) <= 1:
                return [cluster]
                
            # Create resulting subclusters
            result = []
            for subcluster_indices in subclusters:
                # Convert 1-based indices from LM analyzer to 0-based
                adjusted_indices = [idx - 1 for idx in subcluster_indices]
                # Create subcluster with those articles
                subcluster = [cluster[i] for i in adjusted_indices if 0 <= i < len(cluster)]
                if subcluster:  # Only add non-empty subclusters
                    result.append(subcluster)
                    
            # If we somehow lost articles, include the original cluster to be safe
            total_articles = sum(len(c) for c in result)
            if total_articles < len(cluster):
                return [cluster]
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error splitting cluster: {e}")
            return [cluster]  # Return original on error
    
    async def split_cluster_async(self, cluster):
        """Async version of split_cluster."""
        if not self.model or len(cluster) < 3:
            return [cluster]
            
        try:
            # Check coherence first
            coherence = await self.check_cluster_coherence_async(cluster)
            
            # If reasonably coherent, don't split
            if coherence >= 0.6:
                return [cluster]
                
            # Extract text for clustering
            cluster_articles = []
            for article in cluster:
                content = article.get('content', '')
                if len(content) > 500:
                    content = content[:500]  # Truncate long content
                cluster_articles.append({
                    'content': f"{article.get('title', '')} {content}"
                })
                
            # Use LM to suggest subclusters in a non-blocking way
            subclusters = await asyncio.to_thread(
                self.model.cluster_articles,
                cluster_articles,
                text_extractor=lambda a: a.get('content', ''),
                similarity_threshold=0.6
            )
            
            # Process results as before
            if len(subclusters) <= 1:
                return [cluster]
                
            result = []
            for subcluster_indices in subclusters:
                adjusted_indices = [idx - 1 for idx in subcluster_indices]
                subcluster = [cluster[i] for i in adjusted_indices if 0 <= i < len(cluster)]
                if subcluster:
                    result.append(subcluster)
                    
            total_articles = sum(len(c) for c in result)
            if total_articles < len(cluster):
                return [cluster]
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async cluster split: {e}")
            return [cluster]
    
    def extract_topics(self, cluster, max_topics=5):
        """Extract topics from a cluster using the LM."""
        if not self.model or not cluster:
            return []
            
        try:
            topics = self.model.extract_cluster_topics(cluster)
            return topics[:max_topics]
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []
    
    async def extract_topics_async(self, cluster, max_topics=5):
        """Async version of extract_topics."""
        if not self.model or not cluster:
            return []
            
        try:
            # Run topic extraction in a non-blocking way
            topics = await asyncio.to_thread(self.model.extract_cluster_topics, cluster)
            return topics[:max_topics]
        except Exception as e:
            self.logger.error(f"Error in async topic extraction: {e}")
            return []


class EnhancedArticleClusterer(ArticleClusterer):
    """
    Enhanced article clusterer with multi-article language model comparison.
    
    This clusterer can use a language model to refine clusters by understanding
    semantic relationships across multiple articles simultaneously.
    """
    
    def __init__(self, summarizer=None):
        """
        Initialize the enhanced clusterer.
        
        Args:
            summarizer: A summarizer instance that provides access to the language model
        """
        super().__init__()
        self.refiner = LMClusterRefiner(summarizer)
        self.logger = logging.getLogger("EnhancedArticleClusterer")
        
        # Log capabilities
        if self.refiner.can_refine():
            self.logger.info("Initialized with language model refinement capabilities")
        else:
            self.logger.info("Initialized without LM refinement (using basic clustering)")
    
    def _embedding_similarity(self, text1, text2):
        """Calculate embedding similarity between two texts."""
        try:
            # Get embeddings
            emb1 = self.model.encode(text1)
            emb2 = self.model.encode(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _identify_uncertain_clusters(self, clusters):
        """Identify clusters that might benefit from LM refinement."""
        uncertain_clusters = []
        
        for i, cluster in enumerate(clusters):
            # Skip very small or large clusters
            if len(cluster) < 3 or len(cluster) > 10:
                continue
                
            # Use embedding similarity to check coherence
            coherence_score = 0.0
            pairs_count = 0
            
            for j in range(len(cluster)):
                for k in range(j+1, len(cluster)):
                    article1 = cluster[j]
                    article2 = cluster[k]
                    
                    # Get short text representation
                    text1 = f"{article1.get('title', '')} {article1.get('content', '')[:200]}"
                    text2 = f"{article2.get('title', '')} {article2.get('content', '')[:200]}"
                    
                    # Calculate similarity
                    similarity = self._embedding_similarity(text1, text2)
                    coherence_score += similarity
                    pairs_count += 1
            
            # Calculate average coherence
            avg_coherence = coherence_score / max(1, pairs_count)
            
            # If coherence is low, mark for refinement
            if avg_coherence < 0.75:
                uncertain_clusters.append((i, avg_coherence))
                
        return uncertain_clusters
    
    async def _evaluate_cluster_coherence_async(self, cluster_idx, cluster):
        """Helper to evaluate cluster coherence asynchronously."""
        coherence_score = 0.0
        pairs_count = 0
        
        # Create comparison tasks
        tasks = []
        for j in range(len(cluster)):
            for k in range(j+1, len(cluster)):
                article1 = cluster[j]
                article2 = cluster[k]
                
                # Get short text representation
                text1 = f"{article1.get('title', '')} {article1.get('content', '')[:200]}"
                text2 = f"{article2.get('title', '')} {article2.get('content', '')[:200]}"
                
                # Create task to calculate similarity
                task = asyncio.create_task(
                    asyncio.to_thread(self._embedding_similarity, text1, text2)
                )
                tasks.append(task)
        
        # Gather results
        similarities = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate average coherence
        valid_sims = [s for s in similarities if not isinstance(s, Exception)]
        avg_coherence = sum(valid_sims) / max(1, len(valid_sims))
        
        return cluster_idx, avg_coherence
    
    async def _identify_uncertain_clusters_async(self, clusters):
        """Async version of _identify_uncertain_clusters."""
        uncertain_clusters = []
        
        # Process clusters in parallel
        tasks = []
        for i, cluster in enumerate(clusters):
            # Skip very small or large clusters
            if len(cluster) < 3 or len(cluster) > 10:
                continue
                
            # Create task to evaluate cluster coherence
            task = asyncio.create_task(self._evaluate_cluster_coherence_async(i, cluster))
            tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Error evaluating cluster: {result}")
                continue
                
            cluster_idx, coherence = result
            if coherence < 0.75:
                uncertain_clusters.append((cluster_idx, coherence))
                
        return uncertain_clusters
    
    @track_performance
    def cluster_with_summaries(self, articles):
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
            # Use basic clustering as starting point
            preliminary_clusters = super().cluster_articles(articles)
            self.logger.info(f"Initial clustering created {len(preliminary_clusters)} clusters")
            
            # Refine clusters with LM if available
            if self.refiner.can_refine():
                # Identify uncertain clusters
                uncertain_clusters = self._identify_uncertain_clusters(preliminary_clusters)
                self.logger.info(f"Found {len(uncertain_clusters)} potentially uncertain clusters")
                
                # Refine uncertain clusters
                refined_clusters = []
                for i, cluster in enumerate(preliminary_clusters):
                    # Check if this cluster is uncertain
                    is_uncertain = False
                    for uc_idx, coherence in uncertain_clusters:
                        if uc_idx == i:
                            is_uncertain = True
                            break
                    
                    if is_uncertain and len(cluster) >= 3:
                        # Try to refine with LM
                        self.logger.info(f"Attempting to refine cluster {i} (size {len(cluster)})")
                        subclusters = self.refiner.split_cluster(cluster)
                        
                        if len(subclusters) > 1:
                            self.logger.info(f"Split cluster {i} into {len(subclusters)} subclusters")
                            refined_clusters.extend(subclusters)
                        else:
                            refined_clusters.append(cluster)
                    else:
                        # Keep as is
                        refined_clusters.append(cluster)
                        
                final_clusters = refined_clusters
            else:
                final_clusters = preliminary_clusters
            
            # Extract topics for each cluster
            for cluster in final_clusters:
                try:
                    if not cluster:
                        continue
                    
                    # Use LM for topic extraction if available
                    if self.refiner.can_refine():
                        topics = self.refiner.extract_topics(cluster)
                        if topics:
                            for article in cluster:
                                article['cluster_topics'] = topics
                    else:
                        # Use base class topic extraction
                        texts = []
                        for article in cluster:
                            title = article.get('title', '')
                            content = article.get('content', '')
                            texts.append(f"{title} {content}")
                            
                        topics = self._extract_topics(texts)
                        
                        if topics:
                            for article in cluster:
                                article['cluster_topics'] = topics
                except Exception as e:
                    self.logger.error(f"Error extracting topics: {e}")
            
            self.logger.info(f"Enhanced clustering completed: {len(final_clusters)} final clusters")
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in enhanced clustering: {e}")
            # Fallback to basic clustering
            return super().cluster_articles(articles)
    
    @track_performance
    async def cluster_articles_async(self, articles):
        """
        Asynchronous version of enhanced clustering.
        
        Args:
            articles: List of articles to cluster
            
        Returns:
            List of article clusters with topics and summaries
        """
        self.logger.info(f"Starting async enhanced clustering for {len(articles)} articles")
        
        try:
            # Use basic clustering as starting point (asynchronously)
            preliminary_clusters = await super().cluster_articles_async(articles)
            self.logger.info(f"Initial async clustering created {len(preliminary_clusters)} clusters")
            
            # Refine clusters with LM if available
            if self.refiner.can_refine():
                # Identify uncertain clusters asynchronously
                uncertain_clusters = await self._identify_uncertain_clusters_async(preliminary_clusters)
                self.logger.info(f"Found {len(uncertain_clusters)} potentially uncertain clusters")
                
                # Process uncertain clusters concurrently
                refinement_tasks = []
                for cluster_idx, coherence in uncertain_clusters:
                    if len(preliminary_clusters[cluster_idx]) >= 3:
                        self.logger.info(f"Creating task to refine cluster {cluster_idx}")
                        task = asyncio.create_task(
                            self.refiner.split_cluster_async(preliminary_clusters[cluster_idx])
                        )
                        refinement_tasks.append((cluster_idx, task))
                
                # Create a mutable copy of clusters that we can modify
                final_clusters = list(preliminary_clusters)
                
                # Process refinement results
                for cluster_idx, task in refinement_tasks:
                    try:
                        subclusters = await task
                        
                        if len(subclusters) > 1:
                            self.logger.info(f"Split cluster {cluster_idx} into {len(subclusters)} subclusters")
                            # Replace the original cluster with the first subcluster
                            final_clusters[cluster_idx] = subclusters[0]
                            # Add the rest of the subclusters at the end
                            final_clusters.extend(subclusters[1:])
                    except Exception as e:
                        self.logger.error(f"Error refining cluster {cluster_idx}: {e}")
            else:
                final_clusters = preliminary_clusters
            
            # Extract topics for each cluster concurrently
            topic_tasks = []
            for cluster_idx, cluster in enumerate(final_clusters):
                if not cluster:
                    continue
                
                # Choose appropriate topic extraction method
                if self.refiner.can_refine():
                    task = asyncio.create_task(
                        self.refiner.extract_topics_async(cluster)
                    )
                else:
                    # Use base class topic extraction
                    texts = []
                    for article in cluster:
                        title = article.get('title', '')
                        content = article.get('content', '')
                        texts.append(f"{title} {content}")
                    
                    task = asyncio.create_task(
                        self._extract_topics_async(texts)
                    )
                
                topic_tasks.append((cluster_idx, task))
            
            # Process topic extraction results
            for cluster_idx, task in topic_tasks:
                try:
                    topics = await task
                    
                    # Add topics to each article in cluster
                    if topics:
                        for article in final_clusters[cluster_idx]:
                            article['cluster_topics'] = topics
                except Exception as e:
                    self.logger.error(f"Error extracting topics: {e}")
            
            self.logger.info(f"Async enhanced clustering completed: {len(final_clusters)} final clusters")
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in async enhanced clustering: {e}")
            # Fallback to basic clustering
            return await super().cluster_articles_async(articles)


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