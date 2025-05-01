"""
Optimized clustering module for RSS reader with performance enhancements.

This module extends the standard clustering functionality with a performance-optimized
multi-phase approach that balances speed and accuracy.
"""

import logging
import torch
import numpy as np
import re
import json
import asyncio
from typing import List, Dict, Optional, Union, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import time

# Import from the original clustering file to avoid code duplication
from clustering import ArticleClusterer, CONFIG
from utils.performance import track_performance

# Import the LM cluster analyzer
from lm_cluster_analyzer import create_cluster_analyzer


class OptimizedArticleClusterer(ArticleClusterer):
    """
    Optimized article clusterer with better performance characteristics.
    
    This class extends the basic ArticleClusterer with a multi-phase
    approach that prioritizes speed while maintaining accuracy.
    """
    
    def __init__(self, summarizer=None):
        """
        Initialize the optimized article clusterer.
        
        Args:
            summarizer: Optional ArticleSummarizer instance to use for LM-based clustering
        """
        super().__init__()
        self.summarizer = summarizer
        self.logger = logging.getLogger("OptimizedArticleClusterer")
        
        # Create a cluster analyzer for LM-based operations
        self.analyzer = create_cluster_analyzer(summarizer=summarizer)
        
        # Load config for optimized clustering
        self._load_config()
        
        # Initialize embedding cache dict to prevent redundant computations
        self.embedding_memo = {}
        
    def _load_config(self):
        """Load configuration values for optimized clustering."""
        self.confidence_threshold = float(os.environ.get('CLUSTER_CONFIDENCE_THRESHOLD', '0.85'))
        self.max_lm_comparisons = int(os.environ.get('MAX_LM_COMPARISONS', '50'))
        self.batch_size = int(os.environ.get('CLUSTERING_BATCH_SIZE', '100'))
        self.use_incremental = os.environ.get('USE_INCREMENTAL_CLUSTERING', 'true').lower() == 'true'
        
        self.logger.info(f"Loaded optimized clustering configuration: confidence_threshold={self.confidence_threshold}, "
                       f"max_lm_comparisons={self.max_lm_comparisons}, batch_size={self.batch_size}")
        
    def _is_ambiguous_cluster(self, cluster, embeddings=None):
        """
        Determine if a cluster has ambiguous boundaries that need verification.
        
        Args:
            cluster: List of article indices forming a cluster
            embeddings: Optional pre-computed embeddings
            
        Returns:
            bool: True if the cluster is ambiguous and needs verification
        """
        if len(cluster) <= 1:
            # Single-article clusters are never ambiguous
            return False
            
        if len(cluster) > 10:
            # Large clusters are usually reliable
            return False
            
        # Calculate internal similarity
        if embeddings is not None:
            cluster_embeddings = [embeddings[i] for i in cluster]
            similarities = []
            
            # Calculate pairwise similarities within cluster
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    similarity = np.dot(cluster_embeddings[i], cluster_embeddings[j])
                    similarities.append(similarity)
                    
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            # If internal similarity is high, cluster is not ambiguous
            if avg_similarity > self.confidence_threshold:
                return False
                
        # Default to treating small clusters as potentially ambiguous
        return True
        
    async def _process_batch(self, articles):
        """
        Process a batch of articles in parallel.
        
        Args:
            articles: List of articles to process
            
        Returns:
            List of clusters
        """
        # Generate embeddings for the batch
        embeddings = self._generate_embeddings([a.get('content', '') for a in articles])
        
        # Perform initial clustering
        labels = self._perform_initial_clustering(embeddings)
        
        # Convert labels to clusters
        initial_clusters = self._labels_to_clusters(labels, articles)
        
        # Identify ambiguous clusters that need LM verification
        ambiguous_indices = []
        confident_clusters = []
        
        for i, cluster in enumerate(initial_clusters):
            if self._is_ambiguous_cluster(cluster):
                # Add to ambiguous list for further processing
                ambiguous_indices.extend(cluster)
            else:
                # Keep as a confident cluster
                confident_clusters.append([articles[idx] for idx in cluster])
                
        # Process ambiguous articles if there are any
        verified_clusters = []
        if ambiguous_indices and self.summarizer:
            ambiguous_articles = [articles[idx] for idx in ambiguous_indices]
            
            # Only perform LM verification if we have a reasonable number of comparisons
            if len(ambiguous_articles) <= self.max_lm_comparisons:
                try:
                    # Use LM to verify these articles
                    verified_clusters = await self._verify_with_lm(ambiguous_articles)
                except Exception as e:
                    self.logger.error(f"Error in LM verification: {e}")
                    # Fallback: treat each ambiguous article as its own cluster
                    verified_clusters = [[article] for article in ambiguous_articles]
            else:
                # Too many comparisons, use simpler methods
                self.logger.info(f"Skipping LM verification for {len(ambiguous_articles)} articles (exceeds max comparisons)")
                # Apply a more aggressive traditional clustering
                fallback_clusters = self._apply_fallback_clustering(ambiguous_articles)
                verified_clusters = fallback_clusters
        
        # Combine confident and verified clusters
        all_clusters = confident_clusters + verified_clusters
        
        return all_clusters
        
    def _generate_embeddings(self, texts):
        """
        Generate embeddings for texts with caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        # Initialize the model if needed
        if self.model is None:
            self._initialize_model()
            
        # Check cache for existing embeddings
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            # Use hash of text as cache key
            cache_key = hash(text[:1000])  # Use first 1000 chars for reasonable hash
            
            if cache_key in self.embedding_memo:
                # Use cached embedding
                embeddings.append(self.embedding_memo[cache_key])
            else:
                # Mark for embedding
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                
        # Generate new embeddings only for texts not in cache
        if texts_to_embed:
            self.logger.info(f"Generating {len(texts_to_embed)} new embeddings")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            new_embeddings = []
            
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i+batch_size]
                batch_embeddings = self.model.encode(batch)
                new_embeddings.extend(batch_embeddings)
                
            # Update cache with new embeddings
            for i, idx in enumerate(indices_to_embed):
                cache_key = hash(texts[idx][:1000])
                self.embedding_memo[cache_key] = new_embeddings[i]
                embeddings.append(new_embeddings[i])
        
        # Convert to numpy array
        if hasattr(embeddings[0], 'cpu'):
            return torch.stack(embeddings).cpu().numpy()
        else:
            return np.array(embeddings)
            
    def _perform_initial_clustering(self, embeddings):
        """
        Perform initial fast clustering on embeddings.
        
        Args:
            embeddings: Matrix of article embeddings
            
        Returns:
            numpy.ndarray: Cluster labels for each article
        """
        from sklearn.cluster import AgglomerativeClustering
        
        # Calculate adaptive threshold based on embedding distribution
        threshold = self._calculate_adaptive_threshold(embeddings)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='cosine',
            linkage='complete'
        )
        
        return clustering.fit_predict(embeddings)
        
    def _labels_to_clusters(self, labels, items):
        """
        Convert cluster labels to lists of item indices.
        
        Args:
            labels: Array of cluster labels
            items: List of items that were clustered
            
        Returns:
            List of lists, where each inner list contains indices of items in a cluster
        """
        clusters = defaultdict(list)
        
        for i, label in enumerate(labels):
            clusters[label].append(i)
            
        return list(clusters.values())
        
    async def _verify_with_lm(self, articles):
        """
        Verify article relationships using the language model.
        
        Args:
            articles: List of articles to verify
            
        Returns:
            List of verified clusters
        """
        if not self.analyzer:
            return [[article] for article in articles]
            
        # Extract text for each article
        texts = [a.get('content', '') for a in articles]
        
        # Use the cluster_articles method from the analyzer with limit on comparisons
        article_dicts = [{'content': text} for text in texts]
        try:
            result = self.analyzer.cluster_articles(
                articles=article_dicts, 
                text_extractor=lambda a: a.get('content', ''),
                similarity_threshold=0.7,
                max_comparisons=self.max_lm_comparisons
            )
            
            # Convert result to article clusters
            clusters = []
            for cluster_indices in result:
                cluster = [articles[idx-1] for idx in cluster_indices]  # Adjust for 1-based indexing
                clusters.append(cluster)
                
            return clusters
        except Exception as e:
            self.logger.error(f"Error in LM clustering: {e}")
            # Fallback to treating each article as its own cluster
            return [[article] for article in articles]
            
    def _apply_fallback_clustering(self, articles):
        """
        Apply a simpler clustering method when LM verification isn't possible.
        
        Args:
            articles: List of articles to cluster
            
        Returns:
            List of article clusters
        """
        # Use the parent method for fallback
        return super().cluster_articles(articles)
        
    @track_performance
    async def cluster_articles_async(self, articles):
        """
        Cluster articles with an optimized async implementation.
        
        Args:
            articles: List of article dicts to cluster
            
        Returns:
            List of clusters, each a list of article dicts
        """
        if not articles:
            return []
            
        # Process in batches for better memory management
        batches = [articles[i:i+self.batch_size] for i in range(0, len(articles), self.batch_size)]
        self.logger.info(f"Processing {len(articles)} articles in {len(batches)} batches")
        
        # Process batches in parallel for speed
        batch_results = await asyncio.gather(*[self._process_batch(batch) for batch in batches])
        
        # Combine results from all batches
        combined_clusters = []
        for batch_clusters in batch_results:
            combined_clusters.extend(batch_clusters)
            
        # Apply a final merging step if there are many clusters
        # This helps reconcile clusters from different batches
        if len(combined_clusters) > len(articles) / 10:
            self.logger.info(f"Applying final merging to {len(combined_clusters)} clusters")
            final_clusters = self._merge_similar_clusters(combined_clusters)
        else:
            final_clusters = combined_clusters
            
        return final_clusters
        
    def _merge_similar_clusters(self, clusters):
        """
        Merge clusters that are highly similar to reduce fragmentation.
        
        Args:
            clusters: List of clusters to potentially merge
            
        Returns:
            List of merged clusters
        """
        # Skip if only a few clusters
        if len(clusters) <= 3:
            return clusters
            
        # Generate representative embeddings for each cluster
        cluster_embeddings = []
        for cluster in clusters:
            # Use the first few articles in each cluster
            sample = cluster[:3]
            texts = [a.get('content', '') for a in sample]
            combined = " ".join(texts)
            
            # Check embedding cache
            cache_key = hash(combined[:1000])
            if cache_key in self.embedding_memo:
                embedding = self.embedding_memo[cache_key]
            else:
                embedding = self.model.encode(combined)
                self.embedding_memo[cache_key] = embedding
                
            cluster_embeddings.append(embedding)
            
        # Convert to numpy array
        if hasattr(cluster_embeddings[0], 'cpu'):
            cluster_embeddings = torch.stack(cluster_embeddings).cpu().numpy()
        else:
            cluster_embeddings = np.array(cluster_embeddings)
            
        # Calculate similarity matrix
        similarity_matrix = np.matmul(cluster_embeddings, cluster_embeddings.T)
        
        # Find pairs to merge
        merged = [False] * len(clusters)
        final_clusters = []
        
        for i in range(len(clusters)):
            if merged[i]:
                continue
                
            current_cluster = list(clusters[i])
            
            # Look for clusters to merge with this one
            for j in range(i+1, len(clusters)):
                if merged[j]:
                    continue
                    
                # Check similarity
                similarity = similarity_matrix[i, j]
                if similarity > self.confidence_threshold:
                    # Merge clusters
                    current_cluster.extend(clusters[j])
                    merged[j] = True
                    
            final_clusters.append(current_cluster)
            merged[i] = True
            
        return final_clusters
    
    @track_performance
    def cluster_with_summaries(self, articles):
        """
        Enhanced clustering that also uses summary content to further refine clusters.
        This is a complete pipeline that:
        1. Does initial embedding-based clustering
        2. Enhances with LM-based verification (only on ambiguous clusters)
        3. Performs final refinement based on generated summaries
        
        Args:
            articles: List of article dicts to cluster
            
        Returns:
            List of clusters with similar content
        """
        # Use asyncio to run the async clustering
        import asyncio
        
        try:
            # Get event loop or create one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run the async clustering
            start_time = time.time()
            lm_enhanced_clusters = loop.run_until_complete(self.cluster_articles_async(articles))
            elapsed = time.time() - start_time
            
            self.logger.info(f"Async clustering completed in {elapsed:.2f}s with {len(lm_enhanced_clusters)} clusters")
            
            # Only perform summary-based refinement if we have a reasonable number of clusters
            if len(lm_enhanced_clusters) > 1 and len(lm_enhanced_clusters) < 20 and self.summarizer:
                # This phase is optional - only do it if clusters need refinement
                final_clusters = self._detect_duplicate_summaries(lm_enhanced_clusters)
            else:
                final_clusters = lm_enhanced_clusters
                
            # Log clustering statistics
            self._log_clustering_stats(lm_enhanced_clusters, final_clusters)
            
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error in optimized clustering: {e}")
            # Fallback to basic clustering
            return super().cluster_articles(articles)
            
    def cluster_articles(self, articles):
        """
        Override the synchronous clustering method with an optimized version.
        
        Args:
            articles: List of article dicts to cluster
            
        Returns:
            List of clusters, each a list of article dicts
        """
        # This method exists for backward compatibility
        # It uses the async implementation but runs it synchronously
        
        try:
            # Use the enhanced method with summaries
            return self.cluster_with_summaries(articles)
        except Exception as e:
            self.logger.error(f"Error in optimized clustering, falling back to basic: {e}")
            # Fallback to the parent implementation
            return super().cluster_articles(articles)
            
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
            
        # Generate summaries only for clusters that need them
        for cluster in clusters:
            if not cluster:
                continue
                
            has_summary = False
            for article in cluster:
                if article.get('summary') and isinstance(article['summary'], dict) and 'summary' in article['summary']:
                    has_summary = True
                    break
                    
            # Only generate summaries for clusters that don't have them
            # and limit the number of API calls
            if not has_summary and self.summarizer and len(cluster) > 0:
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
        
        # Check for similar summaries, but use a more efficient approach
        merged_clusters = []
        skip_indices = set()
        
        # Pre-calculate article pairs to compare based on publication date
        pairs_to_check = []
        
        for i in range(len(clusters)):
            if i in skip_indices or not clusters[i]:
                continue
                
            for j in range(i+1, len(clusters)):
                if j in skip_indices or not clusters[j]:
                    continue
                    
                # Check publication dates - only compare if within 24 hours
                try:
                    date1 = datetime.fromisoformat(clusters[i][0].get('published', '').replace('Z', '+00:00'))
                    date2 = datetime.fromisoformat(clusters[j][0].get('published', '').replace('Z', '+00:00'))
                    
                    # Only compare if dates are close
                    if abs((date1 - date2).total_seconds()) < 86400:  # 24 hours
                        pairs_to_check.append((i, j))
                except:
                    # Fallback for invalid dates - always check
                    pairs_to_check.append((i, j))
                    
        # Only check the most promising pairs
        max_pairs = min(20, len(pairs_to_check))  # Limit comparisons
        pairs_to_check = pairs_to_check[:max_pairs]
        
        for i, j in pairs_to_check:
            # Skip if already merged
            if i in skip_indices or j in skip_indices:
                continue
                
            # Get summaries for comparison
            summary1 = ""
            if clusters[i][0].get('summary') and isinstance(clusters[i][0]['summary'], dict):
                summary1 = clusters[i][0]['summary'].get('summary', '')
            
            summary2 = ""
            if clusters[j][0].get('summary') and isinstance(clusters[j][0]['summary'], dict):
                summary2 = clusters[j][0]['summary'].get('summary', '')
            
            # Only compare if both have summaries
            if summary1 and summary2:
                # Use a faster comparison method first
                # Check for shared keywords in headlines
                headline1 = clusters[i][0].get('summary', {}).get('headline', '')
                headline2 = clusters[j][0].get('summary', {}).get('headline', '')
                
                # Count shared significant words
                words1 = set(w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', headline1))
                words2 = set(w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', headline2))
                common_words = words1.intersection(words2)
                
                # If they share enough significant words, use LM for final check
                if len(common_words) >= 2:
                    # Use LM for more accurate comparison
                    similarity = self._compare_texts_with_lm(summary1, summary2)
                    
                    if similarity >= similarity_threshold:
                        # Merge the clusters
                        clusters[i].extend(clusters[j])
                        skip_indices.add(j)
                        self.logger.info(f"Merged clusters {i} and {j} due to similar summaries (score: {similarity:.2f})")
        
        # Create final list of merged clusters
        for i, cluster in enumerate(clusters):
            if i not in skip_indices:
                merged_clusters.append(cluster)
        
        return merged_clusters
        
    def _compare_texts_with_lm(self, text1, text2):
        """
        Compare two texts using the language model to determine similarity.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Use the analyzer instead of direct API calls
        if self.analyzer:
            return self.analyzer.compare_article_pair(text1, text2)
        else:
            # Fallback to simpler comparison if no analyzer
            return self._fast_text_similarity(text1, text2)
            
    def _fast_text_similarity(self, text1, text2):
        """
        Calculate text similarity using a fast lexical method.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Use TF-IDF style comparison for speed
        import re
        from collections import Counter
        
        # Extract words, ignoring common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        def tokenize(text):
            # Extract significant words (4+ chars)
            words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
            return [w for w in words if w not in stopwords]
            
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        # Calculate TF-IDF vectors
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        
        all_tokens = set(tokens1 + tokens2)
        
        # Cosine similarity 
        dot_product = sum(counter1.get(token, 0) * counter2.get(token, 0) for token in all_tokens)
        magnitude1 = sum(counter1.get(token, 0) ** 2 for token in all_tokens) ** 0.5
        magnitude2 = sum(counter2.get(token, 0) ** 2 for token in all_tokens) ** 0.5
        
        if magnitude1 > 0 and magnitude2 > 0:
            return dot_product / (magnitude1 * magnitude2)
        else:
            return 0.0
            
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


# Helper function to create an optimized clusterer with the summarizer
def create_optimized_clusterer(summarizer=None):
    """
    Create an optimized article clusterer instance.
    
    Args:
        summarizer: Optional ArticleSummarizer instance
        
    Returns:
        OptimizedArticleClusterer instance
    """
    return OptimizedArticleClusterer(summarizer=summarizer)