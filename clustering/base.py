"""Enhanced clustering functionality using sentence transformers and advanced topic modeling.

This improved version includes:
- Simplified architecture with the same API
- Caching for embeddings
- Better error handling
- UMAP+HDBSCAN clustering pipeline with fallbacks
- Topic extraction
- Performance optimizations
- Asynchronous processing support
- Improved content-focused weighting to reduce false positives from headline terms
"""

import logging
import torch
import numpy as np
import os
import pickle
import hashlib
import re
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union

from common.performance import track_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Simplified configuration
class ClusteringConfig:
    """Simplified configuration with defaults and environment overrides"""
    
    def __init__(self):
        # Core parameters with sensible defaults
        self.model_name = os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-large-v2')
        self.fallback_model_name = os.environ.get('FALLBACK_MODEL', 'distiluse-base-multilingual-cased-v1')
        self.cache_dir = os.environ.get('EMBEDDING_CACHE_DIR', '/tmp/article_embeddings')
        # Reduced threshold for stricter clustering
        self.distance_threshold = float(os.environ.get('DISTANCE_THRESHOLD', 0.15))
        self.min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', 2))
        self.days_threshold = int(os.environ.get('DAYS_THRESHOLD', 7))
        self.use_cache = os.environ.get('USE_EMBEDDING_CACHE', 'true').lower() == 'true'
        
        # Advanced features - enable/disable
        self.use_advanced_clustering = os.environ.get('USE_ADVANCED_CLUSTERING', 'true').lower() == 'true'
        self.min_text_length = int(os.environ.get('MIN_TEXT_LENGTH', 50))
        self.cache_max_size = int(os.environ.get('CACHE_MAX_SIZE', 1000))
        
        # UMAP parameters
        self.umap_components = int(os.environ.get('UMAP_COMPONENTS', 5))
        self.umap_neighbors = int(os.environ.get('UMAP_NEIGHBORS', 15))
        
        # Async parameters
        self.async_batch_size = int(os.environ.get('ASYNC_BATCH_SIZE', 8))
        
        # List of common stopwords - simplified from previous version
        self.stopwords = set([
            'this', 'that', 'with', 'from', 'what', 'when', 'where', 'which', 'about', 
            'have', 'will', 'your', 'their', 'there', 'they', 'these', 'those', 'some', 
            'were', 'after', 'before', 'could', 'should', 'would'
        ])
        
        # Common entities that might cause false clustering
        self.common_entities = set([
            'AI', 'Artificial Intelligence', 'ChatGPT', 'Machine Learning', 'Deep Learning', 
            'Data Science', 'LLM', 'Large Language Model', 'GPT', 'Model', 'Neural Network',
            'ChatBot', 'Tech', 'Technology', 'Report', 'News', 'Today'
        ])

# Create global configuration
CONFIG = ClusteringConfig()

# Create cache directory if it doesn't exist
os.makedirs(CONFIG.cache_dir, exist_ok=True)


class DummyModel:
    """A dummy model that returns random embeddings when SentenceTransformer fails to load."""
    
    def __init__(self):
        self.embedding_dim = 768  # Standard embedding dimension
        logging.warning("Using DummyModel which will return random embeddings")
    
    def encode(self, sentences, **kwargs):
        """Return random embeddings for the input sentences."""
        if isinstance(sentences, str):
            sentences = [sentences]
        batch_size = len(sentences)
        return torch.rand(batch_size, self.embedding_dim)
    
    def to(self, device):
        """Mock method to match SentenceTransformer API."""
        return self


class EmbeddingCache:
    """Cache for article embeddings to avoid recomputing them."""
    
    def __init__(self, cache_dir=CONFIG.cache_dir, max_cache_size=CONFIG.cache_max_size):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.max_cache_size = max_cache_size
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, text):
        """Generate a cache key for a text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key):
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, text):
        """Get embedding for a text from cache."""
        key = self._get_cache_key(text)
        
        # Check memory cache first
        if key in self.memory_cache:
            # LRU-like behavior: move to the "front" by removing and re-adding
            value = self.memory_cache.pop(key)
            self.memory_cache[key] = value
            return value
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                    # Store in memory cache for faster access next time
                    self._add_to_memory_cache(key, embedding)
                    return embedding
            except Exception as e:
                logging.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def _add_to_memory_cache(self, key, value):
        """Add item to memory cache with LRU eviction if needed."""
        # If cache is full, remove oldest item (first item in dict)
        if len(self.memory_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.memory_cache))
            self.memory_cache.pop(oldest_key)
        self.memory_cache[key] = value
    
    def set(self, text, embedding):
        """Store embedding for a text in cache."""
        key = self._get_cache_key(text)
        
        # Store in memory cache
        self._add_to_memory_cache(key, embedding)
        
        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logging.warning(f"Failed to cache embedding: {e}")


class ArticleClusterer:
    """
    Simplified article clusterer with the same public API.
    """
    
    def __init__(self):
        """Initialize the clusterer with necessary components."""
        # Initialize model and cache
        self.model = None
        self.embedding_cache = EmbeddingCache() if CONFIG.use_cache else None
        self.logger = logging.getLogger("ArticleClusterer")
        
        # Load model (will happen on first use if not here)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model with fallback options."""
        if self.model is not None:
            return
            
        try:
            logging.info(f"Loading model: {CONFIG.model_name}")
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(CONFIG.model_name)
                # Move to GPU if available
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(self.device)
                logging.info(f"Model loaded on {self.device}")
            except Exception as primary_error:
                logging.warning(f"Primary model failed: {primary_error}. Trying fallback.")
                try:
                    self.model = SentenceTransformer(CONFIG.fallback_model_name)
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model = self.model.to(self.device)
                    logging.info(f"Fallback model loaded on {self.device}")
                except Exception as fallback_error:
                    logging.error(f"Fallback model also failed: {fallback_error}")
                    self.model = DummyModel()
                    logging.warning("Using dummy model")
        except ImportError:
            logging.error("SentenceTransformer not installed")
            self.model = DummyModel()
    
    def _parse_date(self, date_str):
        """Parse date string with simple format handling."""
        if not date_str:
            return datetime.now()
            
        try:
            from dateutil import parser as date_parser
            return date_parser.parse(date_str)
        except Exception:
            return datetime.now()
    
    def _filter_recent_articles(self, articles, cutoff_date):
        """Filter articles to include only those from the recent period."""
        if CONFIG.days_threshold <= 0:
            return articles
            
        recent_articles = []
        for article in articles:
            try:
                article_date = self._parse_date(article.get('published', ''))
                if article_date >= cutoff_date:
                    recent_articles.append(article)
            except Exception:
                # If date parsing fails, include the article anyway
                recent_articles.append(article)
        return recent_articles
    
    def _prepare_article_texts(self, articles):
        """Prepare article texts for embedding with content-focused weighting."""
        texts = []
        publication_times = []
        
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            
            # Extract potential entities from both title and content
            entity_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b'
            title_entities = re.findall(entity_pattern, title)
            content_entities = re.findall(entity_pattern, content[:500])  # Limit to first 500 chars for efficiency
            
            # Filter out common entities that might cause false clustering
            filtered_title_entities = [e for e in title_entities if e not in CONFIG.common_entities]
            filtered_content_entities = [e for e in content_entities if e not in CONFIG.common_entities]
            
            # Get significant content entities (up to 10)
            significant_entities = ' '.join(filtered_content_entities[:10])
            
            # Build combined text with content-focused weighting
            # Content appears in full, title appears once, significant entities are included
            combined_text = f"{content} {content} {title} {significant_entities}".strip()
            
            # Ensure minimum text length
            if len(combined_text) < CONFIG.min_text_length and title:
                combined_text = f"{title} {title}"
                
            texts.append(combined_text)
            
            # Get publication time
            try:
                pub_time = self._parse_date(article.get('published', ''))
                publication_times.append(pub_time)
            except:
                publication_times.append(datetime.now())
                
        return texts, publication_times
    
    def _get_embeddings(self, texts):
        """Get embeddings for texts, using cache if available."""
        if not self.model:
            self._initialize_model()
            
        embeddings = []
        for text in texts:
            embedding = None
            
            # Try to get from cache
            if self.embedding_cache:
                embedding = self.embedding_cache.get(text)
                
            # If not in cache, compute
            if embedding is None:
                try:
                    embedding = self.model.encode(text)
                    if self.embedding_cache:
                        self.embedding_cache.set(text, embedding)
                except Exception as e:
                    logging.error(f"Error encoding text: {e}")
                    # Create random embedding as fallback
                    embedding = np.random.rand(768)
                    
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    async def _get_embeddings_async(self, texts):
        """Async version of _get_embeddings."""
        if not self.model:
            self._initialize_model()
            
        embeddings = []
        # Process texts in batches for better efficiency
        batch_size = CONFIG.async_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            # Create tasks for cache lookups
            cache_tasks = []
            for text in batch_texts:
                if self.embedding_cache:
                    # Use to_thread to make cache operations non-blocking
                    task = asyncio.create_task(
                        asyncio.to_thread(self.embedding_cache.get, text)
                    )
                    cache_tasks.append((text, task))
                else:
                    cache_tasks.append((text, None))
            
            # Process cache results and compute missing embeddings
            compute_texts = []
            compute_indices = []
            
            for idx, (text, task) in enumerate(cache_tasks):
                if task:
                    try:
                        embedding = await task
                        if embedding is not None:
                            batch_embeddings.append((idx, embedding))
                            continue
                    except Exception as e:
                        self.logger.warning(f"Cache lookup error: {e}")
                
                compute_texts.append(text)
                compute_indices.append(idx)
            
            # Compute embeddings for cache misses
            if compute_texts:
                try:
                    # Run encoding in a separate thread to avoid blocking
                    computed = await asyncio.to_thread(
                        self.model.encode, compute_texts
                    )
                    
                    # Store in cache and add to results
                    for sub_idx, (text_idx, text) in enumerate(zip(compute_indices, compute_texts)):
                        embedding = computed[sub_idx]
                        batch_embeddings.append((text_idx, embedding))
                        
                        if self.embedding_cache:
                            # Cache in background
                            asyncio.create_task(
                                asyncio.to_thread(self.embedding_cache.set, text, embedding)
                            )
                except Exception as e:
                    self.logger.error(f"Error encoding batch: {e}")
                    # Create random embeddings as fallback
                    for text_idx, text in zip(compute_indices, compute_texts):
                        embedding = np.random.rand(768)
                        batch_embeddings.append((text_idx, embedding))
            
            # Sort by original index and add to final results
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
            
            # Short sleep to allow other tasks to run
            await asyncio.sleep(0)
            
        return np.array(embeddings)
    
    def _cluster_embeddings(self, embeddings, threshold=None):
        """Cluster embeddings using UMAP+HDBSCAN if available, with fallback."""
        if threshold is None:
            threshold = CONFIG.distance_threshold
            
        # Ensure positive threshold
        threshold = max(0.001, threshold)
        
        # Try UMAP+HDBSCAN if available and enabled
        if CONFIG.use_advanced_clustering and len(embeddings) >= 5:
            try:
                import umap
                import hdbscan
                
                # UMAP dimensionality reduction
                reducer = umap.UMAP(
                    n_components=CONFIG.umap_components,
                    n_neighbors=CONFIG.umap_neighbors,
                    min_dist=0.1,
                    metric='cosine'
                )
                reduced = reducer.fit_transform(embeddings)
                
                # HDBSCAN clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=CONFIG.min_cluster_size,
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_epsilon=max(0.001, threshold),
                    prediction_data=True
                )
                labels = clusterer.fit_predict(reduced)
                
                # If all points classified as noise, try more permissive settings
                if np.all(labels == -1) and len(embeddings) > 3:
                    logging.info("All points classified as noise, trying more permissive settings")
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=2,
                        min_samples=1,
                        metric='euclidean',
                        cluster_selection_epsilon=max(0.001, threshold * 2),
                        prediction_data=True
                    )
                    labels = clusterer.fit_predict(reduced)
                    
                # If still all noise, fall back to agglomerative
                if np.all(labels == -1):
                    raise ValueError("HDBSCAN classified all points as noise")
                    
                logging.info(f"UMAP+HDBSCAN created clusters with labels: {set(labels)}")
                return labels
                
            except Exception as e:
                logging.warning(f"Advanced clustering failed: {e}. Using fallback.")
        
        # Fallback to Agglomerative Clustering
        try:
            from sklearn.cluster import AgglomerativeClustering
            logging.info("Using Agglomerative Clustering")
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            logging.info(f"Agglomerative clustering created clusters with labels: {set(labels)}")
            return labels
            
        except Exception as e:
            logging.error(f"Clustering failed: {e}. Returning singleton clusters.")
            # Last resort: each item in its own cluster
            return np.arange(len(embeddings))
    
    async def _cluster_embeddings_async(self, embeddings, threshold=None):
        """Async version of _cluster_embeddings."""
        # Run CPU-intensive clustering in a separate thread
        return await asyncio.to_thread(
            self._cluster_embeddings, embeddings, threshold
        )
    
    def _extract_topics(self, cluster_texts, top_n=5):
        """Extract topics from cluster texts using simple keyword extraction."""
        if not cluster_texts:
            return []
            
        try:
            # Try to use sklearn's CountVectorizer for better extraction
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Configure vectorizer with stopwords and common entities to filter out
            stopwords = list(CONFIG.stopwords) + list(CONFIG.common_entities)
            
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words=stopwords,
                ngram_range=(1, 2)
            )
            
            # Combine texts
            all_text = " ".join(cluster_texts)
            
            # Vectorize
            X = vectorizer.fit_transform([all_text])
            
            # Get feature names and counts
            feature_names = vectorizer.get_feature_names_out()
            word_counts = X.toarray()[0]
            
            # Sort by count
            sorted_idx = word_counts.argsort()[::-1]
            
            # Get top words
            top_words = [feature_names[i] for i in sorted_idx[:top_n]]
            return top_words
            
        except ImportError:
            # Fallback to simple regex-based extraction
            logging.info("Using simple regex-based topic extraction")
            
            # Combine all texts
            all_text = " ".join(cluster_texts)
            
            # Find capitalized phrases (potential entities/topics)
            entity_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b'
            entities = re.findall(entity_pattern, all_text)
            
            # Count occurrences and filter common entities
            entity_counts = defaultdict(int)
            for entity in entities:
                if (len(entity) >= 3 and 
                    entity.lower() not in CONFIG.stopwords and
                    entity not in CONFIG.common_entities):
                    entity_counts[entity] += 1
            
            # Get top entities
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [entity for entity, _ in top_entities]
        
        except Exception as e:
            logging.error(f"Topic extraction failed: {e}")
            return []
    
    async def _extract_topics_async(self, cluster_texts, top_n=5):
        """Async version of _extract_topics."""
        return await asyncio.to_thread(
            self._extract_topics, cluster_texts, top_n
        )
    
    def cluster_articles(self, articles):
        """
        Cluster a list of article dicts into groups of similar articles.
        Args:
            articles: List[dict] - articles to cluster (must have 'title', 'content', etc.)
        Returns:
            List[List[dict]]: List of clusters, each a list of article dicts
        """
        if not articles:
            return []
            
        # Filter recent articles if days_threshold is set
        if CONFIG.days_threshold > 0:
            cutoff_date = datetime.now() - timedelta(days=CONFIG.days_threshold)
            articles = self._filter_recent_articles(articles, cutoff_date)
            
        if not articles:
            return []
            
        # Prepare article texts
        texts, publication_times = self._prepare_article_texts(articles)
        
        if not texts:
            return [[article] for article in articles]
            
        # Get embeddings
        embeddings = self._get_embeddings(texts)
        
        # Perform clustering
        labels = self._cluster_embeddings(embeddings)
        
        # Group articles by cluster
        clusters_dict = defaultdict(list)
        for i, label in enumerate(labels):
            clusters_dict[label].append(articles[i])
            
        # Convert to list of clusters
        clusters = list(clusters_dict.values())
        
        return clusters
    
    async def cluster_articles_async(self, articles):
        """
        Async version of cluster_articles.
        """
        if not articles:
            return []
            
        # Filter recent articles if days_threshold is set
        if CONFIG.days_threshold > 0:
            cutoff_date = datetime.now() - timedelta(days=CONFIG.days_threshold)
            articles = self._filter_recent_articles(articles, cutoff_date)
            
        if not articles:
            return []
            
        # Prepare article texts
        texts, publication_times = self._prepare_article_texts(articles)
        
        if not texts:
            return [[article] for article in articles]
            
        # Get embeddings asynchronously
        embeddings = await self._get_embeddings_async(texts)
        
        # Perform clustering asynchronously
        labels = await self._cluster_embeddings_async(embeddings)
        
        # Group articles by cluster
        clusters_dict = defaultdict(list)
        for i, label in enumerate(labels):
            clusters_dict[label].append(articles[i])
            
        # Convert to list of clusters
        clusters = list(clusters_dict.values())
        
        return clusters
    
    @track_performance
    def cluster_with_topics(self, articles, merge_clusters=True):
        """
        Cluster articles with topic extraction for each cluster.
        
        Args:
            articles: List[dict] - articles to cluster
            merge_clusters: bool - whether to merge related clusters (currently ignored)
            
        Returns:
            List[List[dict]]: Clusters of articles with topics
        """
        try:
            # Get clusters
            clusters = self.cluster_articles(articles)
            
            # Extract topics for each cluster
            for cluster in clusters:
                try:
                    if not cluster:
                        continue
                        
                    # Get texts for topic extraction
                    texts = []
                    for article in cluster:
                        title = article.get('title', '')
                        content = article.get('content', '')
                        texts.append(f"{title} {content}")
                        
                    # Extract topics
                    topics = self._extract_topics(texts)
                    
                    # Add topics to each article in cluster
                    if topics:
                        for article in cluster:
                            article['cluster_topics'] = topics
                            
                except Exception as e:
                    logging.warning(f"Error extracting topics for cluster: {e}")
            
            # Log results
            logging.info(f"Created {len(clusters)} clusters")
            for i, cluster in enumerate(clusters):
                topics = cluster[0].get('cluster_topics', []) if cluster else []
                logging.info(f"Cluster {i}: {len(cluster)} articles. Topics: {topics}")
            
            return clusters
            
        except Exception as e:
            logging.error(f"Error in cluster_with_topics: {e}")
            # Fallback to singleton clusters
            return [[article] for article in articles]
    
    @track_performance
    async def cluster_with_topics_async(self, articles, merge_clusters=True):
        """
        Async version of cluster_with_topics.
        """
        try:
            # Get clusters asynchronously
            clusters = await self.cluster_articles_async(articles)
            
            # Extract topics for each cluster concurrently
            topic_tasks = []
            for cluster_idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                    
                # Get texts for topic extraction
                texts = []
                for article in cluster:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    texts.append(f"{title} {content}")
                    
                # Create task for topic extraction
                task = asyncio.create_task(self._extract_topics_async(texts))
                topic_tasks.append((cluster_idx, task))
                
            # Process results as they complete
            for cluster_idx, task in topic_tasks:
                try:
                    topics = await task
                    
                    # Add topics to each article in cluster
                    if topics:
                        for article in clusters[cluster_idx]:
                            article['cluster_topics'] = topics
                except Exception as e:
                    logging.warning(f"Error extracting topics for cluster {cluster_idx}: {e}")
            
            # Log results
            logging.info(f"Created {len(clusters)} clusters asynchronously")
            for i, cluster in enumerate(clusters):
                topics = cluster[0].get('cluster_topics', []) if cluster else []
                logging.info(f"Cluster {i}: {len(cluster)} articles. Topics: {topics}")
            
            return clusters
            
        except Exception as e:
            logging.error(f"Error in async cluster_with_topics: {e}")
            # Fallback to singleton clusters
            return [[article] for article in articles]