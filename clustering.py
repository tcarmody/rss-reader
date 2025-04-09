"""Enhanced article clustering functionality using sentence transformers.

This improved version includes:
- Caching for embeddings
- Better error handling
- Adaptive distance thresholds
- Topic extraction
- Improved date handling
- Enhanced cluster merging
- Performance optimizations
"""

import logging
import torch
import numpy as np
import os
import pickle
import hashlib
import re
from datetime import datetime, timedelta
from collections import defaultdict
from functools import lru_cache
from dateutil import parser as date_parser
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import fasttext
import langdetect
from typing import List, Dict, Tuple, Any, Optional, Union

from utils.performance import track_performance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure constants
CONFIG = {
    'model_name': 'all-mpnet-base-v2',
    'fallback_model_name': 'distiluse-base-multilingual-cased-v1',
    'cache_dir': os.environ.get('EMBEDDING_CACHE_DIR', '/tmp/article_embeddings'),
    'days_threshold': int(os.environ.get('DAYS_THRESHOLD', 14)),
    'base_distance_threshold': float(os.environ.get('DISTANCE_THRESHOLD', 0.08)),
    'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
    'min_text_length': int(os.environ.get('MIN_TEXT_LENGTH', 50)),
    'min_keyword_matches': int(os.environ.get('MIN_KEYWORD_MATCHES', 2)),
    'cluster_match_threshold': float(os.environ.get('CLUSTER_MATCH_THRESHOLD', 0.5)),
    'use_hdbscan': True,
    'circuit_breaker_attempts': int(os.environ.get('CIRCUIT_BREAKER_ATTEMPTS', 3)),
    'stopwords': set(['this', 'that', 'with', 'from', 'what', 'when', 'where', 'which', 'about', 'have', 'will', 'your', 'their', 'there', 'they', 'these', 'those', 'some', 'were', 'after', 'before', 'could', 'should', 'would']),
}

# Create cache directory if it doesn't exist
os.makedirs(CONFIG['cache_dir'], exist_ok=True)


class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent repeated failures."""
    
    def __init__(self, max_failures=CONFIG['circuit_breaker_attempts'], reset_timeout=300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        
    def record_failure(self):
        """Record a failure and check if the circuit is open."""
        current_time = datetime.now()
        
        # Reset if enough time has passed
        if self.last_failure_time and (current_time - self.last_failure_time).total_seconds() > self.reset_timeout:
            self.failures = 0
            
        self.failures += 1
        self.last_failure_time = current_time
        
    def is_open(self):
        """Check if the circuit breaker is open (too many failures)."""
        if self.failures >= self.max_failures:
            # Check if we should try again
            current_time = datetime.now()
            if self.last_failure_time and (current_time - self.last_failure_time).total_seconds() > self.reset_timeout:
                self.failures = 0
                return False
            return True
        return False
    
    def reset(self):
        """Reset the circuit breaker state."""
        self.failures = 0
        self.last_failure_time = None


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
    
    def __init__(self, cache_dir=CONFIG['cache_dir']):
        self.cache_dir = cache_dir
        self.memory_cache = {}
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
            return self.memory_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                    # Store in memory cache for faster access next time
                    self.memory_cache[key] = embedding
                    return embedding
            except Exception as e:
                logging.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def set(self, text, embedding):
        """Store embedding for a text in cache."""
        key = self._get_cache_key(text)
        
        # Store in memory cache
        self.memory_cache[key] = embedding
        
        # Store on disk
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logging.warning(f"Failed to cache embedding: {e}")


class LanguageDetector:
    """Detect language of text to use appropriate models."""
    
    def __init__(self):
        self.model = None
        self.circuit_breaker = CircuitBreaker()
        
    def detect(self, text):
        """Detect the language of a text."""
        if not text or len(text) < 10:
            return "en"  # Default to English for very short texts
            
        if self.circuit_breaker.is_open():
            logging.warning("Language detection circuit breaker is open, defaulting to English")
            return "en"
            
        try:
            return langdetect.detect(text)
        except Exception as e:
            self.circuit_breaker.record_failure()
            logging.warning(f"Language detection failed: {e}. Defaulting to English.")
            return "en"


class TopicExtractor:
    """Extract keywords and topics from article clusters."""
    
    def __init__(self):
        # Enhanced stopwords list with common web terms and non-informative words
        self.extended_stopwords = list(CONFIG['stopwords']) + [
            'com', 'www', 'http', 'https', 'html', 'jpg', 'png', 'pdf',
            'says', 'said', 'according', 'reported', 'report', 'reports',
            'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
            'time', 'times', 'new', 'news', 'latest', 'update', 'updates',
            'first', 'last', 'next', 'previous', 'one', 'two', 'three', 'four', 'five',
            'article', 'story', 'post', 'read', 'view', 'click', 'find', 'get',
            'just', 'like', 'make', 'made', 'take', 'took', 'way', 'use', 'used',
            'know', 'need', 'see', 'look', 'want', 'going', 'come', 'came', 'back'
        ]
        
        # Improved vectorizer with better parameters
        self.vectorizer = CountVectorizer(
            max_features=200,  # Increased to capture more potential topics
            stop_words=self.extended_stopwords,
            min_df=2,  # Term must appear in at least 2 documents
            max_df=0.85,  # Ignore terms that appear in >85% of documents
            ngram_range=(1, 2)  # Include bigrams for more meaningful topics
        )
        
    def extract_topics(self, articles, top_n=5):
        """Extract top keywords and meaningful topics from a group of articles."""
        if not articles:
            return []
            
        # For single article clusters, use a simplified approach
        if len(articles) == 1:
            return self._extract_topics_from_single_article(articles[0], top_n)
            
        # For very small clusters, adjust vectorizer parameters
        if len(articles) < 3:
            return self._extract_topics_from_small_cluster(articles, top_n)
            
        # Standard approach for larger clusters
        return self._extract_topics_from_cluster(articles, top_n)
    
    def _extract_topics_from_single_article(self, article, top_n=5):
        """Extract topics from a single article, focusing on the title."""
        title = self._clean_text(article.get('title', ''))
        content = self._clean_text(article.get('content', ''))
        
        # Extract significant words from title
        title_words = [w.lower() for w in title.split() 
                     if len(w) > 3 and w.lower() not in self.extended_stopwords]
        
        # If we have enough words from the title, prioritize those
        if len(title_words) >= top_n:
            # Count frequencies
            word_freq = defaultdict(int)
            for word in title_words:
                word_freq[word] += 1
                
            # Get top keywords from title
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            topics = [word for word, freq in sorted_words[:top_n]]
            
            # If we have enough topics, return them
            if len(topics) >= min(3, top_n):
                return topics
        
        # If title doesn't provide enough keywords, include content
        all_text = f"{title} {content}"
        words = [w.lower() for w in all_text.split() 
                if len(w) > 3 and w.lower() not in self.extended_stopwords]
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
            
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def _extract_topics_from_small_cluster(self, articles, top_n=5):
        """Extract topics from a small cluster (2-3 articles)."""
        # Collect all text
        all_text = ""
        for article in articles:
            title = self._clean_text(article.get('title', ''))
            content = self._clean_text(article.get('content', ''))
            # Weight title more heavily
            all_text += f"{title} {title} {content} "
        
        # Extract words and filter
        words = [w.lower() for w in all_text.split() 
                if len(w) > 3 and w.lower() not in self.extended_stopwords]
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
            
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def _extract_topics_from_cluster(self, articles, top_n=5):
        """Extract topics from a larger cluster using CountVectorizer."""
        # Weight titles more heavily than content
        texts = []
        titles = []
        for article in articles:
            title = self._clean_text(article.get('title', ''))
            content = self._clean_text(article.get('content', ''))
            
            # Add title and content to their respective lists
            titles.append(title)
            texts.append(f"{title} {title} {content}")  # Weight title 2x
            
        try:
            # Create a custom vectorizer with parameters adjusted for cluster size
            cluster_vectorizer = CountVectorizer(
                max_features=200,
                stop_words=self.extended_stopwords,
                min_df=1,  # Accept terms that appear in at least 1 document
                max_df=1.0,  # Accept terms that appear in all documents
                ngram_range=(1, 2)  # Include bigrams
            )
            
            # Extract topics from full text (titles + content)
            X = cluster_vectorizer.fit_transform(texts)
            word_counts = X.sum(axis=0)
            word_counts = np.asarray(word_counts).flatten()
            
            # Get feature names and their counts
            feature_names = cluster_vectorizer.get_feature_names_out()
            word_count_pairs = [(feature_names[i], word_counts[i]) for i in range(len(feature_names))]
            
            # Filter out low-information terms
            filtered_pairs = []
            for term, count in word_count_pairs:
                # Skip terms that are just numbers or very short
                if term.isdigit() or len(term) < 3:
                    continue
                    
                # Skip terms that are years (e.g., 2025)
                if len(term) == 4 and term.isdigit() and 1900 <= int(term) <= 2100:
                    continue
                    
                filtered_pairs.append((term, count))
            
            # Sort by count and get top terms
            sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)
            top_keywords = [term for term, count in sorted_pairs[:top_n]]
            
            # If we got meaningful topics, return them
            if top_keywords:
                return top_keywords
                
            # Fallback to title-only analysis if the above didn't work well
            raise Exception("Primary topic extraction yielded poor results, falling back")
            
        except Exception as e:
            logging.warning(f"Primary topic extraction failed: {e}. Using fallback method.")
            return self._extract_topics_from_small_cluster(articles, top_n)
    
    def _clean_text(self, text):
        """Clean text by removing URLs, special characters, etc."""
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters but keep spaces and alphanumerics
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class ArticleClusterer:
    """
    Enhanced article clusterer with multiple improvements:
    - Caching for embeddings
    - Adaptive distance thresholds
    - Multiple clustering algorithms
    - Topic extraction
    - Better date handling
    - Enhanced cluster merging
    - Performance optimizations
    """
    
    def __init__(self):
        """Initialize the clusterer with necessary components."""
        # Initialize components
        self.model = None
        self.device = None
        self.embedding_cache = EmbeddingCache()
        self.language_detector = LanguageDetector()
        self.topic_extractor = TopicExtractor()
        self.model_circuit_breaker = CircuitBreaker()
        
        # Load model (will happen on first use if not here)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model and device with better error handling."""
        if self.model is not None:
            return
            
        if self.model_circuit_breaker.is_open():
            logging.warning("Model initialization circuit breaker is open, using dummy model")
            self.model = DummyModel()
            return
            
        try:
            logging.info("Initializing sentence transformer model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.model = SentenceTransformer(CONFIG['model_name'])
                self.model = self.model.to(self.device)
                logging.info(f"Model initialized on device: {self.device}")
            except Exception as primary_error:
                logging.warning(f"Primary model failed to load: {primary_error}. Trying fallback model.")
                try:
                    # Try fallback model
                    self.model = SentenceTransformer(CONFIG['fallback_model_name'])
                    self.model = self.model.to(self.device)
                    logging.info(f"Fallback model initialized on device: {self.device}")
                except Exception as fallback_error:
                    logging.error(f"Fallback model also failed: {fallback_error}")
                    raise
        except Exception as e:
            self.model_circuit_breaker.record_failure()
            logging.error(f"Error initializing model: {str(e)}")
            self.model = DummyModel()  # Fallback to dummy model
            logging.warning("Using dummy model as fallback.")
    
    def _parse_date(self, date_str):
        """Parse date string with multiple formats using dateutil."""
        if not date_str:
            return datetime.now()
            
        try:
            return date_parser.parse(date_str)
        except Exception as e:
            logging.debug(f"Date parsing failed: {e}. Using current date.")
            return datetime.now()
    
    def _calculate_adaptive_threshold(self, embeddings):
        """Calculate an adaptive distance threshold based on embedding distribution."""
        # Start with the base threshold
        threshold = CONFIG['base_distance_threshold']
        
        # If we have enough embeddings, adjust threshold based on distribution
        if len(embeddings) > 10:
            try:
                # Calculate pairwise distances for a sample of embeddings
                sample_size = min(100, len(embeddings))
                indices = np.random.choice(len(embeddings), sample_size, replace=False)
                sample_embeddings = embeddings[indices]
                
                # Calculate pairwise cosine similarities
                similarities = np.matmul(sample_embeddings, sample_embeddings.T)
                
                # Get the distribution of similarities
                sim_mean = np.mean(similarities)
                sim_std = np.std(similarities)
                
                # Adjust threshold based on distribution (higher mean similarity = lower threshold)
                # This will create more distinct clusters when articles are more similar
                adjustment = (sim_mean - 0.5) * 0.2
                
                # For more distinct clusters, we lower the threshold when similarity is high
                adjusted_threshold = threshold - adjustment
                
                # Ensure threshold stays in reasonable bounds, but allow for lower values
                # to create more distinct clusters
                final_threshold = max(0.05, min(0.25, adjusted_threshold))
                
                logging.info(f"Adaptive threshold calculation: base={threshold}, adjusted={final_threshold}")
                return final_threshold
            except Exception as e:
                logging.warning(f"Error calculating adaptive threshold: {e}. Using base threshold.")
                
        return threshold
    
    def _filter_recent_articles(self, articles, cutoff_date):
        """
        Filter articles to include only those from the recent period with improved date parsing.
        """
        recent_articles = []
        for article in articles:
            try:
                article_date = self._parse_date(article.get('published', ''))
                if article_date.replace(tzinfo=None) >= cutoff_date:
                    recent_articles.append(article)
            except Exception as e:
                logging.debug(f"Date parsing completely failed for article: {article.get('title')}. Including it anyway.")
                recent_articles.append(article)  # Include articles with unparseable dates
        return recent_articles
    
    def _prepare_articles_for_clustering(self, articles):
        """
        Prepare article texts and sources for embedding and clustering with language detection.
        """
        texts = []
        sources = []
        languages = []
        publication_times = []
        
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            
            # Weight the title more heavily by repeating it
            combined_text = f"{title} {title} {content}".strip()
            
            # Ensure we have enough text to cluster properly
            if len(combined_text) < CONFIG['min_text_length'] and title:
                # If content is minimal, just use the title repeated
                combined_text = f"{title} {title} {title}"
                
            texts.append(combined_text)
            sources.append(article.get('feed_source', ''))
            
            # Detect language
            lang = self.language_detector.detect(combined_text)
            languages.append(lang)
            
            # Get publication time for time-aware clustering
            try:
                pub_time = self._parse_date(article.get('published', ''))
                publication_times.append(pub_time)
            except:
                publication_times.append(datetime.now())
            
            logging.debug(f"Processing article for clustering: {title} (language: {lang})")
            
        return texts, sources, languages, publication_times
    
    def _get_embeddings(self, texts):
        """
        Generate embeddings for texts with caching to avoid recomputation.
        """
        if not texts:
            return np.array([])
            
        # Check cache first
        embeddings_list = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.embedding_cache.get(text)
            if cached_embedding is not None:
                embeddings_list.append((i, cached_embedding))
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                
        logging.info(f"Found {len(embeddings_list)} cached embeddings, need to encode {len(texts_to_encode)} texts")
        
        # Re-initialize model if needed (e.g., after errors)
        self._initialize_model()
        
        # Encode texts that weren't in cache
        if texts_to_encode:
            try:
                # Use progress bar for larger batches
                show_progress = len(texts_to_encode) > 10
                
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    show_progress_bar=show_progress,
                    batch_size=CONFIG['batch_size'],
                    normalize_embeddings=True
                )
                
                # Cache the new embeddings
                for idx, text, embedding in zip(indices_to_encode, texts_to_encode, new_embeddings):
                    self.embedding_cache.set(text, embedding)
                    embeddings_list.append((idx, embedding))
                    
                self.model_circuit_breaker.reset()  # Reset circuit breaker on success
                    
            except Exception as e:
                logging.error(f"Error generating embeddings: {e}")
                # Generate dummy embeddings for uncached texts
                for idx, text in zip(indices_to_encode, texts_to_encode):
                    dummy_embedding = np.random.rand(768)
                    dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)
                    embeddings_list.append((idx, dummy_embedding))
                    
                self.model_circuit_breaker.record_failure()
        
        # Sort by original index and extract just the embeddings
        embeddings_list.sort()
        final_embeddings = np.array([emb for _, emb in embeddings_list])
        return final_embeddings
    
    def _apply_clustering(self, embeddings, threshold, publication_times=None):
        """
        Apply the appropriate clustering algorithm based on configuration.
        """
        if CONFIG['use_hdbscan']:
            try:
                import hdbscan
                # Install HDBSCAN if not already installed
                try:
                    # HDBSCAN is good for finding clusters of varying densities
                    # Improved parameters for better clustering:
                    # - Lower min_cluster_size to allow smaller clusters
                    # - Adjusted min_samples for better noise handling
                    # - Using 'euclidean' metric since 'cosine' isn't directly supported
                    # - Added cluster_selection_method for better cluster extraction
                    
                    # First normalize the embeddings to unit length for cosine similarity
                    # This allows us to use euclidean distance as a proxy for cosine distance
                    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=2,  # Allow clusters as small as 2 articles
                        min_samples=1,       # More sensitive to noise points
                        metric='euclidean',  # Use euclidean on normalized vectors (equivalent to cosine)
                        cluster_selection_epsilon=threshold * 2,  # Adjust threshold for euclidean
                        cluster_selection_method='eom',  # Excess of mass (better for varying density)
                        prediction_data=True  # Keep prediction data for possible soft clustering
                    )
                    
                    # Fit the model and get cluster labels
                    labels = clusterer.fit_predict(embeddings)
                    
                    # If we got only noise points (label -1), try with more permissive settings
                    if np.all(labels == -1) and len(embeddings) > 3:
                        logging.info("All points classified as noise, trying more permissive clustering")
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=2,
                            min_samples=1,
                            metric='cosine',
                            cluster_selection_epsilon=threshold * 1.5,  # More permissive threshold
                            prediction_data=True
                        )
                        labels = clusterer.fit_predict(embeddings)
                    
                    # If we still have all noise, fall back to agglomerative clustering
                    if np.all(labels == -1) and len(embeddings) > 3:
                        logging.info("HDBSCAN still classified all points as noise, falling back to AgglomerativeClustering")
                        raise Exception("HDBSCAN produced all noise points")
                        
                    unique_labels = set(labels)
                    if -1 in unique_labels:
                        unique_labels.remove(-1)
                    logging.info(f"HDBSCAN created {len(unique_labels)} clusters")
                    return labels
                    
                except Exception as e:
                    logging.warning(f"HDBSCAN clustering failed: {e}. Falling back to AgglomerativeClustering.")
            except ImportError:
                logging.warning("HDBSCAN not installed. Falling back to AgglomerativeClustering.")
        
        # Time-weighted similarity matrix if we have publication times
        if publication_times and len(publication_times) == len(embeddings):
            try:
                # Convert times to relative hours
                base_time = min(publication_times)
                time_diffs = np.array([(t - base_time).total_seconds() / 3600 for t in publication_times])
                
                # Calculate time weight matrix (closer in time = higher weight)
                max_time_diff = max(time_diffs) if len(time_diffs) > 0 else 1
                time_weights = 1 - (np.abs(time_diffs[:, np.newaxis] - time_diffs[np.newaxis, :]) / (max_time_diff + 1e-6)) * 0.3
                
                # Create cosine similarity matrix
                cosine_sim = np.matmul(embeddings, embeddings.T)
                
                # Combine content and time similarity
                combined_sim = cosine_sim * time_weights
                
                # Convert to distance
                distances = 1 - combined_sim
                
                # Use pre-computed distances with AgglomerativeClustering
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    metric='precomputed',
                    linkage='complete'
                ).fit(distances)
                
                return clustering.labels_
            except Exception as e:
                logging.warning(f"Time-weighted clustering failed: {e}. Falling back to standard clustering.")
        
        # Default to standard AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='cosine',
            linkage='complete'
        ).fit(embeddings)
        
        return clustering.labels_
        
    def _merge_similar_clusters(self, clusters, recent_articles):
        """
        Enhanced cluster merging with topic overlap and better duplicate detection.
        """
        merged_clusters = []
        processed_keys = set()
        article_ids = {id(article): i for i, article in enumerate(recent_articles)}

        # Extract topics for each cluster for better merging decisions
        cluster_topics = {}
        for key, articles in clusters.items():
            topics = self.topic_extractor.extract_topics(articles)
            cluster_topics[key] = set(topics)
            logging.debug(f"Cluster {key} topics: {topics}")

        for key1 in clusters:
            if key1 in processed_keys:
                continue

            label1 = int(key1.split('_')[0])
            source1 = key1.split('_', 1)[1] if '_' in key1 else ''
            current_cluster = clusters[key1].copy()
            processed_keys.add(key1)
            
            # Get article indices for later duplicate detection
            current_indices = {article_ids[id(article)] for article in current_cluster if id(article) in article_ids}
            
            # Get publication times
            current_times = []
            for article in current_cluster:
                try:
                    pub_time = self._parse_date(article.get('published', ''))
                    current_times.append(pub_time)
                except:
                    current_times.append(datetime.now())
            
            # Look for similar clusters to merge
            for key2 in clusters:
                if key2 in processed_keys:
                    continue

                label2 = int(key2.split('_')[0])
                source2 = key2.split('_', 1)[1] if '_' in key2 else ''
                
                # Candidate indices for duplicate detection
                candidate_indices = {article_ids[id(article)] for article in clusters[key2] if id(article) in article_ids}
                
                # Skip if we'd be adding the same articles (avoiding duplicates)
                if current_indices.intersection(candidate_indices):
                    continue
                
                # Check if they have the same cluster label 
                # AND are from different sources (avoid duplicate articles)
                # OR they have significant topic overlap
                common_topics = len(cluster_topics[key1].intersection(cluster_topics[key2]))
                if (label1 == label2 and source1 != source2) or (
                    common_topics >= CONFIG['min_keyword_matches']
                ):
                    # Check publication time proximity
                    time_matches = 0
                    for article in clusters[key2]:
                        try:
                            article_time = self._parse_date(article.get('published', ''))
                            # Check if this article was published close to any article in the first cluster
                            for curr_time in current_times:
                                time_diff = abs((article_time - curr_time).total_seconds() / 3600)  # hours
                                if time_diff < 48:  # Within 48 hours
                                    time_matches += 1
                                    break
                        except:
                            pass
                    
                    # Merge if topics match AND timing is close
                    matches = len(cluster_topics[key1].intersection(cluster_topics[key2]))
                    if matches >= CONFIG['min_keyword_matches'] and (
                        time_matches >= len(clusters[key2]) * CONFIG['cluster_match_threshold']
                    ):
                        current_cluster.extend(clusters[key2])
                        processed_keys.add(key2)
                        logging.info(f"Merged clusters with topic overlap: {cluster_topics[key1].intersection(cluster_topics[key2])}")
                        
                        # Update indices and times
                        current_indices.update(candidate_indices)
                        for article in clusters[key2]:
                            try:
                                pub_time = self._parse_date(article.get('published', ''))
                                current_times.append(pub_time)
                            except:
                                pass

            # Only add clusters with at least one article
            if len(current_cluster) > 0:
                # Extract topics for the merged cluster
                if len(current_cluster) > 1:
                    topics = self.topic_extractor.extract_topics(current_cluster)
                    # Add topics to the first article in the cluster for reference
                    if topics and current_cluster:
                        current_cluster[0]['cluster_topics'] = topics
                
                merged_clusters.append(current_cluster)
                
            # Log the cluster titles for debugging
            if len(current_cluster) > 1:
                titles = [a.get('title', 'No title')[:50] for a in current_cluster]
                logging.info(f"Created cluster with {len(current_cluster)} articles: {titles}")
                
        return merged_clusters
    
    @track_performance()
    def cluster_articles(self, articles, days_threshold=None, distance_threshold=None):
        """
        Cluster articles based on semantic similarity with multiple improvements.
        
        Args:
            articles: List of articles to cluster
            days_threshold: Number of days to include in filtering (override config)
            distance_threshold: Similarity threshold for clustering (override config)
            
        Returns:
            list: List of article clusters with topic information
        """
        # Use config values unless overridden
        days_threshold = days_threshold or CONFIG['days_threshold']
        distance_threshold = distance_threshold or CONFIG['base_distance_threshold']
        
        try:
            if not articles:
                logging.warning("No articles to cluster")
                return []

            logging.info(f"Clustering {len(articles)} articles")

            # Filter articles by date
            current_time = datetime.now()
            cutoff_date = current_time - timedelta(days=days_threshold)
            
            recent_articles = self._filter_recent_articles(articles, cutoff_date)
            
            if not recent_articles:
                logging.warning("No recent articles to cluster")
                return []

            logging.info(f"Found {len(recent_articles)} articles from the last {days_threshold} days")

            # Get article texts and metadata for clustering
            texts, sources, languages, publication_times = self._prepare_articles_for_clustering(recent_articles)

            # Get embeddings with caching
            logging.info("Generating embeddings for articles...")
            embeddings = self._get_embeddings(texts)
            
            if len(embeddings) == 0:
                logging.warning("No embeddings were generated")
                return [[article] for article in recent_articles]
                
            # Calculate adaptive threshold if enough articles
            if len(embeddings) > 10:
                threshold = self._calculate_adaptive_threshold(embeddings)
                logging.info(f"Using adaptive distance threshold: {threshold}")
            else:
                threshold = distance_threshold
                logging.info(f"Using fixed distance threshold: {threshold}")

            # Apply clustering
            logging.info("Clustering articles...")
            labels = self._apply_clustering(embeddings, threshold, publication_times)

            # Group articles by cluster, considering source
            clusters = defaultdict(list)
            for idx, label in enumerate(labels):
                if label == -1:  # HDBSCAN noise point
                    # Each noise point gets its own "cluster"
                    clusters[f"{10000 + idx}_{sources[idx]}"].append(recent_articles[idx])
                else:
                    # Create a unique cluster key that includes the source
                    source_key = f"{label}_{sources[idx]}"
                    clusters[source_key].append(recent_articles[idx])

            # Merge similar clusters
            merged_clusters = self._merge_similar_clusters(clusters, recent_articles)

            # Extract topics for each final cluster
            for i, cluster in enumerate(merged_clusters):
                if len(cluster) > 1:  # Only extract topics for actual clusters
                    topics = self.topic_extractor.extract_topics(cluster)
                    logging.info(f"Cluster {i} ({len(cluster)} articles) topics: {topics}")
                    
                    # Add topics to each article in the cluster
                    for article in cluster:
                        article['cluster_topics'] = topics

            # Log clustering results
            logging.info(f"Created {len(merged_clusters)} clusters:")
            for i, cluster in enumerate(merged_clusters):
                titles = [a.get('title', 'No title') for a in cluster]
                topics = cluster[0].get('cluster_topics', []) if cluster else []
                logging.info(f"Cluster {i}: {len(cluster)} articles. Topics: {topics}")
                logging.info(f"Titles: {titles}")

            return merged_clusters

        except Exception as e:
            logging.error(f"Error clustering articles: {str(e)}", exc_info=True)
            # Fallback: return each article in its own cluster
            return [[article] for article in articles]