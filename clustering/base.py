"""Enhanced clustering functionality using sentence transformers and advanced topic modeling.

This improved version includes:
- Caching for embeddings
- Better error handling
- Adaptive distance thresholds
- Advanced topic extraction with BERTopic
- Improved date handling
- Enhanced cluster merging with entity recognition
- Performance optimizations
- UMAP+HDBSCAN clustering pipeline
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
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import fasttext
import langdetect
from typing import List, Dict, Tuple, Any, Optional, Union

from common.performance import track_performance

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
    'days_threshold': int(os.environ.get('DAYS_THRESHOLD', 7)),
    'base_distance_threshold': float(os.environ.get('DISTANCE_THRESHOLD', 0.05)),
    'batch_size': int(os.environ.get('BATCH_SIZE', 32)),
    'min_text_length': int(os.environ.get('MIN_TEXT_LENGTH', 50)),
    'min_keyword_matches': int(os.environ.get('MIN_KEYWORD_MATCHES', 2)),
    'cluster_match_threshold': float(os.environ.get('CLUSTER_MATCH_THRESHOLD', 0.5)),
    'use_hdbscan': True,
    'use_bertopic': True,  # New flag to enable advanced topic modeling
    'use_umap': True,      # New flag to enable UMAP dimensionality reduction
    'use_ner': True,       # New flag to enable Named Entity Recognition
    'circuit_breaker_attempts': int(os.environ.get('CIRCUIT_BREAKER_ATTEMPTS', 3)),
    'umap_components': int(os.environ.get('UMAP_COMPONENTS', 5)),
    'umap_neighbors': int(os.environ.get('UMAP_NEIGHBORS', 15)),
    'stopwords': set(['this', 'that', 'with', 'from', 'what', 'when', 'where', 'which', 'about', 'have', 'will', 'your', 'their', 'there', 'they', 'these', 'those', 'some', 'were', 'after', 'before', 'could', 'should', 'would']),
    'min_epsilon': float(os.environ.get('MIN_EPSILON', 0.001)),  # New config for minimum epsilon value
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
    
    def __init__(self, cache_dir=CONFIG['cache_dir'], max_cache_size=1000):
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.max_cache_size = max_cache_size  # New: limit cache size
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


class NamedEntityRecognizer:
    """Extract named entities from text for better clustering."""
    
    def __init__(self):
        self.model = None
        self.circuit_breaker = CircuitBreaker()
        
    def _initialize_model(self):
        """Lazily initialize NER model."""
        if self.model is not None:
            return True
            
        if self.circuit_breaker.is_open():
            logging.warning("NER model initialization circuit breaker is open")
            return False
            
        try:
            # Try to import and load SpaCy model
            import spacy
            try:
                self.model = spacy.load("en_core_web_sm")
                return True
            except:
                # If en_core_web_sm is not available, try the smaller model
                try:
                    self.model = spacy.load("en_core_web_md")
                    return True
                except:
                    logging.warning("SpaCy models not available. Using fallback entity extraction.")
                    self.circuit_breaker.record_failure()
                    return False
        except ImportError:
            logging.warning("SpaCy not installed. Using fallback entity extraction.")
            self.circuit_breaker.record_failure()
            return False
    
    def extract_entities(self, text):
        """Extract named entities from text."""
        if not text:
            return []
            
        # Try to initialize model if not already done
        if not self._initialize_model():
            # Fallback: use regex-based extraction
            return self._extract_entities_fallback(text)
            
        try:
            doc = self.model(text[:5000])  # Limit text length for performance
            entities = []
            
            for ent in doc.ents:
                # Filter for the most relevant entity types
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "count": 1
                    })
            
            # Merge duplicate entities and count occurrences
            merged_entities = {}
            for entity in entities:
                key = f"{entity['text']}|{entity['type']}"
                if key in merged_entities:
                    merged_entities[key]["count"] += 1
                else:
                    merged_entities[key] = entity
                    
            return list(merged_entities.values())
            
        except Exception as e:
            logging.warning(f"Entity extraction failed: {e}. Using fallback method.")
            self.circuit_breaker.record_failure()
            return self._extract_entities_fallback(text)
    
    def _extract_entities_fallback(self, text):
        """Simple regex-based entity extraction as fallback."""
        if not text:
            return []
            
        # Simple regex for potential named entities (capitalized phrases)
        entity_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b'
        matches = re.findall(entity_pattern, text)
        
        # Filter and count entities
        entity_counts = defaultdict(int)
        for match in matches:
            # Skip very short matches and common words
            if len(match) < 3 or match.lower() in CONFIG['stopwords']:
                continue
            entity_counts[match] += 1
        
        # Convert to list format
        entities = []
        for entity_text, count in entity_counts.items():
            entities.append({
                "text": entity_text,
                "type": "UNKNOWN",
                "count": count
            })
        
        return sorted(entities, key=lambda x: x["count"], reverse=True)[:20]


class AdvancedTopicModeler:
    """Advanced topic modeling with BERTopic and fallback methods."""
    
    def __init__(self):
        self.model = None
        self.circuit_breaker = CircuitBreaker()
        self.fallback_extractor = None
        
    def _initialize_model(self):
        """Lazily initialize BERTopic model."""
        if self.model is not None:
            return True
            
        if self.circuit_breaker.is_open():
            logging.warning("BERTopic initialization circuit breaker is open, using fallback")
            return False
            
        try:
            # Try to import and initialize BERTopic
            try:
                from bertopic import BERTopic
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Configure vectorizer with stopwords and n-gram range
                vectorizer = CountVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )
                
                # Initialize BERTopic with minimal compute requirements
                self.model = BERTopic(
                    vectorizer_model=vectorizer,
                    calculate_probabilities=False,
                    verbose=True
                )
                logging.info("Successfully initialized BERTopic model")
                return True
                
            except ImportError:
                logging.warning("BERTopic not installed. Using fallback topic extraction.")
                self.circuit_breaker.record_failure()
                return False
                
        except Exception as e:
            logging.warning(f"Failed to initialize BERTopic: {e}. Using fallback.")
            self.circuit_breaker.record_failure()
            return False
    
    def extract_topics(self, texts, top_n=5):
        """Extract topics from a collection of texts."""
        if not texts:
            return []
            
        # Initialize fallback extractor if needed
        if self.fallback_extractor is None:
            from sklearn.feature_extraction.text import CountVectorizer
            # Enhanced stopwords list with common web terms and non-informative words
            extended_stopwords = list(CONFIG['stopwords']) + [
                'com', 'www', 'http', 'https', 'html', 'jpg', 'png', 'pdf',
                'says', 'said', 'according', 'reported', 'report', 'reports',
                'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
                'time', 'times', 'new', 'news', 'latest', 'update', 'updates',
                'first', 'last', 'next', 'previous', 'one', 'two', 'three', 'four', 'five',
                'article', 'story', 'post', 'read', 'view', 'click', 'find', 'get',
                'just', 'like', 'make', 'made', 'take', 'took', 'way', 'use', 'used',
                'know', 'need', 'see', 'look', 'want', 'going', 'come', 'came', 'back'
            ]
            
            self.fallback_extractor = CountVectorizer(
                max_features=300,
                stop_words=extended_stopwords,
                min_df=1,
                max_df=0.8,
                ngram_range=(1, 3),
                token_pattern=r'(?u)\b[A-Za-z][A-Za-Z0-9+\-_\.]*\b'
            )
        
        # For very small collections, use fallback immediately
        if len(texts) < 3 or not CONFIG['use_bertopic']:
            return self._extract_topics_fallback(texts, top_n)
            
        # Try to initialize model if not already done
        if not self._initialize_model():
            return self._extract_topics_fallback(texts, top_n)
            
        try:
            # Fit BERTopic model to the texts
            topics, _ = self.model.fit_transform(texts)
            
            # Get the top topic words
            topic_words = self.model.get_topic_info()
            
            # Extract the most common topic (excluding -1 which is noise)
            valid_topics = [t for t in set(topics) if t != -1]
            if not valid_topics:
                return self._extract_topics_fallback(texts, top_n)
                
            # Get the topic words for each topic
            result_topics = []
            for topic_id in valid_topics:
                top_words = self.model.get_topic(topic_id)
                # Extract just the words (not the weights)
                words = [word for word, _ in top_words[:top_n]]
                result_topics.extend(words)
                
            # Limit to top_n unique words
            unique_topics = []
            for topic in result_topics:
                if topic not in unique_topics:
                    unique_topics.append(topic)
                if len(unique_topics) >= top_n:
                    break
                    
            return unique_topics
            
        except Exception as e:
            logging.warning(f"BERTopic extraction failed: {e}. Using fallback method.")
            self.circuit_breaker.record_failure()
            return self._extract_topics_fallback(texts, top_n)
    
    def _extract_topics_fallback(self, texts, top_n=5):
        """Fallback method for topic extraction using CountVectorizer."""
        try:
            # Join all texts
            all_text = " ".join(texts)
            
            # Vectorize
            X = self.fallback_extractor.fit_transform([all_text])
            
            # Get feature names and counts
            feature_names = self.fallback_extractor.get_feature_names_out()
            word_counts = X.toarray()[0]
            
            # Sort by count
            sorted_idx = word_counts.argsort()[::-1]
            
            # Get top words
            top_words = [feature_names[i] for i in sorted_idx[:top_n]]
            return top_words
            
        except Exception as e:
            logging.error(f"Fallback topic extraction also failed: {e}")
            # Last resort: just return empty list
            return []


class AdvancedClusteringPipeline:
    """Advanced clustering pipeline with UMAP+HDBSCAN and fallbacks."""
    
    def __init__(self):
        self.umap_model = None
        self.hdbscan_model = None
        self.circuit_breaker = CircuitBreaker()
        
    def _initialize_models(self):
        """Lazily initialize UMAP and HDBSCAN models."""
        if self.umap_model is not None and self.hdbscan_model is not None:
            return True
            
        if self.circuit_breaker.is_open():
            logging.warning("Advanced clustering pipeline circuit breaker is open")
            return False
            
        try:
            # Try to import and initialize UMAP
            import umap
            import hdbscan
            
            self.umap_model = umap.UMAP(
                n_components=CONFIG['umap_components'],
                n_neighbors=CONFIG['umap_neighbors'],
                min_dist=0.1,
                metric='cosine'
            )
            
            # Ensure epsilon is always positive - FIX: Initialize with minimum value
            epsilon_value = max(CONFIG['min_epsilon'], 0.05)
            
            self.hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=2,
                metric='euclidean',
                cluster_selection_epsilon=epsilon_value,  # Use positive epsilon value
                prediction_data=True
            )
            
            logging.info("Successfully initialized UMAP+HDBSCAN pipeline")
            return True
            
        except ImportError:
            logging.warning("UMAP or HDBSCAN not installed. Using fallback clustering.")
            self.circuit_breaker.record_failure()
            return False
            
        except Exception as e:
            logging.warning(f"Failed to initialize clustering pipeline: {e}")
            self.circuit_breaker.record_failure()
            return False
    
    def cluster(self, embeddings, threshold, publication_times=None):
        """Apply the UMAP+HDBSCAN clustering pipeline with fallbacks."""
        # Ensure threshold is always positive - FIX
        threshold = max(CONFIG['min_epsilon'], threshold)
        
        # Check if we should use the advanced pipeline
        if not CONFIG['use_umap'] or len(embeddings) < 5:
            # For very small datasets, use the simpler method
            return self._apply_fallback_clustering(embeddings, threshold, publication_times)
            
        # Try to initialize models if not already done
        if not self._initialize_models():
            return self._apply_fallback_clustering(embeddings, threshold, publication_times)
            
        try:
            # Apply UMAP dimensionality reduction
            logging.info("Applying UMAP dimensionality reduction")
            reduced_embeddings = self.umap_model.fit_transform(embeddings)
            
            # Apply HDBSCAN clustering
            logging.info("Applying HDBSCAN clustering")
            
            # Ensure epsilon is always positive - FIX: Added this check
            epsilon_value = max(CONFIG['min_epsilon'], threshold * 1.5)
            logging.info(f"Setting cluster_selection_epsilon to {epsilon_value}")
            self.hdbscan_model.cluster_selection_epsilon = epsilon_value
            
            # For small datasets, adjust parameters
            if len(embeddings) < 10:
                self.hdbscan_model.min_cluster_size = 2
                self.hdbscan_model.min_samples = 1
            else:
                self.hdbscan_model.min_cluster_size = 2
                self.hdbscan_model.min_samples = 2
                
            labels = self.hdbscan_model.fit_predict(reduced_embeddings)
            
            # If we got only noise points, try with more permissive settings
            if np.all(labels == -1) and len(embeddings) > 3:
                logging.info("All points classified as noise, trying more permissive clustering")
                self.hdbscan_model.min_cluster_size = 2
                self.hdbscan_model.min_samples = 1
                
                # Ensure epsilon is always positive - FIX: Added this check
                epsilon_value = max(CONFIG['min_epsilon'], threshold * 2)
                logging.info(f"Setting more permissive cluster_selection_epsilon to {epsilon_value}")
                self.hdbscan_model.cluster_selection_epsilon = epsilon_value
                
                labels = self.hdbscan_model.fit_predict(reduced_embeddings)
                
            # If we still have all noise, fall back to agglomerative clustering
            if np.all(labels == -1) and len(embeddings) > 3:
                logging.info("HDBSCAN still classified all points as noise, falling back")
                return self._apply_fallback_clustering(embeddings, threshold, publication_times)
                
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            logging.info(f"UMAP+HDBSCAN created {len(unique_labels)} clusters")
            
            self.circuit_breaker.reset()  # Success, reset circuit breaker
            return labels
            
        except Exception as e:
            logging.warning(f"Advanced clustering failed: {e}. Using fallback method.")
            self.circuit_breaker.record_failure()
            return self._apply_fallback_clustering(embeddings, threshold, publication_times)
    
    def _apply_fallback_clustering(self, embeddings, threshold, publication_times=None):
        """Fallback clustering method using Agglomerative Clustering."""
        logging.info("Using fallback Agglomerative Clustering")
        
        # Ensure threshold is always positive - FIX
        threshold = max(CONFIG['min_epsilon'], threshold)
        
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


class ArticleClusterer:
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
        # Prepare article texts and metadata
        texts, sources, languages, publication_times, entities_by_index = self._prepare_articles_for_clustering(articles)
        if not texts:
            return [[article] for article in articles]
        # Compute or retrieve embeddings
        embeddings = []
        for text in texts:
            cached = self.embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                emb = self.model.encode(text)
                self.embedding_cache.set(text, emb)
                embeddings.append(emb)
        embeddings = torch.stack(embeddings).cpu().numpy() if hasattr(embeddings[0], 'cpu') else np.array(embeddings)
        # Compute adaptive threshold
        threshold = self._calculate_adaptive_threshold(embeddings)
        # Run clustering pipeline
        labels = self.clustering_pipeline.cluster(embeddings, threshold, publication_times)
        # Group articles by cluster label
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(articles[idx])
        # Convert to list, ignore noise (-1) as singletons
        clusters = [group for label, group in clusters_dict.items() if label != -1]
        # Add noise articles as their own clusters
        noise = [articles[idx] for idx, label in enumerate(labels) if label == -1]
        for article in noise:
            clusters.append([article])
        return clusters

    """
    Enhanced article clusterer with multiple improvements:
    - Caching for embeddings
    - Adaptive distance thresholds
    - Multiple clustering algorithms
    - Advanced topic extraction
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
        self.entity_recognizer = NamedEntityRecognizer()
        self.topic_modeler = AdvancedTopicModeler()
        self.clustering_pipeline = AdvancedClusteringPipeline()
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
                # FIX: Add explicit check to ensure adjusted_threshold is positive
                final_threshold = max(CONFIG['min_epsilon'], min(0.25, adjusted_threshold))
                
                logging.info(f"Adaptive threshold calculation: base={threshold}, adjusted={final_threshold}")
                return final_threshold
            except Exception as e:
                logging.warning(f"Error calculating adaptive threshold: {e}. Using base threshold.")
                
        # Ensure return value is never negative or zero - FIX
        return max(CONFIG['min_epsilon'], threshold)
    
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
        entities_by_index = {}  # Store entity data by article index
        
        for i, article in enumerate(articles):
            title = article.get('title', '')
            content = article.get('content', '')
            
            # Extract entities from title and content if enabled
            entities = []
            if CONFIG['use_ner']:
                try:
                    # Weight title entities more heavily
                    title_entities = self.entity_recognizer.extract_entities(title)
                    content_entities = self.entity_recognizer.extract_entities(content)
                    
                    # Combine entities, prioritizing those from the title
                    title_entity_texts = {e["text"]: e for e in title_entities}
                    content_entity_texts = {e["text"]: e for e in content_entities}
                    
                    # Add title entities first (they're more important)
                    for text, entity in title_entity_texts.items():
                        entity["count"] *= 2  # Weight title entities more heavily
                        entities.append(entity)
                    
                    # Add content entities that weren't in the title
                    for text, entity in content_entity_texts.items():
                        if text not in title_entity_texts:
                            entities.append(entity)
                    
                    # Save entities for later use in merging
                    entities_by_index[i] = entities
                    
                    # Create entity text for embedding enhancement
                    entity_text = " ".join([e["text"] for e in entities for _ in range(e["count"])])
                except Exception as e:
                    logging.debug(f"Entity extraction failed: {e}")
                    entity_text = ""
            else:
                # Extract potential entity names and important keywords from the title using simple regex
                import re
                
                # Look for capitalized words and phrases that might be entity names
                entity_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b'
                extracted_entities = re.findall(entity_pattern, title)
                
                # Repeat entity names multiple times to give them more weight
                entity_text = ' '.join([entity for entity in extracted_entities for _ in range(3)])
            
            # Weight the title more heavily by repeating it
            combined_text = f"{title} {title} {entity_text} {content}".strip()
            
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
            
        return texts, sources, languages, publication_times, entities_by_index
    
    @track_performance
    def cluster_with_topics(self, articles, merge_clusters=True):
        """
        Cluster articles with topic extraction for each cluster.
        
        Args:
            articles: List[dict] - articles to cluster
            merge_clusters: bool - whether to merge related clusters
            
        Returns:
            List[List[dict]]: Clusters of articles with topics
        """
        try:
            if not articles:
                return []
                
            # Filter for recent articles if days_threshold is set
            if CONFIG['days_threshold'] > 0:
                cutoff_date = datetime.now() - timedelta(days=CONFIG['days_threshold'])
                articles = self._filter_recent_articles(articles, cutoff_date)
                
            # Basic clustering
            clusters = self.cluster_articles(articles)
            
            # Extract topics for each cluster
            for cluster in clusters:
                try:
                    if not cluster:
                        continue
                        
                    # Extract text for topic modeling
                    texts = []
                    for article in cluster:
                        title = article.get('title', '')
                        content = article.get('content', '')
                        combined = f"{title} {content}"
                        if combined.strip():
                            texts.append(combined)
                            
                    # Get top topics
                    top_topics = self.topic_modeler.extract_topics(texts, top_n=5)
                    
                    # Add topics to cluster info
                    if top_topics:
                        for article in cluster:
                            article['cluster_topics'] = top_topics
                            
                    # Extract top entities if available
                    top_entities = []
                    entity_counter = defaultdict(int)
                    
                    # Collect entities from all articles in cluster
                    for article in cluster:
                        article_entities = article.get('entities', [])
                        for entity in article_entities:
                            entity_key = f"{entity.get('text')}|{entity.get('type')}"
                            entity_counter[entity_key] += entity.get('count', 1)
                    
                    # Get top entities by count
                    top_entity_keys = sorted(entity_counter.keys(), 
                                            key=lambda k: entity_counter[k], 
                                            reverse=True)[:5]
                                            
                    # Convert back to format (text, count)
                    for key in top_entity_keys:
                        text, entity_type = key.split('|')
                        count = entity_counter[key]
                        top_entities.append((text, count, entity_type))
                        
                    # Add entities to cluster info
                    if top_entities:
                        for article in cluster:
                            article['cluster_entities'] = [entity[0] for entity in top_entities]
                except Exception as e:
                    logging.warning(f"Error extracting entities for cluster: {e}")

            # Log clustering results
            logging.info(f"Created {len(clusters)} clusters:")
            for i, cluster in enumerate(clusters):
                titles = [a.get('title', 'No title') for a in cluster]
                topics = cluster[0].get('cluster_topics', []) if cluster else []
                entities = cluster[0].get('cluster_entities', []) if cluster else []
                
                logging.info(f"Cluster {i}: {len(cluster)} articles. Topics: {topics}")
                if entities:
                    logging.info(f"Cluster {i} entities: {entities}")
                logging.info(f"Titles: {titles}")

            return clusters

        except Exception as e:
            logging.error(f"Error clustering articles: {str(e)}", exc_info=True)
            # Fallback: return each article in its own cluster
            return [[article] for article in articles]