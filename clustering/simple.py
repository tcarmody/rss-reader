"""Lightweight clustering using mini embeddings + keywords.

A middle-path approach that combines:
- Small sentence transformer model (80MB vs 2GB+)
- Keyword extraction and overlap scoring
- Simple distance-based clustering
- Reliable fallbacks
"""

import logging
import os
import pickle
import hashlib
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Try to import sentence transformers, fallback gracefully
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using keyword-only clustering")

logger = logging.getLogger(__name__)

class SimpleClustering:
    """Lightweight clustering with embeddings + keywords hybrid approach."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'simple_cluster_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Configuration with environment variable support
        self.similarity_threshold = float(os.environ.get('MIN_SIMILARITY_THRESHOLD', 0.3))
        self.keyword_weight = 0.25  # 25% keyword overlap, 75% semantic similarity (increased semantic weight)
        self.semantic_weight = 0.75
        self.min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', 2))
        self.max_days_old = int(os.environ.get('MAX_CLUSTERING_DAYS', 7))

        # Common entities to filter (expanded to reduce false positives)
        self.common_entities = {
            # AI/ML terms
            'ai', 'artificial intelligence', 'chatgpt', 'gpt', 'gpt-4', 'gpt-5',
            'openai', 'anthropic', 'claude', 'gemini', 'llm', 'genai', 'agi',
            'machine learning', 'deep learning', 'data science', 'model',
            'neural network', 'chatbot', 'large language model',
            # Companies (tech)
            'google', 'microsoft', 'meta', 'amazon', 'apple', 'nvidia', 'tesla',
            'deepmind', 'hugging face', 'mistral', 'cohere', 'facebook',
            'intel', 'amd', 'qualcomm', 'ibm', 'oracle', 'salesforce',
            # AI startups
            'deepseek', 'perplexity', 'inflection', 'character', 'replika',
            # Geographic/Political
            'trump', 'china', 'chinese', 'india', 'indian', 'europe', 'european',
            'america', 'american', 'government', 'federal', 'state', 'national',
            # Generic tech/news terms
            'tech', 'technology', 'startup', 'funding', 'billion', 'million',
            'ceo', 'launch', 'announce', 'update', 'release', 'new', 'report',
            'news', 'today', 'says', 'could', 'will', 'according', 'platform',
            'service', 'product', 'company', 'business', 'industry',
            # News aggregators
            'google news', 'techmeme',
            # Financial/business terms
            'invest', 'investment', 'investor', 'deal', 'agreement', 'partnership',
            'revenue', 'profit', 'stock', 'market', 'share', 'percent', 'growth',
            # Generic actions
            'make', 'makes', 'making', 'develop', 'developing', 'build', 'building',
            'create', 'creating', 'work', 'working', 'help', 'helping', 'use', 'using',
            # Hardware terms
            'chip', 'chips', 'processor', 'semiconductor', 'hardware', 'device',
            # Time/quantity
            'year', 'month', 'week', 'first', 'latest', 'next'
        }

        # Initialize model if available
        self.model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use lightweight model - only 80MB
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded lightweight embedding model")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.model = None
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract meaningful keywords from text, filtering common entities."""
        if not text:
            return []

        # Simple but effective keyword extraction
        # Remove common stop words and short words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'shall', 'can', 'a', 'an', 'this', 'that', 'these', 'those'
        }

        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [
            w for w in text.split()
            if len(w) > 3 and w not in stop_words and w not in self.common_entities
        ]

        # Count frequency and return top keywords
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def calculate_keyword_overlap(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate overlap ratio between two keyword lists."""
        if not keywords1 or not keywords2:
            return 0.0
        
        set1, set2 = set(keywords1), set(keywords2)
        overlap = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return overlap / union if union > 0 else 0.0
    
    def get_embedding_cache_path(self, text: str) -> str:
        """Get cache file path for text embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"embed_{text_hash}.pkl")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching."""
        if not self.model:
            return None
        
        # Check cache first
        cache_path = self.get_embedding_cache_path(text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        # Generate embedding
        try:
            # Use title + first sentence for efficiency
            text_snippet = text[:200]  # Limit text length
            embedding = self.model.encode([text_snippet])[0]
            
            # Cache the result
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
            
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    def calculate_similarity(self, article1: Dict, article2: Dict) -> float:
        """Calculate hybrid similarity score between two articles."""
        # Extract text for comparison
        text1 = f"{article1.get('title', '')} {article1.get('summary', '')}"
        text2 = f"{article2.get('title', '')} {article2.get('summary', '')}"
        
        # Get keywords
        keywords1 = self.extract_keywords(text1)
        keywords2 = self.extract_keywords(text2)
        keyword_sim = self.calculate_keyword_overlap(keywords1, keywords2)
        
        # Get semantic similarity if model available
        semantic_sim = 0.0
        if self.model:
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            
            if emb1 is not None and emb2 is not None:
                # Calculate cosine similarity
                sim_matrix = cosine_similarity([emb1], [emb2])
                semantic_sim = sim_matrix[0][0]
        
        # Combine scores
        if self.model and semantic_sim > 0:
            return self.semantic_weight * semantic_sim + self.keyword_weight * keyword_sim
        else:
            # Fall back to keywords only
            return keyword_sim
    
    def filter_recent_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles to recent ones only."""
        cutoff_date = datetime.now() - timedelta(days=self.max_days_old)
        
        filtered = []
        for article in articles:
            # Try to parse date
            article_date = None
            if 'date_added' in article:
                try:
                    if isinstance(article['date_added'], str):
                        article_date = datetime.fromisoformat(article['date_added'].replace('Z', '+00:00'))
                    else:
                        article_date = article['date_added']
                except:
                    pass
            
            # Include if recent or no date
            if article_date is None or article_date >= cutoff_date:
                filtered.append(article)
        
        return filtered
    
    def cluster_articles(self, articles: List[Dict]) -> List[List[Dict]]:
        """Cluster articles using hybrid similarity approach."""
        if not articles:
            return []
        
        # Filter to recent articles
        recent_articles = self.filter_recent_articles(articles)
        if len(recent_articles) < 2:
            return [[article] for article in recent_articles]
        
        logger.info(f"Clustering {len(recent_articles)} recent articles")
        
        # Calculate pairwise similarities
        n = len(recent_articles)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_similarity(recent_articles[i], recent_articles[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim  # Symmetric
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Use improved clustering - find the best connected components
        clusters = []
        assigned = set()
        
        # First pass: find articles that meet the threshold with any other article
        connections = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    connections[i].append(j)
                    connections[j].append(i)
        
        # Second pass: build clusters from connected components
        for i in range(n):
            if i in assigned:
                continue
            
            # Build cluster using breadth-first search
            cluster_indices = set()
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                if current in assigned:
                    continue
                    
                cluster_indices.add(current)
                assigned.add(current)
                
                # Add connected articles to queue
                for neighbor in connections[current]:
                    if neighbor not in assigned and neighbor not in cluster_indices:
                        queue.append(neighbor)
            
            # Convert indices to articles
            cluster = [recent_articles[idx] for idx in cluster_indices]
            
            # Only keep clusters with minimum size
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
            else:
                # Add as individual articles
                clusters.extend([[article] for article in cluster])
        
        logger.info(f"Created {len(clusters)} clusters")
        return clusters
    
    def extract_cluster_topic(self, cluster: List[Dict]) -> str:
        """Extract a simple topic name for a cluster."""
        if not cluster:
            return "Unknown"
        
        # Collect all keywords from cluster
        all_keywords = []
        for article in cluster:
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            keywords = self.extract_keywords(text, max_keywords=3)
            all_keywords.extend(keywords)
        
        # Find most common keywords
        keyword_counts = Counter(all_keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(2)]
        
        if top_keywords:
            return " ".join(top_keywords).title()
        else:
            # Fallback to first article title
            return cluster[0].get('title', 'Unknown')[:30] + "..."
    
    def cluster_with_topics(self, articles: List[Dict]) -> List[Dict]:
        """Cluster articles and return with topic information."""
        clusters = self.cluster_articles(articles)
        
        result = []
        for i, cluster in enumerate(clusters):
            topic = self.extract_cluster_topic(cluster)
            result.append({
                'cluster_id': i,
                'topic': topic,
                'articles': cluster,
                'size': len(cluster)
            })
        
        # Sort by cluster size (largest first)
        result.sort(key=lambda x: x['size'], reverse=True)
        return result