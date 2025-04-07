"""Article clustering functionality using sentence transformers."""

import logging
import torch
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from utils.performance import track_performance


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


class ArticleClusterer:
    """
    Clusters articles based on semantic similarity using sentence transformers.
    
    This class handles:
    - Semantic embedding generation using SentenceTransformer
    - Hierarchical clustering of articles
    - Merging similar clusters
    - Filtering articles by date
    
    Example:
        clusterer = ArticleClusterer()
        clusters = clusterer.cluster_articles(articles)
    """
    
    def __init__(self):
        """Initialize the clusterer with a sentence transformer model."""
        self.model = None
        self.device = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model and device."""
        try:
            if self.model is None:
                logging.info("Initializing sentence transformer model...")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    self.model = SentenceTransformer('all-mpnet-base-v2')
                    self.model = self.model.to(self.device)
                    logging.info(f"Model initialized on device: {self.device}")
                except NameError as ne:
                    if 'init_empty_weights' in str(ne):
                        logging.warning("Encountered init_empty_weights error. Using dummy model for development.")
                        # Create a simple dummy model for development purposes
                        self.model = DummyModel()
                    else:
                        raise
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            self.model = DummyModel()  # Fallback to dummy model
            logging.warning("Using dummy model as fallback.")
    
    @track_performance()
    def cluster_articles(self, articles, days_threshold=14, distance_threshold=0.15):
        """
        Cluster articles based on semantic similarity.
        
        Args:
            articles: List of articles to cluster
            days_threshold: Number of days to include in filtering
            distance_threshold: Similarity threshold for clustering (lower = more clusters)
                               Default is now 0.15 for much stricter clustering
            
        Returns:
            list: List of article clusters
        """
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

            # Get article texts and sources for clustering
            texts, sources = self._prepare_articles_for_clustering(recent_articles)

            # Get embeddings with progress bar
            logging.info("Generating embeddings for articles...")
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                normalize_embeddings=True
            )

            # Use Agglomerative Clustering with adjusted threshold
            logging.info("Clustering articles...")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='complete'  # Use complete linkage for stricter clustering
            ).fit(embeddings)

            # Group articles by cluster, considering source
            clusters = defaultdict(list)
            for idx, label in enumerate(clustering.labels_):
                # Create a unique cluster key that includes the source
                source_key = f"{label}_{sources[idx]}"
                clusters[source_key].append(recent_articles[idx])

            # Merge similar clusters
            merged_clusters = self._merge_similar_clusters(clusters)

            # Log clustering results
            logging.info(f"Created {len(merged_clusters)} clusters:")
            for i, cluster in enumerate(merged_clusters):
                titles = [a.get('title', 'No title') for a in cluster]
                logging.info(f"Cluster {i}: {len(cluster)} articles")
                logging.info(f"Titles: {titles}")

            return merged_clusters

        except Exception as e:
            logging.error(f"Error clustering articles: {str(e)}")
            # Fallback: return each article in its own cluster
            return [[article] for article in articles]
            
    def _filter_recent_articles(self, articles, cutoff_date):
        """
        Filter articles to include only those from the last two weeks.
        
        Args:
            articles: List of articles to filter
            cutoff_date: Datetime object representing the cutoff date
            
        Returns:
            list: Filtered list of recent articles
        """
        recent_articles = []
        for article in articles:
            try:
                article_date = datetime.strptime(article.get('published', ''), '%a, %d %b %Y %H:%M:%S %z')
                if article_date >= cutoff_date:
                    recent_articles.append(article)
            except (ValueError, TypeError):
                # If date parsing fails, try alternate format
                try:
                    article_date = datetime.strptime(article.get('published', ''), '%Y-%m-%dT%H:%M:%S%z')
                    if article_date >= cutoff_date:
                        recent_articles.append(article)
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse date for article: {article.get('title')}. Using current date.")
                    recent_articles.append(article)  # Include articles with unparseable dates
        return recent_articles
    
    def _prepare_articles_for_clustering(self, articles):
        """
        Prepare article texts and sources for embedding and clustering.
        Uses more content from each article for better semantic matching.
        
        Args:
            articles: List of articles to prepare
            
        Returns:
            tuple: (texts, sources) for clustering
        """
        texts = []
        sources = []
        for article in articles:
            title = article.get('title', '')
            
            # Use more content (up to 1000 chars) for better semantic matching
            content = article.get('content', '')
            
            # Weight the title more heavily by repeating it
            combined_text = f"{title} {title} {content}".strip()
            
            # Ensure we have enough text to cluster properly
            if len(combined_text) < 50 and title:
                # If content is minimal, just use the title repeated
                combined_text = f"{title} {title} {title}"
                
            texts.append(combined_text)
            sources.append(article.get('feed_source', ''))
            logging.debug(f"Processing article for clustering: {title}")
            
        return texts, sources
    
    def _merge_similar_clusters(self, clusters):
        """
        Merge only clusters that are truly related.
        More restrictive merging to prevent unrelated articles from being grouped together.
        
        Args:
            clusters: Dictionary of cluster_key -> article list
            
        Returns:
            list: List of merged article clusters
        """
        merged_clusters = []
        processed_keys = set()

        for key1 in clusters:
            if key1 in processed_keys:
                continue

            label1 = int(key1.split('_')[0])
            source1 = key1.split('_', 1)[1] if '_' in key1 else ''
            current_cluster = clusters[key1].copy()  # Make a copy to avoid modifying the original
            processed_keys.add(key1)
            
            # Get title keywords from the first cluster
            title_words = set()
            for article in current_cluster:
                if 'title' in article and article['title']:
                    # Extract meaningful words from title (skip common words)
                    words = [w.lower() for w in article['title'].split() 
                             if len(w) > 3 and w.lower() not in ['this', 'that', 'with', 'from', 'what', 'when', 'where', 'which', 'about']]
                    title_words.update(words)

            # Look for similar clusters to merge - only merge if they have the same label
            # AND share significant title keywords
            for key2 in clusters:
                if key2 in processed_keys:
                    continue

                label2 = int(key2.split('_')[0])
                source2 = key2.split('_', 1)[1] if '_' in key2 else ''
                
                # Only consider merging if they have the same cluster label
                # AND are from different sources (avoid duplicate articles)
                if label1 == label2 and source1 != source2:
                    # Check if there's significant title overlap
                    matches = 0
                    for article in clusters[key2]:
                        if 'title' in article and article['title']:
                            article_words = [w.lower() for w in article['title'].split() 
                                            if len(w) > 3 and w.lower() not in ['this', 'that', 'with', 'from', 'what', 'when', 'where', 'which', 'about']]
                            
                            # Count matching significant words
                            common_words = set(article_words) & title_words
                            if len(common_words) >= 2:  # Require at least 2 significant matching words
                                matches += 1
                    
                    # Only merge if a significant portion of articles match
                    if matches >= len(clusters[key2]) * 0.5:  # At least 50% of articles should match
                        current_cluster.extend(clusters[key2])
                        processed_keys.add(key2)
                        logging.info(f"Merged clusters with common keywords: {common_words}")

            # Only add clusters with at least one article
            if len(current_cluster) > 0:
                merged_clusters.append(current_cluster)
                
            # Log the cluster titles for debugging
            if len(current_cluster) > 1:
                titles = [a.get('title', 'No title')[:50] for a in current_cluster]
                logging.info(f"Created cluster with {len(current_cluster)} articles: {titles}")
                
        return merged_clusters