"""Article clustering functionality using sentence transformers."""

import logging
import torch
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from rss_reader.utils.performance import track_performance


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
                self.model = SentenceTransformer('all-mpnet-base-v2')
                self.model = self.model.to(self.device)
                logging.info(f"Model initialized on device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise
    
    @track_performance()
    def cluster_articles(self, articles, days_threshold=14, distance_threshold=0.33):
        """
        Cluster articles based on semantic similarity.
        
        Args:
            articles: List of articles to cluster
            days_threshold: Number of days to include in filtering
            distance_threshold: Similarity threshold for clustering (lower = more clusters)
            
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
        
        Args:
            articles: List of articles to prepare
            
        Returns:
            tuple: (texts, sources) for clustering
        """
        texts = []
        sources = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')[:500]  # First 500 chars of content
            source = article.get('feed_source', '')
            combined_text = f"{title} {content}".strip()
            texts.append(combined_text)
            sources.append(source)
            logging.debug(f"Processing article for clustering: {title}")
        return texts, sources
    
    def _merge_similar_clusters(self, clusters):
        """
        Merge small clusters from same source if they're similar.
        
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
            current_cluster = clusters[key1]
            processed_keys.add(key1)

            # Look for similar clusters to merge
            for key2 in clusters:
                if key2 in processed_keys:
                    continue

                label2 = int(key2.split('_')[0])
                # Only merge if they're from different sources
                if label1 == label2:
                    current_cluster.extend(clusters[key2])
                    processed_keys.add(key2)

            if len(current_cluster) > 0:  # Only add non-empty clusters
                merged_clusters.append(current_cluster)
                
        return merged_clusters