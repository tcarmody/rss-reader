"""
Semantic Summary Cache using Vector Database (ChromaDB).

Provides semantic similarity-based caching for article summaries,
enabling cache hits for similar (not just identical) articles.
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Try to import chromadb, but make it optional
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Semantic cache will use fallback mode.")


class SemanticSummaryCache:
    """
    Vector database-backed semantic cache for article summaries.

    Features:
    - Semantic similarity search for cache lookups
    - Cross-article knowledge reuse
    - Foundation for related articles feature
    - Enables summary search functionality

    Falls back to exact matching if ChromaDB is unavailable.
    """

    def __init__(
        self,
        persist_directory: str = "./vector_cache",
        collection_name: str = "summaries",
        similarity_threshold: float = 0.92,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the semantic cache.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the vector collection
            similarity_threshold: Minimum similarity for cache hit (0-1)
            embedding_model: Optional embedding model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

        self.client = None
        self.collection = None
        self._embedder = None
        self._fallback_cache: Dict[str, Dict] = {}

        self._stats = {
            'semantic_hits': 0,
            'exact_hits': 0,
            'misses': 0,
            'stores': 0
        }

        self._initialize()

    def _initialize(self):
        """Initialize the vector database connection."""
        if not CHROMADB_AVAILABLE:
            logger.info("Using fallback exact-match cache (ChromaDB not available)")
            return

        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection with cosine similarity
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(
                f"Initialized SemanticSummaryCache with {self.collection.count()} entries"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            logger.info("Falling back to exact-match cache")
            self.client = None
            self.collection = None

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.embedding_model or 'all-MiniLM-L6-v2'
                self._embedder = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")

            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedder = False  # Mark as failed

        return self._embedder if self._embedder else None

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if unavailable
        """
        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            # Truncate text to reasonable length
            text_truncated = text[:2000]
            embedding = embedder.encode(text_truncated, convert_to_numpy=True)
            return embedding.tolist()

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _generate_id(self, article: Dict[str, str]) -> str:
        """
        Generate a unique ID for an article.

        Args:
            article: Article dictionary

        Returns:
            Unique ID string
        """
        url = article.get('url', '')
        text = article.get('text', article.get('content', ''))[:500]

        key = f"{url}:{text}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def _generate_cache_key(self, text: str) -> str:
        """Generate exact-match cache key."""
        return hashlib.md5(text[:1000].encode('utf-8')).hexdigest()

    async def get_cached_summary(
        self,
        article: Dict[str, str],
        style: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Look up cached summary by semantic similarity.

        Args:
            article: Article dictionary with 'text', 'title', 'url'
            style: Summary style

        Returns:
            Cached summary dict or None if not found
        """
        text = article.get('text', article.get('content', ''))

        # Try semantic search if available
        if self.collection is not None:
            result = await self._semantic_lookup(text, style)
            if result:
                return result

        # Fall back to exact match
        cache_key = self._generate_cache_key(text)
        if cache_key in self._fallback_cache:
            cached = self._fallback_cache[cache_key]
            if cached.get('style') == style:
                self._stats['exact_hits'] += 1
                logger.debug(f"Exact cache hit for article")
                return {
                    **cached,
                    'cache_hit': True,
                    'cache_type': 'exact'
                }

        self._stats['misses'] += 1
        return None

    async def _semantic_lookup(
        self,
        text: str,
        style: str
    ) -> Optional[Dict[str, Any]]:
        """
        Perform semantic similarity lookup.

        Args:
            text: Article text
            style: Summary style

        Returns:
            Cached summary or None
        """
        embedding = self._generate_embedding(text)
        if embedding is None:
            return None

        try:
            # Query for similar articles
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=['documents', 'metadatas', 'distances'],
                where={"style": style} if style != "default" else None
            )

            # Check if we found a similar enough match
            if (results['distances'] and
                results['distances'][0] and
                len(results['distances'][0]) > 0):

                distance = results['distances'][0][0]
                similarity = 1 - distance  # Convert distance to similarity

                if similarity >= self.similarity_threshold:
                    cached = results['metadatas'][0][0]
                    self._stats['semantic_hits'] += 1

                    logger.info(
                        f"Semantic cache hit (similarity: {similarity:.3f})"
                    )

                    return {
                        'summary': cached.get('summary', ''),
                        'headline': cached.get('headline', ''),
                        'style': cached.get('style', 'default'),
                        'cache_hit': True,
                        'cache_type': 'semantic',
                        'similarity': similarity,
                        'original_url': cached.get('url', '')
                    }

            return None

        except Exception as e:
            logger.warning(f"Semantic lookup failed: {e}")
            return None

    async def store_summary(
        self,
        article: Dict[str, str],
        summary: Dict[str, str],
        style: str = "default"
    ) -> bool:
        """
        Store a summary in the cache.

        Args:
            article: Original article
            summary: Generated summary
            style: Summary style

        Returns:
            True if stored successfully
        """
        text = article.get('text', article.get('content', ''))
        url = article.get('url', '')

        # Store in vector database if available
        if self.collection is not None:
            success = await self._store_in_vector_db(article, summary, style)
            if success:
                self._stats['stores'] += 1
                return True

        # Fall back to exact-match storage
        cache_key = self._generate_cache_key(text)
        self._fallback_cache[cache_key] = {
            'summary': summary.get('summary', ''),
            'headline': summary.get('headline', ''),
            'style': style,
            'url': url,
            'created_at': datetime.now().isoformat()
        }
        self._stats['stores'] += 1

        return True

    async def _store_in_vector_db(
        self,
        article: Dict[str, str],
        summary: Dict[str, str],
        style: str
    ) -> bool:
        """
        Store summary in vector database.

        Args:
            article: Original article
            summary: Generated summary
            style: Summary style

        Returns:
            True if stored successfully
        """
        text = article.get('text', article.get('content', ''))
        embedding = self._generate_embedding(text)

        if embedding is None:
            return False

        try:
            article_id = self._generate_id(article)

            # Check if already exists
            existing = self.collection.get(ids=[article_id])
            if existing and existing['ids']:
                # Update existing entry
                self.collection.update(
                    ids=[article_id],
                    embeddings=[embedding],
                    documents=[text[:2000]],
                    metadatas=[{
                        'summary': summary.get('summary', ''),
                        'headline': summary.get('headline', ''),
                        'style': style,
                        'url': article.get('url', ''),
                        'title': article.get('title', ''),
                        'created_at': datetime.now().isoformat()
                    }]
                )
            else:
                # Add new entry
                self.collection.add(
                    ids=[article_id],
                    embeddings=[embedding],
                    documents=[text[:2000]],
                    metadatas=[{
                        'summary': summary.get('summary', ''),
                        'headline': summary.get('headline', ''),
                        'style': style,
                        'url': article.get('url', ''),
                        'title': article.get('title', ''),
                        'created_at': datetime.now().isoformat()
                    }]
                )

            logger.debug(f"Stored summary in vector database: {article_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store in vector database: {e}")
            return False

    async def find_similar_articles(
        self,
        article: Dict[str, str],
        n_results: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find articles similar to the given article.

        Args:
            article: Article to find similar articles for
            n_results: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar article summaries with similarity scores
        """
        if self.collection is None:
            return []

        text = article.get('text', article.get('content', ''))
        embedding = self._generate_embedding(text)

        if embedding is None:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results + 1,  # Extra to exclude self
                include=['documents', 'metadatas', 'distances']
            )

            similar_articles = []
            article_id = self._generate_id(article)

            for i, (distance, metadata) in enumerate(
                zip(results['distances'][0], results['metadatas'][0])
            ):
                # Skip if it's the same article
                if i < len(results['ids'][0]) and results['ids'][0][i] == article_id:
                    continue

                similarity = 1 - distance
                if similarity >= min_similarity:
                    similar_articles.append({
                        'headline': metadata.get('headline', ''),
                        'summary': metadata.get('summary', ''),
                        'url': metadata.get('url', ''),
                        'title': metadata.get('title', ''),
                        'similarity': similarity
                    })

            return similar_articles[:n_results]

        except Exception as e:
            logger.warning(f"Similar article search failed: {e}")
            return []

    async def search_summaries(
        self,
        query: str,
        n_results: int = 10,
        style: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search summaries by semantic query.

        Args:
            query: Search query text
            n_results: Maximum results to return
            style: Optional filter by style

        Returns:
            List of matching summaries
        """
        if self.collection is None:
            return []

        embedding = self._generate_embedding(query)
        if embedding is None:
            return []

        try:
            where_filter = {"style": style} if style else None

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'],
                where=where_filter
            )

            search_results = []
            for i, (distance, metadata) in enumerate(
                zip(results['distances'][0], results['metadatas'][0])
            ):
                similarity = 1 - distance
                search_results.append({
                    'headline': metadata.get('headline', ''),
                    'summary': metadata.get('summary', ''),
                    'url': metadata.get('url', ''),
                    'title': metadata.get('title', ''),
                    'style': metadata.get('style', 'default'),
                    'relevance': similarity
                })

            return search_results

        except Exception as e:
            logger.warning(f"Summary search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_hits = self._stats['semantic_hits'] + self._stats['exact_hits']
        total_requests = total_hits + self._stats['misses']

        return {
            'semantic_hits': self._stats['semantic_hits'],
            'exact_hits': self._stats['exact_hits'],
            'total_hits': total_hits,
            'misses': self._stats['misses'],
            'stores': self._stats['stores'],
            'hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'collection_size': self.collection.count() if self.collection else len(self._fallback_cache),
            'chromadb_available': CHROMADB_AVAILABLE and self.collection is not None
        }

    async def cleanup_old_entries(self, days: int = 30) -> int:
        """
        Remove entries older than specified days.

        Args:
            days: Maximum age in days

        Returns:
            Number of entries removed
        """
        if self.collection is None:
            return 0

        cutoff = datetime.now().isoformat()[:10]  # Current date
        # Note: ChromaDB doesn't support date comparisons in where clause
        # This would require iterating through entries
        # For now, this is a placeholder for future implementation

        logger.info(f"Cleanup requested for entries older than {days} days")
        return 0


# Global instance
_semantic_cache: Optional[SemanticSummaryCache] = None


def get_semantic_cache(
    persist_directory: str = "./vector_cache",
    similarity_threshold: float = 0.92
) -> SemanticSummaryCache:
    """
    Get or create the global semantic cache instance.

    Args:
        persist_directory: Storage directory
        similarity_threshold: Similarity threshold for cache hits

    Returns:
        SemanticSummaryCache instance
    """
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticSummaryCache(
            persist_directory=persist_directory,
            similarity_threshold=similarity_threshold
        )
    return _semantic_cache


async def get_or_create_summary(
    article: Dict[str, str],
    summarizer,
    similarity_threshold: float = 0.92,
    style: str = "default"
) -> Dict[str, Any]:
    """
    Get cached summary or create new one.

    Convenience function that integrates semantic cache with summarizer.

    Args:
        article: Article to summarize
        summarizer: Summarizer instance (FastSummarizer or similar)
        similarity_threshold: Threshold for semantic matching
        style: Summary style

    Returns:
        Summary dictionary with cache_hit flag
    """
    cache = get_semantic_cache(similarity_threshold=similarity_threshold)

    # Try cache first
    cached = await cache.get_cached_summary(article, style)
    if cached:
        return cached

    # Generate new summary
    text = article.get('text', article.get('content', ''))
    title = article.get('title', '')
    url = article.get('url', '')

    summary = summarizer.summarize(
        text=text,
        title=title,
        url=url,
        style=style
    )

    # Store in cache
    await cache.store_summary(article, summary, style)

    return {**summary, 'cache_hit': False}
