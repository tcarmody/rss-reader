"""
Progressive clustering pipeline for incremental refinement.

Provides immediate results with quick clustering, followed by
background refinement for improved accuracy.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ClusteringStage(str, Enum):
    """Stages of progressive clustering."""
    INITIAL = "initial"  # Fast, lightweight clustering
    REFINING = "refining"  # Background refinement in progress
    REFINED = "refined"  # Refinement complete
    FAILED = "failed"  # Refinement failed


@dataclass
class ProgressiveClusterResult:
    """Result from progressive clustering."""
    clusters: List[List[Dict[str, Any]]]
    stage: ClusteringStage
    topics: List[str]
    timestamp: str
    refinement_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'clusters': self.clusters,
            'stage': self.stage.value,
            'topics': self.topics,
            'timestamp': self.timestamp,
            'refinement_info': self.refinement_info
        }


class ProgressiveClusterer:
    """
    Multi-stage progressive clustering pipeline.

    Features:
    - Stage 1: Fast clustering for immediate display
    - Stage 2: Background refinement for accuracy
    - Callback system for UI updates
    - Coherence checking and cluster splitting
    """

    def __init__(
        self,
        simple_clusterer=None,
        enhanced_clusterer=None,
        coherence_threshold: float = 0.7,
        min_cluster_size_for_refinement: int = 3
    ):
        """
        Initialize progressive clusterer.

        Args:
            simple_clusterer: Fast clusterer for initial results
            enhanced_clusterer: Advanced clusterer for refinement
            coherence_threshold: Threshold for cluster coherence
            min_cluster_size_for_refinement: Minimum size to consider for refinement
        """
        self.simple_clusterer = simple_clusterer
        self.enhanced_clusterer = enhanced_clusterer
        self.coherence_threshold = coherence_threshold
        self.min_cluster_size = min_cluster_size_for_refinement

        self._refinement_tasks: Dict[str, asyncio.Task] = {}
        self._results_cache: Dict[str, ProgressiveClusterResult] = {}

        # Lazy load clusterers if not provided
        self._clusterers_loaded = False

        logger.info(
            f"Initialized ProgressiveClusterer "
            f"(coherence_threshold={coherence_threshold})"
        )

    def _ensure_clusterers(self):
        """Lazy-load clusterers if not already loaded."""
        if self._clusterers_loaded:
            return

        if self.simple_clusterer is None:
            try:
                from clustering.simple import SimpleClustering
                self.simple_clusterer = SimpleClustering()
                logger.info("Loaded SimpleClustering for Stage 1")
            except ImportError:
                logger.warning("SimpleClustering not available")

        if self.enhanced_clusterer is None:
            try:
                from clustering.base import ArticleClusterer
                self.enhanced_clusterer = ArticleClusterer()
                logger.info("Loaded ArticleClusterer for Stage 2")
            except ImportError:
                logger.warning("ArticleClusterer not available")

        self._clusterers_loaded = True

    async def cluster_progressive(
        self,
        articles: List[Dict[str, Any]],
        callback: Optional[Callable[[ProgressiveClusterResult], Awaitable[None]]] = None,
        session_id: Optional[str] = None
    ) -> ProgressiveClusterResult:
        """
        Perform progressive clustering with background refinement.

        Args:
            articles: Articles to cluster
            callback: Async callback for progress updates
            session_id: Optional session ID for tracking

        Returns:
            Initial clustering result (refinement continues in background)
        """
        self._ensure_clusterers()

        if not articles:
            return ProgressiveClusterResult(
                clusters=[],
                stage=ClusteringStage.REFINED,
                topics=[],
                timestamp=datetime.now().isoformat()
            )

        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Stage 1: Fast clustering
        logger.info(f"Stage 1: Fast clustering for {len(articles)} articles")

        initial_result = await self._fast_cluster(articles)

        # Send initial result
        if callback:
            await callback(initial_result)

        # Cache initial result
        self._results_cache[session_id] = initial_result

        # Stage 2: Start background refinement
        if self.enhanced_clusterer is not None:
            refinement_task = asyncio.create_task(
                self._refine_clusters_background(
                    session_id=session_id,
                    initial_result=initial_result,
                    articles=articles,
                    callback=callback
                )
            )
            self._refinement_tasks[session_id] = refinement_task
        else:
            # No enhanced clusterer - mark as complete
            initial_result.stage = ClusteringStage.REFINED

        return initial_result

    async def _fast_cluster(
        self,
        articles: List[Dict[str, Any]]
    ) -> ProgressiveClusterResult:
        """
        Perform fast initial clustering.

        Args:
            articles: Articles to cluster

        Returns:
            Initial clustering result
        """
        if self.simple_clusterer is None:
            # Fallback: single cluster with all articles
            return ProgressiveClusterResult(
                clusters=[articles],
                stage=ClusteringStage.INITIAL,
                topics=['All Articles'],
                timestamp=datetime.now().isoformat()
            )

        try:
            # Run clustering in executor to avoid blocking
            loop = asyncio.get_event_loop()
            clusters = await loop.run_in_executor(
                None,
                lambda: self.simple_clusterer.cluster_articles(articles)
            )

            # Extract topics
            topics = self._extract_topics(clusters)

            return ProgressiveClusterResult(
                clusters=clusters,
                stage=ClusteringStage.INITIAL,
                topics=topics,
                timestamp=datetime.now().isoformat(),
                refinement_info={
                    'cluster_count': len(clusters),
                    'method': 'simple'
                }
            )

        except Exception as e:
            logger.error(f"Fast clustering failed: {e}")
            return ProgressiveClusterResult(
                clusters=[articles],
                stage=ClusteringStage.FAILED,
                topics=['All Articles'],
                timestamp=datetime.now().isoformat(),
                refinement_info={'error': str(e)}
            )

    async def _refine_clusters_background(
        self,
        session_id: str,
        initial_result: ProgressiveClusterResult,
        articles: List[Dict[str, Any]],
        callback: Optional[Callable[[ProgressiveClusterResult], Awaitable[None]]]
    ):
        """
        Background task to refine clusters.

        Args:
            session_id: Session identifier
            initial_result: Initial clustering result
            articles: Original articles
            callback: Progress callback
        """
        try:
            # Update status
            initial_result.stage = ClusteringStage.REFINING
            if callback:
                await callback(initial_result)

            logger.info(f"Stage 2: Refining {len(initial_result.clusters)} clusters")

            refined_clusters = []
            splits_performed = 0
            merges_performed = 0

            for i, cluster in enumerate(initial_result.clusters):
                if len(cluster) >= self.min_cluster_size:
                    # Check coherence
                    coherence = await self._check_cluster_coherence(cluster)

                    if coherence < self.coherence_threshold:
                        # Split incoherent cluster
                        logger.info(
                            f"Splitting cluster {i+1} (coherence: {coherence:.2f})"
                        )
                        sub_clusters = await self._split_cluster(cluster)
                        refined_clusters.extend(sub_clusters)
                        splits_performed += 1
                    else:
                        refined_clusters.append(cluster)
                else:
                    refined_clusters.append(cluster)

            # Check for potential merges
            merged_clusters = await self._merge_similar_clusters(refined_clusters)
            merges_performed = len(refined_clusters) - len(merged_clusters)

            # Extract new topics
            topics = self._extract_topics(merged_clusters)

            # Create refined result
            refined_result = ProgressiveClusterResult(
                clusters=merged_clusters,
                stage=ClusteringStage.REFINED,
                topics=topics,
                timestamp=datetime.now().isoformat(),
                refinement_info={
                    'original_count': len(initial_result.clusters),
                    'refined_count': len(merged_clusters),
                    'splits': splits_performed,
                    'merges': merges_performed,
                    'method': 'enhanced'
                }
            )

            # Update cache
            self._results_cache[session_id] = refined_result

            # Send refined result
            if callback:
                await callback(refined_result)

            logger.info(
                f"Refinement complete: {len(initial_result.clusters)} → "
                f"{len(merged_clusters)} clusters "
                f"({splits_performed} splits, {merges_performed} merges)"
            )

        except asyncio.CancelledError:
            logger.info(f"Refinement cancelled for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Refinement failed: {e}")

            # Update with failure status
            initial_result.stage = ClusteringStage.FAILED
            initial_result.refinement_info['error'] = str(e)

            if callback:
                await callback(initial_result)

        finally:
            # Clean up task reference
            if session_id in self._refinement_tasks:
                del self._refinement_tasks[session_id]

    async def _check_cluster_coherence(
        self,
        cluster: List[Dict[str, Any]]
    ) -> float:
        """
        Check the coherence of a cluster.

        Coherence measures how semantically similar the articles are.

        Args:
            cluster: Cluster to check

        Returns:
            Coherence score (0-1)
        """
        if len(cluster) < 2:
            return 1.0

        try:
            # Use enhanced clusterer's similarity calculation
            if hasattr(self.enhanced_clusterer, '_compute_similarity_matrix'):
                loop = asyncio.get_event_loop()

                # Prepare texts
                texts = []
                for article in cluster:
                    title = article.get('title', '')
                    content = article.get('content', article.get('text', ''))[:500]
                    texts.append(f"{title} {content}")

                similarity_matrix = await loop.run_in_executor(
                    None,
                    lambda: self.enhanced_clusterer._compute_similarity_matrix(texts)
                )

                # Average pairwise similarity
                n = len(texts)
                total_sim = 0
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        total_sim += similarity_matrix[i][j]
                        count += 1

                return total_sim / count if count > 0 else 1.0

            # Fallback: use simple keyword overlap
            return self._simple_coherence_check(cluster)

        except Exception as e:
            logger.warning(f"Coherence check failed: {e}")
            return 0.5  # Assume moderate coherence

    def _simple_coherence_check(self, cluster: List[Dict[str, Any]]) -> float:
        """
        Simple coherence check using keyword overlap.

        Args:
            cluster: Cluster to check

        Returns:
            Coherence score (0-1)
        """
        if len(cluster) < 2:
            return 1.0

        # Extract keywords from each article
        all_keywords = []
        for article in cluster:
            text = f"{article.get('title', '')} {article.get('content', '')[:200]}"
            words = set(
                w.lower() for w in text.split()
                if len(w) > 3 and w.isalpha()
            )
            all_keywords.append(words)

        # Calculate average Jaccard similarity
        total_sim = 0
        count = 0
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                intersection = len(all_keywords[i] & all_keywords[j])
                union = len(all_keywords[i] | all_keywords[j])
                if union > 0:
                    total_sim += intersection / union
                    count += 1

        return total_sim / count if count > 0 else 1.0

    async def _split_cluster(
        self,
        cluster: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Split an incoherent cluster into smaller clusters.

        Args:
            cluster: Cluster to split

        Returns:
            List of smaller clusters
        """
        if len(cluster) <= 2:
            return [cluster]

        try:
            if self.enhanced_clusterer is not None:
                loop = asyncio.get_event_loop()
                sub_clusters = await loop.run_in_executor(
                    None,
                    lambda: self.enhanced_clusterer.cluster_articles(cluster)
                )

                # Filter out very small clusters
                return [c for c in sub_clusters if len(c) >= 1]

        except Exception as e:
            logger.warning(f"Cluster split failed: {e}")

        # Fallback: return original cluster
        return [cluster]

    async def _merge_similar_clusters(
        self,
        clusters: List[List[Dict[str, Any]]],
        similarity_threshold: float = 0.8
    ) -> List[List[Dict[str, Any]]]:
        """
        Merge very similar clusters.

        Args:
            clusters: Clusters to potentially merge
            similarity_threshold: Threshold for merging

        Returns:
            Merged clusters
        """
        if len(clusters) <= 1:
            return clusters

        # For now, skip complex merging
        # This could be enhanced with centroid comparison
        return clusters

    def _extract_topics(
        self,
        clusters: List[List[Dict[str, Any]]]
    ) -> List[str]:
        """
        Extract topic labels for clusters.

        Args:
            clusters: Clusters to label

        Returns:
            List of topic labels
        """
        topics = []

        for cluster in clusters:
            if not cluster:
                topics.append("Unknown")
                continue

            # Use most common words from titles
            title_words = {}
            for article in cluster:
                title = article.get('title', '')
                for word in title.split():
                    word = word.lower().strip('.,!?;:')
                    if len(word) > 3 and word.isalpha():
                        title_words[word] = title_words.get(word, 0) + 1

            # Get top 3 words
            sorted_words = sorted(
                title_words.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            if sorted_words:
                topic = ' '.join(w[0].title() for w in sorted_words)
            else:
                topic = f"Topic ({len(cluster)} articles)"

            topics.append(topic)

        return topics

    def get_result(self, session_id: str) -> Optional[ProgressiveClusterResult]:
        """
        Get cached result for a session.

        Args:
            session_id: Session identifier

        Returns:
            Cached result or None
        """
        return self._results_cache.get(session_id)

    def cancel_refinement(self, session_id: str) -> bool:
        """
        Cancel background refinement for a session.

        Args:
            session_id: Session to cancel

        Returns:
            True if cancelled, False if not found
        """
        if session_id in self._refinement_tasks:
            self._refinement_tasks[session_id].cancel()
            return True
        return False

    async def wait_for_refinement(
        self,
        session_id: str,
        timeout: Optional[float] = None
    ) -> Optional[ProgressiveClusterResult]:
        """
        Wait for refinement to complete.

        Args:
            session_id: Session to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Final result or None if timeout/not found
        """
        if session_id not in self._refinement_tasks:
            return self._results_cache.get(session_id)

        try:
            await asyncio.wait_for(
                self._refinement_tasks[session_id],
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Refinement timeout for session {session_id}")
        except asyncio.CancelledError:
            pass

        return self._results_cache.get(session_id)


# Factory function
def create_progressive_clusterer(
    simple_clusterer=None,
    enhanced_clusterer=None,
    coherence_threshold: float = 0.7
) -> ProgressiveClusterer:
    """
    Create a progressive clusterer.

    Args:
        simple_clusterer: Optional simple clusterer
        enhanced_clusterer: Optional enhanced clusterer
        coherence_threshold: Coherence threshold

    Returns:
        ProgressiveClusterer instance
    """
    return ProgressiveClusterer(
        simple_clusterer=simple_clusterer,
        enhanced_clusterer=enhanced_clusterer,
        coherence_threshold=coherence_threshold
    )
