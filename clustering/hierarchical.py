"""
Hierarchical clustering for better organization of large topic groups.

Implements two-level clustering:
- Level 1: Broad topic grouping (loose threshold)
- Level 2: Sub-topic clustering within large groups (strict threshold)
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from clustering.base import ArticleClusterer

logger = logging.getLogger(__name__)


@dataclass
class SubCluster:
    """Represents a sub-cluster within a broader topic."""
    topic: Optional[str]
    articles: List[Dict[str, Any]]

    def __len__(self):
        return len(self.articles)


@dataclass
class ClusterGroup:
    """Represents a hierarchical cluster with optional sub-clusters."""
    topic: str
    sub_clusters: List[SubCluster]
    all_articles: List[Dict[str, Any]]

    def __len__(self):
        return len(self.all_articles)

    def has_subclusters(self) -> bool:
        """Check if this group has meaningful sub-clusters."""
        return len(self.sub_clusters) > 1

    def get_flat_articles(self) -> List[Dict[str, Any]]:
        """Get all articles as a flat list."""
        return self.all_articles


class HierarchicalClusterer:
    """
    Two-level hierarchical clustering for better organization.

    Features:
    - Broad topic clustering for main groups
    - Sub-topic clustering for large groups (>5 articles)
    - Automatic topic extraction at both levels
    - Fallback to flat clustering for small groups
    """

    def __init__(
        self,
        base_clusterer: Optional[ArticleClusterer] = None,
        min_subcluster_size: int = 5,
        broad_threshold: float = 0.25,
        strict_threshold: float = 0.45
    ):
        """
        Initialize hierarchical clusterer.

        Args:
            base_clusterer: Base clustering implementation to use
            min_subcluster_size: Minimum articles to create sub-clusters
            broad_threshold: Distance threshold for broad clustering
            strict_threshold: Distance threshold for sub-clustering
        """
        self.base_clusterer = base_clusterer or ArticleClusterer()
        self.min_subcluster_size = min_subcluster_size
        self.broad_threshold = broad_threshold
        self.strict_threshold = strict_threshold

        logger.info(
            f"Initialized HierarchicalClusterer with "
            f"min_subcluster_size={min_subcluster_size}, "
            f"broad_threshold={broad_threshold}, "
            f"strict_threshold={strict_threshold}"
        )

    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[ClusterGroup]:
        """
        Perform hierarchical clustering on articles.

        Args:
            articles: List of article dictionaries

        Returns:
            List of ClusterGroup objects with hierarchical structure
        """
        if not articles:
            return []

        logger.info(f"Starting hierarchical clustering for {len(articles)} articles")

        # Level 1: Create broad topic clusters (loose threshold)
        broad_clusters = self._create_broad_clusters(articles)

        logger.info(f"Created {len(broad_clusters)} broad clusters")

        # Level 2: Process each broad cluster for sub-clustering
        hierarchical_results = []
        for i, cluster in enumerate(broad_clusters):
            logger.info(
                f"Processing broad cluster {i+1}/{len(broad_clusters)} "
                f"({len(cluster)} articles)"
            )

            # Extract broad topic for this cluster
            broad_topic = self._extract_broad_topic(cluster)

            # Check if cluster is large enough for sub-clustering
            if len(cluster) >= self.min_subcluster_size:
                # Create sub-clusters
                sub_clusters = self._create_sub_clusters(cluster)

                if len(sub_clusters) > 1:
                    # Multiple sub-clusters found
                    logger.info(
                        f"Created {len(sub_clusters)} sub-clusters for '{broad_topic}'"
                    )
                    hierarchical_results.append(
                        ClusterGroup(
                            topic=broad_topic,
                            sub_clusters=sub_clusters,
                            all_articles=cluster
                        )
                    )
                else:
                    # Sub-clustering didn't split - keep as single cluster
                    logger.info(f"No meaningful sub-clusters for '{broad_topic}'")
                    hierarchical_results.append(
                        ClusterGroup(
                            topic=broad_topic,
                            sub_clusters=[SubCluster(topic=None, articles=cluster)],
                            all_articles=cluster
                        )
                    )
            else:
                # Too small for sub-clustering - keep as single cluster
                hierarchical_results.append(
                    ClusterGroup(
                        topic=broad_topic,
                        sub_clusters=[SubCluster(topic=None, articles=cluster)],
                        all_articles=cluster
                    )
                )

        logger.info(
            f"Hierarchical clustering complete: {len(hierarchical_results)} groups"
        )

        return hierarchical_results

    def _create_broad_clusters(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Create broad topic clusters with loose threshold.

        Args:
            articles: Articles to cluster

        Returns:
            List of article clusters (broad topics)
        """
        # Use base clusterer with lower threshold for broader grouping
        # We do this by temporarily modifying the config
        from clustering.base import CONFIG

        original_threshold = CONFIG.distance_threshold
        try:
            # Set looser threshold for broad clustering
            CONFIG.distance_threshold = self.broad_threshold

            # Perform clustering
            clusters = self.base_clusterer.cluster_articles(articles)

            return clusters

        finally:
            # Restore original threshold
            CONFIG.distance_threshold = original_threshold

    def _create_sub_clusters(
        self,
        cluster: List[Dict[str, Any]]
    ) -> List[SubCluster]:
        """
        Create sub-clusters within a broad cluster.

        Args:
            cluster: Articles in the broad cluster

        Returns:
            List of SubCluster objects
        """
        # Use base clusterer with stricter threshold
        from clustering.base import CONFIG

        original_threshold = CONFIG.distance_threshold
        try:
            # Set stricter threshold for sub-clustering
            CONFIG.distance_threshold = self.strict_threshold

            # Perform sub-clustering
            sub_article_groups = self.base_clusterer.cluster_articles(cluster)

            # Convert to SubCluster objects with topics
            sub_clusters = []
            for sub_group in sub_article_groups:
                topic = self._extract_specific_topic(sub_group)
                sub_clusters.append(
                    SubCluster(topic=topic, articles=sub_group)
                )

            return sub_clusters

        finally:
            # Restore original threshold
            CONFIG.distance_threshold = original_threshold

    def _extract_broad_topic(self, cluster: List[Dict[str, Any]]) -> str:
        """
        Extract a broad topic name from cluster.

        Uses the base clusterer's topic extraction with emphasis on
        high-level themes.

        Args:
            cluster: Articles in the cluster

        Returns:
            Topic string
        """
        # Get texts for topic extraction
        texts = []
        for article in cluster:
            title = article.get('title', '')
            content = article.get('content', '')
            # For broad topics, focus more on titles
            texts.append(f"{title} {title} {content[:500]}")

        # Use base clusterer's topic extraction
        topics = self.base_clusterer._extract_topics(texts, top_n=3)

        if topics:
            # Return top 2-3 keywords as broad topic
            return " ".join(topics[:3]).title()
        else:
            # Fallback to first article title
            return cluster[0].get('title', 'Unknown Topic')[:40] + "..."

    def _extract_specific_topic(self, cluster: List[Dict[str, Any]]) -> str:
        """
        Extract a specific sub-topic name from cluster.

        Uses more detailed topic extraction for sub-clusters.

        Args:
            cluster: Articles in the sub-cluster

        Returns:
            Topic string
        """
        # Get texts for topic extraction
        texts = []
        for article in cluster:
            title = article.get('title', '')
            content = article.get('content', '')
            # For specific topics, use more content
            texts.append(f"{title} {content[:1000]}")

        # Use base clusterer's topic extraction
        topics = self.base_clusterer._extract_topics(texts, top_n=5)

        if topics:
            # Return top 2-3 keywords as specific topic
            return " ".join(topics[:2]).title()
        else:
            # Fallback to first article title
            return cluster[0].get('title', 'Unknown')[:30] + "..."

    def to_flat_clusters(
        self,
        hierarchical_groups: List[ClusterGroup]
    ) -> List[List[Dict[str, Any]]]:
        """
        Convert hierarchical structure back to flat clusters.

        Useful for backwards compatibility.

        Args:
            hierarchical_groups: Hierarchical cluster groups

        Returns:
            Flat list of article clusters
        """
        flat_clusters = []

        for group in hierarchical_groups:
            if group.has_subclusters():
                # Expand sub-clusters into separate flat clusters
                for sub in group.sub_clusters:
                    flat_clusters.append(sub.articles)
            else:
                # Single cluster - add as-is
                flat_clusters.append(group.all_articles)

        return flat_clusters

    def cluster_with_topics(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[ClusterGroup]:
        """
        Convenience method that performs clustering and ensures topics.

        Compatible with base clusterer's interface.

        Args:
            articles: Articles to cluster

        Returns:
            List of ClusterGroup objects with topics
        """
        groups = self.cluster_articles(articles)

        # Add cluster_topics to all articles for compatibility
        for group in groups:
            if group.has_subclusters():
                # Add both broad and specific topics
                for sub in group.sub_clusters:
                    for article in sub.articles:
                        article['cluster_topics'] = [group.topic, sub.topic] if sub.topic else [group.topic]
                        article['broad_topic'] = group.topic
                        article['sub_topic'] = sub.topic
            else:
                # Add only broad topic
                for article in group.all_articles:
                    article['cluster_topics'] = [group.topic]
                    article['broad_topic'] = group.topic

        return groups


def create_hierarchical_clusterer(
    min_subcluster_size: int = 5,
    broad_threshold: float = 0.25,
    strict_threshold: float = 0.45
) -> HierarchicalClusterer:
    """
    Factory function to create a hierarchical clusterer.

    Args:
        min_subcluster_size: Minimum articles to create sub-clusters
        broad_threshold: Distance threshold for broad clustering
        strict_threshold: Distance threshold for sub-clustering

    Returns:
        HierarchicalClusterer instance
    """
    return HierarchicalClusterer(
        min_subcluster_size=min_subcluster_size,
        broad_threshold=broad_threshold,
        strict_threshold=strict_threshold
    )
