"""
Cross-cluster relationship mapping for topic discovery.

Builds a relationship graph between clusters to enable
related topic discovery and navigation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Types of relationships between clusters."""
    SAME_STORY = "same_story"  # Different coverage of same event (sim > 0.7)
    RELATED_TOPIC = "related_topic"  # Same broader topic (sim 0.55-0.7)
    TANGENTIAL = "tangential"  # Loosely related (sim 0.4-0.55)
    CONTINUATION = "continuation"  # Follow-up story
    CONTRAST = "contrast"  # Opposing viewpoints


@dataclass
class ClusterNode:
    """Represents a cluster in the relationship graph."""
    id: str
    topic: str
    articles: List[Dict[str, Any]]
    centroid: Optional[List[float]] = None
    keywords: Set[str] = field(default_factory=set)

    def __len__(self):
        return len(self.articles)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'topic': self.topic,
            'article_count': len(self.articles),
            'keywords': list(self.keywords)
        }


@dataclass
class ClusterEdge:
    """Represents a relationship between two clusters."""
    source_id: str
    target_id: str
    similarity: float
    relationship_type: RelationshipType
    shared_keywords: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'similarity': self.similarity,
            'type': self.relationship_type.value,
            'shared_keywords': list(self.shared_keywords)
        }


@dataclass
class ClusterGraph:
    """Graph structure for cluster relationships."""
    nodes: Dict[str, ClusterNode] = field(default_factory=dict)
    edges: List[ClusterEdge] = field(default_factory=list)

    def add_node(self, node: ClusterNode):
        """Add a cluster node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: ClusterEdge):
        """Add a relationship edge to the graph."""
        self.edges.append(edge)

    def get_related_clusters(
        self,
        cluster_id: str,
        min_similarity: float = 0.4
    ) -> List[Tuple[ClusterNode, ClusterEdge]]:
        """
        Get clusters related to the given cluster.

        Args:
            cluster_id: ID of the cluster to find relations for
            min_similarity: Minimum similarity threshold

        Returns:
            List of (related_cluster, edge) tuples
        """
        related = []
        for edge in self.edges:
            if edge.similarity < min_similarity:
                continue

            if edge.source_id == cluster_id:
                if edge.target_id in self.nodes:
                    related.append((self.nodes[edge.target_id], edge))
            elif edge.target_id == cluster_id:
                if edge.source_id in self.nodes:
                    related.append((self.nodes[edge.source_id], edge))

        # Sort by similarity descending
        related.sort(key=lambda x: x[1].similarity, reverse=True)
        return related

    def get_clusters_by_relationship(
        self,
        relationship_type: RelationshipType
    ) -> List[Tuple[ClusterNode, ClusterNode, ClusterEdge]]:
        """
        Get all cluster pairs with a specific relationship type.

        Args:
            relationship_type: Type of relationship to filter by

        Returns:
            List of (cluster_a, cluster_b, edge) tuples
        """
        results = []
        for edge in self.edges:
            if edge.relationship_type == relationship_type:
                if edge.source_id in self.nodes and edge.target_id in self.nodes:
                    results.append((
                        self.nodes[edge.source_id],
                        self.nodes[edge.target_id],
                        edge
                    ))
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges],
            'stats': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'relationship_types': self._count_relationship_types()
            }
        }

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = {}
        for edge in self.edges:
            type_name = edge.relationship_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts


class ClusterRelationshipMapper:
    """
    Maps relationships between clusters.

    Features:
    - Centroid-based similarity calculation
    - Relationship type classification
    - Keyword overlap analysis
    - Graph construction for navigation
    """

    def __init__(
        self,
        same_story_threshold: float = 0.7,
        related_topic_threshold: float = 0.55,
        tangential_threshold: float = 0.4,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the relationship mapper.

        Args:
            same_story_threshold: Similarity threshold for same story
            related_topic_threshold: Threshold for related topics
            tangential_threshold: Threshold for tangential relations
            embedding_model: Optional embedding model name
        """
        self.same_story_threshold = same_story_threshold
        self.related_topic_threshold = related_topic_threshold
        self.tangential_threshold = tangential_threshold
        self.embedding_model = embedding_model

        self._embedder = None
        self._embedder_loaded = False

        logger.info(
            f"Initialized ClusterRelationshipMapper "
            f"(thresholds: same={same_story_threshold}, "
            f"related={related_topic_threshold}, "
            f"tangential={tangential_threshold})"
        )

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder_loaded:
            return self._embedder

        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.embedding_model or 'all-MiniLM-L6-v2'
            self._embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")

        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._embedder = None

        self._embedder_loaded = True
        return self._embedder

    def map_relationships(
        self,
        clusters: List[List[Dict[str, Any]]],
        topics: Optional[List[str]] = None
    ) -> ClusterGraph:
        """
        Build a relationship graph from clusters.

        Args:
            clusters: List of article clusters
            topics: Optional list of topic labels

        Returns:
            ClusterGraph with nodes and edges
        """
        if not clusters:
            return ClusterGraph()

        logger.info(f"Mapping relationships for {len(clusters)} clusters")

        graph = ClusterGraph()

        # Create nodes
        nodes = self._create_nodes(clusters, topics)
        for node in nodes:
            graph.add_node(node)

        # Calculate centroids
        self._calculate_centroids(nodes)

        # Find relationships
        edges = self._find_relationships(nodes)
        for edge in edges:
            graph.add_edge(edge)

        logger.info(
            f"Built relationship graph: {len(graph.nodes)} nodes, "
            f"{len(graph.edges)} edges"
        )

        return graph

    def _create_nodes(
        self,
        clusters: List[List[Dict[str, Any]]],
        topics: Optional[List[str]]
    ) -> List[ClusterNode]:
        """
        Create cluster nodes from article clusters.

        Args:
            clusters: Article clusters
            topics: Optional topic labels

        Returns:
            List of ClusterNode objects
        """
        nodes = []

        for i, cluster in enumerate(clusters):
            # Generate ID
            cluster_id = f"cluster_{i}"

            # Get or generate topic
            if topics and i < len(topics):
                topic = topics[i]
            else:
                topic = self._extract_topic(cluster)

            # Extract keywords
            keywords = self._extract_keywords(cluster)

            node = ClusterNode(
                id=cluster_id,
                topic=topic,
                articles=cluster,
                keywords=keywords
            )
            nodes.append(node)

        return nodes

    def _extract_topic(self, cluster: List[Dict[str, Any]]) -> str:
        """Extract a topic label from cluster articles."""
        if not cluster:
            return "Unknown"

        # Count word frequencies in titles
        word_freq = {}
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'that', 'this', 'these', 'those', 'it', 'its', 'new', 'says'
        }

        for article in cluster:
            title = article.get('title', '')
            words = title.lower().split()
            for word in words:
                word = word.strip('.,!?;:\'"()[]{}')
                if len(word) > 2 and word not in stop_words and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Get top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [w[0].title() for w in sorted_words[:3]]

        return ' '.join(top_words) if top_words else f"Topic {len(cluster)}"

    def _extract_keywords(
        self,
        cluster: List[Dict[str, Any]],
        max_keywords: int = 10
    ) -> Set[str]:
        """
        Extract keywords from cluster articles.

        Args:
            cluster: Article cluster
            max_keywords: Maximum keywords to extract

        Returns:
            Set of keywords
        """
        word_freq = {}
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'that', 'this', 'these', 'those', 'it', 'its', 'new', 'says',
            'said', 'also', 'just', 'like', 'more', 'most', 'other', 'some',
            'such', 'than', 'then', 'there', 'when', 'where', 'which', 'while',
            'who', 'why', 'how', 'what', 'all', 'any', 'both', 'each', 'few',
            'many', 'much', 'own', 'same', 'very', 'can', 'into', 'only',
            'over', 'through', 'under', 'about', 'after', 'before', 'between'
        }

        for article in cluster:
            text = f"{article.get('title', '')} {article.get('content', '')[:300]}"
            words = text.lower().split()

            for word in words:
                word = word.strip('.,!?;:\'"()[]{}')
                if len(word) > 3 and word not in stop_words and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return set(w[0] for w in sorted_words[:max_keywords])

    def _calculate_centroids(self, nodes: List[ClusterNode]):
        """
        Calculate centroid embeddings for all nodes.

        Args:
            nodes: List of cluster nodes
        """
        embedder = self._get_embedder()
        if embedder is None:
            logger.warning("No embedder available; using keyword-based similarity")
            return

        try:
            for node in nodes:
                # Create cluster text representation
                texts = []
                for article in node.articles[:5]:  # Limit for efficiency
                    title = article.get('title', '')
                    content = article.get('content', article.get('text', ''))[:200]
                    texts.append(f"{title} {content}")

                if texts:
                    # Get embeddings and average them
                    embeddings = embedder.encode(texts, convert_to_numpy=True)
                    node.centroid = np.mean(embeddings, axis=0).tolist()

        except Exception as e:
            logger.warning(f"Failed to calculate centroids: {e}")

    def _find_relationships(
        self,
        nodes: List[ClusterNode]
    ) -> List[ClusterEdge]:
        """
        Find relationships between all cluster pairs.

        Args:
            nodes: List of cluster nodes with centroids

        Returns:
            List of relationship edges
        """
        edges = []

        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes[i + 1:], i + 1):
                # Calculate similarity
                similarity = self._calculate_similarity(node_a, node_b)

                if similarity >= self.tangential_threshold:
                    # Classify relationship
                    rel_type = self._classify_relationship(
                        node_a, node_b, similarity
                    )

                    # Find shared keywords
                    shared = node_a.keywords & node_b.keywords

                    edge = ClusterEdge(
                        source_id=node_a.id,
                        target_id=node_b.id,
                        similarity=similarity,
                        relationship_type=rel_type,
                        shared_keywords=shared
                    )
                    edges.append(edge)

        return edges

    def _calculate_similarity(
        self,
        node_a: ClusterNode,
        node_b: ClusterNode
    ) -> float:
        """
        Calculate similarity between two clusters.

        Uses centroid similarity if available, falls back to keyword overlap.

        Args:
            node_a: First cluster
            node_b: Second cluster

        Returns:
            Similarity score (0-1)
        """
        # Try centroid similarity
        if node_a.centroid is not None and node_b.centroid is not None:
            try:
                # Cosine similarity
                a = np.array(node_a.centroid)
                b = np.array(node_b.centroid)

                dot = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)

                if norm_a > 0 and norm_b > 0:
                    return float(dot / (norm_a * norm_b))
            except Exception as e:
                logger.warning(f"Centroid similarity failed: {e}")

        # Fallback to keyword overlap (Jaccard similarity)
        if node_a.keywords and node_b.keywords:
            intersection = len(node_a.keywords & node_b.keywords)
            union = len(node_a.keywords | node_b.keywords)
            if union > 0:
                return intersection / union

        return 0.0

    def _classify_relationship(
        self,
        node_a: ClusterNode,
        node_b: ClusterNode,
        similarity: float
    ) -> RelationshipType:
        """
        Classify the type of relationship between clusters.

        Args:
            node_a: First cluster
            node_b: Second cluster
            similarity: Calculated similarity

        Returns:
            RelationshipType enum value
        """
        if similarity >= self.same_story_threshold:
            return RelationshipType.SAME_STORY
        elif similarity >= self.related_topic_threshold:
            return RelationshipType.RELATED_TOPIC
        else:
            return RelationshipType.TANGENTIAL

    def find_story_threads(
        self,
        graph: ClusterGraph
    ) -> List[List[ClusterNode]]:
        """
        Find connected story threads in the graph.

        A story thread is a chain of same_story or related_topic clusters.

        Args:
            graph: Cluster relationship graph

        Returns:
            List of story threads (each a list of connected clusters)
        """
        visited = set()
        threads = []

        def dfs(node_id: str, thread: List[ClusterNode]):
            """Depth-first search to build thread."""
            if node_id in visited:
                return
            visited.add(node_id)

            if node_id in graph.nodes:
                thread.append(graph.nodes[node_id])

            # Follow strong relationships
            for edge in graph.edges:
                if edge.relationship_type not in [
                    RelationshipType.SAME_STORY,
                    RelationshipType.RELATED_TOPIC
                ]:
                    continue

                if edge.source_id == node_id and edge.target_id not in visited:
                    dfs(edge.target_id, thread)
                elif edge.target_id == node_id and edge.source_id not in visited:
                    dfs(edge.source_id, thread)

        # Start DFS from each unvisited node
        for node_id in graph.nodes:
            if node_id not in visited:
                thread = []
                dfs(node_id, thread)
                if len(thread) > 1:  # Only include multi-cluster threads
                    threads.append(thread)

        # Sort threads by size
        threads.sort(key=len, reverse=True)
        return threads

    def get_related_topics(
        self,
        graph: ClusterGraph,
        cluster_id: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get topics related to a specific cluster.

        Args:
            graph: Cluster relationship graph
            cluster_id: ID of the cluster
            max_results: Maximum related topics to return

        Returns:
            List of related topic dictionaries
        """
        related = graph.get_related_clusters(cluster_id)

        results = []
        for node, edge in related[:max_results]:
            results.append({
                'topic': node.topic,
                'cluster_id': node.id,
                'article_count': len(node.articles),
                'similarity': edge.similarity,
                'relationship': edge.relationship_type.value,
                'shared_keywords': list(edge.shared_keywords)
            })

        return results


# Factory function
def create_relationship_mapper(
    same_story_threshold: float = 0.7,
    related_topic_threshold: float = 0.55,
    tangential_threshold: float = 0.4
) -> ClusterRelationshipMapper:
    """
    Create a cluster relationship mapper.

    Args:
        same_story_threshold: Threshold for same story detection
        related_topic_threshold: Threshold for related topics
        tangential_threshold: Threshold for tangential relations

    Returns:
        ClusterRelationshipMapper instance
    """
    return ClusterRelationshipMapper(
        same_story_threshold=same_story_threshold,
        related_topic_threshold=related_topic_threshold,
        tangential_threshold=tangential_threshold
    )


def build_cluster_graph(
    clusters: List[List[Dict[str, Any]]],
    topics: Optional[List[str]] = None
) -> ClusterGraph:
    """
    Convenience function to build a cluster relationship graph.

    Args:
        clusters: List of article clusters
        topics: Optional topic labels

    Returns:
        ClusterGraph instance
    """
    mapper = create_relationship_mapper()
    return mapper.map_relationships(clusters, topics)
