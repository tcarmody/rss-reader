#!/usr/bin/env python3
"""
Test script for Tier 3 Architectural improvements.

Tests:
1. Semantic Summary Cache (Vector Database)
2. WebSocket Streaming (Connection Manager)
3. Progressive Clustering Pipeline
4. Cross-Cluster Relationship Mapping
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_semantic_cache():
    """Test semantic summary cache with vector database."""
    logger.info("=" * 60)
    logger.info("TEST 1: Semantic Summary Cache")
    logger.info("=" * 60)

    from cache.semantic_cache import SemanticSummaryCache, get_semantic_cache

    # Create cache with test directory
    cache = SemanticSummaryCache(
        persist_directory="./test_vector_cache",
        similarity_threshold=0.85
    )

    # Test Article 1
    article1 = {
        'text': 'OpenAI announced a major breakthrough in artificial intelligence research today. The new model demonstrates unprecedented capabilities in natural language understanding and generation.',
        'title': 'OpenAI Announces AI Breakthrough',
        'url': 'https://example.com/openai-breakthrough'
    }

    # Test Article 2 - Similar to Article 1
    article2 = {
        'text': 'OpenAI revealed significant advances in AI technology. Their latest model shows remarkable improvements in language processing and comprehension abilities.',
        'title': 'OpenAI Reveals AI Advances',
        'url': 'https://example.com/openai-advances'
    }

    # Test Article 3 - Different topic
    article3 = {
        'text': 'Tesla reported record quarterly earnings driven by strong electric vehicle sales in China and Europe. The company exceeded analyst expectations.',
        'title': 'Tesla Reports Record Earnings',
        'url': 'https://example.com/tesla-earnings'
    }

    # Store first article summary
    summary1 = {
        'headline': 'OpenAI Breakthrough in AI',
        'summary': 'OpenAI announced major AI research breakthrough with unprecedented NLP capabilities.'
    }

    stored = await cache.store_summary(article1, summary1, style='default')
    logger.info(f"Stored article 1: {stored}")
    assert stored, "Should store article 1"

    # Check cache lookup for similar article
    cached = await cache.get_cached_summary(article2, style='default')
    logger.info(f"Cache lookup for similar article: {cached is not None}")

    # Note: Semantic matching depends on embeddings being available
    # If ChromaDB/embeddings not installed, will fall back to exact match

    # Check cache for different article (should miss)
    cached_diff = await cache.get_cached_summary(article3, style='default')
    logger.info(f"Cache lookup for different article: {cached_diff is not None}")

    # Get stats
    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")

    assert stats['stores'] >= 1, "Should have at least 1 store"
    logger.info(f"  ChromaDB available: {stats['chromadb_available']}")

    # Test find similar articles
    similar = await cache.find_similar_articles(article1, n_results=3)
    logger.info(f"Similar articles found: {len(similar)}")

    # Test search summaries
    search_results = await cache.search_summaries("AI breakthrough", n_results=5)
    logger.info(f"Search results: {len(search_results)}")

    logger.info("✓ Semantic cache working correctly\n")


async def test_websocket_streaming():
    """Test WebSocket streaming components."""
    logger.info("=" * 60)
    logger.info("TEST 2: WebSocket Streaming Components")
    logger.info("=" * 60)

    from api.websocket_streaming import (
        ConnectionManager,
        StreamMessage,
        MessageType,
        create_connection_manager
    )

    # Test ConnectionManager
    manager = create_connection_manager()
    assert manager.connection_count == 0, "Should start with no connections"
    logger.info(f"Initial connection count: {manager.connection_count}")

    # Test StreamMessage
    message = StreamMessage(
        type=MessageType.SUMMARY,
        timestamp=datetime.now().isoformat(),
        data={
            'index': 0,
            'article_url': 'https://example.com/test',
            'summary': {'headline': 'Test', 'summary': 'Test summary'},
            'progress': 0.5
        }
    )

    json_str = message.to_json()
    logger.info(f"Message JSON length: {len(json_str)}")
    assert 'summary' in json_str, "JSON should contain summary"
    assert 'progress' in json_str, "JSON should contain progress"

    # Test message types
    assert MessageType.CONNECTED.value == "connected"
    assert MessageType.SUMMARY.value == "summary"
    assert MessageType.COMPLETE.value == "complete"
    assert MessageType.CLUSTER_UPDATE.value == "cluster_update"

    logger.info(f"Message types: {[m.value for m in MessageType]}")

    logger.info("✓ WebSocket streaming components working correctly\n")


async def test_progressive_clustering():
    """Test progressive clustering pipeline."""
    logger.info("=" * 60)
    logger.info("TEST 3: Progressive Clustering Pipeline")
    logger.info("=" * 60)

    from clustering.progressive import (
        ProgressiveClusterer,
        ClusteringStage,
        create_progressive_clusterer
    )

    # Create test articles
    today = datetime.now().isoformat()
    articles = [
        # AI Group
        {'title': 'OpenAI GPT-5 Announcement', 'content': 'OpenAI announces GPT-5 with major improvements in reasoning capabilities', 'published': today},
        {'title': 'OpenAI New Model Release', 'content': 'OpenAI releases new language model with enhanced performance', 'published': today},
        {'title': 'Claude AI Improvements', 'content': 'Anthropic updates Claude with better coding and analysis abilities', 'published': today},

        # Quantum Group
        {'title': 'IBM Quantum Breakthrough', 'content': 'IBM achieves major quantum computing milestone with new processor', 'published': today},
        {'title': 'Google Quantum Supremacy', 'content': 'Google demonstrates quantum advantage in complex calculations', 'published': today},

        # EV Group
        {'title': 'Tesla Model Y Update', 'content': 'Tesla announces refreshed Model Y with improved range', 'published': today},
        {'title': 'Rivian New Factory', 'content': 'Rivian opens new manufacturing facility for electric trucks', 'published': today},
    ]

    # Track callback invocations
    callback_results = []

    async def test_callback(result):
        callback_results.append(result)
        logger.info(f"Callback received: stage={result.stage.value}, clusters={len(result.clusters)}")

    # Create progressive clusterer
    clusterer = create_progressive_clusterer(coherence_threshold=0.6)

    # Run progressive clustering
    result = await clusterer.cluster_progressive(
        articles=articles,
        callback=test_callback,
        session_id="test_session"
    )

    logger.info(f"Initial result: {len(result.clusters)} clusters, stage={result.stage.value}")
    logger.info(f"Topics: {result.topics}")

    # Verify initial result
    assert result is not None, "Should return initial result"
    assert len(result.clusters) > 0, "Should create at least one cluster"
    assert result.stage in [ClusteringStage.INITIAL, ClusteringStage.REFINING, ClusteringStage.REFINED], \
        f"Stage should be valid, got {result.stage}"

    # Check topics were extracted
    assert len(result.topics) == len(result.clusters), "Should have topic for each cluster"

    # Wait briefly for potential background refinement
    await asyncio.sleep(0.5)

    # Check cached result
    cached_result = clusterer.get_result("test_session")
    assert cached_result is not None, "Should have cached result"
    logger.info(f"Cached result stage: {cached_result.stage.value}")

    # Verify all articles are in clusters
    total_articles = sum(len(c) for c in result.clusters)
    assert total_articles == len(articles), f"All articles should be clustered: {total_articles} != {len(articles)}"

    logger.info(f"Callback invocations: {len(callback_results)}")
    assert len(callback_results) >= 1, "Should have at least one callback"

    logger.info("✓ Progressive clustering working correctly\n")


async def test_cluster_relationships():
    """Test cross-cluster relationship mapping."""
    logger.info("=" * 60)
    logger.info("TEST 4: Cross-Cluster Relationship Mapping")
    logger.info("=" * 60)

    from clustering.relationships import (
        ClusterRelationshipMapper,
        RelationshipType,
        ClusterGraph,
        create_relationship_mapper,
        build_cluster_graph
    )

    # Create test clusters with related content
    clusters = [
        # Cluster 0: AI Regulation
        [
            {'title': 'EU AI Act Passes', 'content': 'European Union passes comprehensive AI regulation legislation'},
            {'title': 'EU AI Compliance', 'content': 'Companies prepare for EU artificial intelligence compliance requirements'},
        ],
        # Cluster 1: US AI Policy (related to Cluster 0)
        [
            {'title': 'US AI Executive Order', 'content': 'President signs executive order on AI safety and regulation'},
            {'title': 'US AI Guidelines', 'content': 'Federal agencies release AI governance guidelines'},
        ],
        # Cluster 2: AI Research (related to 0 and 1)
        [
            {'title': 'AI Safety Research', 'content': 'Researchers publish AI safety findings with policy implications'},
            {'title': 'AI Ethics Study', 'content': 'New study examines ethical considerations in AI development'},
        ],
        # Cluster 3: Quantum Computing (different topic)
        [
            {'title': 'Quantum Computer Milestone', 'content': 'Researchers achieve quantum computing breakthrough'},
            {'title': 'Quantum Supremacy Claims', 'content': 'New quantum processor demonstrates computational advantage'},
        ],
    ]

    topics = ['EU AI Regulation', 'US AI Policy', 'AI Research', 'Quantum Computing']

    # Build relationship graph
    mapper = create_relationship_mapper(
        same_story_threshold=0.7,
        related_topic_threshold=0.5,
        tangential_threshold=0.3
    )

    graph = mapper.map_relationships(clusters, topics)

    logger.info(f"Graph nodes: {len(graph.nodes)}")
    logger.info(f"Graph edges: {len(graph.edges)}")

    # Verify nodes
    assert len(graph.nodes) == 4, "Should have 4 cluster nodes"

    # Check node properties
    for node_id, node in graph.nodes.items():
        logger.info(f"  Node {node_id}: topic='{node.topic}', articles={len(node)}, keywords={len(node.keywords)}")
        assert len(node.keywords) > 0, "Nodes should have keywords"

    # Check edges (relationships)
    logger.info(f"Relationships found: {len(graph.edges)}")
    for edge in graph.edges:
        logger.info(
            f"  {edge.source_id} <-> {edge.target_id}: "
            f"type={edge.relationship_type.value}, sim={edge.similarity:.2f}"
        )

    # Test get related clusters
    related = graph.get_related_clusters("cluster_0", min_similarity=0.2)
    logger.info(f"Clusters related to cluster_0: {len(related)}")

    # Test get related topics
    related_topics = mapper.get_related_topics(graph, "cluster_0", max_results=3)
    logger.info(f"Related topics for cluster_0: {related_topics}")

    # Test find story threads
    threads = mapper.find_story_threads(graph)
    logger.info(f"Story threads found: {len(threads)}")
    for i, thread in enumerate(threads):
        topics_in_thread = [n.topic for n in thread]
        logger.info(f"  Thread {i+1}: {topics_in_thread}")

    # Test graph serialization
    graph_dict = graph.to_dict()
    assert 'nodes' in graph_dict, "Should have nodes in dict"
    assert 'edges' in graph_dict, "Should have edges in dict"
    assert 'stats' in graph_dict, "Should have stats in dict"

    logger.info(f"Graph stats: {graph_dict['stats']}")

    # Test relationship types
    same_story = graph.get_clusters_by_relationship(RelationshipType.SAME_STORY)
    related_topic = graph.get_clusters_by_relationship(RelationshipType.RELATED_TOPIC)
    tangential = graph.get_clusters_by_relationship(RelationshipType.TANGENTIAL)

    logger.info(f"Same story pairs: {len(same_story)}")
    logger.info(f"Related topic pairs: {len(related_topic)}")
    logger.info(f"Tangential pairs: {len(tangential)}")

    # Convenience function test
    graph2 = build_cluster_graph(clusters, topics)
    assert len(graph2.nodes) == len(graph.nodes), "Convenience function should work"

    logger.info("✓ Cross-cluster relationship mapping working correctly\n")


async def test_integration():
    """Test integration between Tier 3 components."""
    logger.info("=" * 60)
    logger.info("TEST 5: Integration Test")
    logger.info("=" * 60)

    from clustering.progressive import create_progressive_clusterer
    from clustering.relationships import create_relationship_mapper

    # Create test articles
    articles = [
        {'title': 'AI Regulation News', 'content': 'Governments worldwide are implementing AI regulations', 'published': datetime.now().isoformat()},
        {'title': 'AI Policy Updates', 'content': 'New policies aim to ensure safe AI development', 'published': datetime.now().isoformat()},
        {'title': 'Tech Earnings Report', 'content': 'Major tech companies report strong quarterly earnings', 'published': datetime.now().isoformat()},
        {'title': 'Stock Market Analysis', 'content': 'Tech stocks rise on positive earnings outlook', 'published': datetime.now().isoformat()},
    ]

    # Step 1: Progressive clustering
    clusterer = create_progressive_clusterer()
    cluster_result = await clusterer.cluster_progressive(articles)

    logger.info(f"Progressive clustering: {len(cluster_result.clusters)} clusters")

    # Step 2: Relationship mapping
    mapper = create_relationship_mapper()
    graph = mapper.map_relationships(
        cluster_result.clusters,
        cluster_result.topics
    )

    logger.info(f"Relationship graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Step 3: Find story threads
    threads = mapper.find_story_threads(graph)
    logger.info(f"Story threads: {len(threads)}")

    # Verify integration worked
    assert len(cluster_result.clusters) > 0, "Should have clusters"
    assert len(graph.nodes) == len(cluster_result.clusters), "Graph nodes should match clusters"

    logger.info("✓ Integration test passed\n")


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "TIER 3 IMPROVEMENTS TEST SUITE" + " " * 17 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")

    tests = [
        ("Semantic Summary Cache", test_semantic_cache),
        ("WebSocket Streaming", test_websocket_streaming),
        ("Progressive Clustering", test_progressive_clustering),
        ("Cluster Relationships", test_cluster_relationships),
        ("Integration Test", test_integration),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            asyncio.run(test_func())
            passed += 1
        except AssertionError as e:
            logger.error(f"✗ {name} FAILED: {str(e)}\n")
            failed += 1
        except ImportError as e:
            logger.warning(f"⚠ {name} SKIPPED: Missing dependency - {str(e)}\n")
            skipped += 1
        except Exception as e:
            logger.error(f"✗ {name} ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    # Print summary
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 22 + "TEST SUMMARY" + " " * 24 + "║")
    logger.info("╠" + "=" * 58 + "╣")
    logger.info(f"║  Passed:  {passed:2d}" + " " * 46 + "║")
    logger.info(f"║  Failed:  {failed:2d}" + " " * 46 + "║")
    logger.info(f"║  Skipped: {skipped:2d}" + " " * 46 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")

    if failed > 0:
        sys.exit(1)
    else:
        logger.info("🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
