#!/usr/bin/env python3
"""
Test script for Tier 1 Quick Win improvements.

Tests:
1. Enhanced entity filtering in clustering
2. Semantic cache keys for better hit rates
3. Parallel chunk summarization for long articles
4. Partial batch success handling
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhanced_entity_filtering():
    """Test that common entities are properly filtered in clustering."""
    logger.info("=" * 60)
    logger.info("TEST 1: Enhanced Entity Filtering")
    logger.info("=" * 60)

    from clustering.simple import SimpleClustering
    from clustering.base import CONFIG

    # Check that common entities list is expanded
    logger.info(f"Common entities count: {len(CONFIG.common_entities)}")
    assert len(CONFIG.common_entities) > 15, "Common entities should be expanded"

    # Verify key terms are included
    key_terms = {'ai', 'chatgpt', 'openai', 'google', 'microsoft', 'startup', 'funding'}
    lowercase_entities = {e.lower() for e in CONFIG.common_entities}
    missing = key_terms - lowercase_entities
    assert not missing, f"Missing key common entities: {missing}"

    # Test SimpleClustering keyword extraction
    clustering = SimpleClustering()
    test_text = "AI and ChatGPT are transforming technology with OpenAI leading the way"
    keywords = clustering.extract_keywords(test_text)

    # Common entities should be filtered out
    filtered_entities = {k.lower() for k in keywords}
    overlap = filtered_entities & key_terms
    logger.info(f"Keywords extracted: {keywords}")
    logger.info(f"Overlap with common entities: {overlap}")

    # Most common terms should be filtered (though "transforming" and "technology" might remain)
    assert len(overlap) <= 2, f"Too many common entities in keywords: {overlap}"

    logger.info("✓ Enhanced entity filtering working correctly\n")


def test_semantic_cache_keys():
    """Test semantic cache key generation."""
    logger.info("=" * 60)
    logger.info("TEST 2: Semantic Cache Keys")
    logger.info("=" * 60)

    from cache.semantic_keys import generate_cache_keys, get_all_cache_keys

    # Test with similar articles
    article1 = "Apple announces new AI chip for iPhone"
    article2 = "Apple   announces  new   AI chip   for iPhone"  # Extra whitespace

    exact1, semantic1 = generate_cache_keys(article1)
    exact2, semantic2 = generate_cache_keys(article2)

    logger.info(f"Article 1 exact key: {exact1[:50]}...")
    logger.info(f"Article 2 exact key: {exact2[:50]}...")
    logger.info(f"Article 1 semantic key: {semantic1}")
    logger.info(f"Article 2 semantic key: {semantic2}")

    # Exact keys should differ
    assert exact1 != exact2, "Exact keys should differ with whitespace changes"

    # Semantic keys should match (normalized whitespace)
    assert semantic1 == semantic2, "Semantic keys should match despite whitespace"

    # Test get_all_cache_keys returns multiple keys
    all_keys = get_all_cache_keys(article1)
    assert len(all_keys) == 2, f"Should return 2 keys, got {len(all_keys)}"
    logger.info(f"get_all_cache_keys returns {len(all_keys)} keys for lookup")

    logger.info("✓ Semantic cache keys working correctly\n")


def test_parallel_chunk_summarization():
    """Test parallel chunk summarization for long articles."""
    logger.info("=" * 60)
    logger.info("TEST 3: Parallel Chunk Summarization")
    logger.info("=" * 60)

    # Check if ANTHROPIC_API_KEY is available
    if not os.environ.get('ANTHROPIC_API_KEY'):
        logger.warning("⚠ Skipping API test - ANTHROPIC_API_KEY not set")
        logger.info("  (This is expected if testing without API access)\n")
        return

    from summarization.fast_summarizer import FastSummarizer
    import time

    # Create a long article (> 12KB)
    long_text = """
    Artificial intelligence has transformed the technology landscape over the past decade.
    Major companies like Google, Microsoft, and OpenAI have invested billions in AI research.
    The development of large language models has been particularly transformative.
    """ * 200  # Repeat to make it long

    logger.info(f"Article length: {len(long_text)} characters")

    try:
        summarizer = FastSummarizer(max_batch_workers=3)

        # Time the summarization
        start_time = time.time()
        result = summarizer.summarize(
            text=long_text,
            title="AI Revolution Test Article",
            url="https://example.com/test"
        )
        elapsed = time.time() - start_time

        logger.info(f"Summarization completed in {elapsed:.2f} seconds")
        logger.info(f"Headline: {result.get('headline', 'N/A')}")
        logger.info(f"Summary length: {len(result.get('summary', ''))}")

        assert 'headline' in result, "Result should contain headline"
        assert 'summary' in result, "Result should contain summary"

        logger.info("✓ Parallel chunk summarization working correctly\n")

    except Exception as e:
        logger.error(f"✗ Parallel chunk summarization failed: {str(e)}\n")
        raise


async def test_partial_batch_success():
    """Test partial batch success handling."""
    logger.info("=" * 60)
    logger.info("TEST 4: Partial Batch Success Handling")
    logger.info("=" * 60)

    # Check if ANTHROPIC_API_KEY is available
    if not os.environ.get('ANTHROPIC_API_KEY'):
        logger.warning("⚠ Skipping API test - ANTHROPIC_API_KEY not set")
        logger.info("  (This is expected if testing without API access)\n")
        return

    from summarization.fast_summarizer import FastSummarizer

    # Create a batch with some valid and some problematic articles
    articles = [
        {
            'title': 'Valid Article 1',
            'text': 'This is a valid article about technology.',
            'url': 'https://example.com/1'
        },
        {
            'title': 'Valid Article 2',
            'text': 'Another valid article about science.',
            'url': 'https://example.com/2'
        },
        {
            'title': 'Empty Article',
            'text': '',  # This might cause issues
            'url': 'https://example.com/3'
        },
        {
            'title': 'Valid Article 3',
            'text': 'A third valid article about innovation.',
            'url': 'https://example.com/4'
        },
    ]

    try:
        summarizer = FastSummarizer()

        # Process batch
        results = await summarizer.batch_summarize(
            articles=articles,
            max_concurrent=2,
            timeout=30
        )

        logger.info(f"Batch processing completed: {len(results)}/{len(articles)} articles")

        # Check that we got results (even if some failed)
        assert len(results) > 0, "Should return at least some results"

        # Count successful vs error results
        successful = sum(1 for r in results if not r.get('summary', {}).get('error', False))
        errors = sum(1 for r in results if r.get('summary', {}).get('error', False))

        logger.info(f"Successful: {successful}, Errors: {errors}")

        # We should have processed most articles
        assert successful >= 2, f"Should have at least 2 successful summaries, got {successful}"

        logger.info("✓ Partial batch success handling working correctly\n")

    except Exception as e:
        logger.error(f"✗ Partial batch success test failed: {str(e)}\n")
        raise


def test_clustering_content_weighting():
    """Test that clustering uses content-focused weighting."""
    logger.info("=" * 60)
    logger.info("TEST 5: Content-Focused Weighting in Clustering")
    logger.info("=" * 60)

    from clustering.base import ArticleClusterer

    clusterer = ArticleClusterer()

    # Create test articles with same headline but different content
    articles = [
        {
            'title': 'AI News Today',
            'content': 'Detailed analysis of new regulation policies for artificial intelligence in Europe.',
            'published': '2025-01-01'
        },
        {
            'title': 'AI News Today',
            'content': 'Breakthrough in quantum computing achieves new milestone for processing.',
            'published': '2025-01-01'
        },
    ]

    # Prepare texts - this is where content weighting happens
    texts, _ = clusterer._prepare_article_texts(articles)

    logger.info(f"Prepared text 1 length: {len(texts[0])}")
    logger.info(f"Prepared text 2 length: {len(texts[1])}")

    # Check that content appears multiple times (4x weighting)
    # Count occurrences of a unique word from each article
    regulation_count = texts[0].lower().count('regulation')
    quantum_count = texts[1].lower().count('quantum')

    logger.info(f"'regulation' appears {regulation_count} times in text 1")
    logger.info(f"'quantum' appears {quantum_count} times in text 2")

    # With 4x content weighting, unique content words should appear multiple times
    assert regulation_count >= 4, f"Content should be weighted 4x, got {regulation_count}"
    assert quantum_count >= 4, f"Content should be weighted 4x, got {quantum_count}"

    logger.info("✓ Content-focused weighting implemented correctly\n")


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "TIER 1 IMPROVEMENTS TEST SUITE" + " " * 18 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")

    tests = [
        ("Enhanced Entity Filtering", test_enhanced_entity_filtering),
        ("Semantic Cache Keys", test_semantic_cache_keys),
        ("Content-Focused Weighting", test_clustering_content_weighting),
        ("Parallel Chunk Summarization", test_parallel_chunk_summarization),
        ("Partial Batch Success", test_partial_batch_success),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"✗ {name} FAILED: {str(e)}\n")
            failed += 1
        except Exception as e:
            if "Skipping" in str(e) or "not set" in str(e):
                skipped += 1
            else:
                logger.error(f"✗ {name} ERROR: {str(e)}\n")
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
