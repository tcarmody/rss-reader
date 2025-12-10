#!/usr/bin/env python3
"""
Demonstration script for cache and model selection monitoring.

This script shows how to use the new monitoring features added in DOCTRINE_REVIEW.md items 3-4.
Run this script to verify that monitoring is working correctly.
"""

import sys
import logging
from cache.tiered_cache import TieredCache
from models.config import select_model_by_complexity, get_model_usage_stats, log_model_usage_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_cache_monitoring():
    """Demonstrate cache performance monitoring."""
    print("\n" + "="*70)
    print("CACHE PERFORMANCE MONITORING DEMO")
    print("="*70)

    # Create cache instance
    cache = TieredCache(memory_size=10, disk_path="./demo_cache", ttl_days=1)

    # Simulate some cache operations
    print("\n1. Simulating cache operations...")

    # First access (all misses)
    for i in range(5):
        url = f"http://example.com/article{i}"
        result = cache.get(url)
        if result is None:
            # Simulate API call
            summary = f"Summary of article {i}"
            cache.set(url, summary)
            print(f"   Cache MISS for article{i} -> stored summary")

    # Second access (memory hits)
    print("\n2. Accessing same articles (should hit memory cache)...")
    for i in range(5):
        url = f"http://example.com/article{i}"
        result = cache.get(url)
        if result:
            print(f"   Cache HIT for article{i} (from memory)")

    # Third access after clearing memory (disk hits)
    print("\n3. Clearing memory cache, accessing again (should hit disk)...")
    cache.memory_cache.clear()
    for i in range(3):
        url = f"http://example.com/article{i}"
        result = cache.get(url)
        if result:
            print(f"   Cache HIT for article{i} (from disk)")

    # Show statistics
    print("\n" + "-"*70)
    print("CACHE STATISTICS")
    print("-"*70)

    stats = cache.get_stats()

    print(f"\nMemory Cache:")
    print(f"  - Size: {stats['memory']['size']}/{stats['memory']['max_size']}")
    print(f"  - Hits: {stats['memory']['hits']}")
    print(f"  - Misses: {stats['memory']['misses']}")
    print(f"  - Hit Rate: {stats['memory']['hit_rate']:.1%}")

    print(f"\nDisk Cache:")
    print(f"  - Entries: {stats['disk_entries']}")
    print(f"  - Size: {stats['disk_size_mb']} MB")
    print(f"  - Hits: {stats['disk_hits']}")
    print(f"  - Misses: {stats['disk_misses']}")
    print(f"  - Hit Rate: {stats['disk_hit_rate']:.1%}")

    print(f"\nCombined Performance:")
    print(f"  - Total Requests: {stats['total_requests']}")
    print(f"  - Total Hits: {stats['total_hits']}")
    print(f"  - Combined Hit Rate: {stats['combined_hit_rate']:.1%}")
    print(f"  - Memory Serving: {stats['memory_serving_percentage']:.1f}%")
    print(f"  - Disk Serving: {stats['disk_serving_percentage']:.1f}%")
    print(f"  - API Calls Needed: {stats['api_serving_percentage']:.1f}%")

    print("\n4. Using log_stats() method...")
    cache.log_stats()

    # Cleanup
    import shutil
    import os
    if os.path.exists("./demo_cache"):
        shutil.rmtree("./demo_cache")
        print("\n✓ Demo cache cleaned up")


def demo_model_selection_monitoring():
    """Demonstrate model selection monitoring."""
    print("\n" + "="*70)
    print("MODEL SELECTION MONITORING DEMO")
    print("="*70)

    print("\n1. Simulating model selections with different complexity scores...")

    # Simulate various article complexities
    test_cases = [
        (0.2, "http://example.com/simple-news"),
        (0.3, "http://example.com/basic-update"),
        (0.4, "http://example.com/medium-article"),
        (0.5, "http://example.com/standard-post"),
        (0.65, "http://example.com/technical-analysis"),
        (0.7, "http://example.com/research-paper"),
        (0.8, "http://example.com/complex-topic"),
        (0.9, "http://example.com/advanced-research"),
        (0.3, "http://example.com/another-simple"),
        (0.4, "http://example.com/another-medium"),
    ]

    for complexity, url in test_cases:
        model = select_model_by_complexity(complexity, url)
        model_name = "Haiku" if "haiku" in model else "Sonnet"
        print(f"   Complexity {complexity:.2f} -> {model_name}")

    # Show statistics
    print("\n" + "-"*70)
    print("MODEL USAGE STATISTICS")
    print("-"*70)

    stats = get_model_usage_stats()

    print(f"\nModel Distribution:")
    print(f"  - Haiku 4.5:  {stats['haiku_count']:2d} selections ({stats['haiku_percentage']:.1f}%)")
    print(f"  - Sonnet 4.5: {stats['sonnet_count']:2d} selections ({stats['sonnet_percentage']:.1f}%)")
    print(f"  - Total:      {stats['total_selections']:2d} selections")

    print(f"\nComplexity Score Analysis:")
    print(f"  - Average:  {stats['avg_complexity']:.3f}")
    print(f"  - Minimum:  {stats['min_complexity']:.3f}")
    print(f"  - Maximum:  {stats['max_complexity']:.3f}")

    print(f"\nThreshold Validation (from DOCTRINE_REVIEW.md):")
    print(f"  - Target Haiku Usage: 40-60%")
    print(f"  - Actual Haiku Usage: {stats['haiku_percentage']:.1f}%")

    if stats['haiku_percentage'] < 30:
        print(f"  ⚠️  WARNING: Haiku usage <30% - threshold (0.6) may be too high")
    elif stats['haiku_percentage'] > 70:
        print(f"  ⚠️  WARNING: Haiku usage >70% - threshold (0.6) may be too low")
    else:
        print(f"  ✓ Haiku usage within target range")

    print("\n2. Using log_model_usage_stats() method...")
    log_model_usage_stats()

    print("\n✓ Model selection monitoring demo complete")


def main():
    """Run all monitoring demonstrations."""
    print("\n" + "="*70)
    print("MONITORING SYSTEM DEMONSTRATION")
    print("Implementing DOCTRINE_REVIEW.md Items 3-4")
    print("="*70)

    try:
        demo_cache_monitoring()
        demo_model_selection_monitoring()

        print("\n" + "="*70)
        print("✓ ALL MONITORING DEMOS COMPLETED SUCCESSFULLY")
        print("="*70)

        print("\nNext Steps (from DOCTRINE_REVIEW.md):")
        print("1. Run this demo weekly to collect metrics")
        print("2. Track cache hit rates (target: >60% combined)")
        print("3. Monitor model distribution (target: 40-60% Haiku)")
        print("4. Adjust thresholds if metrics fall outside targets")
        print("\nSee TEST_COVERAGE_REPORT.md for full monitoring plan.")

    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
