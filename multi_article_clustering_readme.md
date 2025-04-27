# Multi-Article Clustering Enhancement

This enhancement improves the AI News Digest RSS Reader's clustering capabilities by adding language model (Claude) based multi-article clustering. The new approach allows the system to process multiple articles simultaneously, resulting in more accurate article clusters and improved performance.

## New Features

- **Multi-Article Comparison**: Claude can now analyze multiple articles at once to determine optimal clustering
- **Batch Processing**: Small clusters can be efficiently processed in batches
- **Topic Extraction**: Better identification of key topics within article clusters
- **Cluster Refinement**: Improved cluster merging based on content similarity
- **Enhanced Robustness**: Better error handling and fallback mechanisms

## Changes Overview

1. Enhanced the `EnhancedArticleClusterer` class with a new `_compare_multiple_texts_with_lm` method
2. Created a dedicated `LMClusterAnalyzer` class for all LM-based cluster operations
3. Added robust JSON parsing and error handling
4. Added comprehensive test coverage for the new functionality
5. Created an integration example demonstrating the new capabilities

## File Changes

### Updated Files:

- **enhanced_clustering.py**: Added multi-article comparison capabilities
- **main.py**: Updated to use the enhanced clustering system

### New Files:

- **lm_cluster_analyzer.py**: New utility class for all LM-based cluster operations
- **test_enhanced_clustering.py**: Test suite for the new clustering functionality
- **multi_article_clustering_integration.py**: Example script demonstrating usage

## Using the Enhanced Clustering

The enhanced clustering system is fully backward compatible with the existing system. It's automatically used when processing feeds through the main application. Here's how to use it programmatically:

```python
# Initialize the components
from summarizer import ArticleSummarizer
from enhanced_clustering import create_enhanced_clusterer
from lm_cluster_analyzer import create_cluster_analyzer

# Create the necessary instances
summarizer = ArticleSummarizer()
clusterer = create_enhanced_clusterer(summarizer=summarizer)
analyzer = create_cluster_analyzer(summarizer=summarizer)

# Process a list of articles
clusters = clusterer.cluster_with_summaries(articles)

# Analyze the resulting clusters
for cluster in clusters:
    topics = analyzer.extract_cluster_topics(cluster)
    print(f"Cluster with {len(cluster)} articles")
    print(f"Topics: {', '.join(topics)}")
```

## Performance Impact

The enhanced clustering system improves clustering accuracy at the cost of slight additional processing time. However, the improved batch processing helps mitigate the performance impact:

- Small clusters (1-2 articles) are processed in batches to reduce API calls
- The system falls back to standard clustering when appropriate
- Caching is used extensively to avoid redundant API calls

## Testing the Changes

To run the tests for the new functionality:

```bash
python -m test_enhanced_clustering
```

To see the enhanced clustering in action with sample data:

```bash
python -m multi_article_clustering_integration
```

## Configuration Options

You can adjust the behavior of the enhanced clustering system through the following environment variables:

- `ENABLE_MULTI_ARTICLE_CLUSTERING`: Set to 'false' to disable multi-article clustering (default: true)
- `MAX_ARTICLES_PER_BATCH`: Maximum number of articles to process in a single batch (default: 5)
- `MIN_SIMILARITY_THRESHOLD`: Minimum similarity score to consider articles related (default: 0.7)

These variables can be set in your `.env` file or directly in the environment.

## Implementation Details

The enhanced clustering works in three phases:

1. **Initial Clustering**: Uses embedding-based clustering to create initial article groups
2. **LM Enhancement**: Uses Claude to analyze and refine small clusters 
3. **Summary Verification**: Checks for similar summary content to further refine clusters

The system maintains fallback mechanisms at each phase to ensure resilience and reasonable performance.
