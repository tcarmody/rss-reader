# Enhanced RSS Reader with Multi-Article Clustering

An advanced RSS feed reader with AI-powered summarization, intelligent article clustering, and source extraction capabilities.

## Features

- **Advanced Clustering**: Automatically groups related articles using optimized sentence embeddings and ML-based clustering with HDBSCAN
- **Intelligent Topic Extraction**: Extracts meaningful topics from article clusters using specialized algorithms for different cluster sizes
- **AI Summaries**: Generates concise summaries of articles using the Claude API
- **Source Extraction**: Automatically extracts and follows original source URLs from aggregator sites like Techmeme and Google News
- **Paywall Bypass**: Capability to retrieve content from paywalled sites (configurable)
- **Batch Processing**: Efficient parallel processing of multiple feeds and articles
- **Web Interface**: Clean, simple interface for browsing feed summaries
- **Performance Optimization**: Tiered caching system and optimized batch processing

## Quick Start

### Prerequisites

- Python 3.11 or higher (recommended)
- Anthropic API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rss-reader.git
   cd rss-reader
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv rss_venv
   source rss_venv/bin/activate  # On Windows, use: rss_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # For minimal installation with essential dependencies only:
   # pip install -r essential_requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Running the Server

Start the web server:

```bash
python server.py
```

Or use the provided script (recommended):

```bash
./run_server.sh
```

The server will be available at http://localhost:5005 by default.

## Configuration

### RSS Feeds

Add your RSS feeds to `rss_feeds.txt`, one URL per line. Comments can be added after a `#` character.

Example:
```
https://example.com/feed.xml  # Tech News
https://another-site.com/rss   # Science
```

### Environment Variables

Set these in your `.env` file or environment:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `API_RPM_LIMIT` | Rate limit for API calls (requests per minute) | 50 |
| `CACHE_SIZE` | Size of in-memory cache | 256 |
| `CACHE_DIR` | Directory for cache storage | ./summary_cache |
| `CACHE_TTL_DAYS` | Time-to-live for cache entries (days) | 30 |
| `MAX_BATCH_WORKERS` | Maximum worker processes for batch processing | 3 |
| `ENABLE_PAYWALL_BYPASS` | Enable paywall bypass capabilities | false |

## System Architecture

The system consists of several key components:

### Core Components

- **Reader (`reader.py`)**: Manages feed fetching and processing
- **Summarizer (`summarizer.py`)**: Handles article summarization with Claude API
- **Fast Summarizer (`fast_summarizer.py`)**: Optimized wrapper with enhanced batching
- **Clustering (`clustering.py`)**: Manages article clustering using sentence embeddings with HDBSCAN
- **Topic Extraction (`topic_extraction.py`)**: Extracts meaningful topics from article clusters
- **Server (`server.py`)**: Web interface and API

### Support Modules

- **Batch Processing (`batch.py`, `batch_processing.py`)**: Efficient parallel processing with CompatibilityWrapper
- **Caching (`cache.py`, `tiered_cache.py`)**: Multi-level caching system with SummaryCache
- **Archive Utilities (`utils/archive.py`)**: Handling paywalled content
- **Source Extractor (`utils/source_extractor.py`)**: Extracts original source URLs from aggregator sites
- **Performance Tracking (`utils/performance.py`)**: Track and log performance metrics

## Recent Changes and Improvements

### Clustering and Topic Extraction
- Improved clustering with optimized HDBSCAN algorithm and lower distance threshold (0.08)
- Enhanced topic extraction with specialized methods for different cluster sizes
- Added support for bigrams to capture multi-word topics
- Implemented text cleaning to remove URLs and special characters
- Added extended stopwords list to filter out non-informative terms
- Weighted article titles more heavily than content for better topic relevance

### Source Extraction
- Added specialized handling for Techmeme and Google News aggregator links
- Implemented automatic extraction of original source URLs from aggregator pages
- Added capability to follow links to original sources, bypassing aggregator pages
- Enhanced article fetching to bypass paywalls on original source articles
- Added support for summarizing full content from original sources

### Bug Fixes
- Fixed missing import for SummaryCache in summarizer.py
- Resolved indentation errors in nested try-except blocks in ArticleSummarizer
- Fixed issues with CompatibilityWrapper implementation in batch_processing.py
- Removed incorrectly placed docstring fragment in summarizer.py
- Added proper error handling for initialization failures

### Dependencies
- Added hdbscan, numpy, safetensors, and transformers to requirements.txt
- Updated version specifications for key dependencies
- Added Google Auth dependency to resolve Anthropic API integration issues
- Added support for the latest Claude model (Claude Sonnet 4)

## Troubleshooting

### Missing Dependencies

If you encounter errors about missing dependencies, run:

```bash
pip install -r requirements.txt
```

For a minimal installation with only essential dependencies:

```bash
pip install -r essential_requirements.txt
```

For issues with the Anthropic API and Google Auth, run the fix script:

```bash
./fix_dependencies.sh
```

### Performance Issues

If you experience performance problems with batch processing:

1. Reduce the number of concurrent workers:
   ```
   MAX_BATCH_WORKERS=2
   ```

2. Apply the batch processing fix:
   ```python
   # Add to the top of main.py
   import batch_processing
   batch_processing.apply()
   ```

### Debugging

For more detailed logs, set the logging level to DEBUG:

```bash
export LOG_LEVEL=DEBUG
python server.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic](https://anthropic.com/) for the Claude API
- [SentenceTransformers](https://www.sbert.net/) for text embeddings
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
