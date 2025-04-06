# RSS Reader and Summarizer

A comprehensive RSS feed reader that fetches articles, summarizes them using the Anthropic Claude API, and clusters similar articles together.

## Features

- Fetches articles from multiple RSS feeds
- Automatically summarizes articles using Anthropic Claude API
- Clusters similar articles based on semantic similarity
- Generates a clean HTML report with summaries and links
- Implements caching to avoid redundant API calls
- Processes feeds in batches with rate limiting
- Provides performance tracking and logging

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rss-reader
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Run the RSS reader with default settings:

```bash
python -m rss_reader.main --feeds https://example.com/rss https://example2.com/rss --batch-size 10 --batch-delay 5
```

### Command-line Arguments

- `--feeds`: List of feed URLs to process (separated by spaces)
- `--batch-size`: Number of feeds to process in each batch (default: 25)
- `--batch-delay`: Delay between batches in seconds (default: 15)

### Using as a Library

You can also use the RSS reader as a library in your Python code:

```python
from rss_reader.reader import RSSReader

# Initialize with custom feeds
feeds = ['https://example.com/rss', 'https://example2.com/rss']
reader = RSSReader(feeds=feeds, batch_size=10, batch_delay=5)

# Process feeds and get output file path
output_file = reader.process_feeds()
print(f"Generated summary at: {output_file}")
```

## Project Structure

```
rss_reader/
├── __init__.py                   # Makes the directory a package
├── main.py                       # Entry point with main function
├── reader.py                     # Main RSSReader class
├── summarizer.py                 # ArticleSummarizer class
├── clustering.py                 # Article clustering functionality
├── cache.py                      # SummaryCache implementation
├── batch.py                      # BatchProcessor class
├── utils/
│   ├── __init__.py
│   ├── http.py                   # HTTP utilities
│   ├── performance.py            # Performance tracking
│   ├── config.py                 # Configuration utilities
│   └── rate_limit.py             # Rate limiting functionality
├── templates/
│   └── feed_summary.html         # HTML template
```

## Creating a Feed List

Create a file named `rss_feeds.txt` in the project root with one RSS feed URL per line:

```
https://news.google.com/rss/search?q=artificial+intelligence
https://feeds.feedburner.com/aiweekly
# Lines starting with # are ignored (comments)
https://machinelearning.apple.com/rss.xml
```

## Requirements

- Python 3.8+
- Anthropic API key
- Dependencies listed in requirements.txt

## Performance Notes

- The RSS reader uses concurrent processing to handle multiple feeds efficiently
- Caching reduces redundant API calls for previously processed articles
- Performance logs are stored in the `performance_logs` directory

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT.main
```

This will:
1. Read RSS feed URLs from `rss_feeds.txt`
2. Fetch and process articles from those feeds
3. Generate an HTML summary in the `output` directory

### Custom Usage

Specify your own feeds and settings:

```bash
python -m rss_reader
