# AI News Digest - RSS Reader and Summarizer

A comprehensive RSS feed reader that fetches articles, summarizes them using the Anthropic Claude API, clusters similar articles together, and provides a clean web interface for browsing summaries.

## Features

- **Smart Article Clustering**: Groups similar articles based on semantic similarity
- **AI-Powered Summaries**: Automatically summarizes articles using Anthropic Claude API
- **Interactive Web Interface**: Browse summaries and article clusters through a web browser
- **Responsive Design**: Clean, modern UI that works on desktop and mobile devices
- **Dark Mode Support**: Automatic theme switching based on system preferences
- **Caching System**: Implements caching to avoid redundant API calls
- **Batch Processing**: Processes feeds in batches with rate limiting
- **Performance Tracking**: Detailed logging and performance metrics

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rss-reader
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   This project requires Python 3.8 or newer. All required dependencies are listed in `requirements.txt`:

   - anthropic
   - beautifulsoup4
   - feedparser
   - Flask
   - fasttext
   - hdbscan
   - langdetect
   - numpy
   - python-dateutil
   - python-dotenv
   - psutil
   - requests
   - ratelimit
   - scikit-learn
   - safetensors
   - sentence-transformers
   - torch
   - transformers
   - tqdm

4. **Add your Anthropic API key:**
   Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Updating Dependencies

To update dependencies to the latest compatible versions, run:
```bash
pip install -U -r requirements.txt
```
If you add new packages to your code, be sure to add them to `requirements.txt` as well.

## Using the Web Interface

The recommended way to use the RSS Reader is via the web interface:

1. **Start the web server:**
   ```bash
   python server.py
   ```
   By default, the server runs on [http://localhost:5004](http://localhost:5004).

2. **Open your browser:**
   Go to [http://localhost:5004](http://localhost:5004)

3. **Web interface features:**
   - Refresh feeds to fetch and process the latest articles
   - Browse clustered articles and AI-generated summaries
   - Click article titles to view the original sources
   - See when feeds were last processed (timestamp at top)
   - Add custom RSS feed URLs via the input field (one per line)
   - Adjust batch size and delay settings if needed

## Command-line Usage

You can also run the RSS reader from the command line for batch processing:

```bash
python reader.py
```

This will process the feeds listed in `rss_feeds.txt` and generate a summary HTML file in the `output` directory.

## Using as a Library

You can use the RSS reader programmatically in your own Python scripts:

```python
from reader import RSSReader

feeds = [
    'https://example.com/rss',
    'https://example2.com/rss',
]
reader = RSSReader(feeds=feeds, batch_size=10, batch_delay=5)
output_file = reader.process_feeds()
print(f"Generated summary at: {output_file}")
```

## Project Structure

```
rss-reader/
├── server.py                    # Web server implementation
├── reader.py                    # Main RSSReader class
├── summarizer.py                # ArticleSummarizer class
├── clustering.py                # Article clustering functionality
├── cache.py                     # SummaryCache implementation
├── batch.py                     # BatchProcessor class
├── utils/
│   ├── __init__.py
│   ├── http.py                  # HTTP utilities
│   ├── performance.py           # Performance tracking
│   ├── config.py                # Configuration utilities
│   └── archive.py               # Article content extraction
├── templates/
│   ├── feed-summary.html        # Main summary template
│   ├── welcome.html             # Welcome page template
│   └── error.html               # Error page template
├── static/
│   └── styles.css               # CSS styles for the web interface
├── output/                      # Generated HTML output files
└── rss_feeds.txt                # Default RSS feed list
```

## Creating a Feed List

The application uses a file named `rss_feeds.txt` in the project root with one RSS feed URL per line:

```
https://news.google.com/rss/search?q=artificial+intelligence
https://feeds.feedburner.com/aiweekly
# Lines starting with # are ignored (comments)
https://machinelearning.apple.com/rss.xml
```

## Configuration Options

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
ANTHROPIC_API_KEY=your_api_key_here
```

### Web Server Settings

The web server runs on port 5004 by default. You can modify this in `server.py` if needed.

## Advanced Features

### Article Clustering

The clustering algorithm groups similar articles based on semantic similarity. You can adjust the clustering parameters in `clustering.py`:

- `distance_threshold`: Controls how similar articles must be to be clustered together (lower values create more focused clusters)

### Summary Generation

Summaries are generated using the Anthropic Claude API. The system prompt and style guidelines can be customized in `summarizer.py`.

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your Anthropic API key is correctly set in the `.env` file
2. **Port Conflicts**: If port 5004 is already in use, change the port number in `server.py`
3. **Missing Summaries**: Check the logs for any API errors or rate limiting issues

## Requirements

- Python 3.8 or higher
- Anthropic API key (Claude 3 Haiku model)
- Web browser for accessing the interface
- All dependencies listed in requirements.txt (see above)

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
