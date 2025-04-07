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

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rss-reader
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Using the Web Interface

The easiest way to use the RSS Reader is through its web interface:

1. Start the web server:
   ```bash
   python server.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5004
   ```

3. Using the web interface:
   - Click the **Refresh Feeds** button to fetch and process the latest articles
   - Browse through the clustered articles and their AI-generated summaries
   - Click on article titles to read the original articles
   - The timestamp at the top shows when the feeds were last processed

4. Customizing feeds:
   - You can add custom RSS feed URLs through the web interface
   - Enter one URL per line in the input field on the home page
   - Adjust batch size and delay settings if needed

## Command-line Usage

You can also run the RSS reader directly from the command line:

```bash
python reader.py
```

This will process the default feeds and generate an HTML output file in the `output` directory.

## Using as a Library

You can use the RSS reader as a library in your Python code:

```python
from reader import RSSReader

# Initialize with custom feeds
feeds = ['https://example.com/rss', 'https://example2.com/rss']
reader = RSSReader(feeds=feeds, batch_size=10, batch_delay=5)

# Process feeds and get output file path
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

- Python 3.8+
- Anthropic API key (Claude 3 Haiku model)
- Web browser for accessing the interface
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
