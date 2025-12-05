# Enhanced RSS Reader with Smart Article Clustering

An advanced RSS feed reader with AI-powered summarization, lightweight semantic clustering, and source extraction capabilities.

> **🎉 NEW: Native Mac Application!** This project now includes a sophisticated native macOS app built with Electron. See [MAC_APP.md](MAC_APP.md) for details or jump to [Quick Start for Mac App](#mac-app).

> **🔐 NEW: Multi-User Support!** The app now supports multiple users with individual accounts, personalized RSS feeds, and private bookmark collections. See [Authentication](#authentication) for details.

## Features

- **Multi-User Authentication**: Secure user registration and login with bcrypt password hashing
- **Per-User Data Isolation**: Each user has their own RSS feeds, bookmarks, and settings in separate databases
- **Smart Clustering**: Automatically groups related articles using lightweight semantic embeddings (80MB model) with keyword fallbacks
- **Intelligent Topic Extraction**: Extracts meaningful topics from article clusters using hybrid similarity scoring
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

   **Note**: The new lightweight clustering system requires only ~200MB of dependencies vs the previous 2GB+ heavy ML setup.

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

## Authentication

The application supports multiple users with secure authentication.

### First-Time Setup

1. Start the server and navigate to http://localhost:5005
2. You'll be redirected to the login page
3. Click "Create Account" to register
4. **The first user to register becomes the administrator**
5. New users automatically get the 29 default RSS feeds imported

### Migrating Existing Data

If you have existing bookmarks from before the multi-user update, run the migration script:

```bash
python scripts/migrate_to_multiuser.py
```

This will:
- Create an admin user account
- Migrate existing bookmarks to the admin user
- Import default feeds for the admin user

### User Features

- **Personal RSS Feeds**: Each user manages their own feed subscriptions
- **Private Bookmarks**: Bookmarks are isolated per user
- **User Profile**: Change password and view account stats at `/profile`
- **Remember Me**: Optional 30-day persistent login

### Data Storage

User data is stored in isolated SQLite databases:

```
data/
├── auth.db                    # Shared authentication database
└── users/
    └── {user_id}/
        └── user_data.db       # Per-user bookmarks, feeds, settings
```

## Mac App

### Quick Start for Mac App

Want a native macOS application? Follow these steps:

```bash
# 1. Install Node.js dependencies
cd electron
npm install

# 2. Set up Python environment (if not done already)
cd ..
./run_server.sh --help

# 3. Configure API key in .env file (see above)

# 4. Run the Mac app
cd electron
npm run dev
```

### Building the Mac App

```bash
cd electron

# Build universal binary (Apple Silicon + Intel)
make build-universal

# Or build for specific architecture
make build-arm64  # Apple Silicon only
make build-x64    # Intel only
```

The built app will be in `electron/dist/` as a `.dmg` installer.

### Mac App Features

- **Native macOS Integration**: Menu bar, keyboard shortcuts, dock integration
- **Sophisticated UI**: Mac-native design with vibrancy effects
- **Automatic Server Management**: Python backend starts/stops automatically
- **Dark Mode**: Full support for macOS system dark mode
- **Universal Binary**: Runs natively on both Apple Silicon and Intel

For detailed documentation, see:
- [MAC_APP.md](MAC_APP.md) - Complete Mac app documentation
- [electron/README.md](electron/README.md) - Technical details
- [electron/QUICKSTART.md](electron/QUICKSTART.md) - Step-by-step guide

## Configuration

### RSS Feeds

**For multi-user setups**: Each user manages their own feeds through the web interface at `/feeds`. Default feeds are imported from `rss_feeds.txt` when a new user registers.

**For single-user or initial setup**: Edit `rss_feeds.txt` directly with one URL per line. Comments can be added after a `#` character.

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
| `SECRET_KEY` | Secret key for session encryption | Auto-generated (dev only) |
| `API_RPM_LIMIT` | Rate limit for API calls (requests per minute) | 50 |
| `CACHE_SIZE` | Size of in-memory cache | 256 |
| `CACHE_DIR` | Directory for cache storage | ./summary_cache |
| `CACHE_TTL_DAYS` | Time-to-live for cache entries (days) | 30 |
| `MAX_BATCH_WORKERS` | Maximum worker processes for batch processing | 3 |
| `ENABLE_PAYWALL_BYPASS` | Enable paywall bypass capabilities | false |

**Security Note**: For production deployments, always set a strong `SECRET_KEY` environment variable.

## System Architecture

The system consists of several key components:

### Core Components

- **Reader (`reader/base_reader.py`)**: Manages feed fetching and processing
- **Summarizer (`summarization/`)**: Handles article summarization with Claude API
- **Simple Clustering (`clustering/simple.py`)**: Lightweight semantic clustering with keyword fallbacks
- **Enhanced Clustering (`clustering/enhanced.py`)**: Advanced ML clustering (optional, heavy dependencies)
- **Server (`server.py`)**: FastAPI web interface and API

### Support Modules

- **Batch Processing (`common/batch_processing.py`)**: Efficient parallel processing
- **Caching (`cache/`)**: Multi-level caching system with tiered cache
- **Content Processing (`content/`)**: Archive services, paywall detection, and content extraction
- **Performance Tracking (`common/performance.py`)**: Track and log performance metrics

### Authentication & User Management

- **User Models (`models/user.py`)**: User, UserSession, Feed, and UserSettings models
- **Auth Manager (`services/auth_manager.py`)**: Registration, login, password hashing, session tokens
- **User Data Manager (`services/user_data_manager.py`)**: Per-user database management
- **Auth Middleware (`middleware/auth.py`)**: Route protection with `@require_login` decorator

## Recent Changes and Improvements

### NEW: Multi-User Authentication System
- **User registration and login**: Secure authentication with bcrypt password hashing
- **Per-user data isolation**: Each user has their own SQLite database for complete data separation
- **Session management**: Cookie-based sessions with optional "remember me" persistent tokens
- **Protected routes**: All data-modifying endpoints require authentication
- **First user is admin**: The first registered user automatically becomes an administrator
- **Migration support**: Script to migrate existing single-user data to multi-user system

### NEW: Lightweight Clustering System (v2.0)
- **Replaced heavy ML clustering** (2GB+ dependencies) with lightweight semantic clustering (~200MB)
- **Hybrid approach**: 60% semantic similarity + 40% keyword overlap for better accuracy
- **Smart fallbacks**: Works with keywords-only if embeddings unavailable
- **10x faster startup**: No heavy model loading delays
- **Mini embeddings**: Uses `all-MiniLM-L6-v2` model (80MB) instead of large transformers
- **Better reliability**: Simple, predictable clustering behavior
- **Connected components**: Improved clustering algorithm for better grouping

### Clustering and Topic Extraction (Legacy)
- Previous system used HDBSCAN algorithm with sentence transformers
- Enhanced topic extraction with specialized methods for different cluster sizes
- Support for bigrams to capture multi-word topics
- Text cleaning to remove URLs and special characters
- Extended stopwords list to filter out non-informative terms

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
- **Simplified dependencies**: Removed heavy ML packages (torch, transformers, HDBSCAN, UMAP)
- **Lightweight**: Only sentence-transformers and scikit-learn for clustering
- **Optional heavy clustering**: Previous advanced clustering can be re-enabled by uncommenting dependencies
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
- [SentenceTransformers](https://www.sbert.net/) for lightweight text embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- [scikit-learn](https://scikit-learn.org/) for clustering algorithms
