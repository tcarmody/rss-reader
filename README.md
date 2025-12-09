# Enhanced RSS Reader with AI Summarization

An advanced RSS feed reader with AI-powered summarization, intelligent article clustering, bookmark management, and editorial image prompt generation. Built with FastAPI and Claude AI.

> **🎉 NEW: Native Mac Application!** This project now includes a sophisticated native macOS app built with Electron. See [MAC_APP.md](MAC_APP.md) for details or jump to [Quick Start for Mac App](#mac-app).

> **🔐 NEW: Multi-User Support!** The app now supports multiple users with individual accounts, personalized RSS feeds, and private bookmark collections. See [Authentication](#authentication) for details.

## 📚 Documentation

**Essential Reading:**
- **[README.md](README.md)** (this file) - Project overview, installation, and features
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute (workflow, testing, PR process)
- **[CLAUDE.md](CLAUDE.md)** - Development commands and architecture guide
- **[DOCTRINE.md](DOCTRINE.md)** - Design decisions and architectural rationale

**Specialized Guides:**
- [MAC_APP.md](MAC_APP.md) - Native macOS application details
- [TIER3_IMPLEMENTATION_SUMMARY.md](TIER3_IMPLEMENTATION_SUMMARY.md) - Advanced features documentation

## Features

### Core Capabilities
- **AI-Powered Summarization**: Automatic content summarization using Claude 4.5 (Sonnet & Haiku)
- **Smart Model Selection**: Intelligently switches between Claude Sonnet (complex content) and Haiku (simple content) based on content complexity
- **Lightweight Clustering**: Groups related articles using hybrid semantic embeddings (90% smaller than traditional ML approaches)
- **Multi-User Support**: Per-user caching and session management for concurrent users
- **Bookmark Management**: Save, tag, search, and export articles to JSON/CSV

### Content Processing
- **Multi-Format Support**: RSS/XML feeds, web articles, and PDF documents
- **Source Extraction**: Automatically extracts original URLs from aggregators (Techmeme, Google News, Reddit, Hacker News)
- **Paywall Detection**: Identifies and handles 26+ paywall domains (NYT, WSJ, Bloomberg, etc.)
- **Archive Services**: Integration with Archive.is and Wayback Machine
- **PDF Processing**: Full text extraction from research papers and documents

### Advanced Features
- **Image Prompt Generation**: AI-powered editorial illustration prompts for DALL-E, Midjourney, and Stable Diffusion
- **Batch Processing**: Efficient parallel processing with configurable rate limiting
- **Tiered Caching**: Multi-level caching (Memory → Disk → API) with 30-day TTL
- **Content Toggle**: Switch between AI summaries and original article text
- **Web Interface**: Clean, responsive UI with component-based architecture

## Quick Start

### Prerequisites

- Python 3.11 or higher (recommended)
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rss-reader.git
   cd rss-reader
   ```

2. **Set up environment**:
   ```bash
   # The run_server.sh script handles virtual environment setup automatically
   chmod +x run_server.sh
   ```

3. **Configure API key**:
   Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

4. **Start the server**:
   ```bash
   ./run_server.sh --reload --port 5005
   ```

   The script automatically:
   - Creates and activates a virtual environment
   - Installs dependencies
   - Downloads required spaCy models
   - Installs Playwright browser (for paywall bypass)
   - Starts the FastAPI server

5. **Access the application**:
   Open your browser to [http://localhost:5005](http://localhost:5005)

### Alternative Installation Methods

**Manual installation**:
```bash
python -m venv rss_venv
source rss_venv/bin/activate  # Windows: rss_venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m playwright install chromium
```

**Minimal installation** (essential dependencies only):
```bash
pip install -r essential_requirements.txt
python -m spacy download en_core_web_sm
```

**Direct server start**:
```bash
uvicorn server:app --host 127.0.0.1 --port 5005 --reload
```

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

**Example**:
```
https://example.com/feed.xml  # Tech News
https://another-site.com/rss   # Science
# https://disabled-feed.com/rss  # Commented out
```

The project includes 29 pre-configured AI/tech feeds.

### Environment Variables

Configure via `.env` file or environment:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | - | Yes |
| `API_RPM_LIMIT` | Rate limit (requests per minute) | 50 | No |
| `CACHE_SIZE` | In-memory cache size | 256 | No |
| `CACHE_DIR` | Cache storage directory | ./summary_cache | No |
| `CACHE_TTL_DAYS` | Cache expiration (days) | 30 | No |
| `MAX_BATCH_WORKERS` | Batch processing workers | 3 | No |
| `ENABLE_PAYWALL_BYPASS` | Enable paywall bypass | false | No |
| `LOG_LEVEL` | Logging verbosity | INFO | No |

### Server Options

The [run_server.sh](run_server.sh) script supports:

```bash
./run_server.sh [OPTIONS]

Options:
  --public          Listen on 0.0.0.0 (accessible from network)
  --port PORT       Custom port (default: 5005)
  --reload          Auto-reload on code changes (development mode)
  --workers N       Number of worker processes (production)

Examples:
  ./run_server.sh --reload                    # Development mode
  ./run_server.sh --public --workers 4        # Production mode
  ./run_server.sh --port 8080                 # Custom port
```
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

### High-Level Overview

```
RSS Feeds → Content Extraction → Paywall Detection
                ↓
         Archive Services (if needed)
                ↓
         FastSummarizer (batch processing)
                ↓
    Model Selection (Sonnet vs Haiku)
                ↓
         Claude API Call
                ↓
         Tiered Cache
                ↓
    Article Clustering (hybrid similarity)
                ↓
         Web UI Display
```

### Core Components

#### AI Summarization System
- **[summarization/article_summarizer.py](summarization/article_summarizer.py)**: Base summarizer with caching
- **[summarization/fast_summarizer.py](summarization/fast_summarizer.py)**: Optimized parallel processing (32KB)
- **[models/config.py](models/config.py)**: Model selection logic
  - **Claude 4.5 Sonnet** (`claude-sonnet-4-5-latest`): Complex content (complexity ≥ 0.6)
  - **Claude 4.5 Haiku** (`claude-4-5-haiku-latest`): Simple content (complexity < 0.6)
- Both models use `-latest` versions for automatic improvements

#### Content Processing
- **[content/archive/](content/archive/)**: Archive services and paywall detection
  - [base.py](content/archive/base.py): Abstract interfaces (`ArchiveProvider`, `PaywallDetector`)
  - [providers.py](content/archive/providers.py): Archive.is, Wayback Machine integration
  - [paywall.py](content/archive/paywall.py): Hybrid paywall detection (26+ domains)
  - [specialized/wsj.py](content/archive/specialized/wsj.py): Site-specific bypass logic

- **[content/extractors/](content/extractors/)**: Content extraction utilities
  - [aggregator.py](content/extractors/aggregator.py): Techmeme, Google News, Reddit, Hacker News
  - [pdf.py](content/extractors/pdf.py): PDF text extraction with PyPDF2
  - [source.py](content/extractors/source.py): Generic content utilities and validation

#### Clustering System (v2.0 - Lightweight)
- **[clustering/simple.py](clustering/simple.py)**: Lightweight clustering implementation
  - **90% size reduction**: 200MB vs 2GB+ dependencies
  - **Hybrid similarity**: 60% semantic + 40% keyword matching
  - **Mini embeddings**: Uses `all-MiniLM-L6-v2` (80MB model)
  - **Fallback strategy**: Keywords-only when embeddings unavailable
  - **Connected components**: BFS algorithm for improved grouping

- **[clustering/enhanced.py](clustering/enhanced.py)**: Advanced ML clustering (optional)
  - HDBSCAN-based clustering
  - Requires heavy dependencies (torch, transformers)
  - Available for users who need advanced ML clustering

#### Services Layer
- **[services/bookmark_manager.py](services/bookmark_manager.py)**: Bookmark CRUD operations
  - SQLite database with SQLAlchemy ORM
  - Tag-based filtering and search
  - JSON/CSV export and import
  - Read/unread status tracking

- **[services/image_prompt_generator.py](services/image_prompt_generator.py)**: AI image prompt generation
  - Editorial illustration focus
  - Content analysis with 125+ visual keywords
  - Multiple artistic styles (photojournalistic, editorial, abstract, infographic)
  - 24-hour caching with tiered cache

#### Feed Reader
- **[reader/base_reader.py](reader/base_reader.py)**: Core RSS feed processing (52KB)
- **[reader/enhanced_reader.py](reader/enhanced_reader.py)**: Extended functionality

#### Web Interface
- **[server.py](server.py)**: FastAPI application with session middleware
- **[templates/](templates/)**: Jinja2 templates (8 pages, 10 components)
  - Component-based architecture
  - User-specific data isolation
  - Real-time clustering settings
- **[static/](static/)**: CSS and JavaScript assets

### Support Systems

#### Caching ([cache/](cache/))
- **Three-tier architecture**: Memory → Disk → API fallback
- **User-specific**: Per-user cluster data
- **TTL management**: Configurable expiration (30 days default)
- **Key format**: URL-based with content hash

#### Rate Limiting ([api/rate_limiter.py](api/rate_limiter.py))
- **Fully async**: Unified async architecture
- **Configurable limits**: 50 RPM default
- **Exponential backoff**: Automatic retry on rate limits

#### Batch Processing ([common/batch_processing.py](common/batch_processing.py))
- **Concurrent workers**: 3 default (configurable)
- **Dependency injection**: Cached summarizer instances
- **Error handling**: Graceful failures with retries

#### Utilities ([common/](common/))
- **56.4% code reduction**: From 1,963 to 855 lines
- [config.py](common/config.py), [logging.py](common/logging.py), [http.py](common/http.py)
- [performance.py](common/performance.py), [errors.py](common/errors.py)
- Backward compatibility layers for refactored code

## Recent Changes

### Major Features (2025)

**Multi-User Support** (commit d684b50)
- Per-user cluster data caching
- Session-based user ID management
- Race condition fixes
- Tiered cache architecture

**Async Architecture Unification** (commit 5499f33)
- Fully async rate limiter
- Improved concurrency handling
- Better batch processing performance

**Dependency Injection Pattern** (commit 5a6c2eb)
- Cached summarizer instances
- Better resource management
- Improved testability

**Content Processing Refactor** (May-June 2025)
- **56.4% code reduction** in common/ directory
- Modular architecture with `content/archive/` and `content/extractors/`
- Lazy loading for performance optimization
- Backward compatibility maintained

**PDF Support** (commit 4c04e48)
- Full PDF text extraction pipeline
- PyPDF2 integration
- Metadata extraction (title, author, page count)
- Configurable page limits (default 50 pages)

**Image Prompt Generation** (commit 708fa84)
- Editorial illustration prompt generation
- Multiple artistic styles
- Content analysis with visual keyword detection
- 24-hour caching

**Content Toggle** (commit 2ae2281)
- Switch between AI summary and original text
- Inline display with smooth transitions

**Model Updates** (commit 2a648e5)
- Claude 4.5 Sonnet for complex content
- Claude 4.5 Haiku for simple content
- Both using `-latest` versions

### Clustering Improvements (v2.0)

**Lightweight Clustering System**:
- **90% dependency reduction**: From 2GB+ to 200MB
- **10x faster startup**: No heavy model loading
- **Hybrid approach**: More accurate than keywords alone
- **Fallback strategy**: Works without embeddings
- **Better reliability**: Simple, predictable behavior
- **Connected components**: Improved grouping algorithm

**Previous system** (optional):
- HDBSCAN algorithm with sentence transformers
- Enhanced topic extraction with bigrams
- Still available by uncommenting heavy dependencies

### Bug Fixes

- Fixed cluster data structure mismatches
- Resolved XML/HTML parser conflicts
- Fixed loop variable errors in templates
- Improved error handling throughout
- Fixed missing original content in summaries
- Resolved indentation errors in nested try-except blocks

## API Reference

### Summarization Endpoints

```http
POST /api/summarize
Content-Type: application/json

{
  "url": "https://example.com/article",
  "complexity": 0.7  # Optional: 0.0-1.0, auto-detected if omitted
}

Response: {
  "summary": "Article summary text...",
  "model_used": "claude-sonnet-4-5-latest",
  "cached": false
}
```

### Bookmark Endpoints

```http
# Create bookmark
POST /api/bookmarks
{ "title": "...", "url": "...", "summary": "...", "tags": "tag1,tag2" }

# List bookmarks
GET /api/bookmarks?tags=tag1&read_status=unread

# Get single bookmark
GET /api/bookmarks/{id}

# Update bookmark
PUT /api/bookmarks/{id}
{ "read_status": "read", "tags": "tag1,tag2,tag3" }

# Delete bookmark
DELETE /api/bookmarks/{id}

# Export bookmarks
GET /api/bookmarks/export/json
GET /api/bookmarks/export/csv

# Import bookmarks
POST /api/bookmarks/import
Content-Type: application/json
[{ "title": "...", "url": "...", ... }]
```

### Image Prompt Endpoints

```http
# Generate image prompt
POST /api/generate-image-prompt
{
  "content": "Article text...",
  "title": "Article Title",
  "style": "editorial_illustration"
}

Response: {
  "prompt": "Editorial illustration prompt...",
  "style": "editorial_illustration"
}

# Get available styles
GET /api/image-prompt-styles

Response: {
  "styles": [
    "photojournalistic",
    "editorial_illustration",
    "abstract_conceptual",
    "infographic_style"
  ]
}
```

## Database Schema

**Location**: [data/bookmarks.db](data/bookmarks.db)
**ORM**: SQLAlchemy 2.0+

**Table: bookmarks**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| title | TEXT | Article title |
| url | TEXT | Article URL (unique) |
| summary | TEXT | AI-generated summary |
| content | TEXT | Original article content |
| date_added | DATETIME | Timestamp |
| tags | TEXT | Comma-separated tags |
| read_status | TEXT | "read" or "unread" |

## Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/

# Specific test files
python -m pytest tests/test_batch_processing.py
python -m pytest tests/test_model_selection.py

# With coverage (if installed)
python -m pytest tests/ --cov
```

**Test files**:
- [tests/test_batch_processing.py](tests/test_batch_processing.py)
- [tests/test_model_selection.py](tests/test_model_selection.py)

## Troubleshooting

### Missing Dependencies

If you encounter missing dependency errors:

```bash
# Full installation
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m playwright install chromium

# Minimal installation
pip install -r essential_requirements.txt
python -m spacy download en_core_web_sm
```

### Performance Issues

**Slow batch processing**:
```env
# Reduce workers in .env
MAX_BATCH_WORKERS=2
```

**Memory issues**:
```env
# Reduce cache size
CACHE_SIZE=128
```

**Rate limiting**:
```env
# Adjust API rate limit
API_RPM_LIMIT=30
```

### Debugging

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python server.py
```

Check logs for error patterns, API call traces, and performance metrics.

### Common Issues

**"Anthropic API key not found"**:
- Ensure `.env` file exists in project root
- Verify `ANTHROPIC_API_KEY=sk-ant-...` is set correctly

**"spaCy model not found"**:
```bash
python -m spacy download en_core_web_sm
```

**"Playwright browser not found"**:
```bash
python -m playwright install chromium
```

**XML parser conflicts**:
- Ensure `lxml` is installed: `pip install lxml`
- The application uses `lxml` parser by default

## CLI Usage

Batch process articles from command line:

```bash
# Basic usage
python main.py --input articles.json --output summaries.json

# With custom settings
export API_RPM_LIMIT=30
export MAX_BATCH_WORKERS=2
python main.py --input articles.json --output summaries.json

# Debug mode
export LOG_LEVEL=DEBUG
python main.py --input articles.json --output summaries.json
```

## Performance Characteristics

### Clustering Performance
- **90% size reduction**: 200MB vs 2GB+ dependencies
- **10x faster startup**: No heavy model loading
- **Hybrid similarity**: More accurate than keywords alone
- **Fallback strategy**: Works without embeddings

### Summarization Performance
- **Concurrent processing**: 3 workers default
- **Rate limiting**: 50 RPM default
- **Model selection**: Automatic complexity-based routing
- **Caching**: Multi-level with 30-day TTL

### Response Times (typical)
- **Feed loading**: 1-3 seconds (29 feeds)
- **Article summarization**: 2-5 seconds (Haiku), 5-10 seconds (Sonnet)
- **Clustering**: < 1 second (up to 100 articles)
- **Image prompt generation**: 3-7 seconds

## Migration Notes

### Refactored Modules (Backward Compatible)

The following modules have been refactored but maintain backward compatibility:

**Archive services**:
- Old: `common/archive_service.py`
- New: `content/archive/` directory
- Compatibility: `common/archive_compat.py`

**Source extraction**:
- Old: `common/source_extractor.py`
- New: `content/extractors/` directory
- Compatibility: `common/source_extractor_compat.py`

### Deprecated Patterns

- **Global `latest_data` dictionary**: Use user-specific cache instead
- **Heavy ML clustering**: Use lightweight simple clustering
- **Synchronous rate limiting**: Use async rate limiter

## Contributing

**Contributions are welcome!** We appreciate bug fixes, feature additions, documentation improvements, and more.

### Quick Start for Contributors

Please read [**CONTRIBUTING.md**](CONTRIBUTING.md) for detailed guidelines on:
- Development workflow and setup
- Code style and testing requirements
- Pull request process
- Architecture guidelines

**Key resources for contributors:**
- [**CONTRIBUTING.md**](CONTRIBUTING.md) - Complete contributor guide
- [**CLAUDE.md**](CLAUDE.md) - Development commands and architecture overview
- [**DOCTRINE.md**](DOCTRINE.md) - Design decisions and architectural rationale

### Quick Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/rss-reader.git
cd rss-reader

# Setup and run
./run_server.sh --reload

# Run tests
python -m pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete guide.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic](https://anthropic.com/) - Claude API for AI summarization
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SentenceTransformers](https://www.sbert.net/) - Lightweight text embeddings
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
- [scikit-learn](https://scikit-learn.org/) - Clustering algorithms
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [PyPDF2](https://pypdf2.readthedocs.io/) - PDF text extraction
- [spaCy](https://spacy.io/) - Natural language processing

## Project Status

**Active Development** - Regular updates and improvements

**Latest Release**: See [recent commits](https://github.com/yourusername/rss-reader/commits/)

**Python Version**: 3.11+ (recommended 3.11 specifically)

**Web Framework**: FastAPI (not Flask)

**AI Models**: Claude 4.5 Sonnet and Haiku (both `-latest` versions)

## Support

For issues, questions, or feature requests:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [CLAUDE.md](CLAUDE.md) for development guidance
3. Review [DOCTRINE.md](DOCTRINE.md) for design decisions and architectural rationale
4. Check [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
5. Open an issue on GitHub with detailed information

---

**Built with Claude AI** - Intelligent RSS reading for the modern web
