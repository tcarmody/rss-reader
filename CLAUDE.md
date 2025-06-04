# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Setup and start development server
./run_server.sh --reload --port 5005

# Setup with minimal dependencies (if full install fails)
pip install -r essential_requirements.txt

# Install spaCy model (required for clustering)
python -m spacy download en_core_web_sm

# Install Playwright browser (for paywall bypass)
python -m playwright install chromium
```

### Server Commands
```bash
# Development server with auto-reload
./run_server.sh --reload

# Production server (public access)
./run_server.sh --public --workers 4

# Custom port
./run_server.sh --port 8080

# Direct uvicorn command
uvicorn server:app --host 127.0.0.1 --port 5005 --reload
```

### CLI Processing
```bash
# Batch process articles
python main.py --input articles.json --output summaries.json

# Debug mode
export LOG_LEVEL=DEBUG
python server.py
```

### Testing
```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_batch_processing.py

# Run with coverage (if installed)
python -m pytest tests/ --cov
```

## System Architecture

### Core Architecture Pattern
- **Modular Design**: Separation of concerns across reader, summarization, caching, and clustering
- **Factory Pattern**: Model selection based on content complexity in `models/config.py`
- **Strategy Pattern**: Multiple summarization strategies (ArticleSummarizer vs FastSummarizer)
- **Repository Pattern**: BookmarkManager handles all database operations

### Key Entry Points
- **`server.py`**: FastAPI web application (port 5005)
- **`main.py`**: CLI batch processing interface
- **`run_server.sh`**: Production startup script with environment validation

### Data Flow
```
RSS Feeds → EnhancedRSSReader → FastSummarizer → TieredCache → Web UI
                            ↓
                 ArticleClustering → Topic Extraction
```

### Component Dependencies
- **Reader Layer**: `reader/base_reader.py` → `reader/enhanced_reader.py`
- **Summarization**: `summarization/article_summarizer.py` → `summarization/fast_summarizer.py`
- **Caching**: `cache/base.py` → `cache/memory_cache.py` → `cache/tiered_cache.py`
- **Models**: `models/config.py` handles Claude model selection (Sonnet-4 vs Haiku-3.5)

## Configuration

### Required Environment Variables
```env
ANTHROPIC_API_KEY=required  # Claude API key
```

### Optional Environment Variables
```env
API_RPM_LIMIT=50           # Rate limit for API calls
CACHE_SIZE=256             # In-memory cache size
CACHE_DIR=./summary_cache  # Cache storage directory
CACHE_TTL_DAYS=30         # Cache expiration
MAX_BATCH_WORKERS=3       # Batch processing workers
ENABLE_PAYWALL_BYPASS=false # Paywall bypass capability
LOG_LEVEL=INFO            # Logging level
```

### Configuration Files
- **`rss_feeds.txt`**: RSS feed URLs (one per line, supports `#` comments)
- **`.env`**: Environment variables and API keys
- **`models/config.py`**: AI model selection logic

## Database Schema

### SQLite Database (`data/bookmarks.db`)
- **Table**: `bookmarks`
- **Fields**: `id`, `title`, `url`, `summary`, `content`, `date_added`, `tags`, `read_status`
- **ORM**: SQLAlchemy with declarative base
- **Manager**: `services/bookmark_manager.py`

## AI Model Selection

The system automatically selects between two Claude models based on content complexity:

- **Claude Sonnet 4** (`claude-sonnet-4-20250514`): Complex technical content, complexity ≥ 0.6
- **Claude 3.5 Haiku** (`claude-3-5-haiku-latest`): Simple content, fast processing, complexity < 0.6

Model selection logic is in `models/config.py:select_model_by_complexity()`.

## Critical Dependencies

### Python Version
- **Required**: Python 3.11+ (recommended 3.11 specifically)
- **Virtual Environment**: `rss_venv/` (auto-created by run_server.sh)

### Key Libraries
- **FastAPI**: Web framework (not Flask)
- **Anthropic**: Claude API client
- **sentence-transformers**: Text embeddings for clustering
- **hdbscan**: ML clustering algorithm
- **playwright/selenium**: Paywall bypass (JavaScript rendering)
- **SQLAlchemy 2.0+**: Database ORM

### Post-Install Requirements
```bash
python -m spacy download en_core_web_sm
python -m playwright install chromium
```

## Common Patterns

### Error Handling
- Use `common/errors.py` for custom exception classes
- All API calls should have rate limiting via `api/rate_limiter.py`
- Cache misses fall back to API calls with exponential backoff

### Async Patterns
- FastAPI routes are async by default
- Use `asyncio.gather()` for parallel processing
- Batch operations use `common/batch_processing.py`

### Caching Strategy
- **Tiered**: Memory → Disk → API fallback
- **Key Format**: URL-based with content hash
- **TTL**: 30 days default, configurable
- **Implementation**: `cache/tiered_cache.py`

## File Organization

### Core Modules
- **`api/`**: Rate limiting and API utilities
- **`cache/`**: Multi-level caching system
- **`clustering/`**: ML-based article clustering with HDBSCAN
- **`common/`**: Shared utilities (config, logging, HTTP, performance)
- **`models/`**: Data models and AI model configuration
- **`reader/`**: RSS feed processing and content extraction
- **`services/`**: Business logic (bookmark management)
- **`summarization/`**: Claude API integration and text processing

### Web Interface
- **`templates/`**: Jinja2 templates with component-based structure
- **`static/`**: CSS/JS assets
- **`server.py`**: FastAPI application with session middleware

### Testing
- **`tests/`**: pytest-based test suite
- Run tests before commits to ensure functionality