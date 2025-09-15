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
- **Service Layer**: ImagePromptGenerator for AI-powered image prompt generation

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
- **Image Prompts**: `services/image_prompt_generator.py` → leverages ArticleSummarizer
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

## Supported Content Types

The RSS Reader can process multiple content formats:

### **Web Content**
- HTML articles and blog posts
- RSS/XML feeds
- News articles with paywall bypass
- Aggregator links (Google News, Reddit, etc.)

### **PDF Documents** (NEW)
- Research papers and academic documents
- PDF articles and reports
- Automatic text extraction with PyPDF2
- Metadata extraction (title, author, page count)
- Configurable page limits for large documents

### **Content Processing Features**
- Automatic format detection (URL extension and content-type)
- Text cleaning and normalization
- Content length validation
- Error handling with fallback strategies

## Image Prompt Generation Feature

### Overview
The RSS Reader includes AI-powered image prompt generation that creates detailed prompts for AI image generators like DALL-E, Midjourney, and Stable Diffusion. This feature is available on all pages and supports multiple artistic styles.

### Architecture
- **Service**: `services/image_prompt_generator.py` - Core prompt generation logic
- **API Endpoints**:
  - `POST /api/generate-image-prompt` - Generate prompts from article content
  - `GET /api/image-prompt-styles` - Get available style options
- **Frontend**: 
  - `static/js/image-prompt.js` - Modal interactions and API calls
  - `templates/components/image_prompt_modal.html` - Modal UI component
  - CSS integrated in `static/css/styles.css`

### Supported Styles
- **Photojournalistic**: Realistic news photography style
- **Editorial Illustration**: Artistic illustration for magazines
- **Abstract Conceptual**: Abstract art representation
- **Infographic Style**: Data visualization and clean design

### Integration Points
- **Home Page**: Image prompt buttons on each article in clusters
- **Individual Summary**: Generate prompts for summarized articles
- **Bookmarks Page**: Generate prompts for saved articles

### Performance Features
- **Caching**: 24-hour cache for generated prompts
- **Rate Limiting**: Integrated with existing API rate limits
- **Error Handling**: Graceful fallbacks with simple prompts
- **Content Optimization**: Automatic text truncation and keyword extraction

### Usage
1. Click "Image Prompt" button on any article
2. Select desired artistic style
3. Click "Generate Prompt"
4. Copy generated prompt to clipboard
5. Use with AI image generation tools

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
- **`common/`**: True shared utilities (config, logging, HTTP, performance, batch processing, errors)
- **`content/`**: Content processing and archive services (NEW)
  - **`content/archive/`**: Archive services, paywall detection, and specialized handlers
  - **`content/extractors/`**: Source extraction and content processing utilities
- **`models/`**: Data models and AI model configuration
- **`reader/`**: RSS feed processing and content extraction
- **`services/`**: Business logic (bookmark management, image prompt generation)
- **`summarization/`**: Claude API integration and text processing

### Web Interface
- **`templates/`**: Jinja2 templates with component-based structure
- **`static/`**: CSS/JS assets
- **`server.py`**: FastAPI application with session middleware

### Content Processing (NEW Architecture)
The content/ directory contains the refactored content processing system:

#### Archive Services (`content/archive/`)
- **`base.py`**: Abstract interfaces and data classes
- **`providers.py`**: Archive service implementations (Archive.is, Wayback Machine, etc.)
- **`paywall.py`**: Paywall detection with hybrid detection methods
- **`specialized/wsj.py`**: WSJ-specific bypass logic

#### Extractors (`content/extractors/`)
- **`base.py`**: Abstract extractor interfaces and base classes
- **`aggregator.py`**: Consolidated aggregator extraction (Techmeme, Google News, Reddit, etc.)
- **`source.py`**: Generic content utilities, validation, and cleaning
- **`pdf.py`**: PDF text extraction with PyPDF2 (supports research papers, documents)

### Testing
- **`tests/`**: pytest-based test suite
- Run tests before commits to ensure functionality

### Architectural Improvements
- **Modular Design**: Clear separation between archive services and content extraction
- **Lazy Loading**: Performance optimization with on-demand component initialization
- **56.4% Reduction**: common/ directory reduced from 1,963 to 855 lines
- **Enhanced Functionality**: More robust paywall detection and content extraction
- **Backward Compatibility**: Existing code continues to work during transition