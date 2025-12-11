# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚡ CRITICAL: Virtual Environment Requirement

**ALWAYS activate the virtual environment before running ANY Python code or commands.**

### Virtual Environment Activation

```bash
# ALWAYS start commands with this prefix
source rss_venv/bin/activate && python <your_command>

# Examples:
source rss_venv/bin/activate && python -m pytest tests/
source rss_venv/bin/activate && python main.py --input articles.json
source rss_venv/bin/activate && python -c "import some_module; print('test')"
```

### Why This Matters

- The project **requires Python 3.11 specifically** (not system Python, not Python 3.12+)
- The virtual environment (`rss_venv/`) is configured for Python 3.11
- Without activation, you'll get `ModuleNotFoundError` for numpy, playwright, etc.
- ALL dependencies (numpy, BeautifulSoup, playwright, etc.) are installed in this venv
- System Python may be a different version and will NOT have the required packages

### Testing and Verification

When testing or verifying implementations:
1. **ALWAYS** prefix with `source rss_venv/bin/activate &&`
2. **NEVER** use bare `python`, `python3`, or `python3.11` commands without activation
3. The venv activation ensures Python 3.11 is used with all dependencies
4. This applies to: pytest, python -c, python -m, python scripts, etc.

### Common Mistakes to Avoid

```bash
# ❌ WRONG - uses system Python (may be wrong version, no packages)
python -c "from clustering.simple import SimpleClustering"
python3 -c "from clustering.simple import SimpleClustering"
python3.11 -c "from clustering.simple import SimpleClustering"

# ✅ CORRECT - activates Python 3.11 venv with all dependencies
source rss_venv/bin/activate && python -c "from clustering.simple import SimpleClustering"
```

### Python Version Verification

After activating the venv, you can verify the correct Python version:
```bash
source rss_venv/bin/activate && python --version
# Should output: Python 3.11.x
```

## 🚨 IMPORTANT: Documenting Design Decisions

**Before making significant design or architectural changes, consider updating [DOCTRINE.md](DOCTRINE.md).**

### When to Update DOCTRINE.md

Update DOCTRINE.md when making:
- **Major Architecture Changes**: New patterns, refactors affecting multiple modules
- **Technology Decisions**: Switching frameworks, databases, AI models, or libraries
- **Performance Trade-offs**: Changes that trade one metric for another
- **Breaking Changes**: Deprecating features or patterns
- **New Design Patterns**: Introducing new architectural approaches

### How to Update DOCTRINE.md

1. **Document the WHY**: Explain rationale, not just the change itself
2. **Document TRADE-OFFS**: List pros and cons explicitly
3. **Specify "When to Reconsider"**: Future conditions that might invalidate this decision
4. **Provide Context**: Help future maintainers understand the situation at decision time

### Example Entry

```markdown
## Feature Name

### Decision: Brief description of what was decided

**Rationale**:
1. Why this approach was chosen
2. What problem it solves
3. What alternatives were considered

**Implementation**: Link to relevant code

**Trade-offs**:
- **Pro**: Benefit 1
- **Pro**: Benefit 2
- **Con**: Drawback 1
- **Mitigation**: How drawbacks are addressed

**When to Reconsider**:
- Condition that might change the decision
- Metrics to monitor
```

DOCTRINE.md serves as institutional memory - it helps future maintainers (human or agentic) understand why decisions were made and evaluate them against new requirements.


## Development Commands

### Environment Setup
```bash
# Setup and start development server
./run_server.sh --reload --port 5005

# Setup with minimal dependencies (if full install fails)
source rss_venv/bin/activate && pip install -r essential_requirements.txt

# Install spaCy model (required for clustering)
source rss_venv/bin/activate && python -m spacy download en_core_web_sm

# Install Playwright browser (for JavaScript-rendered content extraction)
source rss_venv/bin/activate && python -m playwright install chromium
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
source rss_venv/bin/activate && python main.py --input articles.json --output summaries.json

# Debug mode
export LOG_LEVEL=DEBUG
source rss_venv/bin/activate && python server.py
```

### Testing
```bash
# Run tests (ALWAYS activate venv first)
source rss_venv/bin/activate && python -m pytest tests/

# Run specific test files
source rss_venv/bin/activate && python -m pytest tests/test_batch_processing.py
source rss_venv/bin/activate && python -m pytest tests/test_model_selection.py

# Run with coverage (if installed)
source rss_venv/bin/activate && python -m pytest tests/ --cov
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
- **`rss_feeds.txt`**: RSS feed URLs (one per line, supports `#` comments) - 29 AI/tech feeds pre-configured
- **`.env`**: Environment variables and API keys (required: ANTHROPIC_API_KEY)
- **`models/config.py`**: AI model selection logic
- **`requirements.txt`**: Full dependencies (~73 packages)
- **`essential_requirements.txt`**: Minimal dependencies (~23 core packages)

## Database Schema

### SQLite Database (`data/bookmarks.db`)
- **Table**: `bookmarks`
- **Fields**: `id`, `title`, `url`, `summary`, `content`, `date_added`, `tags`, `read_status`
- **ORM**: SQLAlchemy with declarative base
- **Manager**: `services/bookmark_manager.py`

## AI Model Selection

The system automatically selects between two Claude models based on content complexity:

- **Claude 4.5 Sonnet** (`claude-sonnet-4-5-latest`): Complex technical content, complexity ≥ 0.6
- **Claude 4.5 Haiku** (`claude-4-5-haiku-latest`): Simple content, fast processing, complexity < 0.6

Both models use `-latest` versions to automatically receive improvements and updates.

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
- **Required**: Python 3.11 specifically (NOT 3.12+, NOT system Python)
- **Virtual Environment**: `rss_venv/` (auto-created by run_server.sh)
- **CRITICAL**: Always activate venv before running Python commands (see section above)

### Key Libraries
- **FastAPI**: Web framework (not Flask)
- **Anthropic**: Claude API client
- **sentence-transformers**: Text embeddings for clustering
- **hdbscan**: ML clustering algorithm
- **playwright/selenium**: Paywall bypass (JavaScript rendering)
- **SQLAlchemy 2.0+**: Database ORM

### Post-Install Requirements
```bash
# Remember to activate venv first!
source rss_venv/bin/activate && python -m spacy download en_core_web_sm
source rss_venv/bin/activate && python -m playwright install chromium
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
- **56.4% Code Reduction**: common/ directory reduced from 1,963 to 855 lines
- **Enhanced Functionality**: More robust paywall detection and content extraction
- **Backward Compatibility**: Existing code continues to work during transition

## Performance Characteristics

### Clustering Optimization
- **Lightweight Dependencies**: 200MB vs previous 2GB+ (90% reduction)
- **Hybrid Similarity**: 60% semantic + 40% keyword matching
- **Fallback Strategy**: Keywords-only when embeddings unavailable

### Batch Processing
- **Rate Limiting**: 50 RPM default (configurable via API_RPM_LIMIT)
- **Concurrent Workers**: 3 default (configurable via MAX_BATCH_WORKERS)
- **Implementation**: `common/batch_processing.py`