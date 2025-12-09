# DOCTRINE.md

**Design Principles and Architectural Decisions**

This document records the reasoning behind major design and architecture decisions in the Data Points AI RSS Reader. It serves as institutional memory for future maintainers (human or agentic) to understand why choices were made and evaluate them against new requirements.

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [AI Model Strategy](#ai-model-strategy)
3. [Clustering Architecture](#clustering-architecture)
4. [Content Processing](#content-processing)
5. [Feature Decisions](#feature-decisions)
6. [Caching Strategy](#caching-strategy)
7. [Multi-User Architecture](#multi-user-architecture)
8. [Technology Choices](#technology-choices)
9. [Performance Trade-offs](#performance-trade-offs)
10. [Mac Application Design](#mac-application-design)
11. [Metrics and Monitoring](#metrics-and-monitoring)

---

## Core Philosophy

### Principle: Progressive Enhancement with Graceful Degradation

**Decision**: Build features that provide immediate value with basic functionality, then enhance progressively without breaking core experience.

**Rationale**:
- Users should see results immediately, not wait for perfect processing
- Optional dependencies (ChromaDB, heavy ML models) enhance but don't block
- Features work at multiple quality levels depending on available resources

**Examples**:
- **Progressive Clustering** (Tier 3): Fast clustering in <2s, refined results in background
- **Semantic Cache**: Falls back to exact-match when ChromaDB unavailable
- **Lightweight Clustering**: Works without 2GB+ ML dependencies, 90% size reduction

**When to Reconsider**: If progressive enhancement adds excessive complexity or maintenance burden, evaluate whether immediate complete results would be simpler.

---

## AI Model Strategy

### Decision: Dual-Model Architecture (Sonnet vs Haiku)

**Current Models**:
- **Claude 4.5 Sonnet** (`claude-sonnet-4-5`): Complex content (complexity ≥ 0.6)
- **Claude 4.5 Haiku** (`claude-haiku-4-5`): Simple content (complexity < 0.6)

**Rationale**:
1. **Cost Optimization**: Haiku is significantly cheaper for simple articles
2. **Speed Optimization**: Haiku processes 2-3x faster for straightforward content
3. **Quality Preservation**: Sonnet handles technical/complex content requiring deep analysis
4. **Automatic Selection**: Content complexity scoring removes manual model selection

**Implementation**: `models/config.py:select_model_by_complexity()`

**Trade-offs**:
- **Pro**: 40-60% cost reduction on typical RSS feeds (many simple news articles)
- **Pro**: Faster processing for bulk operations
- **Con**: Complexity scoring adds minimal overhead (~50ms)
- **Con**: Risk of misclassification (mitigated by conservative 0.6 threshold)

**When to Reconsider**:
- If Haiku quality degrades for simple content
- If model pricing changes significantly
- If complexity scoring proves unreliable (monitor misclassification rates)
- If new Claude models offer better cost/quality trade-offs

**Why `-latest` Versions Were Removed** (Commit 2a648e5):
- Original intent: Automatically receive improvements
- Reality: Claude removed `-latest` endpoints for 4.5 models
- Current approach: Explicit version IDs (`claude-sonnet-4-5`, `claude-haiku-4-5`)
- Future: Monitor for API versioning changes and update accordingly

---

## Clustering Architecture

### Decision: Lightweight Hybrid Clustering (v2.0)

**Architecture**:
- **Primary**: Simple clustering with 60% semantic + 40% keyword matching
- **Optional**: Enhanced HDBSCAN clustering for users with heavy ML dependencies
- **Dependencies**: 200MB vs previous 2GB+ (90% reduction)

**Rationale**:
1. **Startup Performance**: 10x faster without heavy model loading
2. **Deployment Size**: Dramatically smaller Docker images and downloads
3. **Reliability**: Simple algorithms are more predictable and debuggable
4. **Sufficient Quality**: Hybrid approach adequate for most RSS clustering needs

**Implementation**:
- `clustering/simple.py`: Lightweight default clustering
- `clustering/enhanced.py`: Optional HDBSCAN clustering (requires torch, transformers)

**Trade-offs**:
- **Pro**: Fast startup (<1s vs 10-15s with heavy ML)
- **Pro**: Easier to deploy and maintain
- **Pro**: More predictable behavior
- **Con**: Slightly lower clustering quality vs full HDBSCAN
- **Mitigation**: Progressive clustering refines results in background

**When to Reconsider**:
- If clustering quality becomes insufficient for user needs
- If smaller embedding models (e.g., quantized) become available
- If users consistently need HDBSCAN-level clustering

**Historical Context**: Original system used HDBSCAN with sentence transformers and 2GB+ dependencies. Users rarely needed this precision for RSS article grouping, and startup time became a major pain point.

---

## Content Processing

### Decision: Modular Extractor Architecture (Content Refactor, May-June 2025)

**Structure**:
```
content/
  archive/          # Archive services and paywall detection
    base.py         # Abstract interfaces
    providers.py    # Archive.is, Wayback Machine
    paywall.py      # Hybrid paywall detection
    specialized/    # Site-specific handlers (WSJ, etc.)
  extractors/       # Content extraction
    aggregator.py   # Techmeme, Google News, Reddit, etc.
    pdf.py          # PDF text extraction
    source.py       # Generic utilities
```

**Rationale**:
1. **Separation of Concerns**: Archive services ≠ content extraction
2. **Code Reduction**: 56.4% reduction in common/ directory (1,963 → 855 lines)
3. **Lazy Loading**: Only load components when needed
4. **Extensibility**: Easy to add new aggregators or archive services

**Trade-offs**:
- **Pro**: Cleaner module boundaries
- **Pro**: Better testability (mock individual extractors)
- **Pro**: Performance optimization via lazy loading
- **Con**: More files to navigate
- **Mitigation**: Backward compatibility layers maintained during transition

**When to Reconsider**:
- If module structure becomes too granular (too many small files)
- If backward compatibility layers remain after 6+ months
- If lazy loading doesn't provide meaningful performance benefits

**Historical Context**: Original code had monolithic `common/` directory with mixed responsibilities. Refactoring separated concerns and removed dead code accumulated over development.

---

## Feature Decisions

### Decision: PDF Document Support

**Implementation**: `content/extractors/pdf.py` with PyPDF2

**Rationale**:
1. **User Need**: Research papers and technical reports are common in AI/tech RSS feeds
2. **Workflow Integration**: Users want to summarize PDFs alongside web articles
3. **Archive Access**: Many paywalled articles available as PDFs
4. **Academic Content**: AI research papers (ArXiv, academic journals) often linked as PDFs

**Why PyPDF2**:
- **Lightweight**: No system dependencies (unlike pdfminer or Poppler)
- **Pure Python**: Cross-platform compatibility
- **Sufficient Quality**: Text extraction good enough for summarization
- **Metadata Support**: Extracts title, author, page count
- **Active Maintenance**: Regular updates and bug fixes

**Features**:
- Automatic format detection via URL extension and content-type headers
- Configurable page limits (default: 50 pages to prevent huge documents)
- Text cleaning and normalization
- Metadata extraction for context

**Trade-offs**:
- **Pro**: Seamless integration with existing summarization pipeline
- **Pro**: No additional system dependencies to install
- **Pro**: Handles most common PDF formats
- **Con**: Limited support for scanned PDFs (OCR not included)
- **Con**: Complex PDFs with unusual layouts may extract poorly
- **Mitigation**: Text cleaning helps with minor extraction issues

**When to Reconsider**:
- If OCR support becomes necessary (consider adding pytesseract)
- If extraction quality issues are common (evaluate pdfplumber or pdfminer.six)
- If performance becomes an issue with large PDFs (consider async extraction)

**Alternatives Considered**:
- **pdfminer.six**: More accurate but slower and heavier dependencies
- **pdfplumber**: Better table extraction but overkill for our use case
- **PyMuPDF**: Faster but GPL license incompatible with MIT project

---

### Decision: AI-Powered Image Prompt Generation

**Implementation**: `services/image_prompt_generator.py`

**Rationale**:
1. **Editorial Value**: News articles benefit from compelling visual accompaniment
2. **Content Marketing**: Users want to generate social media visuals for articles
3. **Creative Workflow**: Bridges RSS reader → AI image generators (DALL-E, Midjourney, Stable Diffusion)
4. **Differentiation**: Unique feature not found in typical RSS readers

**Why Editorial Illustration Focus**:
- **Target Audience**: Tech/AI content creators and journalists
- **Professional Output**: Editorial style suitable for blogs, newsletters, presentations
- **Prompt Quality**: Claude excels at analyzing content for visual metaphors
- **Versatility**: Multiple styles (photojournalistic, abstract, infographic) for different contexts

**Architecture Decisions**:
- **Content Analysis**: Extracts 125+ visual keywords from article text
- **Style Templates**: Pre-defined prompt structures for consistent quality
- **Caching**: 24-hour cache per article+style combination
- **Integration**: Modal UI component available on all article views

**Supported Styles**:
- **Photojournalistic**: Realistic news photography (for current events)
- **Editorial Illustration**: Magazine-style artwork (for features/analysis)
- **Abstract Conceptual**: Metaphorical representation (for complex topics)
- **Infographic Style**: Data visualization approach (for technical content)

**Trade-offs**:
- **Pro**: Unique value-add for content creators
- **Pro**: Leverages existing Claude API integration
- **Pro**: Low overhead (prompt generation is fast)
- **Con**: Additional API costs (mitigated by caching)
- **Con**: Requires user to have access to image generation tools
- **Mitigation**: Optional feature, doesn't interfere with core functionality

**When to Reconsider**:
- If API costs for prompt generation become significant (monitor usage)
- If users request different style categories (easy to add)
- If direct image generation integration becomes feasible (DALL-E API integration)
- If prompt quality degrades (tune keyword extraction or templates)

**User Feedback Integration**:
- Initially built for editorial use cases
- Style variety added based on different content types in feeds
- Clipboard copy feature for easy workflow integration

---

## Caching Strategy

### Decision: Three-Tier Cache Architecture

**Layers**:
1. **Memory Cache**: Fast access for recent items (configurable size, default 256)
2. **Disk Cache**: Persistent storage with 30-day TTL
3. **API Fallback**: Claude API when cache misses

**Semantic Cache Layer** (Tier 3):
- **Vector Database**: ChromaDB with sentence-transformers
- **Similarity Threshold**: 0.92 for semantic matches
- **Graceful Degradation**: Falls back to exact-match without ChromaDB

**Rationale**:
1. **Cost Optimization**: Avoid redundant API calls (Claude API costs per token)
2. **Performance**: Memory cache provides sub-millisecond access
3. **Persistence**: Disk cache survives restarts
4. **Semantic Reuse**: Similar articles share summaries (30-40% additional hits)

**Implementation**:
- `cache/tiered_cache.py`: Main caching logic
- `cache/semantic_cache.py`: Vector database semantic matching

**Trade-offs**:
- **Pro**: 60% base cache hit rate, 85% with semantic matching
- **Pro**: Significant cost reduction for repeated/similar content
- **Con**: Disk space usage (mitigated by TTL and cleanup)
- **Con**: Complexity of multi-tier coordination

**When to Reconsider**:
- If cache hit rates drop below 50% (indicates poor cache effectiveness)
- If disk usage becomes problematic (tune TTL or implement smarter eviction)
- If semantic cache similarity threshold needs tuning (monitor quality)

**User-Specific Caching** (Multi-User Support):
- Each user has isolated cache data
- Prevents cache poisoning across users
- Cluster data cached per user session

---

## Multi-User Architecture

### Decision: Session-Based User Isolation

**Architecture**:
- **Shared Auth Database**: `data/auth.db` for all users
- **Per-User Data**: `data/users/{user_id}/user_data.db` for bookmarks, feeds, settings
- **Session Management**: FastAPI session middleware with secure cookies
- **First-User Admin**: First registered user becomes administrator

**Rationale**:
1. **Privacy**: User data completely isolated (bookmarks, feeds, cache)
2. **Scalability**: Per-user databases prevent single-database bottlenecks
3. **Security**: Sessions use secure cookies, password hashing with bcrypt
4. **Simplicity**: SQLite per-user simpler than complex multi-tenant SQL schema

**Trade-offs**:
- **Pro**: Clear data boundaries, no cross-user leaks
- **Pro**: Easy backup (copy user directories)
- **Pro**: Simple to implement and reason about
- **Con**: More database files (not a problem at expected scale)
- **Con**: No cross-user analytics (intentional privacy feature)

**When to Reconsider**:
- If user count exceeds ~1000 (consider PostgreSQL with proper multi-tenancy)
- If cross-user features needed (e.g., shared feeds, collaborative bookmarks)
- If database per-user causes operational problems

**Migration Path**: `scripts/migrate_to_multiuser.py` for existing single-user data.

---

## Technology Choices

### Framework: FastAPI (Not Flask)

**Decision**: Use FastAPI for web framework

**Rationale**:
1. **Async Native**: Proper async/await support throughout
2. **Type Safety**: Pydantic models with automatic validation
3. **Performance**: Significantly faster than Flask for I/O-bound tasks
4. **WebSocket Support**: Native WebSocket for streaming features (Tier 3)
5. **OpenAPI**: Automatic API documentation

**Trade-offs**:
- **Pro**: Better suited for concurrent AI API calls
- **Pro**: Modern Python patterns (type hints, async)
- **Con**: Smaller ecosystem than Flask
- **Con**: Learning curve for developers unfamiliar with async

**When to Reconsider**: If async complexity becomes unmanageable, but this is unlikely given the I/O-bound nature of the application.

### ORM: SQLAlchemy 2.0+

**Decision**: Use SQLAlchemy for database operations

**Rationale**:
1. **Maturity**: Battle-tested ORM with excellent SQLite support
2. **Version 2.0+**: Modern API with better type hints
3. **Flexibility**: Can use raw SQL when needed
4. **Migration Path**: Easy to switch to PostgreSQL if needed

**Trade-offs**:
- **Pro**: Standard Python ORM, widely understood
- **Pro**: Good for simple bookmark/feed management
- **Con**: Overhead for simple queries (acceptable for this use case)

**When to Reconsider**: If performance profiling shows ORM is a bottleneck (unlikely for this application).

### Parsing: lxml for RSS/XML

**Decision**: Use lxml parser for BeautifulSoup

**Rationale**:
1. **Performance**: Faster than html.parser
2. **Robustness**: Better handling of malformed XML/HTML
3. **Standards**: Proper XML namespace support

**Historical Issue**: Had XML/HTML parser conflicts, resolved by explicitly using lxml.

### UI Layer: Jinja2 Templates (Not SPA Framework)

**Decision**: Use server-side Jinja2 templates instead of React/Vue/Angular

**Rationale**:
1. **Simplicity**: Single codebase, no separate frontend build process
2. **SEO-Friendly**: Server-rendered HTML with full content immediately available
3. **Performance**: No large JavaScript bundle downloads, faster initial page load
4. **Python Integration**: Templates directly access Python data structures
5. **Progressive Enhancement**: Can add JavaScript for interactivity without framework lock-in

**Architecture**:
- **Component-Based**: `templates/components/` for reusable UI elements
- **Base Template**: `templates/base.html` with consistent layout
- **8 Pages**: Home, summary, bookmarks, feeds, profile, etc.
- **10 Components**: Navigation, modals, article cards, cluster displays

**When This Works Well**:
- **Read-Heavy App**: RSS reader is primarily consumption, not heavy interaction
- **Moderate Interactivity**: Image prompts, bookmarks, content toggle work fine with vanilla JS
- **Small Team**: No need for separate frontend/backend expertise
- **Quick Iterations**: Template changes don't require rebuild/redeploy cycle

**Trade-offs**:
- **Pro**: Fast development, no build tooling complexity
- **Pro**: Better initial page load performance
- **Pro**: Easier to maintain for small team
- **Pro**: Works great with FastAPI's template support
- **Con**: More full-page reloads (mitigated by targeted AJAX)
- **Con**: Less sophisticated state management
- **Con**: Harder to build complex interactive features (not needed yet)

**JavaScript Strategy**:
- **Vanilla JS**: Simple interactions without framework overhead
- **Targeted Enhancement**: Image prompt modal, bookmark actions, WebSocket updates
- **Progressive**: Works without JavaScript for core functionality

**When to Reconsider**:
- If app becomes write-heavy (e.g., rich text editing, complex forms)
- If real-time collaboration features needed
- If mobile app requires API-first architecture
- If team grows and frontend specialization emerges
- If WebSocket features require complex client-side state management

**Why Not React/Vue/Angular**:
- **Build Complexity**: Would need webpack/vite, transpilation, dependency management
- **Over-Engineering**: Current interaction patterns don't justify SPA complexity
- **Performance**: SPA bundle size would slow initial load for content-focused app
- **Maintenance**: Additional language (TypeScript), tooling, and expertise required

**Future Path**: If interactive features grow, consider:
1. **HTMX**: Enhance templates with declarative AJAX (minimal JS)
2. **Alpine.js**: Lightweight reactivity without full framework
3. **API First**: Add JSON endpoints, incrementally migrate to SPA

### Embedding Models: sentence-transformers

**Decision**: Use sentence-transformers with `all-MiniLM-L6-v2` for embeddings

**Rationale**:
1. **Size**: 80MB model vs 500MB+ alternatives
2. **Quality**: Sufficient for RSS article clustering
3. **Speed**: Fast inference for real-time clustering
4. **No GPU Required**: Works well on CPU

**When to Reconsider**: If newer, smaller models (e.g., quantized) offer better performance.

---

## Performance Trade-offs

### Batch Processing Limits

**Configuration**:
- **API_RPM_LIMIT**: 50 requests per minute (default)
- **MAX_BATCH_WORKERS**: 3 concurrent workers (default)

**Rationale**:
1. **API Rate Limits**: Claude API has rate limits
2. **Resource Management**: Too many workers overwhelm memory
3. **Cost Control**: Prevents runaway API costs

**Trade-offs**:
- **Pro**: Prevents rate limit errors
- **Pro**: Predictable resource usage
- **Con**: Slower for large batches
- **Tuning**: Adjust based on API tier and available resources

**When to Reconsider**:
- If Claude API rate limits change
- If users consistently process huge batches (consider queuing system)

### WebSocket Streaming (Tier 3)

**Decision**: Use WebSocket for real-time summary delivery

**Rationale**:
1. **User Experience**: 70%+ perceived latency reduction
2. **Progress Feedback**: Users see results as they complete
3. **Error Handling**: Graceful per-article error reporting

**Implementation**: `api/websocket_streaming.py`

**Trade-offs**:
- **Pro**: Much better UX for batch operations
- **Pro**: Enables progressive rendering
- **Con**: More complex than REST API
- **Con**: Requires WebSocket-compatible proxy configuration

**When to Reconsider**: If WebSocket connection management becomes problematic, consider Server-Sent Events (SSE) as simpler alternative.

---

## Mac Application Design

### Decision: Electron Wrapper for Native App

**Architecture**:
- **Electron Main Process**: Manages window, Python server lifecycle
- **Python Backend**: FastAPI server runs as subprocess
- **Renderer Process**: Loads web UI with Mac-native styles

**Rationale**:
1. **Code Reuse**: Leverages existing web UI
2. **Native Feel**: Electron provides Mac integration (menu bar, shortcuts, dock)
3. **Cross-Platform**: Could extend to Windows/Linux with minimal changes
4. **Maintenance**: Single codebase for web and desktop

**Trade-offs**:
- **Pro**: Fast development (wrap existing app)
- **Pro**: Native OS integration
- **Con**: Larger download size (~200MB)
- **Con**: More complex deployment (bundle Python + Node)

**Universal Binary**: Builds for both Apple Silicon and Intel Macs (recommended approach).

**When to Reconsider**:
- If Electron overhead becomes problematic (consider Tauri for Rust-based alternative)
- If Python bundling causes distribution issues
- If native Swift/SwiftUI app would provide better UX

---

### Decision: Python Bundling Strategy

**Approach**: Bundle Python runtime and dependencies with Electron app

**Why Bundle Python**:
1. **User Experience**: No "install Python 3.11 first" requirement
2. **Version Control**: Guaranteed Python 3.11+ with exact dependencies
3. **Consistency**: Same environment across all user machines
4. **Simplicity**: Single .dmg installer, no multi-step setup
5. **Isolation**: Doesn't interfere with system Python or other apps

**Implementation**:
- **Python Location**: Bundled in `app.asar` or adjacent to Electron binary
- **Virtual Environment**: Pre-created venv with all dependencies
- **Subprocess Management**: Electron spawns Python server on startup, kills on quit
- **Port Management**: Dynamic port assignment to avoid conflicts

**Trade-offs**:
- **Pro**: One-click installation for end users
- **Pro**: No version mismatches or dependency conflicts
- **Pro**: Professional app experience
- **Pro**: Works offline after installation
- **Con**: Large app size (~200-300MB vs ~50MB without Python)
- **Con**: Updates require full app download (not just Python code)
- **Con**: Build complexity (separate builds for arm64 and x64)
- **Mitigation**: Universal binary reduces user confusion, GitHub releases handle distribution

**Alternatives Considered**:
- **System Python**: Requires user installation, version conflicts likely
- **Conda Distribution**: 500MB+ size, overkill for this app
- **Python.org Installer**: Separate download, breaks one-click install UX
- **Docker Container**: Overkill, requires Docker Desktop, worse UX

**Build Process**:
```bash
# Universal binary for both architectures
make build-universal

# Or architecture-specific
make build-arm64  # Apple Silicon
make build-x64    # Intel
```

**Distribution Strategy**:
- **GitHub Releases**: .dmg files for easy installation
- **Universal Binary**: Single download works on all Macs (recommended)
- **Auto-Updates**: Electron's auto-updater for future versions

**When to Reconsider**:
- If app size becomes prohibitive (>500MB)
- If Python-only updates are frequent (consider API server separation)
- If targeting professional developers (may prefer system Python)
- If cross-platform support requires different bundling approaches

**User Benefits**:
- **No Terminal**: Users never see command line or installation steps
- **Native Feel**: Dock icon, menu bar, keyboard shortcuts work as expected
- **Reliable**: Bundled dependencies mean it "just works"
- **Professional**: Feels like a real Mac app, not a web wrapper

---

## Metrics and Monitoring

This section documents key performance indicators that validate architectural decisions. Monitor these metrics to determine when decisions should be reconsidered.

### Cache Performance

**Target Metrics**:
- **Base Cache Hit Rate**: >60% (memory + disk)
- **Semantic Cache Hit Rate**: >85% (with ChromaDB)
- **Cache Hit Quality**: <5% user reports of stale/incorrect cached summaries

**How to Monitor**:
```python
# In production, log cache statistics
from cache.tiered_cache import TieredCache

cache = TieredCache()
stats = cache.get_stats()
# Log: hit_rate, memory_hits, disk_hits, api_calls, size
```

**When to Act**:
- **Hit rate <50%**: Cache TTL too aggressive or content too diverse
- **Hit rate >95%**: Cache TTL too conservative, wasting disk space
- **Quality issues**: Similarity threshold too low (tune semantic cache)

**Related Decisions**: [Caching Strategy](#caching-strategy), [Semantic Cache](#semantic-cache-layer-tier-3)

---

### Model Selection Accuracy

**Target Metrics**:
- **Misclassification Rate**: <5% of articles assigned to wrong model
- **Cost Efficiency**: 40-60% of articles use Haiku (cheaper model)
- **User Satisfaction**: No complaints about summary quality degradation

**How to Monitor**:
```python
# Track model selection distribution
from models.config import select_model_by_complexity

# Log: complexity_score, selected_model, article_url
# Periodically sample and manually verify model appropriateness
```

**How to Validate**:
1. Sample 50 articles per week
2. Manually assess if Haiku/Sonnet selection was appropriate
3. Adjust threshold (currently 0.6) if misclassification >5%

**When to Act**:
- **Haiku usage <30%**: Threshold too high, wasting money on Sonnet
- **Haiku usage >70%**: Threshold too low, may sacrifice quality
- **Quality complaints**: Haiku may not be sufficient, lower threshold

**Related Decisions**: [AI Model Strategy](#ai-model-strategy)

---

### Clustering Quality

**Target Metrics**:
- **Cluster Coherence**: >0.7 average intra-cluster similarity
- **Topic Accuracy**: >80% of clusters have sensible topics (manual validation)
- **User Engagement**: Clustering visible on homepage, not disabled by users

**How to Monitor**:
```python
# In clustering code, track coherence
from clustering.progressive import create_progressive_clusterer

clusterer = create_progressive_clusterer(coherence_threshold=0.7)
# Log: coherence_scores, cluster_sizes, refinement_splits, refinement_merges
```

**How to Validate**:
1. Weekly sample of 20 clustered article sets
2. Manually verify topics make sense
3. Check for obvious mis-clustered articles

**When to Act**:
- **Coherence <0.6**: Clustering too aggressive, unrelated articles grouped
- **Many small clusters**: Similarity threshold too strict
- **Few large clusters**: Similarity threshold too loose

**Related Decisions**: [Clustering Architecture](#clustering-architecture), [Progressive Clustering](#decision-progressive-clustering-pipeline)

---

### API Costs

**Target Metrics**:
- **Cost per Article**: <$0.005 average (including cache hits)
- **Daily Cost**: Monitor for unexpected spikes
- **Cache Effectiveness**: 85% hit rate keeps costs sustainable

**How to Monitor**:
```python
# Track API usage
# Log: tokens_used, model_used, cost_estimate, cache_hit
# Monthly report: total_cost, articles_processed, cost_per_article
```

**When to Act**:
- **Cost per article >$0.01**: Cache not working or too many complex articles
- **Sudden spikes**: Investigate cache invalidation or unusual feed content
- **Cost trending up**: Re-evaluate model selection or cache TTL

**Related Decisions**: [AI Model Strategy](#ai-model-strategy), [Caching Strategy](#caching-strategy)

---

### WebSocket Reliability

**Target Metrics**:
- **Connection Success Rate**: >95%
- **Message Delivery**: >99% of summary messages delivered
- **Reconnection Time**: <3 seconds on disconnect

**How to Monitor**:
```python
# In WebSocket handler
from api.websocket_streaming import ConnectionManager

manager = ConnectionManager()
# Log: connections_opened, connections_closed, messages_sent, errors
```

**When to Act**:
- **Success rate <90%**: Connection management issues, investigate proxy config
- **Frequent disconnects**: Heartbeat interval too long, network instability
- **Slow reconnection**: Server overload or network issues

**Related Decisions**: [WebSocket Streaming](#decision-websocket-streaming-tier-3)

---

### Progressive Clustering Performance

**Target Metrics**:
- **Stage 1 Time**: <2 seconds for initial fast clustering
- **Stage 2 Time**: <10 seconds for background refinement
- **Refinement Value**: ≥10% of clusters refined (split or merged)

**How to Monitor**:
```python
# In progressive clustering
from clustering.progressive import create_progressive_clusterer

# Log: stage1_duration, stage2_duration, initial_clusters, final_clusters, splits, merges
```

**When to Act**:
- **Stage 1 >3s**: Fast clustering too slow, review algorithm
- **Stage 2 >30s**: Refinement too complex, adjust coherence checks
- **No refinements**: Coherence threshold too permissive, not adding value

**Related Decisions**: [Progressive Clustering](#decision-progressive-clustering-pipeline)

---

### Mac App Performance

**Target Metrics**:
- **Startup Time**: <5 seconds from launch to UI ready
- **Memory Usage**: <500MB total (Electron + Python)
- **CPU Usage**: <10% idle, <50% during summarization

**How to Monitor**:
- macOS Activity Monitor
- Electron `process.memoryUsage()`
- User feedback on performance

**When to Act**:
- **Startup >10s**: Python server slow to start, optimize dependencies
- **Memory >1GB**: Memory leak investigation needed
- **High idle CPU**: Background task issues or event loop problems

**Related Decisions**: [Mac Application Design](#mac-application-design)

---

### Monitoring Best Practices

**Logging Strategy**:
1. **Production**: Log all metrics to structured logs (JSON)
2. **Development**: DEBUG level for detailed traces
3. **User Privacy**: Never log article content, only URLs and metadata

**Alerting Thresholds**:
- **Cache hit rate drops below 50%**: Investigate within 24 hours
- **API costs 2x expected**: Immediate investigation
- **Clustering fails >5% of time**: Review error patterns

**Review Cadence**:
- **Daily**: API costs, error rates
- **Weekly**: Cache performance, model selection distribution
- **Monthly**: Clustering quality, user engagement metrics
- **Quarterly**: Full architecture review with metrics dashboard

**Tool Suggestions**:
- **Simple Setup**: Log to files, periodic manual review
- **Intermediate**: Prometheus + Grafana for dashboards
- **Advanced**: DataDog, New Relic for full observability

---

## Design Patterns

### Factory Pattern: Model Selection

**Usage**: `models/config.py:select_model_by_complexity()`

**Why**: Centralized model selection logic makes it easy to adjust thresholds or add new models without changing caller code.

### Strategy Pattern: Multiple Summarizers

**Implementation**:
- `ArticleSummarizer`: Base summarizer with caching
- `FastSummarizer`: Optimized parallel processing

**Why**: Different summarization strategies for different use cases (single article vs batch).

### Repository Pattern: Bookmark Management

**Implementation**: `services/bookmark_manager.py`

**Why**: Encapsulates all database operations, makes testing easier, provides clean API.

### Service Layer: Business Logic

**Examples**:
- `ImagePromptGenerator`: AI-powered image prompt generation
- `BookmarkManager`: Bookmark CRUD operations

**Why**: Separates business logic from web routes, improves testability and reusability.

---

## Testing Philosophy

### Test Coverage Strategy

**Current State**: Tests for critical paths (batch processing, model selection, Tier 3 features)

**Rationale**:
1. **Critical Paths**: Test features users rely on most
2. **Integration Tests**: Test component interaction, not just units
3. **Pragmatic Coverage**: Not pursuing 100% coverage, focus on valuable tests

**When to Increase Testing**:
- Before refactoring complex modules
- When bugs appear in untested areas
- For features with complex business logic

---

## Future Considerations

### When to Migrate from SQLite

**Current**: SQLite with per-user databases

**Consider PostgreSQL if**:
- User count exceeds ~1000 active users
- Need cross-user analytics or features
- Complex query performance becomes an issue
- Need better concurrent write performance

**Migration Path**: SQLAlchemy makes this relatively straightforward.

### When to Add Queuing System

**Current**: Direct API calls with rate limiting

**Consider Redis/Celery if**:
- Processing batches of 100+ articles regularly
- Need background job processing
- Want to distribute work across multiple servers

### When to Split Services

**Current**: Monolithic FastAPI application

**Consider Microservices if**:
- Services need independent scaling (e.g., summarization vs clustering)
- Team grows beyond 5-10 developers
- Different services have different deployment requirements

**Note**: Don't prematurely optimize with microservices.

---

## Deprecation Policy

### Backward Compatibility

**Current Approach**: Maintain compatibility layers during transitions

**Examples**:
- `common/archive_compat.py`: Compatibility for refactored archive services
- `common/source_extractor_compat.py`: Compatibility for refactored extractors

**Policy**:
1. Add new implementation with compatibility layer
2. Update documentation to recommend new approach
3. Remove compatibility layer after 6+ months or major version bump
4. Provide migration guides for breaking changes

### Deprecated Patterns

**Global `latest_data` Dictionary** → User-specific cache
- **Why Deprecated**: Multi-user support requires isolation
- **Migration**: Use session-based caching

**Heavy ML Clustering** → Lightweight simple clustering
- **Why Deprecated**: 90% dependency reduction, 10x faster startup
- **Migration**: Still available in `clustering/enhanced.py` for users who need it

**Synchronous Rate Limiting** → Async rate limiter
- **Why Deprecated**: FastAPI is async-native
- **Migration**: Use `api/rate_limiter.py` async methods

---

## Documentation Maintenance

### When to Update This Document

**Required Updates**:
1. **Major Architecture Changes**: New patterns, removed patterns, significant refactors
2. **Technology Decisions**: Switching frameworks, databases, AI models
3. **Trade-off Discoveries**: When monitoring reveals unexpected trade-offs
4. **Deprecations**: When marking features/patterns as deprecated

**How to Update**:
1. Explain the **why** behind decisions, not just the **what**
2. Document **trade-offs** explicitly (pros and cons)
3. Specify **when to reconsider** decisions
4. Provide **historical context** for future maintainers

**Update Prompt in CLAUDE.md**: See CLAUDE.md for instructions on keeping DOCTRINE.md current.

---

## Version History

- **2025-12-09**: Initial DOCTRINE.md creation (Priority 1-2 Updates)
  - Documented core philosophy, AI model strategy, clustering architecture
  - Added content processing, caching strategy, multi-user architecture
  - Documented technology choices, performance trade-offs
  - Added Mac application design, testing philosophy, future considerations
  - **Feature Decisions**: Added PDF support rationale (PyPDF2 choice, alternatives)
  - **Feature Decisions**: Added AI-powered image prompt generation decision
  - **Technology Choices**: Added Jinja2 templates vs SPA framework rationale
  - **Mac App**: Expanded with Python bundling strategy and distribution approach
  - **Metrics and Monitoring**: Added comprehensive monitoring framework with target metrics
    - Cache performance, model selection accuracy, clustering quality
    - API costs tracking, WebSocket reliability, progressive clustering performance
    - Mac app performance metrics, monitoring best practices

---

## Contributing to This Document

When making significant design or architecture decisions:

1. **Before Implementation**: Consider documenting the decision and rationale here
2. **During Implementation**: Note trade-offs and alternatives considered
3. **After Implementation**: Update with actual performance characteristics and lessons learned
4. **When Reverting**: Document why the original decision didn't work out

This document is a living record. Keep it updated so future maintainers (including your future self) understand the reasoning behind choices.
