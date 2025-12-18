# RESTART: RSS Reader Redesign from First Principles

This document outlines how to rebuild the RSS reader with lessons learned from the current implementation. It preserves valuable features (semantic caching, JavaScript rendering, paywall bypass) while dramatically simplifying the architecture and improving the user experience.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Core Modules](#core-modules)
5. [Advanced Features](#advanced-features)
6. [User Experience Design](#user-experience-design)
7. [Native Mac App](#native-mac-app)
8. [Data Model](#data-model)
9. [Implementation Phases](#implementation-phases)
10. [Migration Path](#migration-path)

---

## Design Philosophy

### Guiding Principles

1. **Summaries are the product** - AI summaries should be visible by default, not hidden behind clicks
2. **Chronological by default** - Time-based organization is universally understood; clustering is optional
3. **Progressive disclosure** - Simple surface, power features on demand
4. **Native-first** - Build for macOS properly, not as a web wrapper
5. **Minimal viable complexity** - Every feature must justify its existence

### What We're Keeping

- Claude API integration (Sonnet/Haiku model selection)
- Tiered caching (memory + disk)
- JavaScript rendering for JS-heavy sites (Playwright)
- Semantic caching for similar content detection
- Archive service integration for paywalled content
- SQLite for local persistence

### What We're Simplifying

- One clustering implementation (keyword-based), not six
- One cache class with pluggable backends
- Flat module structure, not deep nesting
- Single configuration file, not scattered settings

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Native Swift App                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Views     │  │   State     │  │   Native Services   │  │
│  │  (SwiftUI)  │  │ (Observable)│  │ (Notifications,     │  │
│  │             │  │             │  │  Spotlight, etc.)   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│                    ┌─────▼─────┐                             │
│                    │  Bridge   │                             │
│                    │  (JSON)   │                             │
│                    └─────┬─────┘                             │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Python Backend                             │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                      FastAPI                             │ │
│  │  /articles  /summarize  /feeds  /bookmarks  /settings   │ │
│  └─────────────────────────┬───────────────────────────────┘ │
│                            │                                  │
│  ┌────────────┬────────────┼────────────┬────────────┐       │
│  │            │            │            │            │       │
│  ▼            ▼            ▼            ▼            ▼       │
│ Feeds    Summarizer     Cache      Database    Fetcher       │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Advanced Features (Optional)                │ │
│  │  Semantic Cache │ JS Renderer │ Archive Services        │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Communication Flow

1. **Swift app** handles all UI rendering and user interaction
2. **Python backend** runs as a local server (localhost:5005)
3. **JSON API** for all communication - clean separation
4. **Background processing** for feed fetching and summarization

---

## Project Structure

```
rss-reader/
├── app/                        # Native Swift/SwiftUI app
│   ├── RSSReader.xcodeproj
│   ├── RSSReader/
│   │   ├── App/
│   │   │   ├── RSSReaderApp.swift
│   │   │   └── AppDelegate.swift
│   │   ├── Views/
│   │   │   ├── MainView.swift           # Three-pane layout
│   │   │   ├── FeedListView.swift       # Left sidebar
│   │   │   ├── ArticleListView.swift    # Middle pane
│   │   │   ├── ArticleDetailView.swift  # Right pane / reading
│   │   │   ├── SettingsView.swift       # Preferences window
│   │   │   └── Components/
│   │   │       ├── ArticleRow.swift
│   │   │       ├── SummaryView.swift
│   │   │       └── SearchBar.swift
│   │   ├── Models/
│   │   │   ├── Article.swift
│   │   │   ├── Feed.swift
│   │   │   └── AppState.swift
│   │   ├── Services/
│   │   │   ├── APIClient.swift          # Talks to Python backend
│   │   │   ├── PythonServer.swift       # Manages backend lifecycle
│   │   │   └── NotificationService.swift
│   │   └── Resources/
│   │       ├── Assets.xcassets
│   │       └── DefaultFeeds.json
│   └── RSSReaderTests/
│
├── backend/                    # Python backend (FastAPI)
│   ├── server.py              # API endpoints (~400 lines)
│   ├── feeds.py               # Feed fetching and parsing (~200 lines)
│   ├── summarizer.py          # Claude API integration (~300 lines)
│   ├── cache.py               # Unified caching (~200 lines)
│   ├── database.py            # SQLite operations (~150 lines)
│   ├── fetcher.py             # HTTP + content extraction (~250 lines)
│   │
│   ├── advanced/              # Optional power features
│   │   ├── __init__.py
│   │   ├── semantic_cache.py  # ChromaDB similarity matching (~300 lines)
│   │   ├── js_renderer.py     # Playwright integration (~200 lines)
│   │   ├── archive.py         # Archive.is, Wayback Machine (~250 lines)
│   │   └── clustering.py      # Keyword-based grouping (~200 lines)
│   │
│   └── tests/
│       ├── test_feeds.py
│       ├── test_summarizer.py
│       └── test_cache.py
│
├── data/                       # Runtime data (gitignored)
│   ├── articles.db
│   ├── cache/
│   └── chromadb/              # Semantic cache vectors
│
├── README.md                   # User documentation
├── CONTRIBUTING.md             # Developer setup
├── ARCHITECTURE.md             # This document's successor
├── requirements.txt            # Python dependencies (~20 packages)
└── Makefile                    # Common commands
```

### Line Count Targets

| Module | Target Lines | Purpose |
|--------|--------------|---------|
| `server.py` | ~400 | API endpoints, routing |
| `feeds.py` | ~200 | RSS parsing, feed management |
| `summarizer.py` | ~300 | Claude API, model selection |
| `cache.py` | ~200 | Memory + disk caching |
| `database.py` | ~150 | SQLite CRUD operations |
| `fetcher.py` | ~250 | HTTP requests, content extraction |
| `advanced/*` | ~950 | Optional power features |
| **Total Backend** | **~2,450** | vs ~15,000 currently |

---

## Core Modules

### 1. Server (`backend/server.py`)

Minimal FastAPI application with clear endpoints:

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="RSS Reader API")

# ─────────────────────────────────────────────────────────────
# Articles
# ─────────────────────────────────────────────────────────────

@app.get("/articles")
async def list_articles(
    feed_id: int | None = None,
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0
) -> list[Article]:
    """Get articles, optionally filtered by feed or read status."""
    pass

@app.get("/articles/{article_id}")
async def get_article(article_id: int) -> ArticleDetail:
    """Get single article with full summary."""
    pass

@app.post("/articles/{article_id}/read")
async def mark_read(article_id: int) -> None:
    """Mark article as read."""
    pass

@app.post("/articles/{article_id}/bookmark")
async def toggle_bookmark(article_id: int) -> None:
    """Toggle bookmark status."""
    pass

# ─────────────────────────────────────────────────────────────
# Summarization
# ─────────────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize_url(url: str, background: BackgroundTasks) -> dict:
    """Summarize a single URL (can be any webpage, not just feeds)."""
    pass

@app.post("/summarize/batch")
async def summarize_batch(urls: list[str]) -> dict:
    """Summarize multiple URLs."""
    pass

# ─────────────────────────────────────────────────────────────
# Feeds
# ─────────────────────────────────────────────────────────────

@app.get("/feeds")
async def list_feeds() -> list[Feed]:
    """List all subscribed feeds."""
    pass

@app.post("/feeds")
async def add_feed(url: str, name: str | None = None) -> Feed:
    """Subscribe to a new feed."""
    pass

@app.delete("/feeds/{feed_id}")
async def remove_feed(feed_id: int) -> None:
    """Unsubscribe from a feed."""
    pass

@app.post("/feeds/refresh")
async def refresh_feeds(background: BackgroundTasks) -> dict:
    """Trigger feed refresh (runs in background)."""
    pass

# ─────────────────────────────────────────────────────────────
# Search & Filtering
# ─────────────────────────────────────────────────────────────

@app.get("/search")
async def search(q: str, limit: int = 20) -> list[Article]:
    """Full-text search across articles and summaries."""
    pass

@app.get("/articles/grouped")
async def grouped_articles(group_by: str = "date") -> dict:
    """Get articles grouped by date, topic, or source."""
    pass

# ─────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────

@app.get("/settings")
async def get_settings() -> Settings:
    pass

@app.put("/settings")
async def update_settings(settings: Settings) -> Settings:
    pass

@app.get("/status")
async def health_check() -> dict:
    """API health check."""
    return {"status": "ok", "version": "2.0.0"}
```

### 2. Summarizer (`backend/summarizer.py`)

Clean Claude API integration:

```python
import anthropic
from dataclasses import dataclass
from enum import Enum

class Model(Enum):
    SONNET = "claude-sonnet-4-5-20250514"
    HAIKU = "claude-haiku-4-5-20250514"

@dataclass
class Summary:
    title: str
    one_liner: str          # 1 sentence for feed view
    full_summary: str       # 3-5 paragraphs
    key_points: list[str]   # Bullet points
    model_used: Model
    cached: bool = False

class Summarizer:
    def __init__(self, api_key: str, cache: "Cache"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cache = cache

    def summarize(self, content: str, url: str) -> Summary:
        # Check cache first
        if cached := self.cache.get(url):
            return cached

        # Select model based on content complexity
        model = self._select_model(content)

        # Generate summary
        response = self.client.messages.create(
            model=model.value,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": self._build_prompt(content)
            }]
        )

        summary = self._parse_response(response, model)
        self.cache.set(url, summary)
        return summary

    def _select_model(self, content: str) -> Model:
        """Simple complexity heuristic."""
        word_count = len(content.split())
        has_technical_terms = any(term in content.lower() for term in [
            "algorithm", "neural", "quantum", "blockchain", "protocol"
        ])

        if word_count > 2000 or has_technical_terms:
            return Model.SONNET
        return Model.HAIKU

    def _build_prompt(self, content: str) -> str:
        return f"""Summarize this article. Provide:
1. A one-sentence summary (max 150 characters)
2. A full summary (3-5 paragraphs)
3. Key points as bullet points (3-5 items)

Article:
{content[:15000]}  # Truncate for token limits
"""

    def _parse_response(self, response, model: Model) -> Summary:
        # Parse structured response
        pass
```

### 3. Cache (`backend/cache.py`)

Unified caching with pluggable backends:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib

@dataclass
class CacheEntry:
    key: str
    value: any
    created_at: datetime
    expires_at: datetime | None

class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> any | None: pass

    @abstractmethod
    def set(self, key: str, value: any, ttl: int | None = None) -> None: pass

    @abstractmethod
    def delete(self, key: str) -> None: pass

class MemoryCache(CacheBackend):
    """Fast in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> any | None:
        if entry := self._cache.get(key):
            if entry.expires_at and entry.expires_at < datetime.now():
                del self._cache[key]
                return None
            return entry.value
        return None

    def set(self, key: str, value: any, ttl: int | None = None) -> None:
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self._cache[key] = CacheEntry(key, value, datetime.now(), expires_at)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def _evict_oldest(self):
        oldest = min(self._cache.values(), key=lambda e: e.created_at)
        del self._cache[oldest.key]

class DiskCache(CacheBackend):
    """Persistent disk cache for summaries."""

    def __init__(self, cache_dir: Path, ttl_days: int = 30):
        self.cache_dir = cache_dir
        self.ttl_days = ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hashed}.json"

    def get(self, key: str) -> any | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            created = datetime.fromisoformat(data["created_at"])
            if datetime.now() - created > timedelta(days=self.ttl_days):
                path.unlink()
                return None
            return data["value"]
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, key: str, value: any, ttl: int | None = None) -> None:
        path = self._key_to_path(key)
        data = {
            "key": key,
            "value": value,
            "created_at": datetime.now().isoformat()
        }
        path.write_text(json.dumps(data))

    def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        path.unlink(missing_ok=True)

class TieredCache:
    """Two-tier cache: memory (fast) -> disk (persistent)."""

    def __init__(self, cache_dir: Path, memory_size: int = 256, ttl_days: int = 30):
        self.memory = MemoryCache(max_size=memory_size)
        self.disk = DiskCache(cache_dir, ttl_days=ttl_days)

    def get(self, key: str) -> any | None:
        # Check memory first
        if value := self.memory.get(key):
            return value

        # Fall back to disk
        if value := self.disk.get(key):
            # Promote to memory
            self.memory.set(key, value)
            return value

        return None

    def set(self, key: str, value: any, ttl: int | None = None) -> None:
        self.memory.set(key, value, ttl)
        self.disk.set(key, value, ttl)

    def delete(self, key: str) -> None:
        self.memory.delete(key)
        self.disk.delete(key)
```

---

## Advanced Features

These are optional modules that provide power features for specific use cases.

### 1. Semantic Cache (`backend/advanced/semantic_cache.py`)

For detecting when you've already summarized similar content:

```python
"""
Semantic Cache - Find similar previously-summarized content.

USE CASE: When an article covers the same story as one you've already read,
return the existing summary instead of calling Claude again.

REQUIREMENTS: pip install chromadb sentence-transformers

HOW IT WORKS:
1. When summarizing, we embed the article text
2. Before calling Claude, we search for similar embeddings
3. If similarity > threshold, return cached summary
4. Otherwise, generate new summary and store embedding
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(
        self,
        db_path: Path,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.threshold = similarity_threshold
        self.encoder = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name="article_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def find_similar(self, content: str) -> tuple[str, float] | None:
        """
        Find a similar article that's already been summarized.

        Returns: (cached_summary_key, similarity_score) or None
        """
        embedding = self.encoder.encode(content[:5000])  # Truncate for speed

        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=1
        )

        if not results["ids"][0]:
            return None

        # ChromaDB returns distance, convert to similarity
        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity >= self.threshold:
            return (results["ids"][0][0], similarity)

        return None

    def store(self, key: str, content: str) -> None:
        """Store embedding for future similarity searches."""
        embedding = self.encoder.encode(content[:5000])

        self.collection.upsert(
            ids=[key],
            embeddings=[embedding.tolist()],
            documents=[content[:1000]]  # Store snippet for debugging
        )

    def delete(self, key: str) -> None:
        """Remove embedding."""
        try:
            self.collection.delete(ids=[key])
        except Exception:
            pass  # Ignore if not found


# Integration with main summarizer:
#
# class Summarizer:
#     def __init__(self, ..., semantic_cache: SemanticCache | None = None):
#         self.semantic_cache = semantic_cache
#
#     def summarize(self, content: str, url: str) -> Summary:
#         # Check semantic similarity first
#         if self.semantic_cache:
#             if similar := self.semantic_cache.find_similar(content):
#                 similar_key, score = similar
#                 if cached := self.cache.get(similar_key):
#                     cached.cached = True
#                     return cached
#
#         # ... generate summary ...
#
#         # Store embedding for future similarity matches
#         if self.semantic_cache:
#             self.semantic_cache.store(url, content)
```

### 2. JavaScript Renderer (`backend/advanced/js_renderer.py`)

For sites that require JavaScript to display content:

```python
"""
JavaScript Renderer - Fetch content from JS-heavy sites.

USE CASE: Some sites (especially news aggregators, SPAs, and paywalled sites)
require JavaScript execution to display article content.

REQUIREMENTS:
- pip install playwright
- python -m playwright install chromium

WHEN TO USE:
- Initial fetch returns mostly empty/placeholder content
- Site is known to be JS-heavy (React, Vue, etc.)
- User explicitly requests "full fetch"

PERFORMANCE NOTES:
- Adds 2-5 seconds to fetch time
- Uses ~100MB memory per browser instance
- Browser is reused across requests for efficiency
"""

from playwright.async_api import async_playwright, Browser, Page
from contextlib import asynccontextmanager
import asyncio

class JSRenderer:
    def __init__(self, headless: bool = True, timeout_ms: int = 30000):
        self.headless = headless
        self.timeout = timeout_ms
        self._browser: Browser | None = None
        self._playwright = None

    async def start(self):
        """Initialize browser (call once at app startup)."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless
        )

    async def stop(self):
        """Cleanup browser (call at app shutdown)."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @asynccontextmanager
    async def _get_page(self):
        """Get a new page, ensuring cleanup."""
        page = await self._browser.new_page()
        try:
            yield page
        finally:
            await page.close()

    async def fetch(self, url: str) -> str:
        """
        Fetch page content after JavaScript execution.

        Returns: HTML content as string
        """
        if not self._browser:
            await self.start()

        async with self._get_page() as page:
            # Navigate and wait for content to load
            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            # Wait a bit more for dynamic content
            await asyncio.sleep(1)

            # Get rendered HTML
            content = await page.content()
            return content

    async def fetch_with_scroll(self, url: str, scroll_count: int = 3) -> str:
        """
        Fetch content with scrolling (for infinite scroll pages).
        """
        if not self._browser:
            await self.start()

        async with self._get_page() as page:
            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            # Scroll to trigger lazy loading
            for _ in range(scroll_count):
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
                await asyncio.sleep(0.5)

            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")

            return await page.content()


# Integration with fetcher:
#
# class Fetcher:
#     def __init__(self, js_renderer: JSRenderer | None = None):
#         self.js_renderer = js_renderer
#
#     async def fetch(self, url: str, force_js: bool = False) -> str:
#         # Try simple HTTP first
#         if not force_js:
#             content = await self._simple_fetch(url)
#             if self._has_content(content):
#                 return content
#
#         # Fall back to JS rendering
#         if self.js_renderer:
#             return await self.js_renderer.fetch(url)
#
#         raise FetchError("Content requires JavaScript but renderer not available")
```

### 3. Archive Services (`backend/advanced/archive.py`)

For bypassing paywalls via archive services:

```python
"""
Archive Services - Access paywalled content via archives.

USE CASE: When a site requires subscription/payment, try to find
the content in web archives instead.

SERVICES:
1. Archive.is / Archive.today - Best for recent news articles
2. Wayback Machine - Best for older content, more comprehensive
3. Google Cache - Sometimes works, less reliable

ETHICAL NOTE: This is for personal reading of content you'd otherwise
have access to (e.g., limited free articles). Not for circumventing
legitimate subscription requirements.
"""

import aiohttp
from dataclasses import dataclass
from urllib.parse import quote_plus

@dataclass
class ArchiveResult:
    url: str
    content: str
    source: str  # "archive.is", "wayback", "google"
    archive_date: str | None

class ArchiveService:
    """Base class for archive services."""

    name: str = "base"

    async def find(self, url: str) -> ArchiveResult | None:
        raise NotImplementedError

class ArchiveIs(ArchiveService):
    """Archive.is / Archive.today"""

    name = "archive.is"
    base_url = "https://archive.is"

    async def find(self, url: str) -> ArchiveResult | None:
        search_url = f"{self.base_url}/newest/{url}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(search_url, timeout=10) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        return ArchiveResult(
                            url=str(resp.url),
                            content=content,
                            source=self.name,
                            archive_date=None  # Parse from page if needed
                        )
            except Exception:
                pass

        return None

class WaybackMachine(ArchiveService):
    """Internet Archive's Wayback Machine"""

    name = "wayback"
    api_url = "https://archive.org/wayback/available"

    async def find(self, url: str) -> ArchiveResult | None:
        async with aiohttp.ClientSession() as session:
            try:
                # Check availability
                async with session.get(
                    self.api_url,
                    params={"url": url},
                    timeout=10
                ) as resp:
                    data = await resp.json()

                snapshot = data.get("archived_snapshots", {}).get("closest")
                if not snapshot or not snapshot.get("available"):
                    return None

                archive_url = snapshot["url"]

                # Fetch archived content
                async with session.get(archive_url, timeout=15) as resp:
                    content = await resp.text()

                return ArchiveResult(
                    url=archive_url,
                    content=content,
                    source=self.name,
                    archive_date=snapshot.get("timestamp")
                )

            except Exception:
                pass

        return None

class ArchiveManager:
    """Try multiple archive services in order."""

    def __init__(self):
        self.services = [
            ArchiveIs(),
            WaybackMachine(),
        ]

    async def find(self, url: str) -> ArchiveResult | None:
        """Try each service until one succeeds."""
        for service in self.services:
            if result := await service.find(url):
                return result
        return None

    def is_likely_paywalled(self, url: str) -> bool:
        """Heuristic: check if URL is from a known paywalled site."""
        paywalled_domains = [
            "wsj.com",
            "nytimes.com",
            "ft.com",
            "economist.com",
            "bloomberg.com",
            "washingtonpost.com",
            "theathletic.com",
            "businessinsider.com",
        ]
        return any(domain in url for domain in paywalled_domains)


# Integration with fetcher:
#
# class Fetcher:
#     def __init__(self, archive_manager: ArchiveManager | None = None):
#         self.archives = archive_manager
#
#     async def fetch(self, url: str) -> str:
#         # Try direct fetch first
#         content = await self._simple_fetch(url)
#
#         # If looks paywalled and we have archive access, try that
#         if self._looks_paywalled(content) and self.archives:
#             if archived := await self.archives.find(url):
#                 return archived.content
#
#         return content
```

### 4. Clustering (`backend/advanced/clustering.py`)

Simple keyword-based article grouping:

```python
"""
Clustering - Group articles by topic.

USE CASE: When viewing many articles, group related ones together
so users can see "5 articles about GPT-5" instead of 5 separate items.

APPROACH: Simple keyword extraction + Jaccard similarity.
No ML models, no embeddings, no heavy dependencies.

WHY NOT ML:
- ML clustering (HDBSCAN, etc.) requires heavy dependencies
- Overkill for <100 articles
- Keyword matching is interpretable and fast
- "Good enough" for the use case
"""

from collections import Counter
from dataclasses import dataclass
import re

@dataclass
class Cluster:
    id: str
    name: str
    articles: list[str]  # Article IDs
    keywords: list[str]

class ArticleClusterer:
    def __init__(
        self,
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.3,
        max_clusters: int = 10
    ):
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters

        # Common words to ignore
        self.stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now",
            "new", "says", "said", "year", "years", "today", "according",
        }

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract significant keywords from text."""
        # Tokenize and clean
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())

        # Filter stopwords
        words = [w for w in words if w not in self.stopwords]

        # Count frequencies
        counts = Counter(words)

        # Return top keywords
        return [word for word, _ in counts.most_common(max_keywords)]

    def similarity(self, keywords1: list[str], keywords2: list[str]) -> float:
        """Jaccard similarity between keyword sets."""
        set1, set2 = set(keywords1), set(keywords2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    def cluster(self, articles: list[dict]) -> list[Cluster]:
        """
        Group articles into clusters.

        articles: List of {"id": str, "title": str, "content": str}
        """
        # Extract keywords for each article
        article_keywords = {}
        for article in articles:
            text = f"{article['title']} {article.get('content', '')}"
            article_keywords[article["id"]] = self.extract_keywords(text)

        # Build clusters greedily
        clusters = []
        used = set()

        for article in articles:
            if article["id"] in used:
                continue

            # Find similar articles
            cluster_articles = [article["id"]]
            cluster_keywords = set(article_keywords[article["id"]])

            for other in articles:
                if other["id"] in used or other["id"] == article["id"]:
                    continue

                sim = self.similarity(
                    article_keywords[article["id"]],
                    article_keywords[other["id"]]
                )

                if sim >= self.similarity_threshold:
                    cluster_articles.append(other["id"])
                    cluster_keywords.update(article_keywords[other["id"]])

            # Only keep clusters with minimum size
            if len(cluster_articles) >= self.min_cluster_size:
                # Generate cluster name from top keywords
                keyword_counts = Counter()
                for aid in cluster_articles:
                    keyword_counts.update(article_keywords[aid])
                top_keywords = [w for w, _ in keyword_counts.most_common(3)]

                clusters.append(Cluster(
                    id=f"cluster_{len(clusters)}",
                    name=" / ".join(top_keywords).title(),
                    articles=cluster_articles,
                    keywords=top_keywords
                ))
                used.update(cluster_articles)

        # Limit number of clusters
        clusters = clusters[:self.max_clusters]

        # Add remaining articles as "unclustered"
        unclustered = [a["id"] for a in articles if a["id"] not in used]
        if unclustered:
            clusters.append(Cluster(
                id="unclustered",
                name="Other Articles",
                articles=unclustered,
                keywords=[]
            ))

        return clusters
```

---

## User Experience Design

### Core UX Principles

1. **Summaries visible by default** - The AI summary is the product
2. **Chronological as primary view** - Clustering is optional
3. **Progressive disclosure** - Simple surface, power on demand
4. **Keyboard-first** - Reading apps need great keyboard support
5. **Native Mac patterns** - Three-pane, Quick Look, Spotlight

### Main Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│  RSS Reader                                        🔍 Search   ⚙️      │
├────────────┬────────────────────────────┬───────────────────────────────┤
│            │                            │                               │
│  FEEDS     │  TODAY                     │  OpenAI Announces GPT-5       │
│            │                            │                               │
│  All (47)  │  ● OpenAI Announces GPT-5  │  The Verge · 2 hours ago      │
│  Unread    │    The Verge · 2h          │                               │
│  Saved ★   │    OpenAI revealed its     │  ─────────────────────────    │
│            │    next-generation model   │                               │
│  ────────  │    with significantly...   │  OpenAI has unveiled GPT-5,   │
│            │                       [→]  │  the latest iteration of its  │
│  SOURCES   │                            │  large language model. The    │
│            │  ● Google DeepMind Study   │  new model demonstrates       │
│  □ The     │    MIT Tech · 4h           │  significant improvements in  │
│    Verge   │    Researchers published   │  reasoning, coding, and       │
│  □ Ars     │    findings on...          │  multimodal capabilities...   │
│    Tech    │                       [→]  │                               │
│  □ MIT     │                            │  Key Points:                  │
│    Tech    │  ─── Earlier Today ───     │  • 2x context window          │
│  □ Wired   │                            │  • Native multimodal input    │
│            │  ○ Apple ML Framework      │  • Improved reasoning         │
│  ────────  │    ...                     │  • Available via API today    │
│            │                            │                               │
│  + Add     │  ○ Microsoft Copilot       │  [Read Original] [Bookmark]   │
│    Feed    │    ...                     │                               │
│            │                            │                               │
└────────────┴────────────────────────────┴───────────────────────────────┘

Legend:
● = Unread article
○ = Read article
★ = Bookmarked
[→] = Expand/collapse
```

### View States

#### 1. Feed View (Default)
- Chronological list
- 2-line summary preview
- Unread indicator (filled dot)
- Time grouping ("Today", "Yesterday", "Last Week")

#### 2. Article View (Right Pane)
- Full AI summary
- Key points as bullets
- Link to original article
- Bookmark action
- Share action (hidden in menu)

#### 3. Grouped View (Optional Toggle)
- Toggle: "Group by Topic" in view menu
- Shows clusters with count badges
- Expand cluster to see articles

#### 4. Search Results
- Instant search as you type
- Highlights matches
- Searches: titles, summaries, content

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑` / `↓` or `j` / `k` | Navigate articles |
| `Enter` or `→` | Open article in reading pane |
| `Space` | Toggle summary expanded/collapsed |
| `o` | Open original in browser |
| `b` | Toggle bookmark |
| `r` | Mark as read |
| `u` | Mark as unread |
| `⌘F` | Focus search |
| `⌘R` | Refresh feeds |
| `⌘,` | Open preferences |
| `⌘1` | Show all articles |
| `⌘2` | Show unread only |
| `⌘3` | Show bookmarked |
| `/` | Quick search |
| `?` | Show keyboard shortcuts |

### Empty States

#### No Articles
```
┌─────────────────────────────────────┐
│                                     │
│      ☀️ You're all caught up!       │
│                                     │
│      No new articles since your     │
│      last visit 3 hours ago.        │
│                                     │
│      [Refresh Now]                  │
│                                     │
└─────────────────────────────────────┘
```

#### No Feeds
```
┌─────────────────────────────────────┐
│                                     │
│      📰 Welcome to RSS Reader       │
│                                     │
│      Add some feeds to get started. │
│                                     │
│      [Add Feed]  [Import OPML]      │
│                                     │
│      Or start with our suggestions: │
│      □ Tech News (5 feeds)          │
│      □ AI & ML (4 feeds)            │
│      □ Science (3 feeds)            │
│                                     │
└─────────────────────────────────────┘
```

#### Search - No Results
```
┌─────────────────────────────────────┐
│                                     │
│      🔍 No results for "quantum"    │
│                                     │
│      Try different keywords or      │
│      check your spelling.           │
│                                     │
└─────────────────────────────────────┘
```

### Settings

Preferences window (native macOS style):

```
┌─────────────────────────────────────────────────┐
│  General  │  Feeds  │  AI  │  Advanced         │
├───────────┴─────────┴──────┴────────────────────┤
│                                                 │
│  APPEARANCE                                     │
│  ─────────────────────────────────────          │
│  Theme: [System ▼]                              │
│  □ Show article previews in list                │
│  □ Show read articles (dimmed)                  │
│                                                 │
│  READING                                        │
│  ─────────────────────────────────────          │
│  Default view: [Chronological ▼]                │
│  □ Mark as read when opened                     │
│  □ Mark as read after 3 seconds                 │
│                                                 │
│  REFRESH                                        │
│  ─────────────────────────────────────          │
│  Auto-refresh: [Every 30 minutes ▼]             │
│  □ Refresh on app launch                        │
│  □ Show notification for new articles           │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Mobile Considerations (Future)

If building an iOS app later, the same API supports:

- Single-column layout
- Swipe gestures (left=read, right=bookmark)
- Pull to refresh
- Share sheet integration

---

## Native Mac App

### SwiftUI Views Structure

```swift
// Main three-pane layout
struct MainView: View {
    @StateObject var appState = AppState()

    var body: some View {
        NavigationSplitView {
            FeedListView()
        } content: {
            ArticleListView()
        } detail: {
            ArticleDetailView()
        }
        .navigationSplitViewStyle(.balanced)
        .searchable(text: $appState.searchQuery)
        .onAppear { appState.refresh() }
    }
}

// Left sidebar: feeds and filters
struct FeedListView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        List(selection: $appState.selectedFeed) {
            Section("Filters") {
                Label("All", systemImage: "tray.full")
                    .tag(Filter.all)
                Label("Unread", systemImage: "circle.fill")
                    .tag(Filter.unread)
                Label("Saved", systemImage: "star.fill")
                    .tag(Filter.saved)
            }

            Section("Feeds") {
                ForEach(appState.feeds) { feed in
                    Label(feed.name, systemImage: "dot.radiowaves.up.forward")
                        .badge(feed.unreadCount)
                        .tag(Filter.feed(feed.id))
                }
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Feeds")
        .toolbar {
            Button(action: { appState.showAddFeed = true }) {
                Image(systemName: "plus")
            }
        }
    }
}

// Middle pane: article list
struct ArticleListView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        List(selection: $appState.selectedArticle) {
            ForEach(appState.groupedArticles) { group in
                Section(group.title) {
                    ForEach(group.articles) { article in
                        ArticleRow(article: article)
                            .tag(article.id)
                    }
                }
            }
        }
        .listStyle(.inset)
        .navigationTitle(appState.currentFilterName)
    }
}

// Article row with inline summary preview
struct ArticleRow: View {
    let article: Article

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .fill(article.isRead ? Color.clear : Color.accentColor)
                    .frame(width: 8, height: 8)

                Text(article.title)
                    .font(.headline)
                    .lineLimit(2)
            }

            Text("\(article.source) · \(article.timeAgo)")
                .font(.caption)
                .foregroundStyle(.secondary)

            if let preview = article.summaryPreview {
                Text(preview)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
}

// Right pane: full article detail
struct ArticleDetailView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        if let article = appState.selectedArticleDetail {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    Text(article.title)
                        .font(.title)

                    HStack {
                        Text(article.source)
                        Text("·")
                        Text(article.timeAgo)
                    }
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                    Divider()

                    // AI Summary
                    Text(article.fullSummary)
                        .font(.body)

                    // Key Points
                    if !article.keyPoints.isEmpty {
                        Text("Key Points")
                            .font(.headline)

                        ForEach(article.keyPoints, id: \.self) { point in
                            HStack(alignment: .top) {
                                Text("•")
                                Text(point)
                            }
                        }
                    }

                    Divider()

                    // Actions
                    HStack {
                        Button("Read Original") {
                            NSWorkspace.shared.open(article.url)
                        }

                        Button(article.isBookmarked ? "Bookmarked" : "Bookmark") {
                            appState.toggleBookmark(article.id)
                        }
                    }
                }
                .padding()
            }
        } else {
            ContentUnavailableView(
                "Select an Article",
                systemImage: "doc.text",
                description: Text("Choose an article from the list to read its summary.")
            )
        }
    }
}
```

### App Menus

```swift
@main
struct RSSReaderApp: App {
    var body: some Scene {
        WindowGroup {
            MainView()
        }
        .commands {
            // File menu
            CommandGroup(replacing: .newItem) {
                Button("Add Feed...") { }
                    .keyboardShortcut("n", modifiers: .command)

                Button("Import OPML...") { }
                    .keyboardShortcut("i", modifiers: [.command, .shift])
            }

            // View menu
            CommandMenu("View") {
                Button("Show All") { }
                    .keyboardShortcut("1", modifiers: .command)

                Button("Show Unread") { }
                    .keyboardShortcut("2", modifiers: .command)

                Button("Show Bookmarked") { }
                    .keyboardShortcut("3", modifiers: .command)

                Divider()

                Toggle("Group by Topic", isOn: .constant(false))
                    .keyboardShortcut("g", modifiers: .command)
            }

            // Article menu
            CommandMenu("Article") {
                Button("Open Original") { }
                    .keyboardShortcut("o", modifiers: .command)

                Button("Toggle Bookmark") { }
                    .keyboardShortcut("b", modifiers: .command)

                Button("Mark as Read") { }
                    .keyboardShortcut("r", modifiers: .command)

                Divider()

                Button("Refresh Feeds") { }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
            }
        }

        Settings {
            SettingsView()
        }
    }
}
```

### Native Integrations

```swift
// Spotlight indexing
import CoreSpotlight

func indexArticle(_ article: Article) {
    let attributeSet = CSSearchableItemAttributeSet(contentType: .text)
    attributeSet.title = article.title
    attributeSet.contentDescription = article.summaryPreview
    attributeSet.keywords = article.keywords

    let item = CSSearchableItem(
        uniqueIdentifier: article.id,
        domainIdentifier: "com.app.articles",
        attributeSet: attributeSet
    )

    CSSearchableIndex.default().indexSearchableItems([item])
}

// Notifications
import UserNotifications

func notifyNewArticles(count: Int) {
    let content = UNMutableNotificationContent()
    content.title = "New Articles"
    content.body = "\(count) new articles available"
    content.sound = .default

    let request = UNNotificationRequest(
        identifier: UUID().uuidString,
        content: content,
        trigger: nil
    )

    UNUserNotificationCenter.current().add(request)
}

// Dock badge
func updateDockBadge(unreadCount: Int) {
    NSApplication.shared.dockTile.badgeLabel =
        unreadCount > 0 ? "\(unreadCount)" : nil
}
```

---

## Data Model

### SQLite Schema

```sql
-- Feeds table
CREATE TABLE feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    category TEXT,
    last_fetched TIMESTAMP,
    fetch_error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Articles table
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    published_at TIMESTAMP,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Content
    content TEXT,
    content_hash TEXT,  -- For deduplication

    -- Summary (generated)
    summary_short TEXT,     -- 1 sentence
    summary_full TEXT,      -- Full summary
    key_points TEXT,        -- JSON array
    model_used TEXT,        -- "sonnet" or "haiku"
    summarized_at TIMESTAMP,

    -- User state
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    is_bookmarked BOOLEAN DEFAULT FALSE,
    bookmarked_at TIMESTAMP,

    -- Indexing
    keywords TEXT,  -- JSON array for clustering/search

    UNIQUE(feed_id, url)
);

-- Full-text search
CREATE VIRTUAL TABLE articles_fts USING fts5(
    title,
    content,
    summary_full,
    content='articles',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER articles_ai AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts(rowid, title, content, summary_full)
    VALUES (new.id, new.title, new.content, new.summary_full);
END;

-- Settings table (key-value)
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_articles_feed ON articles(feed_id);
CREATE INDEX idx_articles_published ON articles(published_at DESC);
CREATE INDEX idx_articles_unread ON articles(is_read, published_at DESC);
CREATE INDEX idx_articles_bookmarked ON articles(is_bookmarked, bookmarked_at DESC);
```

### Swift Models

```swift
struct Feed: Identifiable, Codable {
    let id: Int
    let url: URL
    var name: String
    var category: String?
    var lastFetched: Date?
    var unreadCount: Int = 0
}

struct Article: Identifiable, Codable {
    let id: Int
    let feedId: Int
    let url: URL
    let title: String
    let author: String?
    let publishedAt: Date?

    // Summary
    var summaryShort: String?
    var summaryFull: String?
    var keyPoints: [String]?

    // State
    var isRead: Bool
    var isBookmarked: Bool

    // Computed
    var summaryPreview: String? { summaryShort }
    var source: String { /* derive from feed */ }
    var timeAgo: String { /* relative time */ }
}

struct ArticleGroup: Identifiable {
    let id: String
    let title: String  // "Today", "Yesterday", or cluster name
    let articles: [Article]
}
```

---

## Implementation Phases

### Phase 1: Core Backend (Week 1-2)
- [ ] FastAPI server with basic endpoints
- [ ] Feed fetching and parsing
- [ ] SQLite database setup
- [ ] Basic disk cache

**Deliverable**: API that can fetch feeds and return articles as JSON

### Phase 2: Summarization (Week 3-4)
- [ ] Claude API integration
- [ ] Model selection logic
- [ ] Tiered caching (memory + disk)
- [ ] Background summarization

**Deliverable**: Articles get AI summaries automatically

### Phase 3: Swift App Shell (Week 5-6)
- [ ] Xcode project setup
- [ ] Three-pane layout
- [ ] API client
- [ ] Python server management

**Deliverable**: Native app that displays articles from backend

### Phase 4: Core UX (Week 7-8)
- [ ] Article list with previews
- [ ] Reading pane with full summary
- [ ] Read/unread tracking
- [ ] Bookmarking

**Deliverable**: Fully functional reading experience

### Phase 5: Search & Navigation (Week 9-10)
- [ ] Full-text search
- [ ] Keyboard shortcuts
- [ ] Feed management
- [ ] Preferences window

**Deliverable**: Power user features complete

### Phase 6: Advanced Features (Week 11-14)
- [ ] Semantic cache integration
- [ ] JavaScript rendering
- [ ] Archive services for paywalls
- [ ] Optional clustering view

**Deliverable**: All features from current app, reimplemented

### Phase 7: Polish (Week 15-16)
- [ ] Spotlight integration
- [ ] Notifications
- [ ] Dock badge
- [ ] App icon and branding
- [ ] Performance optimization

**Deliverable**: Release-ready application

---

## Migration Path

If migrating from the current codebase:

### What to Keep
- `rss_feeds.txt` - Feed list
- `data/*.db` - Article history and bookmarks
- `summary_cache/` - Cached summaries
- `.env` - API keys

### What to Port
- `summarization/fast_summarizer.py` → `backend/summarizer.py` (simplify)
- `cache/tiered_cache.py` → `backend/cache.py` (keep as-is)
- `clustering/simple.py` → `backend/advanced/clustering.py` (keep as-is)
- `content/archive/` → `backend/advanced/archive.py` (consolidate)

### What to Discard
- All Jinja2 templates (replaced by SwiftUI)
- All JavaScript (replaced by SwiftUI)
- `server.py` complexity (rewrite simpler)
- Multiple clustering implementations (keep only simple)
- Compatibility layers

### Database Migration

```python
# Script to migrate existing data
def migrate():
    old_db = sqlite3.connect("data/bookmarks.db")
    new_db = sqlite3.connect("data/articles.db")

    # Copy feeds
    # Copy articles with bookmarks
    # Preserve read states
    # Link to existing cache files
```

---

## Additional Core Modules

### 4. Feed Parser (`backend/feeds.py`)

```python
"""
Feed Parser - Fetch and parse RSS/Atom feeds.

Handles:
- RSS 2.0 and Atom 1.0 formats
- Feed autodiscovery from HTML pages
- Error handling and retry logic
- Rate limiting per domain
"""

import feedparser
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
import aiohttp
import asyncio

@dataclass
class FeedItem:
    url: str
    title: str
    author: str | None
    published: datetime | None
    content: str

@dataclass
class Feed:
    url: str
    title: str
    description: str | None
    items: list[FeedItem]
    last_fetched: datetime

class FeedParser:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._domain_last_fetch: dict[str, float] = {}
        self._min_interval = 1.0  # Min seconds between requests to same domain

    async def fetch(self, url: str) -> Feed:
        """Fetch and parse a feed URL."""
        # Rate limit per domain
        domain = urlparse(url).netloc
        await self._rate_limit(domain)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.timeout) as resp:
                content = await resp.text()

        return self._parse(url, content)

    async def fetch_multiple(self, urls: list[str]) -> list[Feed | Exception]:
        """Fetch multiple feeds concurrently."""
        tasks = [self.fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _parse(self, url: str, content: str) -> Feed:
        """Parse feed content using feedparser."""
        parsed = feedparser.parse(content)

        items = []
        for entry in parsed.entries:
            # Extract content (prefer content over summary)
            content_text = ""
            if hasattr(entry, "content") and entry.content:
                content_text = entry.content[0].value
            elif hasattr(entry, "summary"):
                content_text = entry.summary

            # Parse published date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])

            items.append(FeedItem(
                url=entry.get("link", ""),
                title=entry.get("title", "Untitled"),
                author=entry.get("author"),
                published=published,
                content=content_text
            ))

        return Feed(
            url=url,
            title=parsed.feed.get("title", "Unknown Feed"),
            description=parsed.feed.get("description"),
            items=items,
            last_fetched=datetime.now()
        )

    async def _rate_limit(self, domain: str):
        """Ensure minimum interval between requests to same domain."""
        import time
        now = time.time()
        if domain in self._domain_last_fetch:
            elapsed = now - self._domain_last_fetch[domain]
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
        self._domain_last_fetch[domain] = time.time()

    def discover_feed(self, html: str, base_url: str) -> str | None:
        """Find feed URL from HTML page (autodiscovery)."""
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        soup = BeautifulSoup(html, "html.parser")

        # Look for RSS/Atom link tags
        for link in soup.find_all("link", rel="alternate"):
            link_type = link.get("type", "")
            if "rss" in link_type or "atom" in link_type:
                href = link.get("href")
                if href:
                    return urljoin(base_url, href)

        return None
```

### 5. Content Fetcher (`backend/fetcher.py`)

```python
"""
Content Fetcher - Extract article content from URLs.

Handles:
- HTTP fetching with proper headers
- HTML content extraction (removes boilerplate)
- Integration with JS renderer and archive services
- PDF extraction
"""

import aiohttp
from bs4 import BeautifulSoup
from dataclasses import dataclass
import re

@dataclass
class FetchResult:
    url: str
    title: str
    content: str
    author: str | None = None
    published: str | None = None
    source: str = "direct"  # "direct", "archive", "js_render"

class Fetcher:
    def __init__(
        self,
        js_renderer: "JSRenderer | None" = None,
        archive_manager: "ArchiveManager | None" = None,
        timeout: int = 30
    ):
        self.js_renderer = js_renderer
        self.archives = archive_manager
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) RSS Reader/2.0"
        }

    async def fetch(self, url: str, force_js: bool = False) -> FetchResult:
        """
        Fetch and extract content from URL.

        Tries in order:
        1. Simple HTTP fetch
        2. JS rendering (if content looks incomplete)
        3. Archive services (if content looks paywalled)
        """
        # Try simple fetch first
        if not force_js:
            try:
                result = await self._simple_fetch(url)
                if self._has_content(result.content):
                    return result
            except Exception:
                pass

        # Try JS rendering if available
        if self.js_renderer:
            try:
                html = await self.js_renderer.fetch(url)
                result = self._extract_content(url, html)
                result.source = "js_render"
                if self._has_content(result.content):
                    return result
            except Exception:
                pass

        # Try archive services if looks paywalled
        if self.archives and self._looks_paywalled(url):
            try:
                archived = await self.archives.find(url)
                if archived:
                    result = self._extract_content(url, archived.content)
                    result.source = f"archive:{archived.source}"
                    return result
            except Exception:
                pass

        # Return whatever we got
        return await self._simple_fetch(url)

    async def _simple_fetch(self, url: str) -> FetchResult:
        """Basic HTTP fetch."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(url, timeout=self.timeout) as resp:
                html = await resp.text()
        return self._extract_content(url, html)

    def _extract_content(self, url: str, html: str) -> FetchResult:
        """Extract article content from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Try to find article content
        article = (
            soup.find("article") or
            soup.find(class_=re.compile(r"article|post|content|entry", re.I)) or
            soup.find("main") or
            soup.body
        )

        # Extract text
        content = article.get_text(separator="\n", strip=True) if article else ""

        # Clean up whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Extract title
        title = ""
        if title_tag := soup.find("title"):
            title = title_tag.get_text(strip=True)
        elif h1 := soup.find("h1"):
            title = h1.get_text(strip=True)

        # Extract author
        author = None
        if author_meta := soup.find("meta", {"name": "author"}):
            author = author_meta.get("content")

        return FetchResult(
            url=url,
            title=title,
            content=content,
            author=author
        )

    def _has_content(self, content: str) -> bool:
        """Check if content extraction was successful."""
        # Minimum content threshold
        return len(content) > 500

    def _looks_paywalled(self, url: str) -> bool:
        """Heuristic check for paywalled domains."""
        paywalled = [
            "wsj.com", "nytimes.com", "ft.com", "economist.com",
            "bloomberg.com", "washingtonpost.com", "theathletic.com"
        ]
        return any(domain in url for domain in paywalled)
```

### 6. Database (`backend/database.py`)

```python
"""
Database - SQLite operations for articles and feeds.

Uses raw SQLite for simplicity (no ORM).
Includes FTS5 for full-text search.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
import json

@dataclass
class DBArticle:
    id: int
    feed_id: int
    url: str
    title: str
    content: str | None
    summary_short: str | None
    summary_full: str | None
    key_points: list[str] | None
    is_read: bool
    is_bookmarked: bool
    published_at: datetime | None
    created_at: datetime

@dataclass
class DBFeed:
    id: int
    url: str
    name: str
    category: str | None
    last_fetched: datetime | None
    unread_count: int = 0

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT,
                    last_fetched TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary_short TEXT,
                    summary_full TEXT,
                    key_points TEXT,
                    is_read BOOLEAN DEFAULT FALSE,
                    is_bookmarked BOOLEAN DEFAULT FALSE,
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                    title, content, summary_full,
                    content='articles', content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
                    INSERT INTO articles_fts(rowid, title, content, summary_full)
                    VALUES (new.id, new.title, new.content, new.summary_full);
                END;

                CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
                    INSERT INTO articles_fts(articles_fts, rowid, title, content, summary_full)
                    VALUES ('delete', old.id, old.title, old.content, old.summary_full);
                    INSERT INTO articles_fts(rowid, title, content, summary_full)
                    VALUES (new.id, new.title, new.content, new.summary_full);
                END;

                CREATE INDEX IF NOT EXISTS idx_articles_feed ON articles(feed_id);
                CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
            """)

    # Feed operations

    def add_feed(self, url: str, name: str, category: str | None = None) -> int:
        """Add a new feed. Returns feed ID."""
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO feeds (url, name, category) VALUES (?, ?, ?)",
                (url, name, category)
            )
            return cursor.lastrowid

    def get_feeds(self) -> list[DBFeed]:
        """Get all feeds with unread counts."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT f.*, COUNT(CASE WHEN a.is_read = 0 THEN 1 END) as unread_count
                FROM feeds f
                LEFT JOIN articles a ON f.id = a.feed_id
                GROUP BY f.id
                ORDER BY f.name
            """).fetchall()
            return [self._row_to_feed(row) for row in rows]

    def delete_feed(self, feed_id: int):
        """Delete feed and its articles."""
        with self._conn() as conn:
            conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))

    # Article operations

    def add_article(
        self,
        feed_id: int,
        url: str,
        title: str,
        content: str | None = None,
        published_at: datetime | None = None
    ) -> int:
        """Add a new article. Returns article ID."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO articles
                   (feed_id, url, title, content, published_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (feed_id, url, title, content, published_at)
            )
            return cursor.lastrowid

    def get_articles(
        self,
        feed_id: int | None = None,
        unread_only: bool = False,
        bookmarked_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> list[DBArticle]:
        """Get articles with optional filters."""
        query = "SELECT * FROM articles WHERE 1=1"
        params = []

        if feed_id:
            query += " AND feed_id = ?"
            params.append(feed_id)
        if unread_only:
            query += " AND is_read = 0"
        if bookmarked_only:
            query += " AND is_bookmarked = 1"

        query += " ORDER BY published_at DESC NULLS LAST LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_article(row) for row in rows]

    def get_article(self, article_id: int) -> DBArticle | None:
        """Get single article by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM articles WHERE id = ?", (article_id,)
            ).fetchone()
            return self._row_to_article(row) if row else None

    def update_summary(
        self,
        article_id: int,
        summary_short: str,
        summary_full: str,
        key_points: list[str]
    ):
        """Update article summary."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE articles SET
                   summary_short = ?, summary_full = ?, key_points = ?
                   WHERE id = ?""",
                (summary_short, summary_full, json.dumps(key_points), article_id)
            )

    def mark_read(self, article_id: int, is_read: bool = True):
        """Mark article as read/unread."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE articles SET is_read = ? WHERE id = ?",
                (is_read, article_id)
            )

    def toggle_bookmark(self, article_id: int) -> bool:
        """Toggle bookmark status. Returns new status."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT is_bookmarked FROM articles WHERE id = ?", (article_id,)
            ).fetchone()
            new_status = not row["is_bookmarked"]
            conn.execute(
                "UPDATE articles SET is_bookmarked = ? WHERE id = ?",
                (new_status, article_id)
            )
            return new_status

    def search(self, query: str, limit: int = 20) -> list[DBArticle]:
        """Full-text search across articles."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT a.* FROM articles a
                JOIN articles_fts fts ON a.id = fts.rowid
                WHERE articles_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [self._row_to_article(row) for row in rows]

    # Helpers

    def _row_to_article(self, row: sqlite3.Row) -> DBArticle:
        key_points = None
        if row["key_points"]:
            try:
                key_points = json.loads(row["key_points"])
            except json.JSONDecodeError:
                pass

        return DBArticle(
            id=row["id"],
            feed_id=row["feed_id"],
            url=row["url"],
            title=row["title"],
            content=row["content"],
            summary_short=row["summary_short"],
            summary_full=row["summary_full"],
            key_points=key_points,
            is_read=bool(row["is_read"]),
            is_bookmarked=bool(row["is_bookmarked"]),
            published_at=row["published_at"],
            created_at=row["created_at"]
        )

    def _row_to_feed(self, row: sqlite3.Row) -> DBFeed:
        return DBFeed(
            id=row["id"],
            url=row["url"],
            name=row["name"],
            category=row["category"],
            last_fetched=row["last_fetched"],
            unread_count=row.get("unread_count", 0)
        )
```

---

## Swift API Client and Server Management

### API Client (`app/RSSReader/Services/APIClient.swift`)

```swift
import Foundation

/// Errors from API calls
enum APIError: Error, LocalizedError {
    case networkError(Error)
    case invalidResponse
    case serverError(Int, String)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .networkError(let error): return "Network error: \(error.localizedDescription)"
        case .invalidResponse: return "Invalid response from server"
        case .serverError(let code, let message): return "Server error \(code): \(message)"
        case .decodingError(let error): return "Decoding error: \(error.localizedDescription)"
        }
    }
}

/// Client for communicating with Python backend
actor APIClient {
    private let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder

    init(baseURL: URL = URL(string: "http://127.0.0.1:5005")!) {
        self.baseURL = baseURL
        self.session = URLSession.shared

        self.decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
    }

    // MARK: - Articles

    func getArticles(
        feedId: Int? = nil,
        unreadOnly: Bool = false,
        limit: Int = 50
    ) async throws -> [Article] {
        var components = URLComponents(url: baseURL.appendingPathComponent("/articles"), resolvingAgainstBaseURL: true)!
        var queryItems: [URLQueryItem] = []

        if let feedId = feedId {
            queryItems.append(URLQueryItem(name: "feed_id", value: String(feedId)))
        }
        if unreadOnly {
            queryItems.append(URLQueryItem(name: "unread_only", value: "true"))
        }
        queryItems.append(URLQueryItem(name: "limit", value: String(limit)))

        components.queryItems = queryItems

        return try await get(components.url!)
    }

    func getArticle(id: Int) async throws -> ArticleDetail {
        return try await get(baseURL.appendingPathComponent("/articles/\(id)"))
    }

    func markRead(articleId: Int) async throws {
        try await post(baseURL.appendingPathComponent("/articles/\(articleId)/read"), body: Empty())
    }

    func toggleBookmark(articleId: Int) async throws {
        try await post(baseURL.appendingPathComponent("/articles/\(articleId)/bookmark"), body: Empty())
    }

    // MARK: - Feeds

    func getFeeds() async throws -> [Feed] {
        return try await get(baseURL.appendingPathComponent("/feeds"))
    }

    func addFeed(url: String, name: String? = nil) async throws -> Feed {
        struct AddFeedRequest: Encodable {
            let url: String
            let name: String?
        }
        return try await post(
            baseURL.appendingPathComponent("/feeds"),
            body: AddFeedRequest(url: url, name: name)
        )
    }

    func deleteFeed(id: Int) async throws {
        try await delete(baseURL.appendingPathComponent("/feeds/\(id)"))
    }

    func refreshFeeds() async throws {
        try await post(baseURL.appendingPathComponent("/feeds/refresh"), body: Empty())
    }

    // MARK: - Search

    func search(query: String, limit: Int = 20) async throws -> [Article] {
        var components = URLComponents(url: baseURL.appendingPathComponent("/search"), resolvingAgainstBaseURL: true)!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "limit", value: String(limit))
        ]
        return try await get(components.url!)
    }

    // MARK: - Health

    func healthCheck() async throws -> Bool {
        struct Status: Decodable {
            let status: String
        }
        let result: Status = try await get(baseURL.appendingPathComponent("/status"))
        return result.status == "ok"
    }

    // MARK: - HTTP Methods

    private func get<T: Decodable>(_ url: URL) async throws -> T {
        let (data, response) = try await session.data(from: url)
        try validateResponse(response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func post<T: Decodable, B: Encodable>(_ url: URL, body: B) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func post<B: Encodable>(_ url: URL, body: B) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response, data: data)
    }

    private func delete(_ url: URL) async throws {
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"

        let (data, response) = try await session.data(for: request)
        try validateResponse(response, data: data)
    }

    private func validateResponse(_ response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw APIError.serverError(httpResponse.statusCode, message)
        }
    }
}

private struct Empty: Encodable {}
```

### Python Server Manager (`app/RSSReader/Services/PythonServer.swift`)

```swift
import Foundation

/// Manages the Python backend process lifecycle
@MainActor
class PythonServer: ObservableObject {
    @Published var isRunning = false
    @Published var error: String?

    private var process: Process?
    private let port: Int
    private let apiClient: APIClient

    init(port: Int = 5005) {
        self.port = port
        self.apiClient = APIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
    }

    func start() async {
        guard !isRunning else { return }

        // Find Python and project paths
        guard let projectPath = findProjectPath() else {
            error = "Could not find project directory"
            return
        }

        let venvPython = projectPath.appendingPathComponent("rss_venv/bin/python")
        let serverScript = projectPath.appendingPathComponent("backend/server.py")

        guard FileManager.default.fileExists(atPath: venvPython.path) else {
            error = "Python virtual environment not found at \(venvPython.path)"
            return
        }

        guard FileManager.default.fileExists(atPath: serverScript.path) else {
            error = "Server script not found at \(serverScript.path)"
            return
        }

        // Start the process
        let process = Process()
        process.executableURL = venvPython
        process.arguments = [
            "-m", "uvicorn",
            "server:app",
            "--host", "127.0.0.1",
            "--port", String(port)
        ]
        process.currentDirectoryURL = projectPath.appendingPathComponent("backend")

        // Set up environment
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env

        // Capture output
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                print("📝 Python: \(output)")
            }
        }

        do {
            try process.run()
            self.process = process

            // Wait for server to be ready
            try await waitForServer()
            isRunning = true
            error = nil

        } catch {
            self.error = "Failed to start server: \(error.localizedDescription)"
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        isRunning = false
    }

    private func waitForServer(timeout: TimeInterval = 30) async throws {
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            do {
                if try await apiClient.healthCheck() {
                    return
                }
            } catch {
                // Server not ready yet
            }
            try await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds
        }

        throw NSError(domain: "PythonServer", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Server did not start within \(timeout) seconds"
        ])
    }

    private func findProjectPath() -> URL? {
        // When running from Xcode, look relative to bundle
        if let bundlePath = Bundle.main.resourceURL?
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent() {

            // Check if this looks like our project
            let serverPath = bundlePath.appendingPathComponent("backend/server.py")
            if FileManager.default.fileExists(atPath: serverPath.path) {
                return bundlePath
            }
        }

        // Fallback: look in common locations
        let possiblePaths = [
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Workspace/rss-reader"),
            URL(fileURLWithPath: "/Users/\(NSUserName())/Workspace/rss-reader")
        ]

        for path in possiblePaths {
            let serverPath = path.appendingPathComponent("backend/server.py")
            if FileManager.default.fileExists(atPath: serverPath.path) {
                return path
            }
        }

        return nil
    }

    deinit {
        process?.terminate()
    }
}
```

---

## Configuration Files

### Requirements (`backend/requirements.txt`)

```
# Core
fastapi>=0.100.0
uvicorn>=0.23.0
python-dotenv>=0.19.0
pydantic>=2.0.0

# HTTP & Parsing
aiohttp>=3.8.0
feedparser>=6.0.0
beautifulsoup4>=4.11.0
lxml>=4.9.0

# AI
anthropic>=0.50.0,<0.51.0

# Database
# (using built-in sqlite3)

# Advanced features (optional)
playwright>=1.30.0          # For JS rendering
chromadb>=0.4.0             # For semantic cache
sentence-transformers>=2.2.0  # For embeddings

# Development
pytest>=7.0.0
```

### Makefile

```makefile
.PHONY: setup run test clean

# Setup development environment
setup:
	python3.11 -m venv rss_venv
	./rss_venv/bin/pip install -r backend/requirements.txt
	./rss_venv/bin/python -m playwright install chromium
	@echo "✅ Setup complete. Run 'make run' to start the server."

# Run development server
run:
	./rss_venv/bin/python -m uvicorn backend.server:app --reload --port 5005

# Run tests
test:
	./rss_venv/bin/python -m pytest backend/tests/ -v

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf backend/__pycache__ backend/**/__pycache__
	rm -rf data/cache/*

# Initialize database
init-db:
	./rss_venv/bin/python -c "from backend.database import Database; Database('data/articles.db')"

# Full rebuild
rebuild: clean setup init-db
	@echo "✅ Rebuild complete."
```

### Environment Variables (`.env.example`)

```bash
# Required
ANTHROPIC_API_KEY=your-api-key-here

# Optional
PORT=5005
LOG_LEVEL=INFO
CACHE_DIR=./data/cache
DB_PATH=./data/articles.db

# Rate limiting
API_RPM_LIMIT=50

# Advanced features
ENABLE_JS_RENDERER=true
ENABLE_SEMANTIC_CACHE=true
ENABLE_ARCHIVE_SERVICES=true
```

---

## Complete Summarizer with Response Parsing

Here's the complete `_parse_response` method that was stubbed:

```python
# In backend/summarizer.py

def _parse_response(self, response, model: Model) -> Summary:
    """Parse Claude's response into structured Summary."""
    text = response.content[0].text

    # Default values
    one_liner = ""
    full_summary = ""
    key_points = []

    # Split response into sections
    lines = text.strip().split("\n")
    current_section = None
    current_content = []

    for line in lines:
        line_lower = line.lower().strip()

        # Detect section headers
        if "one-sentence" in line_lower or "summary:" in line_lower[:15]:
            if current_section == "full":
                full_summary = "\n".join(current_content).strip()
            current_section = "one_liner"
            current_content = []
        elif "full summary" in line_lower or "detailed" in line_lower:
            if current_section == "one_liner":
                one_liner = " ".join(current_content).strip()
            current_section = "full"
            current_content = []
        elif "key point" in line_lower or "bullet" in line_lower:
            if current_section == "full":
                full_summary = "\n".join(current_content).strip()
            current_section = "points"
            current_content = []
        elif current_section == "points" and line.strip().startswith(("•", "-", "*", "1", "2", "3", "4", "5")):
            # Extract bullet point
            point = line.strip().lstrip("•-*0123456789.").strip()
            if point:
                key_points.append(point)
        elif current_section:
            current_content.append(line)

    # Handle final section
    if current_section == "one_liner" and not one_liner:
        one_liner = " ".join(current_content).strip()
    elif current_section == "full" and not full_summary:
        full_summary = "\n".join(current_content).strip()

    # Fallback: if parsing failed, use entire response
    if not full_summary:
        full_summary = text
    if not one_liner:
        # Take first sentence
        one_liner = text.split(".")[0] + "." if "." in text else text[:150]

    return Summary(
        title="",  # Will be set by caller from article
        one_liner=one_liner[:200],  # Enforce length limit
        full_summary=full_summary,
        key_points=key_points[:5],  # Max 5 points
        model_used=model
    )
```

---

## Summary

This redesign prioritizes:

1. **Simplicity** - 2,500 lines of Python instead of 15,000+
2. **Native UX** - True Mac app, not a web wrapper
3. **Summaries first** - The AI value prop is visible immediately
4. **Progressive complexity** - Power features available but not required
5. **Maintainability** - Clear structure, minimal dependencies

The advanced features (semantic caching, JS rendering, archive services) are preserved but isolated in their own module, documented clearly, and optional.

The goal is an app that feels like Apple made it, powered by Claude.
