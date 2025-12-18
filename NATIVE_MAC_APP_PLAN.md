# Plan: Full Native Swift Mac App - RSS Reader

## Executive Summary

This document provides a comprehensive roadmap for rewriting the RSS reader as a **fully native Swift/SwiftUI macOS application**. The goal is full feature parity with a smaller, faster, truly native app.

**User Goals**: Native look & feel, smaller app size, better performance, full feature parity
**Approach**: Complete Swift rewrite (no Python dependency)
**Timeline Context**: Exploratory phase - understanding scope before committing

---

## Current Architecture to Replace

| Layer | Current Tech | Lines of Code | Swift Replacement |
|-------|------------|---------------|-------------------|
| Web Server | FastAPI | ~1,700 | Not needed |
| Templates | Jinja2 | ~2,000 | SwiftUI Views |
| Frontend JS | Vanilla JS | ~1,500 | SwiftUI + AppKit |
| RSS Parsing | feedparser | ~2,000 | FeedKit |
| Summarization | Anthropic SDK | ~3,000 | URLSession + REST |
| Clustering | HDBSCAN/sklearn | ~4,000 | Accelerate/custom |
| Caching | Custom tiered | ~1,000 | NSCache + FileManager |
| Database | SQLAlchemy | ~500 | Core Data or GRDB |
| Paywall Bypass | Playwright | ~2,000 | WKWebView |
| Electron Shell | Node.js | ~2,500 | Native Cocoa |

**Total: ~23,000 lines Python/JS → ~15,000-20,000 lines Swift** (estimated)

---

## Why Full Swift Rewrite Makes Sense

### Benefits
1. **App Size**: ~50MB vs ~300MB+ (Electron) or ~500MB+ (embedded Python)
2. **Performance**: Native compilation, no interpreter overhead
3. **Memory**: ~50-100MB vs ~400MB+ for Electron
4. **Native Feel**: Real macOS citizen (menus, shortcuts, Spotlight, Handoff)
5. **Future-Proof**: Apple's preferred stack, long-term support
6. **App Store**: Full compatibility, sandboxing support

### Challenges to Acknowledge
1. **Development Time**: 6-12 months realistic for full feature parity
2. **Learning Curve**: SwiftUI patterns differ from web development
3. **Clustering**: No direct HDBSCAN equivalent - will need custom implementation
4. **Paywall Bypass**: WKWebView is more limited than Playwright

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Working app shell with navigation and data models

1. **Xcode Project Setup**
   - macOS App template with SwiftUI
   - Target: macOS 13+ (Ventura) for best SwiftUI support
   - Configure signing & capabilities

2. **Data Models** (Swift structs/classes)
   ```swift
   struct Article: Identifiable, Codable { ... }
   struct Cluster: Identifiable { ... }
   struct Bookmark: Identifiable { ... }
   struct Feed: Identifiable { ... }
   struct Summary: Codable { ... }
   ```

3. **Core UI Structure**
   ```
   MainContentView (NavigationSplitView - 3-pane)
   ├── Sidebar: ClusterListView
   ├── Content: ArticleListView
   └── Detail: ArticleDetailView
   ```

4. **Database Layer**
   - GRDB.swift (recommended) or Core Data
   - Schema matching current SQLite structure

**Deliverable**: App launches, shows empty UI, persists data

---

### Phase 2: RSS Feed Processing (Weeks 5-8)

**Goal**: Fetch, parse, and display RSS feeds

1. **FeedKit Integration**
   ```swift
   // Swift Package: https://github.com/nmdias/FeedKit
   import FeedKit

   func fetchFeed(url: URL) async throws -> [Article]
   ```

2. **Content Extraction**
   - Basic HTML parsing with SwiftSoup
   - Article content extraction logic
   - URL validation and normalization

3. **Feed Management UI**
   - Add/remove feeds
   - Feed categories
   - Refresh controls

4. **Background Refresh**
   - BackgroundTasks framework
   - Configurable refresh intervals

**Deliverable**: App fetches and displays RSS articles

---

### Phase 3: Claude API Integration (Weeks 9-12)

**Goal**: Article summarization via Anthropic API

1. **API Client**
   ```swift
   class AnthropicClient {
       func summarize(content: String, model: Model) async throws -> Summary
       func generateImagePrompt(content: String) async throws -> String
   }

   enum Model {
       case sonnet  // Complex content
       case haiku   // Simple content
   }
   ```

2. **Complexity Detection**
   - Port `models/config.py` logic to Swift
   - Text analysis for model selection

3. **Rate Limiting**
   - Token bucket implementation
   - Request queuing

4. **Caching Layer**
   ```swift
   class SummaryCache {
       private let memoryCache: NSCache<NSString, Summary>
       private let diskCache: FileManager // ~/Library/Caches/
   }
   ```

**Deliverable**: Articles can be summarized with caching

---

### Phase 4: Clustering (Weeks 13-18)

**Goal**: Group articles by topic

This is the most challenging component - no direct Swift HDBSCAN equivalent.

**Options**:

1. **Option A: Keyword-Based Clustering** (Simpler, recommended to start)
   - Port `clustering/simple.py` logic
   - TF-IDF with NaturalLanguage framework
   - Custom similarity scoring
   - ~60-70% as good as current system

2. **Option B: Embeddings + Custom Clustering** (Better results)
   - Use Apple's NaturalLanguage framework for embeddings
   - Implement simplified DBSCAN in Swift
   - Or use CreateML for on-device ML

3. **Option C: Server-Side Clustering** (Hybrid fallback)
   - Keep a small Python microservice for clustering only
   - Call it from Swift app
   - Adds complexity but preserves quality

**Recommendation**: Start with Option A, enhance later if needed

**Key Files to Port**:
- `clustering/simple.py` - keyword extraction, similarity scoring
- `clustering/base.py` - cluster data structures and interfaces

**Deliverable**: Articles grouped into topic clusters

---

### Phase 5: Advanced Features (Weeks 19-24)

**Goal**: Paywall bypass, PDF extraction, bookmarks, image prompts

1. **Paywall Bypass via WKWebView**
   ```swift
   class PaywallBypassService {
       func fetchWithJavaScript(url: URL) async throws -> String
       func tryArchiveServices(url: URL) async throws -> String
   }
   ```
   - WKWebView for JS-rendered content
   - Archive.is / Wayback Machine integration
   - Less capable than Playwright but covers most cases

2. **PDF Extraction**
   - PDFKit framework (built into macOS)
   - Extract text from PDF documents
   - Metadata extraction

3. **Bookmarks System**
   - Full CRUD operations
   - Tags and search
   - Import/export (JSON, CSV)

4. **Image Prompt Generation**
   - Port prompt templates from `services/image_prompt_generator.py`
   - Style selection UI

5. **Aggregator Link Resolution**
   - Port logic from `content/extractors/aggregator.py`
   - Handle Techmeme, Google News, Reddit links

**Deliverable**: Full feature parity with current app

---

### Phase 6: Native macOS Integration (Weeks 25-28)

**Goal**: Deep system integration that makes it feel like a first-class Mac app

1. **Menu Bar**
   - Full menu structure with SwiftUI Commands
   - Keyboard shortcuts (Cmd+R refresh, Cmd+S summarize, etc.)
   - Context menus on articles

2. **Spotlight Integration**
   - CSSearchableIndex for articles and bookmarks
   - Quick Look previews

3. **Notifications**
   - UserNotifications framework
   - New article alerts
   - Summary completion notifications

4. **Share Extension**
   - Share articles to other apps
   - Receive URLs from Safari, other apps

5. **Dock Badge**
   - Unread article count

6. **Handoff & iCloud** (optional, nice-to-have)
   - Continue reading on other devices
   - Sync bookmarks via CloudKit

**Deliverable**: Feels like a native macOS citizen

---

### Phase 7: Polish & Release (Weeks 29-32)

1. **UI Polish**
   - App icon and assets
   - Dark mode support (automatic with SwiftUI)
   - Animations and transitions
   - Accessibility (VoiceOver support)

2. **Testing**
   - Unit tests for business logic
   - UI tests for critical flows
   - Beta testing via TestFlight

3. **Distribution**
   - Code signing with Developer ID
   - Notarization for Gatekeeper
   - Mac App Store submission OR direct download DMG

---

## Swift Libraries Required

| Purpose | Library | Notes |
|---------|---------|-------|
| RSS Parsing | FeedKit | Swift Package |
| HTML Parsing | SwiftSoup | BeautifulSoup equivalent |
| Database | GRDB.swift | SQLite wrapper, or use Core Data |
| Networking | URLSession | Built-in, async/await |
| JSON | Codable | Built-in |
| NLP | NaturalLanguage | Built-in Apple framework |
| PDF | PDFKit | Built-in |
| WebView | WKWebView | Built-in, for JS-rendered content |

---

## Project Structure

```
RSSReader/
├── RSSReader.xcodeproj
├── RSSReader/
│   ├── App/
│   │   ├── RSSReaderApp.swift          # App entry point
│   │   └── AppDelegate.swift           # Lifecycle, menus
│   ├── Views/
│   │   ├── MainContentView.swift       # 3-pane layout
│   │   ├── Sidebar/
│   │   │   ├── ClusterListView.swift
│   │   │   └── FeedListView.swift
│   │   ├── Content/
│   │   │   ├── ArticleListView.swift
│   │   │   └── ArticleRowView.swift
│   │   ├── Detail/
│   │   │   ├── ArticleDetailView.swift
│   │   │   └── SummaryView.swift
│   │   ├── Bookmarks/
│   │   │   └── BookmarksView.swift
│   │   ├── Settings/
│   │   │   └── SettingsView.swift
│   │   └── Components/
│   │       ├── LoadingView.swift
│   │       └── ErrorView.swift
│   ├── Models/
│   │   ├── Article.swift
│   │   ├── Cluster.swift
│   │   ├── Bookmark.swift
│   │   ├── Feed.swift
│   │   └── Summary.swift
│   ├── Services/
│   │   ├── FeedService.swift           # RSS fetching
│   │   ├── AnthropicClient.swift       # Claude API
│   │   ├── ClusteringService.swift     # Article grouping
│   │   ├── CacheService.swift          # Tiered caching
│   │   ├── BookmarkService.swift       # Bookmark CRUD
│   │   ├── PaywallService.swift        # Archive services
│   │   └── ImagePromptService.swift    # AI image prompts
│   ├── Database/
│   │   ├── DatabaseManager.swift
│   │   └── Migrations.swift
│   ├── Utilities/
│   │   ├── HTMLParser.swift
│   │   ├── TextProcessor.swift
│   │   └── RateLimiter.swift
│   └── Resources/
│       ├── Assets.xcassets
│       └── DefaultFeeds.json
└── RSSReaderTests/
    └── ...
```

---

## Effort Estimate Summary

| Phase | Duration | Key Challenge |
|-------|----------|---------------|
| Foundation | 4 weeks | SwiftUI learning curve |
| RSS Processing | 4 weeks | Content extraction |
| Claude API | 4 weeks | Rate limiting, caching |
| Clustering | 6 weeks | Algorithm reimplementation |
| Advanced Features | 6 weeks | Paywall bypass limitations |
| Native Integration | 4 weeks | Spotlight, extensions |
| Polish & Release | 4 weeks | Testing, distribution |
| **Total** | **~32 weeks** | ~8 months part-time |

*At "side project pace" (few hours/week): 12-18 months*
*At dedicated effort: 6-9 months*
*Full-time sprint: 4-5 months*

---

## Getting Started: First Steps

If you decide to proceed, here's how to start:

1. **Create Xcode project**: File → New → Project → macOS → App (SwiftUI)

2. **Add first dependency**:
   - File → Add Package Dependencies
   - Add FeedKit: `https://github.com/nmdias/FeedKit`

3. **Build basic data model**: Start with `Article.swift`

4. **Create simple 3-pane UI**: Use `NavigationSplitView`

5. **Fetch one RSS feed**: Prove the architecture works

This gives you a working prototype in 1-2 weekends to validate the approach.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Clustering quality loss | Start with simpler algorithm, iterate |
| Paywall bypass limitations | Keep archive service fallbacks |
| SwiftUI learning curve | Start with simpler views, add complexity |
| Scope creep | Define MVP, defer nice-to-haves |
| Motivation over long project | Set milestones, celebrate progress |

---

## Decision: Proceed or Not?

This is a significant project. Before committing:

**Good reasons to proceed**:
- You want to learn Swift/SwiftUI deeply
- The native Mac experience is worth the investment
- You'll use this app daily for years
- App size/performance really matters to you

**Reasons to reconsider**:
- Current Electron app works well enough
- Limited time to dedicate
- Feature velocity is more important than native feel
- You might want cross-platform (iOS) later

The current Electron app is functional. A full Swift rewrite is a "want" not a "need" - make sure the investment aligns with your goals.
