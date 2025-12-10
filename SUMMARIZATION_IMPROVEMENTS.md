# Summarization & Clustering Improvements Proposal

This document analyzes the current backend summarization and clustering logic, identifies strengths and limitations, and proposes improvements across multiple dimensions.

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Strengths of Current Approach](#strengths-of-current-approach)
3. [Limitations & Pain Points](#limitations--pain-points)
4. [Proposed Improvements](#proposed-improvements)
   - [Tier 1: Quick Wins](#tier-1-quick-wins)
   - [Tier 2: Medium-Term Enhancements](#tier-2-medium-term-enhancements)
   - [Tier 3: Architectural Changes](#tier-3-architectural-changes)
5. [Implementation Recommendations](#implementation-recommendations)

---

## Current Architecture Overview

### Data Flow

```
RSS Feeds → EnhancedRSSReader → FastSummarizer → Clustering → UI
                                      ↓
                          Model Selection (Complexity)
                                      ↓
                          Sonnet 4.5 (≥0.6) / Haiku 4.5 (<0.6)
```

### Component Responsibilities

| Component | File | Purpose |
|-----------|------|---------|
| **BaseSummarizer** | `summarization/base.py` | Core API calls, retry logic, error handling |
| **ArticleSummarizer** | `summarization/article_summarizer.py` | Inherits base, adds memory cache |
| **FastSummarizer** | `summarization/fast_summarizer.py` | Batch processing, tiered cache, model selection |
| **SimpleClustering** | `clustering/simple.py` | Lightweight 60% semantic + 40% keyword hybrid |
| **ArticleClusterer** | `clustering/base.py` | UMAP + HDBSCAN with fallbacks |
| **EnhancedArticleClusterer** | `clustering/enhanced.py` | LM-based refinement and coherence checking |

### Model Selection Logic

Content complexity is estimated using:
- Word count (30% weight)
- Average word length (25% weight)
- Average sentence length (25% weight)
- Technical term density (20% weight)

Articles scoring ≥0.6 use Claude Sonnet 4.5; below 0.6 use Haiku 4.5.

---

## Strengths of Current Approach

### 1. Modular Architecture
- Clean separation between summarization and clustering
- Multiple clustering backends allow choosing speed vs. accuracy tradeoffs
- Pluggable model selection supports future model additions

### 2. Cost Optimization
- Automatic model routing saves costs on simple content
- Tiered caching (memory → disk) prevents redundant API calls
- Batch processing reduces per-article overhead

### 3. Robustness
- Multiple fallback clustering algorithms (HDBSCAN → Agglomerative → keyword-only)
- Graceful degradation when embeddings unavailable
- Exponential backoff retry logic for API failures

### 4. Performance Features
- Rate limiting prevents API throttling (50 RPM default)
- Async support for non-blocking operations
- Concurrent batch processing (3 workers default)

### 5. Flexibility
- Multiple summary styles (default, bullet, newswire)
- Long article chunking with meta-summarization
- User-configurable clustering parameters

---

## Limitations & Pain Points

### Summarization Issues

| Issue | Impact | Root Cause |
|-------|--------|------------|
| **Cache fragility** | Low hit rate on similar articles | Exact text hash required; whitespace differences cause misses |
| **Long article overhead** | N+1 API calls for >12KB articles | Sequential chunk summarization before meta-summary |
| **No summary refinement** | Can't improve poor summaries | One-shot generation with no feedback loop |
| **Fixed temperature** | No creativity control | Hardcoded 0.3 temperature for all summaries |
| **Headline extraction fragility** | Summaries sometimes miss key info | Relies on first-line parsing, which can fail |

### Clustering Issues

| Issue | Impact | Root Cause |
|-------|--------|------------|
| **Headline false clustering** | Unrelated articles grouped together | Common terms ("AI", "ChatGPT") appear across unrelated stories |
| **Flat cluster structure** | No topic hierarchy | Can't show sub-topics within clusters |
| **O(N²) complexity** | Slow for large feeds | Full pairwise similarity matrix in SimpleClustering |
| **Static thresholds** | Clusters too big or too small | Single distance_threshold for all content types |
| **7-day recency filter** | Old content never re-clustered | `_filter_recent_articles()` excludes older content |
| **No cross-cluster relationships** | Related clusters not linked | Each cluster is independent |

### Integration Issues

| Issue | Impact | Root Cause |
|-------|--------|------------|
| **Cold start latency** | First request takes 10-30s | Models lazy-loaded on first use |
| **Memory growth** | Unbounded cache in multi-user | No per-user memory limits |
| **Batch error handling** | Entire batch fails on one error | No partial success return |
| **Cache stampede** | Multiple requests duplicate work | No request coalescing on cache miss |

---

## Proposed Improvements

### Tier 1: Quick Wins

These improvements can be implemented with minimal architectural changes.

#### 1.1 Semantic Cache Keys

**Problem:** Current cache uses exact text hash, causing misses on near-identical content.

**Solution:** Add semantic fingerprinting alongside exact hash.

```python
# Current approach
cache_key = hashlib.md5(f"{text}:{model}:{temp}:{style}".encode()).hexdigest()

# Proposed approach
def generate_cache_key(text: str, model: str, style: str) -> tuple[str, str]:
    """Generate both exact and semantic cache keys."""
    exact_key = hashlib.md5(f"{text}:{model}:{style}".encode()).hexdigest()

    # Semantic key: normalized text + title keywords
    normalized = normalize_whitespace(text.lower())
    title_keywords = extract_top_keywords(text[:500], n=5)
    semantic_key = hashlib.md5(f"{normalized[:2000]}:{title_keywords}".encode()).hexdigest()

    return exact_key, semantic_key
```

**Expected Impact:** 15-25% improvement in cache hit rate.

---

#### 1.2 Parallel Chunk Summarization

**Problem:** Long articles (>12KB) are summarized sequentially, causing latency.

**Solution:** Parallelize chunk summarization with asyncio.gather().

```python
async def summarize_long_article(self, text: str, style: str) -> dict:
    chunks = self._split_into_chunks(text, max_size=8000)

    # Current: sequential
    # for chunk in chunks:
    #     summaries.append(await self.summarize_chunk(chunk))

    # Proposed: parallel with semaphore
    semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls

    async def summarize_with_limit(chunk):
        async with semaphore:
            return await self.summarize_chunk(chunk)

    chunk_summaries = await asyncio.gather(
        *[summarize_with_limit(c) for c in chunks]
    )

    return await self.create_meta_summary(chunk_summaries, style)
```

**Expected Impact:** 60-70% reduction in long article summarization time.

---

#### 1.3 Enhanced Common Entity Filtering

**Problem:** Headlines with "AI", "ChatGPT", "OpenAI" cluster together regardless of actual topic.

**Solution:** Expand the common entity filter and apply title deweighting.

```python
# Current filter (clustering/base.py)
COMMON_ENTITIES = {'AI', 'ChatGPT', 'GPT', 'OpenAI', 'Google', 'Microsoft'}

# Proposed expanded filter
COMMON_ENTITIES = {
    # AI/ML terms
    'AI', 'ChatGPT', 'GPT', 'GPT-4', 'GPT-5', 'OpenAI', 'Anthropic', 'Claude',
    'Gemini', 'LLM', 'GenAI', 'AGI', 'machine learning', 'deep learning',

    # Companies
    'Google', 'Microsoft', 'Meta', 'Amazon', 'Apple', 'NVIDIA', 'Tesla',
    'DeepMind', 'Hugging Face', 'Mistral', 'Cohere',

    # Generic tech terms
    'startup', 'funding', 'billion', 'million', 'CEO', 'launch', 'announce',
    'update', 'release', 'new', 'report', 'says', 'could', 'will'
}

# Title deweighting in similarity calculation
def calculate_similarity(article1, article2):
    title_sim = cosine_similarity(embed(article1.title), embed(article2.title))
    content_sim = cosine_similarity(embed(article1.content), embed(article2.content))

    # Current: equal weight
    # return 0.5 * title_sim + 0.5 * content_sim

    # Proposed: content-heavy weighting
    return 0.2 * title_sim + 0.8 * content_sim
```

**Expected Impact:** 40-50% reduction in false-positive clusters.

---

#### 1.4 Partial Batch Success

**Problem:** One failed article fails the entire batch.

**Solution:** Wrap individual summarizations and return partial results.

```python
async def batch_summarize(self, articles: list[dict]) -> list[dict]:
    results = []
    errors = []

    for article in articles:
        try:
            summary = await self.summarize(article)
            results.append({
                'article': article,
                'summary': summary,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'article': article,
                'summary': None,
                'status': 'error',
                'error': str(e)
            })
            errors.append(e)

    if errors:
        logger.warning(f"Batch completed with {len(errors)} errors")

    return results
```

**Expected Impact:** Improved reliability; users see available summaries instead of complete failure.

---

### Tier 2: Medium-Term Enhancements

These require more significant implementation effort but offer substantial improvements.

#### 2.1 Hierarchical Clustering

**Problem:** Large clusters (10+ articles) are hard to navigate; no sub-topic organization.

**Solution:** Implement two-level clustering with parent topics and sub-clusters.

```python
class HierarchicalClusterer:
    def cluster_articles(self, articles: list) -> list[ClusterGroup]:
        # Level 1: Broad topic clustering (low threshold)
        broad_clusters = self.base_cluster(articles, threshold=0.25)

        hierarchical_results = []
        for cluster in broad_clusters:
            if len(cluster) > 5:
                # Level 2: Sub-topic clustering (higher threshold)
                sub_clusters = self.base_cluster(cluster, threshold=0.45)
                hierarchical_results.append(ClusterGroup(
                    topic=self.extract_broad_topic(cluster),
                    sub_clusters=[
                        SubCluster(
                            topic=self.extract_specific_topic(sc),
                            articles=sc
                        )
                        for sc in sub_clusters
                    ]
                ))
            else:
                hierarchical_results.append(ClusterGroup(
                    topic=self.extract_topic(cluster),
                    sub_clusters=[SubCluster(topic=None, articles=cluster)]
                ))

        return hierarchical_results
```

**UI Representation:**
```
▼ Artificial Intelligence Regulation (12 articles)
    ▼ EU AI Act Implementation (5 articles)
        - Article 1...
        - Article 2...
    ▼ US Executive Orders (4 articles)
        - Article 3...
    ▼ China AI Governance (3 articles)
        - Article 4...
```

**Expected Impact:** Improved navigation for large topic clusters; better content discovery.

---

#### 2.2 Adaptive Model Selection

**Problem:** Fixed complexity threshold (0.6) doesn't account for content type or user preferences.

**Solution:** Multi-factor model selection with learning.

```python
class AdaptiveModelSelector:
    def __init__(self):
        self.selection_history = []

    def select_model(self, article: dict, user_prefs: dict = None) -> str:
        factors = {
            'complexity': self.estimate_complexity(article['text']),
            'length': len(article['text']) / 10000,  # Normalize to 0-1
            'technical_domain': self.detect_domain(article['text']),
            'source_quality': self.get_source_tier(article.get('url', '')),
            'user_preference': user_prefs.get('model_preference', 'auto') if user_prefs else 'auto'
        }

        # Weighted scoring
        score = (
            factors['complexity'] * 0.35 +
            factors['length'] * 0.15 +
            factors['technical_domain'] * 0.25 +
            factors['source_quality'] * 0.15 +
            (0.1 if factors['user_preference'] == 'quality' else 0)
        )

        if score >= 0.55:
            return 'claude-sonnet-4-5-latest'
        else:
            return 'claude-haiku-4-5-latest'

    def detect_domain(self, text: str) -> float:
        """Return 0-1 score for technical domain detection."""
        domains = {
            'research': ['arxiv', 'paper', 'study', 'findings', 'methodology'],
            'finance': ['earnings', 'revenue', 'valuation', 'IPO', 'quarterly'],
            'policy': ['regulation', 'legislation', 'compliance', 'mandate'],
            'product': ['launch', 'feature', 'update', 'available', 'users']
        }

        text_lower = text.lower()
        domain_scores = {}
        for domain, keywords in domains.items():
            domain_scores[domain] = sum(1 for k in keywords if k in text_lower) / len(keywords)

        # Research and policy benefit more from Sonnet
        return max(domain_scores.get('research', 0), domain_scores.get('policy', 0))
```

**Expected Impact:** Better model utilization; reduced costs on simple content; improved quality on complex content.

---

#### 2.3 Summary Quality Feedback Loop

**Problem:** No mechanism to improve summaries based on user feedback.

**Solution:** Implement regeneration with feedback and optional fine-tuning signals.

```python
class SummaryRefiner:
    def __init__(self, summarizer: FastSummarizer):
        self.summarizer = summarizer
        self.feedback_store = FeedbackStore()

    async def regenerate_with_feedback(
        self,
        article: dict,
        original_summary: str,
        feedback: str,
        feedback_type: str  # 'too_long', 'too_short', 'missed_key_point', 'inaccurate'
    ) -> dict:

        refinement_prompts = {
            'too_long': "Provide a more concise summary, focusing only on the key point.",
            'too_short': "Expand the summary to include more context and details.",
            'missed_key_point': f"The summary missed this key point: {feedback}. Regenerate including it.",
            'inaccurate': f"This part was inaccurate: {feedback}. Correct and regenerate."
        }

        enhanced_prompt = f"""
Original article: {article['text'][:8000]}

Previous summary: {original_summary}

Feedback: {refinement_prompts[feedback_type]}

Generate an improved summary addressing the feedback.
"""

        refined = await self.summarizer.summarize({
            'text': enhanced_prompt,
            'title': article.get('title', '')
        }, force_model='claude-sonnet-4-5-latest')  # Use better model for refinement

        # Store feedback for potential fine-tuning
        self.feedback_store.record(
            article_id=article.get('url'),
            original=original_summary,
            refined=refined,
            feedback_type=feedback_type,
            user_feedback=feedback
        )

        return refined
```

**Expected Impact:** User-driven quality improvement; data collection for future model fine-tuning.

---

#### 2.4 Request Coalescing

**Problem:** Multiple simultaneous requests for the same article cause duplicate API calls.

**Solution:** Implement request coalescing with async locks.

```python
class CoalescingSummarizer:
    def __init__(self, base_summarizer: FastSummarizer):
        self.base = base_summarizer
        self.in_flight: dict[str, asyncio.Future] = {}
        self.lock = asyncio.Lock()

    async def summarize(self, article: dict) -> dict:
        cache_key = self._get_cache_key(article)

        async with self.lock:
            # Check if request already in flight
            if cache_key in self.in_flight:
                # Wait for existing request
                return await self.in_flight[cache_key]

            # Create new future for this request
            future = asyncio.Future()
            self.in_flight[cache_key] = future

        try:
            # Perform actual summarization
            result = await self.base.summarize(article)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self.lock:
                del self.in_flight[cache_key]
```

**Expected Impact:** Eliminates duplicate API calls during high-concurrency scenarios (e.g., page refresh spam).

---

### Tier 3: Architectural Changes

These are larger changes that would significantly enhance capabilities but require careful planning.

#### 3.1 Vector Database Integration

**Problem:** Current caching is exact-match only; similar articles can't share summaries.

**Solution:** Replace TieredCache with vector database for semantic similarity search.

```python
# Using Chroma as example (lightweight, runs locally)
import chromadb

class SemanticSummaryCache:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./vector_cache")
        self.collection = self.client.get_or_create_collection(
            name="summaries",
            metadata={"hnsw:space": "cosine"}
        )

    async def get_or_create_summary(
        self,
        article: dict,
        summarizer: FastSummarizer,
        similarity_threshold: float = 0.92
    ) -> dict:
        # Generate embedding for article
        article_embedding = await self.embed(article['text'][:2000])

        # Search for similar cached summaries
        results = self.collection.query(
            query_embeddings=[article_embedding],
            n_results=1,
            include=['documents', 'metadatas', 'distances']
        )

        if results['distances'][0] and results['distances'][0][0] < (1 - similarity_threshold):
            # Found similar article with cached summary
            cached = results['metadatas'][0][0]
            return {
                'summary': cached['summary'],
                'headline': cached['headline'],
                'cache_hit': True,
                'similarity': 1 - results['distances'][0][0]
            }

        # No similar article found; generate new summary
        summary = await summarizer.summarize(article)

        # Store in vector database
        self.collection.add(
            embeddings=[article_embedding],
            documents=[article['text'][:2000]],
            metadatas=[{
                'summary': summary['summary'],
                'headline': summary['headline'],
                'url': article.get('url', ''),
                'created_at': datetime.now().isoformat()
            }],
            ids=[self._generate_id(article)]
        )

        return {**summary, 'cache_hit': False}
```

**Benefits:**
- Semantic cache hits for similar articles
- Cross-article knowledge reuse
- Foundation for "related articles" feature
- Enables summary search functionality

**Expected Impact:** 30-40% additional cache hits; new feature possibilities.

---

#### 3.2 Streaming Summary Delivery

**Problem:** Users wait for entire batch to complete before seeing any results.

**Solution:** WebSocket-based streaming delivery as summaries complete.

```python
# Server-side (FastAPI)
from fastapi import WebSocket

@app.websocket("/ws/summarize")
async def websocket_summarize(websocket: WebSocket):
    await websocket.accept()

    data = await websocket.receive_json()
    articles = data['articles']

    async def stream_summaries():
        for i, article in enumerate(articles):
            try:
                summary = await fast_summarizer.summarize(article)
                await websocket.send_json({
                    'type': 'summary',
                    'index': i,
                    'article_url': article.get('url'),
                    'summary': summary,
                    'progress': (i + 1) / len(articles)
                })
            except Exception as e:
                await websocket.send_json({
                    'type': 'error',
                    'index': i,
                    'article_url': article.get('url'),
                    'error': str(e)
                })

        await websocket.send_json({'type': 'complete'})

    await stream_summaries()
    await websocket.close()

# Client-side (JavaScript)
const ws = new WebSocket('ws://localhost:5005/ws/summarize');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'summary') {
        // Immediately render this summary
        renderSummary(data.index, data.summary);
        updateProgressBar(data.progress);
    } else if (data.type === 'complete') {
        hideLoadingIndicator();
    }
};

ws.send(JSON.stringify({ articles: articleList }));
```

**Expected Impact:** Perceived latency reduction of 70%+; better user experience during batch processing.

---

#### 3.3 Multi-Stage Clustering Pipeline

**Problem:** Users must choose between fast (inaccurate) and slow (accurate) clustering.

**Solution:** Progressive clustering with background refinement.

```python
class ProgressiveClusterer:
    def __init__(self):
        self.simple = SimpleClustering()
        self.enhanced = EnhancedArticleClusterer()
        self.refinement_queue = asyncio.Queue()

    async def cluster_progressive(
        self,
        articles: list,
        callback: Callable[[list], None]
    ) -> list:
        # Stage 1: Fast clustering (immediate)
        quick_clusters = self.simple.cluster_articles(articles)
        callback(quick_clusters)  # Send to UI immediately

        # Stage 2: Background refinement
        asyncio.create_task(self._refine_clusters(quick_clusters, callback))

        return quick_clusters

    async def _refine_clusters(
        self,
        initial_clusters: list,
        callback: Callable[[list], None]
    ):
        """Background task to refine clusters."""
        refined_clusters = []

        for cluster in initial_clusters:
            if len(cluster) > 3:  # Only refine larger clusters
                # Check coherence
                coherence = await self.enhanced.check_cluster_coherence_async(cluster)

                if coherence < 0.7:
                    # Split incoherent cluster
                    sub_clusters = await self.enhanced.split_cluster_async(cluster)
                    refined_clusters.extend(sub_clusters)
                else:
                    refined_clusters.append(cluster)
            else:
                refined_clusters.append(cluster)

        # Send refined results to UI
        callback(refined_clusters)
```

**UI Flow:**
1. User requests feed → Instant display with quick clusters
2. Background refinement runs → UI updates with improved clusters
3. Visual indicator shows "Refining clusters..." → "Clusters optimized"

**Expected Impact:** Best of both worlds: instant results with eventual accuracy.

---

#### 3.4 Cross-Cluster Relationship Mapping

**Problem:** Related clusters aren't connected; users can't discover related topics.

**Solution:** Build a cluster relationship graph.

```python
class ClusterRelationshipMapper:
    def map_relationships(self, clusters: list[Cluster]) -> ClusterGraph:
        graph = ClusterGraph()

        # Add all clusters as nodes
        for cluster in clusters:
            graph.add_node(cluster.id, cluster)

        # Calculate pairwise cluster similarities
        for i, cluster_a in enumerate(clusters):
            centroid_a = self._get_cluster_centroid(cluster_a)

            for j, cluster_b in enumerate(clusters[i+1:], i+1):
                centroid_b = self._get_cluster_centroid(cluster_b)

                similarity = cosine_similarity(centroid_a, centroid_b)

                if similarity > 0.4:  # Related threshold
                    relationship_type = self._classify_relationship(
                        cluster_a, cluster_b, similarity
                    )
                    graph.add_edge(cluster_a.id, cluster_b.id, {
                        'similarity': similarity,
                        'type': relationship_type
                    })

        return graph

    def _classify_relationship(self, a: Cluster, b: Cluster, sim: float) -> str:
        """Classify the type of relationship between clusters."""
        if sim > 0.7:
            return 'same_story'  # Different coverage of same event
        elif sim > 0.55:
            return 'related_topic'  # Same broader topic
        else:
            return 'tangential'  # Loosely related
```

**UI Representation:**
```
┌─────────────────────────────────────────────────────────┐
│  AI Regulation                                          │
│  ├── Related: Tech Policy (0.62)                       │
│  └── Same Story: EU AI Act Vote (0.78)                 │
└─────────────────────────────────────────────────────────┘
```

**Expected Impact:** Improved content discovery; users can explore related topics easily.

---

## Implementation Recommendations

### Priority Matrix

| Improvement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| Semantic cache keys | Medium | Low | **P0** |
| Parallel chunk summarization | High | Low | **P0** |
| Enhanced entity filtering | High | Low | **P0** |
| Partial batch success | Medium | Low | **P0** |
| Hierarchical clustering | High | Medium | **P1** |
| Request coalescing | Medium | Medium | **P1** |
| Adaptive model selection | Medium | Medium | **P2** |
| Summary feedback loop | Medium | Medium | **P2** |
| Vector database | High | High | **P2** |
| Streaming delivery | High | High | **P3** |
| Progressive clustering | High | High | **P3** |
| Cross-cluster relationships | Medium | High | **P3** |

### Suggested Implementation Order

**Phase 1 (1-2 weeks):** Quick Wins
1. Enhanced entity filtering (immediate clustering improvement)
2. Parallel chunk summarization (performance win)
3. Partial batch success (reliability improvement)
4. Semantic cache keys (cache efficiency)

**Phase 2 (2-4 weeks):** Core Enhancements
5. Hierarchical clustering (UX improvement)
6. Request coalescing (scalability)
7. Adaptive model selection (cost optimization)

**Phase 3 (4-8 weeks):** Architectural Evolution
8. Vector database integration (foundation for future features)
9. Streaming summary delivery (UX transformation)
10. Progressive clustering pipeline (best-of-both-worlds)

### Migration Considerations

- **Backward Compatibility:** All Tier 1 changes are additive; no breaking changes
- **Database Migration:** Vector database (Tier 3) requires data migration strategy
- **API Changes:** Streaming delivery requires new WebSocket endpoints alongside existing REST
- **Testing:** Each phase should include comprehensive regression testing

---

## Conclusion

The current summarization and clustering system is well-architected but has room for improvement in three key areas:

1. **Accuracy:** False-positive clustering due to common entity terms
2. **Performance:** Sequential processing of long articles; no request coalescing
3. **User Experience:** No progressive loading; flat cluster structure

The proposed improvements address these issues progressively, from quick wins that can be implemented immediately to architectural changes that enable new capabilities. The recommended approach is to start with Phase 1 improvements, which offer high impact with low effort, while planning for the more substantial Phase 2 and 3 changes.
