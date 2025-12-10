# DOCTRINE.md Review & Recommendations

**Date**: 2025-12-09
**Project Stats**: 71 Python files, ~19,699 lines of code, 1.8MB cache

---

## Executive Summary

Based on the current project complexity (~20K lines, 71 files, established architecture), here are the **top 5 decisions worth revisiting** and **5 that are still solid**.

### 🔴 High Priority: Consider Revisiting

1. **Backward Compatibility Layers** - Should be removed (>6 months old)
2. **Testing Coverage** - Should be expanded before next major refactor
3. **Cache TTL** - Monitor if 30 days is optimal
4. **Model Selection Threshold** - Validate 0.6 complexity threshold
5. **SQLite Scalability** - Plan for potential user growth

### 🟢 Low Priority: Still Appropriate

1. **Lightweight Clustering** - Working well, no need to change
2. **Jinja2 Templates** - Still appropriate for this use case
3. **Dual-Model Strategy** - Cost/quality balance validated
4. **Python Bundling (Mac App)** - User experience benefit confirmed
5. **Progressive Enhancement** - Core philosophy still sound

---

## Detailed Analysis

### 🔴 1. Backward Compatibility Layers (HIGH PRIORITY)

**Current State**: `common/archive_compat.py`, `common/source_extractor_compat.py`

**DOCTRINE.md Says**:
> "When to Reconsider: If backward compatibility layers remain after 6+ months"

**Analysis**:
- Content refactor was done in May-June 2025
- We're now 6+ months past that refactor
- **Action Needed**: Remove compatibility layers

**Recommendation**:
```python
# Files to remove:
- common/archive_compat.py
- common/source_extractor_compat.py

# Update imports throughout codebase to use new modules:
from content.archive.providers import ArchiveProvider  # not common.archive_compat
from content.extractors.aggregator import AggregatorExtractor  # not common.source_extractor_compat
```

**Effort**: 2-3 hours
**Risk**: Low (compatibility layers are just forwarding imports)
**Benefit**: Cleaner codebase, removes technical debt

---

### 🔴 2. Testing Coverage (MEDIUM-HIGH PRIORITY)

**Current State**: Tests for critical paths (batch processing, model selection, Tier 3 features)

**DOCTRINE.md Says**:
> "When to Increase Testing: Before refactoring complex modules"

**Analysis**:
- Project has grown to ~20K lines
- Major refactors have happened (content/, Tier 3 features)
- Current test coverage unknown (no coverage reports mentioned)

**Recommendation**:

1. **Measure Current Coverage**:
```bash
pip install pytest-cov
python -m pytest tests/ --cov --cov-report=html
open htmlcov/index.html
```

2. **Target Areas for New Tests** (based on DOCTRINE coverage goals):
   - `cache/tiered_cache.py` - Target 80-90% (Critical)
   - `models/config.py` - Target 80-90% (Critical)
   - `summarization/` - Target 60-80% (Important)
   - `clustering/simple.py` - Target 60-80% (Important)
   - `content/extractors/` - Target 40-60% (Standard)

3. **Add Integration Tests**:
   - Full RSS feed → summarization → clustering pipeline
   - Multi-user cache isolation
   - Archive service fallbacks

**Effort**: 1-2 weeks for comprehensive coverage
**Risk**: None (only adds tests)
**Benefit**: Safer refactoring, catch bugs earlier, document expected behavior

---

### 🔴 3. Cache TTL and Performance (MEDIUM PRIORITY)

**Current State**: 30-day TTL, 1.8MB cache size

**DOCTRINE.md Says**:
> "When to Act:
> - Hit rate <50%: Cache TTL too aggressive
> - Hit rate >95%: Cache TTL too conservative, wasting disk space"

**Analysis**:
- Cache is only 1.8MB (very small)
- No monitoring data for cache hit rates
- Unknown if 30-day TTL is optimal

**Recommendation**:

1. **Add Cache Monitoring** (Priority 1):
```python
# Add to summarization code
import logging
logger = logging.getLogger(__name__)

cache_stats = cache.get_stats()
logger.info(f"Cache stats: {cache_stats}")
# Log: hit_rate, memory_hits, disk_hits, api_calls
```

2. **Track for 1 Week**, then analyze:
   - If hit rate <50%: Decrease TTL (15 days?)
   - If hit rate >95%: Increase TTL or reduce cache size limit
   - If cache >100MB: Consider eviction strategy

3. **Consider Semantic Cache**:
   - DOCTRINE mentions Tier 3 semantic cache
   - Check if `cache/semantic_cache.py` is actually being used
   - If not, consider enabling it for 30-40% additional cache hits

**Effort**: 2-4 hours for monitoring, 1 week for data collection
**Risk**: Low
**Benefit**: Optimized API costs, better performance

---

### 🔴 4. Model Selection Threshold (MEDIUM PRIORITY)

**Current State**: 0.6 complexity threshold (Haiku vs Sonnet)

**DOCTRINE.md Says**:
> "When to Act:
> - Haiku usage <30%: Threshold too high, wasting money on Sonnet
> - Haiku usage >70%: Threshold too low, may sacrifice quality
> - Quality complaints: Haiku may not be sufficient"

**Analysis**:
- Threshold set to 0.6 in commit `2448af4`
- No monitoring data on actual Haiku/Sonnet distribution
- No validation of classification accuracy

**Recommendation**:

1. **Add Model Selection Logging**:
```python
# In models/config.py or summarization code
logger.info(f"Selected model: {model}, complexity: {complexity_score:.2f}, url: {url}")
```

2. **Collect Data for 1 Week**:
   - Track Haiku vs Sonnet usage percentage
   - Track which articles get which model
   - Manually review 50 articles to validate classification

3. **Adjust Threshold if Needed**:
   - Target: 40-60% Haiku usage
   - If Haiku <30%: Lower threshold (try 0.5)
   - If Haiku >70%: Raise threshold (try 0.65)

**Effort**: 2-3 hours for logging, 1 week data collection, 2 hours analysis
**Risk**: Low (just monitoring)
**Benefit**: Optimized API costs, validate architectural assumption

---

### 🔴 5. SQLite Scalability (LOW-MEDIUM PRIORITY)

**Current State**: SQLite with per-user databases, multi-user support enabled

**DOCTRINE.md Says**:
> "Consider PostgreSQL if:
> - User count exceeds ~1000 active users
> - Need cross-user analytics or features
> - Complex query performance becomes an issue"

**Analysis**:
- Multi-user support implemented (commit `c54a985`)
- Per-user databases in `data/users/{user_id}/`
- Unknown actual user count
- SQLite generally good for <1000 users

**Recommendation**:

**If <10 users** (personal/small team use):
- ✅ Current approach is perfect
- No action needed

**If 10-100 users**:
- ✅ Current approach still fine
- Monitor query performance
- Consider adding indexes if slow

**If 100-1000 users**:
- ⚠️ Start planning PostgreSQL migration
- SQLite will start showing limitations
- Consider connection pooling

**If >1000 users**:
- 🔴 Migrate to PostgreSQL
- Per-user databases won't scale
- Need proper multi-tenant architecture

**Current Action**:
1. Track actual user count
2. If trending toward 100+ users, start PostgreSQL planning
3. SQLAlchemy makes migration relatively straightforward

**Effort**: N/A (just monitoring)
**Risk**: Low (SQLite is fine for current scale)
**Benefit**: Proactive planning

---

## 🟢 Decisions That Are Still Sound

### 1. Lightweight Clustering ✅

**Status**: **Keep as-is**

**Why It's Still Good**:
- 90% dependency reduction validated
- 10x faster startup confirmed
- User needs met (article grouping sufficient)
- No complaints about clustering quality

**DOCTRINE Check**:
> "When to Reconsider: If clustering quality becomes insufficient for user needs"

**Verdict**: Quality is sufficient. Enhanced clustering still available if needed.

---

### 2. Jinja2 Templates (Not SPA) ✅

**Status**: **Keep as-is**

**Why It's Still Good**:
- RSS reader is read-heavy (consumption, not heavy interaction)
- Small team (no frontend specialists needed)
- Fast development cycle
- Server-side rendering appropriate for content-focused app

**DOCTRINE Check**:
> "When to Reconsider:
> - If app becomes write-heavy (e.g., rich text editing)
> - If real-time collaboration features needed"

**Verdict**: Not write-heavy, no collaboration features needed. Templates working well.

**Only Reconsider If**: You add features like:
- Rich text article editing
- Real-time collaborative feeds
- Complex interactive dashboards
- Mobile app requiring API-first

---

### 3. Dual-Model Strategy (Sonnet + Haiku) ✅

**Status**: **Keep as-is** (with monitoring from #4 above)

**Why It's Still Good**:
- Cost optimization strategy sound
- Quality preservation for complex content
- Automatic selection removes manual decisions

**DOCTRINE Check**:
> "When to Reconsider:
> - If Haiku quality degrades for simple content
> - If model pricing changes significantly"

**Verdict**: Strategy sound, just needs validation through monitoring.

---

### 4. Python Bundling for Mac App ✅

**Status**: **Keep as-is**

**Why It's Still Good**:
- One-click installation UX critical for non-technical users
- 200-300MB size acceptable for Mac apps
- Version consistency guaranteed
- Professional app experience

**DOCTRINE Check**:
> "When to Reconsider:
> - If app size becomes prohibitive (>500MB)
> - If Python-only updates are frequent"

**Verdict**: App size <500MB, updates not frequent. Strategy working.

---

### 5. Progressive Enhancement Philosophy ✅

**Status**: **Keep as-is**

**Why It's Still Good**:
- Core principle, not just implementation detail
- Features work at multiple quality levels
- Graceful degradation validated (ChromaDB, heavy ML optional)

**DOCTRINE Check**:
> "When to Reconsider: If progressive enhancement adds excessive complexity"

**Verdict**: Complexity managed well. Philosophy guides good decisions.

---

## Action Plan

### Immediate (This Week)

1. **Remove Compatibility Layers** (2-3 hours)
   - Delete `common/archive_compat.py`, `common/source_extractor_compat.py`
   - Update imports throughout codebase
   - Run tests to verify

2. **Add Cache Monitoring** (2 hours)
   - Log cache hit rates
   - Log model selection distribution
   - Set up weekly review

### Short-Term (1-2 Weeks)

3. **Measure Test Coverage** (1 hour)
   - Run pytest with --cov
   - Generate coverage report
   - Identify gaps in critical modules

4. **Collect Metrics** (passive)
   - Cache performance (hit rates, size)
   - Model selection distribution (Haiku vs Sonnet)
   - User count (if multi-user)

### Medium-Term (1 Month)

5. **Expand Test Coverage** (1-2 weeks)
   - Add tests for critical modules (80-90% coverage)
   - Add integration tests for main pipelines
   - Add performance tests for clustering

6. **Analyze Metrics** (2-3 hours)
   - Cache: Adjust TTL if needed
   - Models: Adjust threshold if needed
   - Document findings in DOCTRINE.md

### Long-Term (3-6 Months)

7. **Quarterly Architecture Review**
   - Review all "When to Reconsider" conditions
   - Update DOCTRINE.md with new decisions
   - Plan any major changes needed

---

## Metrics to Track

Based on DOCTRINE.md Metrics and Monitoring section, implement these:

### High Priority Metrics

| Metric | Target | How to Check | Frequency |
|--------|--------|--------------|-----------|
| Cache Hit Rate | >60% base, >85% semantic | Log cache.get_stats() | Weekly |
| Model Distribution | 40-60% Haiku | Log model selection | Weekly |
| Test Coverage | 80-90% critical modules | pytest --cov | Monthly |

### Medium Priority Metrics

| Metric | Target | How to Check | Frequency |
|--------|--------|--------------|-----------|
| API Cost per Article | <$0.005 | Log token usage | Monthly |
| Clustering Coherence | >0.7 | Manual validation | Monthly |
| User Count | <1000 (SQLite OK) | Count auth.db users | Monthly |

### Low Priority Metrics

| Metric | Target | How to Check | Frequency |
|--------|--------|--------------|-----------|
| Cache Size | <500MB | du -sh summary_cache/ | Quarterly |
| Mac App Startup | <5 seconds | Manual testing | Quarterly |

---

## Risk Assessment

### High Risk (Address Soon)

- **Compatibility layers still present**: Remove to avoid confusion
- **No cache monitoring**: Can't validate performance assumptions
- **Unknown test coverage**: Unsafe for refactoring

### Medium Risk (Monitor)

- **Model selection unvalidated**: May be wasting money
- **No metrics collection**: Can't detect degradation
- **SQLite at unknown scale**: May hit limits unexpectedly

### Low Risk (Current State OK)

- **Clustering strategy**: Working well
- **Template choice**: Appropriate for use case
- **Python bundling**: Meeting user needs

---

## Summary

**Top 3 Immediate Actions**:

1. ✅ **Remove compatibility layers** (6+ months old)
2. 📊 **Add cache and model monitoring** (validate assumptions)
3. 📈 **Measure test coverage** (prepare for future refactoring)

**Top 3 Things That Are Working Well**:

1. ✅ Lightweight clustering (90% size reduction validated)
2. ✅ Jinja2 templates (appropriate for RSS reader)
3. ✅ Dual-model strategy (good cost/quality trade-off)

**Key Insight**: Most architectural decisions are sound. Main gap is **monitoring and validation** of assumptions. Once metrics are in place, you'll have data to make informed decisions about any adjustments needed.

---

## Next DOCTRINE.md Update

After implementing these recommendations, update DOCTRINE.md:

1. **Remove Compatibility Layers section** (no longer relevant)
2. **Add Monitoring Results** to Metrics section
3. **Update Coverage Goals** with actual coverage numbers
4. **Document Threshold Adjustments** if model selection changes
5. **Add "Lessons Learned"** section for discoveries

---

**Generated**: 2025-12-09
**Next Review**: 2025-03-09 (quarterly)
