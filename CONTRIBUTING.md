# Contributing to Data Points AI RSS Reader

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Testing Guidelines](#testing-guidelines)
6. [Code Style](#code-style)
7. [Pull Request Process](#pull-request-process)
8. [Documentation](#documentation)
9. [Release Process](#release-process)

---

## Code of Conduct

### Our Standards

- **Be Respectful**: Treat all contributors with respect and professionalism
- **Be Collaborative**: Work together to improve the project
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Remember that contributors have varying levels of experience

### Unacceptable Behavior

- Harassment, discrimination, or derogatory comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- **Python 3.11+** (3.11 specifically recommended)
- **Node.js 18+** (for Mac app development)
- **Git** for version control
- **Anthropic API Key** ([Get one here](https://console.anthropic.com/))

### Initial Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/rss-reader.git
   cd rss-reader
   ```

2. **Set Up Python Environment**
   ```bash
   # The run_server.sh script handles everything
   ./run_server.sh --reload --port 5005

   # Or manually:
   python -m venv rss_venv
   source rss_venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m playwright install chromium
   ```

3. **Configure API Key**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=your_key_here" > .env
   ```

4. **Verify Installation**
   ```bash
   # Run tests
   python -m pytest tests/

   # Start development server
   ./run_server.sh --reload

   # Visit http://localhost:5005
   ```

### Understanding the Codebase

**Before diving in, read these documents:**
- **[README.md](README.md)**: Project overview, features, quick start
- **[CLAUDE.md](CLAUDE.md)**: Development commands, architecture, common patterns
- **[DOCTRINE.md](DOCTRINE.md)**: Design decisions and architectural rationale

**Key architectural documents:**
- [TIER3_IMPLEMENTATION_SUMMARY.md](TIER3_IMPLEMENTATION_SUMMARY.md): Advanced features
- [MAC_APP.md](MAC_APP.md): Native macOS application details

---

## Development Workflow

### Branch Strategy

- **`main`**: Production-ready code
- **Feature branches**: `feature/your-feature-name`
- **Bug fixes**: `fix/issue-description`
- **Documentation**: `docs/what-youre-documenting`

### Typical Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make Changes**
   - Write code following [Code Style](#code-style)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run test suite
   python -m pytest tests/

   # Run specific tests
   python -m pytest tests/test_batch_processing.py

   # Manual testing
   ./run_server.sh --reload
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description

   - Detailed point 1
   - Detailed point 2

   Closes #123"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open Pull Request**
   - Use the PR template
   - Link related issues
   - Request review

---

## Architecture Guidelines

### Core Principles

**Before making architectural changes, read [DOCTRINE.md](DOCTRINE.md)** to understand existing design decisions.

#### 1. Progressive Enhancement with Graceful Degradation
- Features should work at multiple quality levels
- Optional dependencies enhance but don't block functionality
- Fail gracefully when dependencies unavailable

#### 2. Separation of Concerns
- Keep modules focused on single responsibilities
- Use clear interfaces between components
- Avoid tight coupling

#### 3. Performance-Conscious Design
- Cache aggressively (tiered caching strategy)
- Use async/await for I/O-bound operations
- Monitor resource usage and costs

### Module Structure

```
api/              # Rate limiting, API utilities
cache/            # Multi-level caching system
clustering/       # Article clustering algorithms
common/           # Shared utilities (config, logging, HTTP, etc.)
content/          # Content processing
  archive/        # Archive services, paywall detection
  extractors/     # Content extraction utilities
models/           # Data models, AI model configuration
reader/           # RSS feed processing
services/         # Business logic (bookmarks, image prompts)
summarization/    # Claude API integration
templates/        # Jinja2 templates (component-based)
static/           # CSS/JS assets
tests/            # Test suite
```

### When to Update DOCTRINE.md

**Always update [DOCTRINE.md](DOCTRINE.md) when making:**
- Major architecture changes
- Technology decisions (frameworks, databases, libraries)
- Performance trade-offs
- Breaking changes or deprecations
- New design patterns

See [CLAUDE.md](CLAUDE.md) for the update template and guidelines.

---

## Testing Guidelines

### Test Organization

```
tests/
├── test_batch_processing.py    # Batch operations
├── test_model_selection.py     # AI model selection
├── test_tier3_improvements.py  # Advanced features
└── ...
```

### Writing Tests

**Use pytest with clear test names:**
```python
def test_cache_hit_rate_with_semantic_matching():
    """Test that semantic cache improves hit rate by 20-30%."""
    # Arrange
    cache = SemanticCache()
    articles = create_test_articles()

    # Act
    hit_rate = cache.test_hit_rate(articles)

    # Assert
    assert hit_rate > 0.85, "Semantic cache should achieve >85% hit rate"
```

### Test Coverage Goals

**Priority test areas:**
1. **Critical Paths**: Features users rely on daily
2. **Integration Points**: Component interactions
3. **Edge Cases**: Unusual inputs, error conditions
4. **Performance**: Verify performance assumptions

**We don't pursue 100% coverage**—focus on valuable tests.

### Running Tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_batch_processing.py

# With coverage (if installed)
python -m pytest tests/ --cov

# Verbose output
python -m pytest tests/ -v

# Stop on first failure
python -m pytest tests/ -x
```

### Test Data

- Use fixtures for reusable test data
- Never commit sensitive data (API keys, personal info)
- Mock external API calls (Claude API, archive services)

---

## Code Style

### Python Style Guide

**Follow PEP 8 with these conventions:**

#### Type Hints
```python
# Use type hints for function signatures
def summarize_article(url: str, complexity: float = 0.5) -> dict[str, Any]:
    """Summarize article using appropriate Claude model."""
    pass
```

#### Docstrings
```python
def select_model_by_complexity(complexity_score: float) -> str:
    """
    Select the appropriate model based on content complexity.

    Args:
        complexity_score: Content complexity score (0.0-1.0)

    Returns:
        Model identifier string (e.g., 'claude-sonnet-4-5')

    Raises:
        ValueError: If complexity_score is out of range
    """
    pass
```

#### Async Patterns
```python
# Use async/await for I/O operations
async def fetch_and_summarize(url: str) -> dict:
    """Fetch article and generate summary."""
    content = await fetch_content(url)
    summary = await generate_summary(content)
    return summary
```

#### Error Handling
```python
# Use custom exceptions from common/errors.py
from common.errors import RateLimitError, CacheError

try:
    summary = await summarizer.summarize(article)
except RateLimitError:
    logger.warning("Rate limit hit, retrying with backoff")
    await asyncio.sleep(5)
    summary = await summarizer.summarize(article)
```

### Import Organization

```python
# Standard library
import asyncio
import logging
from typing import Dict, List, Optional

# Third-party
from fastapi import FastAPI, HTTPException
from anthropic import Anthropic

# Local
from cache.tiered_cache import TieredCache
from common.config import get_config
from models.config import select_model_by_complexity
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `TieredCache`, `ArticleSummarizer`)
- **Functions/Variables**: `snake_case` (e.g., `get_cached_summary`, `complexity_score`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MODEL`, `CACHE_TTL_DAYS`)
- **Private**: Prefix with `_` (e.g., `_internal_helper`)

### Avoid Common Pitfalls

❌ **Don't:**
- Add unnecessary dependencies
- Create global state
- Over-engineer simple features
- Add features not explicitly requested
- Use emojis (unless user explicitly requests)

✅ **Do:**
- Keep it simple
- Use existing patterns
- Add tests for new features
- Update documentation
- Consider performance implications

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass (`python -m pytest tests/`)
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, CLAUDE.md, DOCTRINE.md)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear and descriptive

### PR Template

```markdown
## Description
Brief description of changes and why they're needed.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manually tested feature in development environment

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings or errors
- [ ] DOCTRINE.md updated (if architectural change)

## Related Issues
Closes #123
Related to #456

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Additional Notes
[Any additional context or notes for reviewers]
```

### Review Process

1. **Automated Checks**: CI/CD runs tests (if configured)
2. **Code Review**: Maintainer reviews code
3. **Discussion**: Address feedback and make changes
4. **Approval**: Maintainer approves PR
5. **Merge**: Squash and merge to `main`

### After Merge

- Delete your feature branch
- Update your fork's `main` branch
- Close related issues if not auto-closed

---

## Documentation

### What to Document

**Always update documentation for:**
- New features or capabilities
- Changed behavior or APIs
- New configuration options
- Breaking changes
- Deprecations

### Where to Document

| Type | Location |
|------|----------|
| User-facing features | [README.md](README.md) |
| Development guide | [CLAUDE.md](CLAUDE.md) |
| Design decisions | [DOCTRINE.md](DOCTRINE.md) |
| API endpoints | Docstrings + OpenAPI (automatic) |
| Complex algorithms | Code comments + docstrings |

### Documentation Style

- **Clear and Concise**: Get to the point quickly
- **Examples**: Show, don't just tell
- **Context**: Explain the "why" not just the "what"
- **Up-to-Date**: Keep docs in sync with code

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backwards-compatible
- **Patch (0.0.X)**: Bug fixes, backwards-compatible

### Release Checklist

For maintainers releasing new versions:

1. **Update Version Numbers**
   - `package.json` (if applicable)
   - `README.md` (version badges)

2. **Update Documentation**
   - CHANGELOG.md with release notes
   - README.md with new features
   - DOCTRINE.md with new decisions (if any)

3. **Test Thoroughly**
   ```bash
   python -m pytest tests/
   ./run_server.sh --reload  # Manual testing
   ```

4. **Create Release**
   - Tag release: `git tag -a v1.0.0 -m "Release v1.0.0"`
   - Push tag: `git push origin v1.0.0`
   - Create GitHub release with notes

5. **Mac App Release** (if applicable)
   ```bash
   cd electron
   make build-universal
   # Upload .dmg to GitHub releases
   ```

---

## Getting Help

### Resources

- **Documentation**: Start with [README.md](README.md), [CLAUDE.md](CLAUDE.md), [DOCTRINE.md](DOCTRINE.md)
- **Issues**: Search existing issues for similar problems
- **Discussions**: GitHub Discussions for questions and ideas

### Asking Questions

**Good question format:**
```markdown
**What I'm trying to do:**
[Clear description of goal]

**What I tried:**
[Steps taken, code attempted]

**What happened:**
[Error message, unexpected behavior]

**Environment:**
- Python version: 3.11
- OS: macOS 14.0
- Browser: Chrome 120
```

### Reporting Bugs

**Use the issue template:**
```markdown
**Bug Description:**
Clear and concise description

**To Reproduce:**
1. Step 1
2. Step 2
3. See error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- Python: 3.11.5
- OS: macOS 14.0
- Installation method: pip

**Logs:**
[Paste relevant logs]

**Screenshots:**
[If applicable]
```

---

## Feature Requests

### Proposing Features

**Before proposing, check:**
1. Does it align with project goals? (AI-powered RSS reader)
2. Is it already possible with existing features?
3. Has someone else proposed it?

**Good feature request:**
```markdown
**Feature Description:**
Clear description of proposed feature

**Use Case:**
Why is this valuable? Who benefits?

**Proposed Solution:**
How might this work?

**Alternatives Considered:**
Other approaches you've thought about

**Additional Context:**
Screenshots, examples, mockups
```

---

## Code of Conduct Enforcement

### Reporting Issues

Report unacceptable behavior to project maintainers privately.

### Consequences

1. **Warning**: Private or public warning
2. **Temporary Ban**: Temporary ban from project
3. **Permanent Ban**: Permanent ban from project

---

## Recognition

### Contributors

All contributors are recognized in:
- GitHub contributor list
- Release notes (for significant contributions)
- Special thanks in README (for major features)

### Becoming a Maintainer

Active contributors may be invited to become maintainers based on:
- Quality and quantity of contributions
- Community involvement and helpfulness
- Understanding of project architecture and goals
- Demonstrated judgment and professionalism

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License, the same license covering this project.

---

## Thank You!

Your contributions make this project better for everyone. Whether you're fixing a typo, adding a feature, or improving documentation, we appreciate your time and effort.

**Happy coding!** 🚀
