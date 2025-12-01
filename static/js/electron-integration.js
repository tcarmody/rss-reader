/**
 * Electron Integration Script
 * Handles Mac app features like badge updates, recent items, export, and find
 */

(function() {
  // Only run if in Electron
  if (!window.electronAPI || !window.electronAPI.isElectron) {
    return;
  }

  console.log('Electron integration loaded');

  // ====================
  // Badge Management
  // ====================

  /**
   * Update dock badge with unread count
   */
  function updateBadgeCount() {
    // Count articles that haven't been clicked
    const articles = document.querySelectorAll('.article-item, .cluster-article');
    const unreadCount = Array.from(articles).filter(article => {
      const link = article.querySelector('a[href]');
      return link && !link.classList.contains('visited');
    }).length;

    window.electronAPI.updateBadge(unreadCount);
  }

  // Update badge on page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateBadgeCount);
  } else {
    updateBadgeCount();
  }

  // Update badge when articles are clicked
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a[href]');
    if (link && link.href.startsWith('http')) {
      link.classList.add('visited');
      setTimeout(updateBadgeCount, 100);
    }
  });

  // ====================
  // Recent Articles
  // ====================

  /**
   * Add article to recent items when clicked
   */
  document.addEventListener('click', (e) => {
    const articleLink = e.target.closest('a[href]');
    if (!articleLink || !articleLink.href.startsWith('http')) return;

    // Find article title
    const articleContainer = articleLink.closest('.article-item, .cluster-article, .bookmark-item');
    if (!articleContainer) return;

    const titleElement = articleContainer.querySelector('.article-title, h3, h4');
    const sourceElement = articleContainer.querySelector('.feed-source, .source');

    if (titleElement) {
      const article = {
        title: titleElement.textContent.trim(),
        link: articleLink.href,
        source: sourceElement ? sourceElement.textContent.trim() : '',
        timestamp: new Date().toISOString()
      };

      window.electronAPI.addRecentArticle(article);
    }
  });

  // ====================
  // Export Functionality
  // ====================

  /**
   * Handle export requests from menu
   */
  window.electronAPI.onExportRequest((format) => {
    console.log('Export requested:', format);

    // Get all clusters data from the page
    const clusters = [];
    const clusterElements = document.querySelectorAll('.cluster');

    clusterElements.forEach(clusterEl => {
      const clusterArticles = [];
      const articleElements = clusterEl.querySelectorAll('.article-item, .cluster-article');

      articleElements.forEach(articleEl => {
        const titleEl = articleEl.querySelector('.article-title, h3, h4');
        const linkEl = articleEl.querySelector('a[href]');
        const sourceEl = articleEl.querySelector('.feed-source, .source');
        const publishedEl = articleEl.querySelector('.published-date, time');
        const summaryEl = articleEl.querySelector('.article-summary, .summary');

        if (titleEl && linkEl) {
          clusterArticles.push({
            title: titleEl.textContent.trim(),
            link: linkEl.href,
            feed_source: sourceEl ? sourceEl.textContent.trim() : '',
            published: publishedEl ? publishedEl.textContent.trim() : '',
            summary: {
              summary: summaryEl ? summaryEl.textContent.trim() : ''
            }
          });
        }
      });

      if (clusterArticles.length > 0) {
        clusters.push(clusterArticles);
      }
    });

    // If no clusters found, try to get bookmarks
    if (clusters.length === 0) {
      const bookmarkElements = document.querySelectorAll('.bookmark-item');
      const bookmarks = [];

      bookmarkElements.forEach(bookmarkEl => {
        const titleEl = bookmarkEl.querySelector('.bookmark-title, h3');
        const linkEl = bookmarkEl.querySelector('a[href]');
        const summaryEl = bookmarkEl.querySelector('.bookmark-summary, .summary');

        if (titleEl && linkEl) {
          bookmarks.push({
            title: titleEl.textContent.trim(),
            link: linkEl.href,
            feed_source: new URL(linkEl.href).hostname,
            published: '',
            summary: {
              summary: summaryEl ? summaryEl.textContent.trim() : ''
            }
          });
        }
      });

      if (bookmarks.length > 0) {
        clusters.push(bookmarks);
      }
    }

    if (clusters.length === 0) {
      alert('No articles found to export. Please process some feeds first.');
      return;
    }

    // Call export function
    window.electronAPI.exportArticles(format, clusters)
      .then(() => {
        console.log('Export completed');
      })
      .catch(err => {
        console.error('Export failed:', err);
        alert('Export failed: ' + err.message);
      });
  });

  // ====================
  // Find in Page
  // ====================

  let findOverlay = null;
  let findInput = null;
  let currentSearchTerm = '';
  let currentResultIndex = 0;
  let searchResults = [];

  /**
   * Create find overlay
   */
  function createFindOverlay() {
    if (findOverlay) return;

    findOverlay = document.createElement('div');
    findOverlay.id = 'electron-find-overlay';
    findOverlay.innerHTML = `
      <div class="find-container">
        <input type="text" id="find-input" placeholder="Find in page..." />
        <span id="find-results">0 of 0</span>
        <button id="find-prev" title="Previous (Cmd+Shift+G)">▲</button>
        <button id="find-next" title="Next (Cmd+G)">▼</button>
        <button id="find-close" title="Close (Esc)">✕</button>
      </div>
    `;

    // Add styles
    const style = document.createElement('style');
    style.textContent = `
      #electron-find-overlay {
        position: fixed;
        top: 60px;
        right: 20px;
        z-index: 10000;
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 8px 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        display: none;
      }

      #electron-find-overlay.visible {
        display: block;
      }

      .find-container {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      #find-input {
        padding: 4px 8px;
        border: 1px solid rgba(0, 0, 0, 0.2);
        border-radius: 4px;
        width: 200px;
        font-size: 13px;
      }

      #find-input:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
      }

      #find-results {
        font-size: 12px;
        color: #666;
        min-width: 60px;
        text-align: center;
      }

      #electron-find-overlay button {
        padding: 4px 8px;
        border: 1px solid rgba(0, 0, 0, 0.2);
        border-radius: 4px;
        background: white;
        cursor: pointer;
        font-size: 11px;
        transition: all 0.2s;
      }

      #electron-find-overlay button:hover {
        background: #f3f4f6;
        border-color: #3b82f6;
      }

      #find-close {
        color: #ef4444;
        font-weight: bold;
      }

      .find-highlight {
        background-color: #fef08a !important;
        border-radius: 2px;
        padding: 0 2px;
      }

      .find-highlight-current {
        background-color: #fbbf24 !important;
      }

      @media (prefers-color-scheme: dark) {
        #electron-find-overlay {
          background: rgba(31, 41, 55, 0.98);
          border-color: rgba(255, 255, 255, 0.1);
        }

        #find-input {
          background: rgba(55, 65, 81, 0.95);
          border-color: rgba(255, 255, 255, 0.2);
          color: white;
        }

        #find-results {
          color: #9ca3af;
        }

        #electron-find-overlay button {
          background: rgba(55, 65, 81, 0.95);
          color: white;
          border-color: rgba(255, 255, 255, 0.2);
        }

        #electron-find-overlay button:hover {
          background: rgba(75, 85, 99, 0.95);
        }
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(findOverlay);

    findInput = document.getElementById('find-input');

    // Event listeners
    findInput.addEventListener('input', performSearch);
    document.getElementById('find-prev').addEventListener('click', findPrevious);
    document.getElementById('find-next').addEventListener('click', findNext);
    document.getElementById('find-close').addEventListener('click', closeFindOverlay);

    // Keyboard shortcuts
    findInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        if (e.shiftKey) {
          findPrevious();
        } else {
          findNext();
        }
      } else if (e.key === 'Escape') {
        closeFindOverlay();
      }
    });
  }

  /**
   * Perform search
   */
  function performSearch() {
    const term = findInput.value.toLowerCase();
    currentSearchTerm = term;

    // Clear previous highlights
    clearHighlights();
    searchResults = [];
    currentResultIndex = 0;

    if (!term) {
      updateResultsCount();
      return;
    }

    // Find all text nodes and highlight matches
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: function(node) {
          // Skip script, style, and our find overlay
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (parent.closest('#electron-find-overlay, script, style')) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );

    const nodesToHighlight = [];
    let node;
    while (node = walker.nextNode()) {
      const text = node.nodeValue;
      if (text.toLowerCase().includes(term)) {
        nodesToHighlight.push(node);
      }
    }

    // Highlight matches
    nodesToHighlight.forEach(node => {
      const parent = node.parentElement;
      const text = node.nodeValue;
      const lowerText = text.toLowerCase();
      let lastIndex = 0;
      const fragment = document.createDocumentFragment();

      while (true) {
        const index = lowerText.indexOf(term, lastIndex);
        if (index === -1) break;

        // Add text before match
        if (index > lastIndex) {
          fragment.appendChild(document.createTextNode(text.substring(lastIndex, index)));
        }

        // Add highlighted match
        const span = document.createElement('span');
        span.className = 'find-highlight';
        span.textContent = text.substring(index, index + term.length);
        fragment.appendChild(span);
        searchResults.push(span);

        lastIndex = index + term.length;
      }

      // Add remaining text
      if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
      }

      parent.replaceChild(fragment, node);
    });

    // Highlight first result
    if (searchResults.length > 0) {
      highlightResult(0);
    }

    updateResultsCount();
  }

  /**
   * Highlight specific result
   */
  function highlightResult(index) {
    // Remove current highlight
    searchResults.forEach(el => el.classList.remove('find-highlight-current'));

    if (index >= 0 && index < searchResults.length) {
      currentResultIndex = index;
      const element = searchResults[index];
      element.classList.add('find-highlight-current');
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    updateResultsCount();
  }

  /**
   * Find next result
   */
  function findNext() {
    if (searchResults.length === 0) return;
    const nextIndex = (currentResultIndex + 1) % searchResults.length;
    highlightResult(nextIndex);
  }

  /**
   * Find previous result
   */
  function findPrevious() {
    if (searchResults.length === 0) return;
    const prevIndex = (currentResultIndex - 1 + searchResults.length) % searchResults.length;
    highlightResult(prevIndex);
  }

  /**
   * Clear all highlights
   */
  function clearHighlights() {
    document.querySelectorAll('.find-highlight').forEach(span => {
      const parent = span.parentNode;
      parent.replaceChild(document.createTextNode(span.textContent), span);
      parent.normalize();
    });
  }

  /**
   * Update results count display
   */
  function updateResultsCount() {
    const resultsSpan = document.getElementById('find-results');
    if (searchResults.length > 0) {
      resultsSpan.textContent = `${currentResultIndex + 1} of ${searchResults.length}`;
    } else {
      resultsSpan.textContent = currentSearchTerm ? 'No results' : '0 of 0';
    }
  }

  /**
   * Show find overlay
   */
  function showFindOverlay() {
    createFindOverlay();
    findOverlay.classList.add('visible');
    findInput.focus();
    findInput.select();
  }

  /**
   * Close find overlay
   */
  function closeFindOverlay() {
    if (findOverlay) {
      findOverlay.classList.remove('visible');
      clearHighlights();
      searchResults = [];
      currentResultIndex = 0;
      currentSearchTerm = '';
    }
  }

  // Listen for find commands from menu
  window.electronAPI.onTriggerFind(() => {
    showFindOverlay();
  });

  window.electronAPI.onFindNext(() => {
    if (findOverlay && findOverlay.classList.contains('visible')) {
      findNext();
    } else {
      showFindOverlay();
    }
  });

  window.electronAPI.onFindPrevious(() => {
    if (findOverlay && findOverlay.classList.contains('visible')) {
      findPrevious();
    } else {
      showFindOverlay();
    }
  });

  // Global keyboard shortcut (backup)
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'f') {
      e.preventDefault();
      showFindOverlay();
    } else if (e.key === 'Escape' && findOverlay && findOverlay.classList.contains('visible')) {
      closeFindOverlay();
    }
  });

  console.log('Electron integration complete');
})();
