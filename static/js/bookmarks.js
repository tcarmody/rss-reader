// static/js/bookmarks.js - Bookmarks page functionality

class BookmarksPage {
  constructor() {
    this.summaryModal = null;
    this.importModal = null;
    this.init();
  }

  init() {
    this.createModals();
    this.initFilters();
    this.initBookmarkActions();
    this.initBulkActions();
    this.initModals();
  }

  createModals() {
    this.summaryModal = document.getElementById('summaryModal');
    this.importModal = document.getElementById('importModal');
  }

  initFilters() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const bookmarkItems = document.querySelectorAll('.bookmark-item');
    const tagSearch = document.getElementById('tag-search');
    
    // Filter by read status
    filterButtons.forEach(button => {
      button.addEventListener('click', () => {
        // Update active button
        filterButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        const filter = button.dataset.filter;
        this.applyFilters(filter, tagSearch.value);
      });
    });
    
    // Filter by tags
    tagSearch.addEventListener('input', () => {
      const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;
      this.applyFilters(activeFilter, tagSearch.value);
    });
  }

  applyFilters(statusFilter, tagFilter) {
    const bookmarkItems = document.querySelectorAll('.bookmark-item');
    const searchTerm = tagFilter.toLowerCase();
    
    bookmarkItems.forEach(item => {
      let showItem = true;
      
      // Apply status filter
      if (statusFilter === 'read' && !item.classList.contains('read')) {
        showItem = false;
      } else if (statusFilter === 'unread' && !item.classList.contains('unread')) {
        showItem = false;
      }
      
      // Apply tag filter
      if (showItem && searchTerm) {
        const tags = item.dataset.tags.toLowerCase();
        if (!tags.includes(searchTerm)) {
          showItem = false;
        }
      }
      
      item.style.display = showItem ? '' : 'none';
    });
  }

  initBookmarkActions() {
    // Mark as read/unread
    document.querySelectorAll('.read-toggle').forEach(button => {
      button.addEventListener('click', async () => {
        const id = button.dataset.id;
        const currentStatus = button.dataset.status === 'true';
        const newStatus = !currentStatus;
        
        try {
          const response = await fetch(`/api/bookmarks/${id}/read?status=${newStatus}`, {
            method: 'PUT'
          });
          
          if (response.ok) {
            this.updateBookmarkStatus(button, newStatus);
          } else {
            throw new Error('Failed to update status');
          }
        } catch (error) {
          console.error('Error updating read status:', error);
          window.dataPointsAI.showNotification('Error updating bookmark status', 'error');
        }
      });
    });
    
    // Delete bookmark
    document.querySelectorAll('.delete-btn').forEach(button => {
      button.addEventListener('click', async () => {
        const bookmarkId = button.dataset.id;
        if (confirm('Are you sure you want to delete this bookmark?')) {
          try {
            const response = await fetch(`/api/bookmarks/${bookmarkId}`, {
              method: 'DELETE',
            });
            
            if (response.ok) {
              this.removeBookmarkFromDOM(button);
            } else {
              const errorData = await response.json();
              throw new Error(errorData.detail || 'Failed to delete bookmark');
            }
          } catch (error) {
            console.error('Error deleting bookmark:', error);
            window.dataPointsAI.showNotification('Error deleting bookmark', 'error');
          }
        }
      });
    });
    
    // Individual summarize
    document.querySelectorAll('.summarize-btn').forEach(button => {
      button.addEventListener('click', async () => {
        const bookmarkId = button.dataset.id;
        const url = button.dataset.url;
        
        await this.summarizeBookmark(bookmarkId, url, false);
      });
    });
  }

  updateBookmarkStatus(button, newStatus) {
    const bookmarkItem = button.closest('.bookmark-item');
    
    if (newStatus) {
      bookmarkItem.classList.remove('unread');
      bookmarkItem.classList.add('read');
      button.textContent = 'Mark as unread';
    } else {
      bookmarkItem.classList.remove('read');
      bookmarkItem.classList.add('unread');
      button.textContent = 'Mark as read';
    }
    
    button.dataset.status = String(newStatus);
  }

  removeBookmarkFromDOM(button) {
    const bookmarkItem = button.closest('.bookmark-item');
    bookmarkItem.remove();
    
    // Check if there are no more bookmarks
    const remainingBookmarks = document.querySelectorAll('.bookmark-item');
    if (remainingBookmarks.length === 0) {
      this.showEmptyState();
    }
  }

  showEmptyState() {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <p>You haven't saved any articles yet.</p>
      <p>Browse the <a href="/">feed</a> and click "Save for later" on articles you want to read later.</p>
    `;
    document.querySelector('.bookmarks-list').appendChild(emptyState);
  }

  initBulkActions() {
    const checkboxes = document.querySelectorAll('.bookmark-checkbox');
    const selectionCount = document.getElementById('selection-count');
    const summarizeAllBtn = document.getElementById('summarize-all-btn');
    
    // Checkbox selection
    checkboxes.forEach(checkbox => {
      checkbox.addEventListener('change', () => {
        this.updateSelectionCount();
      });
    });
    
    // Bulk summarize
    summarizeAllBtn.addEventListener('click', async () => {
      const checkedBoxes = document.querySelectorAll('.bookmark-checkbox:checked');
      
      if (checkedBoxes.length === 0) {
        window.dataPointsAI.showNotification('Please select at least one article to summarize.', 'warning');
        return;
      }
      
      await this.bulkSummarize(checkedBoxes);
    });
  }

  updateSelectionCount() {
    const checkedBoxes = document.querySelectorAll('.bookmark-checkbox:checked');
    const selectionCount = document.getElementById('selection-count');
    const count = checkedBoxes.length;
    
    if (count > 0) {
      selectionCount.textContent = `${count} article${count === 1 ? '' : 's'} selected`;
      selectionCount.style.display = 'block';
    } else {
      selectionCount.style.display = 'none';
    }
  }

  async summarizeBookmark(bookmarkId, url, isBulk = false) {
    if (!isBulk) {
      this.showSummaryModal();
      this.setSummaryModalContent('loading', 'Generating summary...');
    }
    
    try {
      const bookmarkItem = document.querySelector(`[data-id="${bookmarkId}"]`);
      const summaryElement = bookmarkItem?.querySelector('.bookmark-summary');
      const titleElement = bookmarkItem?.querySelector('.bookmark-title a');
      
      const storedSummary = summaryElement?.textContent.trim() || '';
      const title = titleElement?.textContent.trim() || '';
      
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          url: url,
          title: title,
          stored_summary: storedSummary,
          stored_content: ''
        }),
      });
      
      const result = await response.json();
      
      if (response.ok) {
        // Update bookmark summary in database
        await this.updateBookmarkSummary(bookmarkId, result.summary);
        
        // Update DOM
        this.updateBookmarkSummaryInDOM(bookmarkId, result.summary);
        
        if (!isBulk) {
          this.setSummaryModalContent('success', null, result);
        }
        
        return result;
      } else {
        throw new Error(result.detail || 'Failed to generate summary');
      }
    } catch (error) {
      console.error('Error summarizing article:', error);
      if (!isBulk) {
        this.setSummaryModalContent('error', error.message);
      }
      throw error;
    }
  }

  async bulkSummarize(checkedBoxes) {
    this.showSummaryModal();
    this.setSummaryModalContent('loading', `Generating summaries for ${checkedBoxes.length} article${checkedBoxes.length === 1 ? '' : 's'}...`);
    
    try {
      const urls = [];
      const bookmarkIds = [];
      
      checkedBoxes.forEach(checkbox => {
        urls.push(checkbox.dataset.url);
        bookmarkIds.push(checkbox.dataset.id);
      });
      
      const response = await fetch('/api/summarize-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls, bookmark_ids: bookmarkIds }),
      });
      
      const results = await response.json();
      
      if (response.ok) {
        let summariesHTML = '<div class="batch-summaries">';
        
        results.summaries.forEach((result, index) => {
          summariesHTML += `
            <div class="summary-item">
              <h3>${result.title || `Article ${index + 1}`}</h3>
              <div class="summary-text">${result.summary}</div>
              <div class="summary-meta">
                <a href="${urls[index]}" target="_blank" rel="noopener noreferrer">View Original Article</a>
              </div>
            </div>
          `;
          
          // Update bookmark in database and DOM
          this.updateBookmarkSummary(bookmarkIds[index], result.summary);
          this.updateBookmarkSummaryInDOM(bookmarkIds[index], result.summary);
        });
        
        summariesHTML += '</div>';
        this.setSummaryModalContent('html', null, null, summariesHTML);
      } else {
        throw new Error(results.detail || 'Failed to generate summaries');
      }
    } catch (error) {
      console.error('Error in bulk summarization:', error);
      this.setSummaryModalContent('error', error.message);
    }
  }

  async updateBookmarkSummary(bookmarkId, summary) {
    try {
      await fetch(`/api/bookmarks/${bookmarkId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ summary }),
      });
    } catch (error) {
      console.error('Error updating bookmark summary:', error);
    }
  }

  updateBookmarkSummaryInDOM(bookmarkId, summary) {
    const bookmarkItem = document.querySelector(`[data-id="${bookmarkId}"]`);
    if (!bookmarkItem) return;
    
    let summaryElement = bookmarkItem.querySelector('.bookmark-summary');
    
    if (summaryElement) {
      summaryElement.textContent = summary;
    } else {
      summaryElement = document.createElement('div');
      summaryElement.className = 'bookmark-summary';
      summaryElement.textContent = summary;
      bookmarkItem.insertBefore(summaryElement, bookmarkItem.querySelector('.bookmark-actions'));
    }
  }

  initModals() {
    // Summary modal
    const summaryCloseBtn = this.summaryModal?.querySelector('.close');
    summaryCloseBtn?.addEventListener('click', () => {
      this.hideSummaryModal();
    });
    
    // Import modal
    const importBtn = document.querySelector('.import-btn');
    const importCloseBtn = this.importModal?.querySelector('.close');
    const importForm = document.getElementById('importForm');
    
    importBtn?.addEventListener('click', () => {
      this.showImportModal();
    });
    
    importCloseBtn?.addEventListener('click', () => {
      this.hideImportModal();
    });
    
    importForm?.addEventListener('submit', async (e) => {
      e.preventDefault();
      await this.handleImport(new FormData(importForm));
    });
    
    // Close modals when clicking outside
    window.addEventListener('click', (event) => {
      if (event.target === this.summaryModal) {
        this.hideSummaryModal();
      } else if (event.target === this.importModal) {
        this.hideImportModal();
      }
    });
  }

  showSummaryModal() {
    if (this.summaryModal) {
      this.summaryModal.style.display = 'block';
    }
  }

  hideSummaryModal() {
    if (this.summaryModal) {
      this.summaryModal.style.display = 'none';
    }
  }

  showImportModal() {
    if (this.importModal) {
      this.importModal.style.display = 'block';
    }
  }

  hideImportModal() {
    if (this.importModal) {
      this.importModal.style.display = 'none';
    }
  }

  setSummaryModalContent(type, message, result = null, htmlContent = null) {
    const summaryContent = document.getElementById('summary-content');
    if (!summaryContent) return;
    
    switch (type) {
      case 'loading':
        summaryContent.innerHTML = `
          <div class="loading-spinner">
            <div class="spinner"></div>
            <p>${message}</p>
          </div>
        `;
        break;
        
      case 'success':
        summaryContent.innerHTML = `
          <h3>${result.title || 'Article Summary'}</h3>
          <div class="summary-text">${result.summary}</div>
          <div class="summary-meta">
            <a href="${result.url}" target="_blank" rel="noopener noreferrer">View Original Article</a>
          </div>
        `;
        break;
        
      case 'error':
        summaryContent.innerHTML = `
          <div class="error-message">Error: ${message}</div>
        `;
        break;
        
      case 'html':
        summaryContent.innerHTML = htmlContent;
        break;
    }
  }

  async handleImport(formData) {
    try {
      const response = await fetch('/api/bookmarks/import', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (response.ok) {
        window.dataPointsAI.showNotification(
          `Successfully imported ${result.imported} bookmarks. The page will now reload.`, 
          'success'
        );
        this.hideImportModal();
        setTimeout(() => window.location.reload(), 1000);
      } else {
        throw new Error(result.detail || 'Import failed');
      }
    } catch (error) {
      console.error('Error importing bookmarks:', error);
      window.dataPointsAI.showNotification('An error occurred while importing bookmarks.', 'error');
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new BookmarksPage();
});