// static/js/common.js - Shared functionality across all pages

class DataPointsAI {
  constructor() {
    this.init();
  }

  init() {
    this.initToggleHandlers();
    this.initFormSubmissionStates();
    this.initAccessibility();
  }

  // Handle toggle switches (like time range options)
  initToggleHandlers() {
    document.querySelectorAll('input[data-toggle-target]').forEach(toggle => {
      const targetId = toggle.getAttribute('data-toggle-target');
      const target = document.getElementById(targetId);
      
      if (target) {
        toggle.addEventListener('change', () => {
          target.style.display = toggle.checked ? 'flex' : 'none';
          target.setAttribute('aria-hidden', !toggle.checked);
        });
        
        // Set initial state
        target.style.display = toggle.checked ? 'flex' : 'none';
        target.setAttribute('aria-hidden', !toggle.checked);
      }
    });
  }

  // Add loading states to form submissions
  initFormSubmissionStates() {
    document.querySelectorAll('form').forEach(form => {
      form.addEventListener('submit', (e) => {
        const submitButton = form.querySelector('button[type="submit"]');
        if (submitButton && !submitButton.disabled) {
          this.setButtonLoading(submitButton, true);
          
          // Reset after a reasonable timeout (fallback)
          setTimeout(() => {
            this.setButtonLoading(submitButton, false);
          }, 30000);
        }
      });
    });
  }

  // Add loading spinner to button
  setButtonLoading(button, isLoading) {
    if (isLoading) {
      button.disabled = true;
      const originalText = button.innerHTML;
      button.setAttribute('data-original-text', originalText);
      button.innerHTML = `${originalText} <span class="loading-spinner"></span>`;
    } else {
      button.disabled = false;
      const originalText = button.getAttribute('data-original-text');
      if (originalText) {
        button.innerHTML = originalText;
        button.removeAttribute('data-original-text');
      }
    }
  }

  // Enhanced accessibility features
  initAccessibility() {
    // Add keyboard navigation for custom dropdowns
    this.initDropdownKeyboard();
    
    // Improve focus management
    this.initFocusManagement();
    
    // Add skip links if they don't exist
    this.addSkipLinks();
  }

  initDropdownKeyboard() {
    document.querySelectorAll('.dropdown').forEach(dropdown => {
      const toggle = dropdown.querySelector('.dropdown-toggle');
      const menu = dropdown.querySelector('.dropdown-menu');
      
      if (toggle && menu) {
        toggle.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.toggleDropdown(dropdown);
          }
        });
      }
    });
  }

  initFocusManagement() {
    // Trap focus in modals
    document.querySelectorAll('[role="dialog"]').forEach(modal => {
      modal.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.closeModal(modal);
        } else if (e.key === 'Tab') {
          this.trapFocus(e, modal);
        }
      });
    });
  }

  trapFocus(e, container) {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        lastElement.focus();
        e.preventDefault();
      }
    } else {
      if (document.activeElement === lastElement) {
        firstElement.focus();
        e.preventDefault();
      }
    }
  }

  addSkipLinks() {
    if (!document.querySelector('.skip-links')) {
      const skipLinks = document.createElement('div');
      skipLinks.className = 'skip-links';
      skipLinks.innerHTML = `
        <a href="#main-content" class="skip-link">Skip to main content</a>
        <a href="#navigation" class="skip-link">Skip to navigation</a>
      `;
      document.body.insertBefore(skipLinks, document.body.firstChild);
    }
  }

  // Utility methods
  async apiRequest(url, options = {}) {
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    try {
      const response = await fetch(url, { ...defaultOptions, ...options });
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
      }
      
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.setAttribute('role', 'alert');
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      notification.remove();
    }, 5000);
  }

  // Bookmark functionality (shared across pages)
  async saveBookmark(url, title, summary = '') {
    try {
      const formData = new FormData();
      formData.append('url', url);
      formData.append('title', title);
      formData.append('summary', summary);

      const response = await fetch('/api/bookmarks', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (response.ok) {
        this.showNotification('Article bookmarked successfully!', 'success');
        return result;
      } else {
        throw new Error(result.detail || 'Failed to bookmark article');
      }
    } catch (error) {
      this.showNotification(`Error: ${error.message}`, 'error');
      throw error;
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.dataPointsAI = new DataPointsAI();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DataPointsAI;
}