/**
 * Content Toggle Functionality
 * Handles switching between summary and original article content
 */

class ContentToggle {
  constructor() {
    this.init();
  }

  init() {
    // Initialize all content toggle containers
    document.querySelectorAll('.content-toggle-container').forEach(container => {
      this.setupToggle(container);
    });
  }

  setupToggle(container) {
    const toggleButtons = container.querySelectorAll('.toggle-btn');
    const contentViews = container.querySelectorAll('.content-view');
    const contentLengthEl = container.querySelector('.content-length');

    // Set up button click handlers
    toggleButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        e.preventDefault();
        this.switchView(container, button.dataset.view);
      });
    });

    // Initialize content length display
    this.updateContentLength(container);

    // Set up keyboard navigation
    this.setupKeyboardNavigation(container);
  }

  switchView(container, targetView) {
    const toggleButtons = container.querySelectorAll('.toggle-btn');
    const contentViews = container.querySelectorAll('.content-view');

    // Update button states
    toggleButtons.forEach(button => {
      const isActive = button.dataset.view === targetView;
      button.classList.toggle('active', isActive);
      button.setAttribute('aria-checked', isActive.toString());
    });

    // Update content visibility with fade effect
    contentViews.forEach(view => {
      const isTargetView = view.dataset.view === targetView;

      if (isTargetView) {
        // Fade in the target view
        view.style.opacity = '0';
        view.classList.add('active');

        // Trigger reflow for smooth animation
        void view.offsetHeight;

        view.style.transition = 'opacity 0.2s ease-in-out';
        view.style.opacity = '1';
      } else {
        // Fade out and hide other views
        view.style.transition = 'opacity 0.2s ease-in-out';
        view.style.opacity = '0';

        setTimeout(() => {
          if (!view.dataset.view === targetView) {
            view.classList.remove('active');
          }
        }, 200);
      }
    });

    // Update content length display
    this.updateContentLength(container, targetView);

    // Log analytics event (if analytics exists)
    this.logToggleEvent(targetView);
  }

  updateContentLength(container, currentView = 'summary') {
    const contentLengthEl = container.querySelector('.content-length');
    if (!contentLengthEl) return;

    const summaryLength = parseInt(contentLengthEl.dataset.summaryLength) || 0;
    const originalLength = parseInt(contentLengthEl.dataset.originalLength) || 0;

    let lengthText = '';
    if (currentView === 'summary') {
      if (summaryLength > 0) {
        lengthText = `${this.formatLength(summaryLength)} summary`;
      } else {
        lengthText = 'No summary';
      }
    } else {
      if (originalLength > 0) {
        lengthText = `${this.formatLength(originalLength)} original`;
      } else {
        lengthText = 'No original content';
      }
    }

    contentLengthEl.textContent = lengthText;
  }

  formatLength(length) {
    if (length < 1000) {
      return `${length} chars`;
    } else if (length < 10000) {
      return `${(length / 1000).toFixed(1)}K chars`;
    } else {
      return `${Math.round(length / 1000)}K chars`;
    }
  }

  setupKeyboardNavigation(container) {
    const toggleButtons = container.querySelectorAll('.toggle-btn');

    toggleButtons.forEach((button, index) => {
      button.addEventListener('keydown', (e) => {
        switch (e.key) {
          case 'ArrowLeft':
          case 'ArrowUp':
            e.preventDefault();
            const prevIndex = index === 0 ? toggleButtons.length - 1 : index - 1;
            toggleButtons[prevIndex].focus();
            toggleButtons[prevIndex].click();
            break;

          case 'ArrowRight':
          case 'ArrowDown':
            e.preventDefault();
            const nextIndex = index === toggleButtons.length - 1 ? 0 : index + 1;
            toggleButtons[nextIndex].focus();
            toggleButtons[nextIndex].click();
            break;

          case 'Enter':
          case ' ':
            e.preventDefault();
            button.click();
            break;
        }
      });
    });
  }

  logToggleEvent(view) {
    // Log toggle events for analytics (if available)
    if (typeof gtag === 'function') {
      gtag('event', 'content_toggle', {
        'custom_parameter': view,
        'event_category': 'engagement'
      });
    }

    // Log to console for debugging
    console.log(`Content view switched to: ${view}`);
  }

  // Public method to programmatically switch views
  switchToView(containerId, view) {
    const container = document.querySelector(`[data-cluster-id="${containerId}"]`);
    if (container) {
      this.switchView(container, view);
    }
  }

  // Public method to get current view
  getCurrentView(containerId) {
    const container = document.querySelector(`[data-cluster-id="${containerId}"]`);
    if (container) {
      const activeButton = container.querySelector('.toggle-btn.active');
      return activeButton ? activeButton.dataset.view : 'summary';
    }
    return null;
  }
}

// Initialize content toggle functionality when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  // Create global instance
  window.contentToggle = new ContentToggle();
});

// Re-initialize when new content is dynamically added
function reinitializeContentToggles() {
  if (window.contentToggle) {
    window.contentToggle.init();
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ContentToggle;
}