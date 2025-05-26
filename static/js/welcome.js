// static/js/welcome.js - Welcome page specific functionality

class WelcomePage {
  constructor() {
    this.init();
  }

  init() {
    this.initTabNavigation();
    this.initFormHandlers();
  }

  // Tab functionality for Feed Management
  initTabNavigation() {
    const tabs = document.querySelectorAll('.tabs .tab');
    const tabPanels = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
      tab.addEventListener('click', (event) => {
        event.preventDefault();
        
        // Deactivate all tabs and hide all panels
        tabs.forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
          t.setAttribute('tabindex', '-1');
        });
        
        tabPanels.forEach(panel => {
          panel.classList.remove('active');
          panel.setAttribute('hidden', '');
        });

        // Activate the clicked tab and show its panel
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        tab.removeAttribute('tabindex');
        
        const targetPanelId = tab.getAttribute('aria-controls');
        const targetPanel = document.getElementById(targetPanelId);
        if (targetPanel) {
          targetPanel.classList.add('active');
          targetPanel.removeAttribute('hidden');
        }
      });

      // Keyboard navigation
      tab.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
          e.preventDefault();
          const currentIndex = Array.from(tabs).indexOf(tab);
          const newIndex = e.key === 'ArrowRight' 
            ? (currentIndex + 1) % tabs.length 
            : (currentIndex - 1 + tabs.length) % tabs.length;
          tabs[newIndex].click();
          tabs[newIndex].focus();
        }
      });
    });
  }

  // Form submission handlers
  initFormHandlers() {
    this.updateHiddenFeedParams('defaultFeedsForm');
    this.updateHiddenFeedParams('customFeedsForm');

    // Add event listeners to feed forms
    const defaultFeedsForm = document.getElementById('defaultFeedsForm');
    if (defaultFeedsForm) {
      defaultFeedsForm.addEventListener('submit', () => {
        this.updateHiddenFeedParams('defaultFeedsForm');
      });
    }

    const customFeedsForm = document.getElementById('customFeedsForm');
    if (customFeedsForm) {
      customFeedsForm.addEventListener('submit', () => {
        this.updateHiddenFeedParams('customFeedsForm');
      });
    }
  }

  // Update hidden feed parameters before form submission
  updateHiddenFeedParams(formId) {
    const form = document.getElementById(formId);
    if (!form) return;

    const globalBatchSize = document.getElementById('global_batch_size');
    const globalBatchDelay = document.getElementById('global_batch_delay');
    const globalPerFeedLimit = document.getElementById('global_per_feed_limit');

    // Update or create hidden fields
    this.updateHiddenField(form, 'batch_size', globalBatchSize?.value || '25');
    this.updateHiddenField(form, 'batch_delay', globalBatchDelay?.value || '15');
    this.updateHiddenField(form, 'per_feed_limit', globalPerFeedLimit?.value || '25');
  }

  updateHiddenField(form, name, value) {
    let field = form.querySelector(`input[name="${name}"]`);
    
    if (!field) {
      field = document.createElement('input');
      field.type = 'hidden';
      field.name = name;
      form.appendChild(field);
    }
    
    field.value = value;
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new WelcomePage();
});