// static/js/settings_modal.js - Settings modal functionality

class SettingsModal {
  constructor() {
    this.settingsButton = document.getElementById('settingsButton');
    this.closeSettingsButton = document.getElementById('closeSettings');
    this.settingsMenu = document.getElementById('settingsMenu');
    this.settingsContainer = document.getElementById('settingsContainer');
    
    this.firstFocusableElement = null;
    this.lastFocusableElement = null;
    
    this.init();
  }

  init() {
    if (!this.settingsButton || !this.settingsMenu) return;
    
    this.bindEvents();
    this.setFocusableElements();
  }

  bindEvents() {
    this.settingsButton.addEventListener('click', () => {
      this.toggleModal();
    });
    
    if (this.closeSettingsButton) {
      this.closeSettingsButton.addEventListener('click', () => {
        this.closeModal();
      });
    }
    
    // Close on escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) {
        this.closeModal();
      }
    });
    
    // Optional: Close when clicking outside (be careful with accessibility)
    // document.addEventListener('click', (e) => {
    //   if (this.isOpen() && !this.settingsContainer.contains(e.target)) {
    //     this.closeModal();
    //   }
    // });
  }

  setFocusableElements() {
    const focusableElementsString = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
    const focusableElements = this.settingsMenu.querySelectorAll(focusableElementsString);
    
    if (focusableElements.length > 0) {
      this.firstFocusableElement = focusableElements[0];
      this.lastFocusableElement = focusableElements[focusableElements.length - 1];
    }
  }

  toggleModal() {
    if (this.isOpen()) {
      this.closeModal();
    } else {
      this.openModal();
    }
  }

  openModal() {
    this.settingsMenu.removeAttribute('hidden');
    this.settingsButton.setAttribute('aria-expanded', 'true');
    
    // Focus management
    this.setFocusableElements();
    if (this.firstFocusableElement) {
      this.firstFocusableElement.focus();
    }
    
    // Add focus trap
    document.addEventListener('keydown', this.trapFocus.bind(this));
  }

  closeModal() {
    this.settingsMenu.setAttribute('hidden', '');
    this.settingsButton.setAttribute('aria-expanded', 'false');
    
    // Return focus to the button that opened the modal
    this.settingsButton.focus();
    
    // Remove focus trap
    document.removeEventListener('keydown', this.trapFocus.bind(this));
  }

  isOpen() {
    return !this.settingsMenu.hasAttribute('hidden');
  }

  trapFocus(e) {
    if (!this.isOpen()) return;
    
    if (e.key === 'Tab') {
      if (e.shiftKey) { // Shift + Tab
        if (document.activeElement === this.firstFocusableElement) {
          this.lastFocusableElement.focus();
          e.preventDefault();
        }
      } else { // Tab
        if (document.activeElement === this.lastFocusableElement) {
          this.firstFocusableElement.focus();
          e.preventDefault();
        }
      }
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new SettingsModal();
});