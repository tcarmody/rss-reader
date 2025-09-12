/**
 * Image Prompt Generation Functionality
 * Handles modal interactions, API calls, and UI state management
 */

// Global variables for managing state
let currentArticleData = null;
let isGenerating = false;

/**
 * Initialize image prompt functionality when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
  setupImagePromptListeners();
  setupKeyboardNavigation();
});

/**
 * Set up event listeners for image prompt functionality
 */
function setupImagePromptListeners() {
  // Style button selection
  document.addEventListener('click', function(e) {
    if (e.target.closest('.style-btn')) {
      const clickedButton = e.target.closest('.style-btn');
      selectStyleButton(clickedButton);
    }
  });
  
  // Modal overlay click to close
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal-overlay')) {
      closeImagePromptModal();
    }
  });
  
  // Escape key to close modal
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      closeImagePromptModal();
    }
  });
}

/**
 * Set up keyboard navigation for accessibility
 */
function setupKeyboardNavigation() {
  document.addEventListener('keydown', function(e) {
    if (!document.getElementById('imagePromptModal').classList.contains('modal-open')) {
      return;
    }
    
    // Arrow key navigation for style buttons
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      navigateStyleButtons(e.key === 'ArrowRight');
      e.preventDefault();
    }
  });
}

/**
 * Navigate between style buttons using keyboard
 */
function navigateStyleButtons(forward = true) {
  const buttons = document.querySelectorAll('.style-btn');
  const activeButton = document.querySelector('.style-btn.active');
  const currentIndex = Array.from(buttons).indexOf(activeButton);
  
  let newIndex = forward ? currentIndex + 1 : currentIndex - 1;
  if (newIndex >= buttons.length) newIndex = 0;
  if (newIndex < 0) newIndex = buttons.length - 1;
  
  selectStyleButton(buttons[newIndex]);
  buttons[newIndex].focus();
}

/**
 * Open the image prompt modal with article data
 */
function openImagePromptModal(articleData) {
  currentArticleData = articleData;
  
  const modal = document.getElementById('imagePromptModal');
  const titleEl = document.getElementById('promptArticleTitle');
  const previewEl = document.getElementById('promptArticlePreview');
  
  // Populate article information
  titleEl.textContent = articleData.title || 'Untitled Article';
  
  // Create content preview (truncate if too long)
  const content = articleData.content || articleData.summary || '';
  const preview = content.length > 200 
    ? content.substring(0, 200) + '...' 
    : content;
  previewEl.textContent = preview || 'No content preview available.';
  
  // Reset modal state
  resetModalState();
  
  // Show modal
  modal.classList.add('modal-open');
  modal.setAttribute('aria-hidden', 'false');
  
  // Focus management
  const firstFocusable = modal.querySelector('.style-btn');
  if (firstFocusable) {
    firstFocusable.focus();
  }
  
  // Trap focus within modal
  trapFocus(modal);
}

/**
 * Close the image prompt modal
 */
function closeImagePromptModal() {
  const modal = document.getElementById('imagePromptModal');
  
  modal.classList.remove('modal-open');
  modal.setAttribute('aria-hidden', 'true');
  
  // Clear state
  currentArticleData = null;
  isGenerating = false;
  
  // Reset modal content
  resetModalState();
  
  // Return focus to trigger element if available
  const triggerElement = document.activeElement;
  if (triggerElement && triggerElement.blur) {
    triggerElement.blur();
  }
}

/**
 * Reset modal to initial state
 */
function resetModalState() {
  // Clear generated prompt
  document.getElementById('generatedPromptText').value = '';
  
  // Reset button states
  document.getElementById('generatePromptBtn').disabled = false;
  document.getElementById('copyPromptBtn').disabled = true;
  
  // Hide loading/error states
  document.getElementById('promptLoadingState').style.display = 'none';
  document.getElementById('promptErrorState').style.display = 'none';
  
  // Reset style selection to first option
  const firstStyle = document.querySelector('.style-btn');
  if (firstStyle) {
    selectStyleButton(firstStyle);
  }
}

/**
 * Select a style button and update UI
 */
function selectStyleButton(button) {
  // Remove active class from all buttons
  document.querySelectorAll('.style-btn').forEach(btn => {
    btn.classList.remove('active');
    btn.setAttribute('aria-checked', 'false');
  });
  
  // Add active class to selected button
  button.classList.add('active');
  button.setAttribute('aria-checked', 'true');
}

/**
 * Generate image prompt using the API
 */
async function generateImagePrompt() {
  if (!currentArticleData || isGenerating) {
    return;
  }
  
  isGenerating = true;
  
  // Get selected style
  const selectedStyle = document.querySelector('.style-btn.active')?.dataset.style || 'photojournalistic';
  
  // Show loading state
  showLoadingState();
  
  try {
    // Prepare form data
    const formData = new FormData();
    formData.append('title', currentArticleData.title || '');
    formData.append('content', currentArticleData.content || currentArticleData.summary || '');
    formData.append('style', selectedStyle);
    
    // Make API request
    const response = await fetch('/api/generate-image-prompt', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    if (result.status === 'success') {
      showGeneratedPrompt(result);
    } else {
      throw new Error(result.error || 'Unknown error occurred');
    }
    
  } catch (error) {
    console.error('Error generating image prompt:', error);
    showErrorState(error.message);
  } finally {
    isGenerating = false;
  }
}

/**
 * Show loading state in modal
 */
function showLoadingState() {
  document.getElementById('promptLoadingState').style.display = 'flex';
  document.getElementById('promptErrorState').style.display = 'none';
  document.getElementById('generatePromptBtn').disabled = true;
  
  // Clear previous prompt
  document.getElementById('generatedPromptText').value = '';
  document.getElementById('copyPromptBtn').disabled = true;
}

/**
 * Show generated prompt in modal
 */
function showGeneratedPrompt(result) {
  // Hide loading state
  document.getElementById('promptLoadingState').style.display = 'none';
  document.getElementById('promptErrorState').style.display = 'none';
  
  // Show generated prompt
  const promptTextarea = document.getElementById('generatedPromptText');
  promptTextarea.value = result.prompt;
  
  // Enable buttons
  document.getElementById('generatePromptBtn').disabled = false;
  document.getElementById('copyPromptBtn').disabled = false;
  
  // Focus the textarea for easy selection
  promptTextarea.focus();
  promptTextarea.select();
  
  console.log('Generated image prompt:', result);
}

/**
 * Show error state in modal
 */
function showErrorState(errorMessage) {
  // Hide loading state
  document.getElementById('promptLoadingState').style.display = 'none';
  
  // Show error state
  const errorState = document.getElementById('promptErrorState');
  const errorMessageEl = document.getElementById('promptErrorMessage');
  
  errorState.style.display = 'block';
  errorMessageEl.textContent = errorMessage || 'An unexpected error occurred.';
  
  // Re-enable generate button
  document.getElementById('generatePromptBtn').disabled = false;
}

/**
 * Retry prompt generation after error
 */
function retryPromptGeneration() {
  document.getElementById('promptErrorState').style.display = 'none';
  generateImagePrompt();
}

/**
 * Copy generated prompt to clipboard
 */
async function copyPromptToClipboard() {
  const promptText = document.getElementById('generatedPromptText').value;
  
  if (!promptText.trim()) {
    return;
  }
  
  try {
    await navigator.clipboard.writeText(promptText);
    showCopyToast();
  } catch (error) {
    // Fallback for older browsers
    console.warn('Clipboard API not available, using fallback');
    fallbackCopyToClipboard(promptText);
  }
}

/**
 * Fallback copy method for older browsers
 */
function fallbackCopyToClipboard(text) {
  const textarea = document.getElementById('generatedPromptText');
  textarea.select();
  textarea.setSelectionRange(0, 99999); // For mobile devices
  
  try {
    document.execCommand('copy');
    showCopyToast();
  } catch (error) {
    console.error('Fallback copy failed:', error);
    alert('Copy failed. Please select the text manually and copy it.');
  }
}

/**
 * Show copy success toast notification
 */
function showCopyToast() {
  const toast = document.getElementById('copyToast');
  toast.style.display = 'block';
  
  // Auto-hide after 3 seconds
  setTimeout(() => {
    toast.style.display = 'none';
  }, 3000);
}

/**
 * Trap focus within modal for accessibility
 */
function trapFocus(element) {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const firstFocusable = focusableElements[0];
  const lastFocusable = focusableElements[focusableElements.length - 1];
  
  element.addEventListener('keydown', function(e) {
    if (e.key === 'Tab') {
      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          lastFocusable.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          firstFocusable.focus();
          e.preventDefault();
        }
      }
    }
  });
}

/**
 * Helper function to create article data object for different contexts
 */
function createArticleData(title, content, summary = '', url = '') {
  return {
    title: title || '',
    content: content || summary || '',
    summary: summary || '',
    url: url || ''
  };
}

/**
 * Convenience function to open modal from cluster data
 */
function openImagePromptForCluster(clusterData) {
  if (!clusterData || !clusterData.length) {
    console.warn('No cluster data provided');
    return;
  }
  
  // Use the first article's data, or combine multiple articles
  const article = clusterData[0];
  const articleData = createArticleData(
    article.title,
    article.content,
    article.summary,
    article.link || article.url
  );
  
  openImagePromptModal(articleData);
}

/**
 * Convenience function to open modal from bookmark data
 */
function openImagePromptForBookmark(bookmarkData) {
  const articleData = createArticleData(
    bookmarkData.title,
    bookmarkData.content,
    bookmarkData.summary,
    bookmarkData.url
  );
  
  openImagePromptModal(articleData);
}

// Export functions for use in other scripts
window.ImagePrompt = {
  openModal: openImagePromptModal,
  closeModal: closeImagePromptModal,
  openForCluster: openImagePromptForCluster,
  openForBookmark: openImagePromptForBookmark,
  createArticleData: createArticleData
};