/**
 * Preload script for Data Points AI RSS Reader
 *
 * This script runs in the renderer process before the web page is loaded.
 * It provides a secure bridge between the renderer and main process.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Get application version
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),

  // Get server URL
  getServerUrl: () => ipcRenderer.invoke('get-server-url'),

  // Open external link
  openExternal: (url) => ipcRenderer.invoke('open-external', url),

  // Show item in folder
  showItemInFolder: (path) => ipcRenderer.invoke('show-item-in-folder', path),

  // Platform information
  platform: process.platform,

  // Check if running in Electron
  isElectron: true
});

// Inject custom styles for native Mac feel
window.addEventListener('DOMContentLoaded', () => {
  // Add custom CSS class to body
  document.body.classList.add('electron-app');

  // Create custom styles for Mac-native look
  const style = document.createElement('style');
  style.textContent = `
    /* Mac-native app styling */
    .electron-app {
      -webkit-user-select: none;
      user-select: none;
      cursor: default;
    }

    /* Allow text selection in specific areas */
    .electron-app input,
    .electron-app textarea,
    .electron-app .article-title,
    .electron-app .article-summary,
    .electron-app p,
    .electron-app code,
    .electron-app pre {
      -webkit-user-select: text;
      user-select: text;
      cursor: text;
    }

    /* Smooth scrolling */
    .electron-app {
      scroll-behavior: smooth;
    }

    /* Mac-style scrollbars */
    .electron-app ::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }

    .electron-app ::-webkit-scrollbar-track {
      background: transparent;
    }

    .electron-app ::-webkit-scrollbar-thumb {
      background: rgba(0, 0, 0, 0.2);
      border-radius: 10px;
    }

    .electron-app ::-webkit-scrollbar-thumb:hover {
      background: rgba(0, 0, 0, 0.3);
    }

    /* Adjust for traffic light buttons */
    .electron-app .nav {
      padding-left: 80px !important;
    }

    /* Mac-native window controls */
    .electron-app .nav-container {
      position: relative;
    }

    /* Improve button hover states */
    .electron-app .button {
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .electron-app .button:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .electron-app .button:active {
      transform: translateY(0);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Sophisticated card animations */
    .electron-app .cluster {
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .electron-app .cluster:hover {
      transform: translateY(-4px);
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }

    /* Enhanced focus states for accessibility */
    .electron-app *:focus-visible {
      outline: 2px solid var(--color-primary-600);
      outline-offset: 2px;
      box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
    }

    /* Native macOS vibrancy effect */
    .electron-app .nav,
    .electron-app .card,
    .electron-app .modal-content {
      backdrop-filter: blur(20px);
    }

    /* Improve form inputs for native feel */
    .electron-app input,
    .electron-app textarea,
    .electron-app select {
      transition: all 0.2s ease;
    }

    .electron-app input:focus,
    .electron-app textarea:focus,
    .electron-app select:focus {
      transform: scale(1.01);
    }

    /* Loading states */
    .electron-app .loading {
      position: relative;
      pointer-events: none;
    }

    .electron-app .loading::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 255, 255, 0.8);
      border-radius: inherit;
      animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 0.6; }
      50% { opacity: 1; }
    }

    /* Dark mode enhancements */
    @media (prefers-color-scheme: dark) {
      .electron-app ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
      }

      .electron-app ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
      }

      .electron-app .loading::after {
        background: rgba(31, 41, 55, 0.8);
      }
    }
  `;
  document.head.appendChild(style);

  // Log that we're running in Electron
  console.log('Running in Electron');
});
