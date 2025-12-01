/**
 * Data Points AI RSS Reader - Main Electron Process
 *
 * This file manages the application lifecycle, window creation,
 * Python server integration, and native Mac features.
 */

const { app, BrowserWindow, Menu, shell, dialog, ipcMain, nativeTheme, Tray } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const log = require('electron-log');
const Store = require('electron-store');
const PythonBridge = require('./python-bridge');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Initialize persistent storage with comprehensive defaults
const store = new Store({
  defaults: {
    windowBounds: { width: 1400, height: 900, x: undefined, y: undefined },
    windowMaximized: false,
    windowFullScreen: false,
    serverPort: 5005,
    darkMode: false,
    firstRun: true,
    recentArticles: [],
    lastFeedRefresh: null,
    cacheSize: 256,
    autoRefreshEnabled: false,
    autoRefreshInterval: 30, // minutes
    notificationsEnabled: false,
    minimizeToTray: false,
    startMinimized: false,
    pythonPath: null, // Custom Python path if needed
    lastVisitedPage: '/',
    exportDefaultFormat: 'md',
    exportDefaultPath: null
  },
  // Schema validation
  schema: {
    windowBounds: {
      type: 'object',
      properties: {
        width: { type: 'number', minimum: 800 },
        height: { type: 'number', minimum: 600 },
        x: { type: ['number', 'null'] },
        y: { type: ['number', 'null'] }
      }
    },
    serverPort: {
      type: 'number',
      minimum: 1024,
      maximum: 65535
    },
    cacheSize: {
      type: 'number',
      minimum: 64,
      maximum: 2048
    },
    autoRefreshInterval: {
      type: 'number',
      minimum: 5,
      maximum: 1440
    }
  }
});

// Global references
let mainWindow = null;
let pythonProcess = null;
let serverPort = store.get('serverPort');
let isQuitting = false;
let tray = null;
let unreadCount = 0;
let recentArticles = [];
let pythonBridge = null;

/**
 * Get the path to Python resources
 */
function getAppPath() {
  if (app.isPackaged) {
    // In production, Python files are in resources/app
    return path.join(process.resourcesPath, 'app');
  } else {
    // In development, go up one directory from electron folder
    return path.join(__dirname, '..');
  }
}

/**
 * Get the Python executable path
 */
function getPythonCommand() {
  const appPath = getAppPath();

  if (app.isPackaged) {
    // In production, use system Python or bundled Python
    return 'python3';
  } else {
    // In development, use venv if available
    const venvPath = path.join(appPath, 'rss_venv', 'bin', 'python');
    if (fs.existsSync(venvPath)) {
      return venvPath;
    }
    return 'python3';
  }
}

/**
 * Start the Python FastAPI server
 */
function startPythonServer() {
  return new Promise((resolve, reject) => {
    const appPath = getAppPath();
    const pythonCmd = getPythonCommand();
    const serverScript = path.join(appPath, 'server.py');

    log.info('Starting Python server...');
    log.info('App path:', appPath);
    log.info('Python command:', pythonCmd);
    log.info('Server script:', serverScript);
    log.info('Server port:', serverPort);

    // Check if server script exists
    if (!fs.existsSync(serverScript)) {
      const error = `Server script not found: ${serverScript}`;
      log.error(error);
      reject(new Error(error));
      return;
    }

    // Set environment variables
    const env = {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      TOKENIZERS_PARALLELISM: 'false',
      LOG_LEVEL: app.isPackaged ? 'INFO' : 'DEBUG'
    };

    // Start the server process
    pythonProcess = spawn(
      pythonCmd,
      [
        '-m', 'uvicorn',
        'server:app',
        '--host', '127.0.0.1',
        '--port', serverPort.toString(),
        '--log-level', app.isPackaged ? 'info' : 'debug'
      ],
      {
        cwd: appPath,
        env: env,
        stdio: ['pipe', 'pipe', 'pipe']
      }
    );

    // Handle server output
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      log.info('[Python]:', output);

      // Check if server is ready
      if (output.includes('Uvicorn running on') || output.includes('Application startup complete')) {
        log.info('Python server is ready');
        resolve();
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString().trim();
      log.warn('[Python Error]:', output);
    });

    pythonProcess.on('error', (error) => {
      log.error('Failed to start Python server:', error);
      reject(error);
    });

    pythonProcess.on('close', (code) => {
      log.info(`Python server exited with code ${code}`);
      if (!isQuitting && code !== 0 && code !== null) {
        dialog.showErrorBox(
          'Server Error',
          `The backend server stopped unexpectedly (exit code: ${code}). The application will now close.`
        );
        app.quit();
      }
    });

    // Timeout if server doesn't start in 30 seconds
    setTimeout(() => {
      if (pythonProcess && !pythonProcess.killed) {
        resolve(); // Assume it started successfully
      }
    }, 30000);
  });
}

/**
 * Stop the Python server gracefully
 */
function stopPythonServer() {
  return new Promise((resolve) => {
    if (pythonProcess && !pythonProcess.killed) {
      log.info('Stopping Python server...');

      pythonProcess.on('close', () => {
        log.info('Python server stopped');
        resolve();
      });

      // Try graceful shutdown first
      pythonProcess.kill('SIGTERM');

      // Force kill after 5 seconds
      setTimeout(() => {
        if (pythonProcess && !pythonProcess.killed) {
          log.warn('Force killing Python server');
          pythonProcess.kill('SIGKILL');
          resolve();
        }
      }, 5000);
    } else {
      resolve();
    }
  });
}

/**
 * Create the main application window
 */
function createWindow() {
  const bounds = store.get('windowBounds');
  const wasMaximized = store.get('windowMaximized');
  const startMinimized = store.get('startMinimized');

  // Window configuration
  const windowOptions = {
    width: bounds.width,
    height: bounds.height,
    minWidth: 1000,
    minHeight: 600,
    backgroundColor: '#f9fafb',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 20, y: 20 },
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: true,
      allowRunningInsecureContent: false,
      // Enable performance optimizations
      backgroundThrottling: true,
      enableWebSQL: false
    },
    show: false // Don't show until ready
  };

  // Restore window position if it was saved and is still valid
  if (bounds.x !== undefined && bounds.y !== undefined) {
    windowOptions.x = bounds.x;
    windowOptions.y = bounds.y;
  }

  mainWindow = new BrowserWindow(windowOptions);

  // Set window title
  mainWindow.setTitle('Data Points AI RSS Reader');

  // Restore maximized state
  if (wasMaximized) {
    mainWindow.maximize();
  }

  // Load the last visited page or home
  const lastPage = store.get('lastVisitedPage', '/');
  const serverUrl = `http://127.0.0.1:${serverPort}${lastPage}`;
  log.info('Loading URL:', serverUrl);
  mainWindow.loadURL(serverUrl);

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();

    // Show welcome dialog on first run
    if (store.get('firstRun')) {
      setTimeout(() => {
        dialog.showMessageBox(mainWindow, {
          type: 'info',
          title: 'Welcome to Data Points AI',
          message: 'Welcome to Data Points AI RSS Reader!',
          detail: 'This is a sophisticated AI-powered RSS reader that automatically summarizes and clusters related articles.\n\nTo get started:\n1. Add your RSS feeds or use the default feeds\n2. Configure your preferences in Settings\n3. Click "Process Feeds" to begin\n\nYour articles will be intelligently organized by topic with AI-generated summaries.',
          buttons: ['Get Started']
        });
        store.set('firstRun', false);
      }, 1000);
    }
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });

  // Handle navigation
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(`http://127.0.0.1:${serverPort}`)) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  // Track window state changes
  mainWindow.on('maximize', () => {
    store.set('windowMaximized', true);
  });

  mainWindow.on('unmaximize', () => {
    store.set('windowMaximized', false);
  });

  mainWindow.on('enter-full-screen', () => {
    store.set('windowFullScreen', true);
  });

  mainWindow.on('leave-full-screen', () => {
    store.set('windowFullScreen', false);
  });

  // Save window position and size on move/resize (debounced)
  let saveStateTimeout;
  const saveWindowState = () => {
    clearTimeout(saveStateTimeout);
    saveStateTimeout = setTimeout(() => {
      if (!mainWindow) return;

      const bounds = mainWindow.getBounds();
      const isMaximized = mainWindow.isMaximized();
      const isFullScreen = mainWindow.isFullScreen();

      // Only save bounds if not maximized or fullscreen
      if (!isMaximized && !isFullScreen) {
        store.set('windowBounds', {
          width: bounds.width,
          height: bounds.height,
          x: bounds.x,
          y: bounds.y
        });
      }

      store.set('windowMaximized', isMaximized);
      store.set('windowFullScreen', isFullScreen);
    }, 500); // Debounce by 500ms
  };

  mainWindow.on('resize', saveWindowState);
  mainWindow.on('move', saveWindowState);

  // Track navigation to save last visited page
  mainWindow.webContents.on('did-navigate', (event, url) => {
    try {
      const urlObj = new URL(url);
      if (urlObj.hostname === '127.0.0.1' && urlObj.port === serverPort.toString()) {
        store.set('lastVisitedPage', urlObj.pathname);
      }
    } catch (err) {
      // Ignore invalid URLs
    }
  });

  // Save window bounds on close
  mainWindow.on('close', (event) => {
    const minimizeToTray = store.get('minimizeToTray', false);

    if (!isQuitting && minimizeToTray) {
      event.preventDefault();
      mainWindow.hide();

      // On macOS, keep app running in dock
      if (process.platform === 'darwin') {
        app.dock.hide();
      }
      return false;
    }

    if (!isQuitting) {
      // Save final state before closing
      saveWindowState();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Open DevTools in development
  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

/**
 * Create the application menu
 */
function createMenu() {
  // Load recent articles from store on startup
  if (recentArticles.length === 0) {
    recentArticles = store.get('recentArticles', []);
  }

  // Build recent items submenu
  const recentItemsSubmenu = recentArticles.length > 0
    ? recentArticles.map(article => ({
        label: article.title.substring(0, 50) + (article.title.length > 50 ? '...' : ''),
        click: () => {
          if (mainWindow && article.link) {
            shell.openExternal(article.link);
          }
        }
      }))
    : [{ label: 'No Recent Articles', enabled: false }];

  // Add "Clear Recent Items" option
  if (recentArticles.length > 0) {
    recentItemsSubmenu.push(
      { type: 'separator' },
      {
        label: 'Clear Recent Items',
        click: () => {
          recentArticles = [];
          store.set('recentArticles', []);
          createMenu();
        }
      }
    );
  }

  const template = [
    {
      label: app.name,
      submenu: [
        {
          label: 'About Data Points AI',
          click: async () => {
            const { version } = require('./package.json');
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About Data Points AI RSS Reader',
              message: `Data Points AI RSS Reader v${version}`,
              detail: 'AI-powered RSS reader with intelligent article clustering and summarization.\n\nPowered by Anthropic Claude API\n\n© 2024 Data Points AI',
              buttons: ['OK']
            });
          }
        },
        { type: 'separator' },
        {
          label: 'Preferences...',
          accelerator: 'Cmd+,',
          click: () => {
            mainWindow.loadURL(`http://127.0.0.1:${serverPort}/`);
          }
        },
        { type: 'separator' },
        {
          label: 'Check for Updates...',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'Check for Updates',
              message: 'You are using the latest version',
              detail: 'Automatic updates will be available in a future release.',
              buttons: ['OK']
            });
          }
        },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: 'Cmd+Q',
          click: () => {
            isQuitting = true;
            app.quit();
          }
        }
      ]
    },
    {
      label: 'File',
      submenu: [
        {
          label: 'Process Feeds',
          accelerator: 'Cmd+R',
          click: () => {
            mainWindow.loadURL(`http://127.0.0.1:${serverPort}/`);
          }
        },
        {
          label: 'Summarize URL',
          accelerator: 'Cmd+N',
          click: () => {
            mainWindow.loadURL(`http://127.0.0.1:${serverPort}/summarize`);
          }
        },
        { type: 'separator' },
        {
          label: 'View Bookmarks',
          accelerator: 'Cmd+B',
          click: () => {
            mainWindow.loadURL(`http://127.0.0.1:${serverPort}/bookmarks`);
          }
        },
        { type: 'separator' },
        {
          label: 'Recent Articles',
          submenu: recentItemsSubmenu
        },
        { type: 'separator' },
        {
          label: 'Export',
          submenu: [
            {
              label: 'Export as Markdown...',
              accelerator: 'Cmd+Shift+E',
              click: async () => {
                // Request clusters data from renderer
                mainWindow.webContents.send('request-export-data', 'md');
              }
            },
            {
              label: 'Export as Plain Text...',
              click: async () => {
                mainWindow.webContents.send('request-export-data', 'txt');
              }
            },
            {
              label: 'Export as JSON...',
              click: async () => {
                mainWindow.webContents.send('request-export-data', 'json');
              }
            }
          ]
        },
        { type: 'separator' },
        {
          label: 'Close Window',
          accelerator: 'Cmd+W',
          role: 'close'
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'pasteAndMatchStyle' },
        { role: 'delete' },
        { role: 'selectAll' },
        { type: 'separator' },
        {
          label: 'Find',
          accelerator: 'Cmd+F',
          click: () => {
            // Trigger find in page
            mainWindow.webContents.send('trigger-find');
          }
        },
        {
          label: 'Find Next',
          accelerator: 'Cmd+G',
          click: () => {
            mainWindow.webContents.send('find-next');
          }
        },
        {
          label: 'Find Previous',
          accelerator: 'Cmd+Shift+G',
          click: () => {
            mainWindow.webContents.send('find-previous');
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
        { type: 'separator' },
        {
          label: 'Toggle Dark Mode',
          accelerator: 'Cmd+Shift+D',
          click: () => {
            const isDark = store.get('darkMode');
            store.set('darkMode', !isDark);
            nativeTheme.themeSource = !isDark ? 'dark' : 'light';
          }
        }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { type: 'separator' },
        { role: 'front' },
        { type: 'separator' },
        {
          label: 'Main Window',
          click: () => {
            if (mainWindow) {
              mainWindow.show();
              if (process.platform === 'darwin') {
                app.dock.show();
              }
            }
          }
        }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: async () => {
            await shell.openExternal('https://github.com/tcarmody/rss-reader');
          }
        },
        {
          label: 'Report Issue',
          click: async () => {
            await shell.openExternal('https://github.com/tcarmody/rss-reader/issues');
          }
        },
        { type: 'separator' },
        {
          label: 'View Logs',
          click: () => {
            shell.openPath(log.transports.file.getFile().path);
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

/**
 * Update dock badge with unread count
 */
function updateBadge(count) {
  unreadCount = count;
  if (process.platform === 'darwin') {
    if (count > 0) {
      app.dock.setBadge(count.toString());
    } else {
      app.dock.setBadge('');
    }
  }
}

/**
 * Add article to recent items
 */
function addRecentArticle(article) {
  // Add to beginning of array
  recentArticles.unshift(article);

  // Keep only last 10 items
  if (recentArticles.length > 10) {
    recentArticles = recentArticles.slice(0, 10);
  }

  // Rebuild menu to show updated recent items
  createMenu();

  // Save to store for persistence
  store.set('recentArticles', recentArticles);
}

/**
 * Create system tray icon and menu
 */
function createTray() {
  // Use a simple template icon for the tray (macOS will handle dark mode)
  // For now, we'll use the app icon. In production, you'd want a 16x16 template icon
  const trayIconPath = path.join(__dirname, 'assets', 'icon.icns');

  // Create tray icon
  tray = new Tray(trayIconPath);
  tray.setToolTip('Data Points AI RSS Reader');

  // Create context menu for tray
  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show App',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          if (process.platform === 'darwin') {
            app.dock.show();
          }
        }
      }
    },
    { type: 'separator' },
    {
      label: 'Process Feeds',
      click: () => {
        if (mainWindow) {
          mainWindow.loadURL(`http://127.0.0.1:${serverPort}/`);
          mainWindow.show();
        }
      }
    },
    {
      label: 'Summarize URL',
      click: () => {
        if (mainWindow) {
          mainWindow.loadURL(`http://127.0.0.1:${serverPort}/summarize`);
          mainWindow.show();
        }
      }
    },
    {
      label: 'View Bookmarks',
      click: () => {
        if (mainWindow) {
          mainWindow.loadURL(`http://127.0.0.1:${serverPort}/bookmarks`);
          mainWindow.show();
        }
      }
    },
    { type: 'separator' },
    {
      label: `Unread: ${unreadCount}`,
      enabled: false
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);

  tray.setContextMenu(contextMenu);

  // Click on tray icon shows/hides window
  tray.on('click', () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide();
        if (process.platform === 'darwin') {
          app.dock.hide();
        }
      } else {
        mainWindow.show();
        if (process.platform === 'darwin') {
          app.dock.show();
        }
      }
    }
  });
}

/**
 * Update tray menu with current unread count
 */
function updateTrayMenu() {
  if (!tray) return;

  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show App',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          if (process.platform === 'darwin') {
            app.dock.show();
          }
        }
      }
    },
    { type: 'separator' },
    {
      label: 'Process Feeds',
      click: () => {
        if (mainWindow) {
          mainWindow.loadURL(`http://127.0.0.1:${serverPort}/`);
          mainWindow.show();
        }
      }
    },
    {
      label: 'Summarize URL',
      click: () => {
        if (mainWindow) {
          mainWindow.loadURL(`http://127.0.0.1:${serverPort}/summarize`);
          mainWindow.show();
        }
      }
    },
    {
      label: 'View Bookmarks',
      click: () => {
        if (mainWindow) {
          mainWindow.loadURL(`http://127.0.0.1:${serverPort}/bookmarks`);
          mainWindow.show();
        }
      }
    },
    { type: 'separator' },
    {
      label: `Unread: ${unreadCount}`,
      enabled: false
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);

  tray.setContextMenu(contextMenu);
}

/**
 * Export articles to various formats
 */
async function exportArticles(format, clusters) {
  const { dialog } = require('electron');

  // Ask user where to save
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Export Articles',
    defaultPath: `articles-${new Date().toISOString().split('T')[0]}.${format}`,
    filters: [
      { name: format.toUpperCase(), extensions: [format] }
    ]
  });

  if (result.canceled) return;

  const filePath = result.filePath;
  let content = '';

  if (format === 'md') {
    // Markdown format
    content = '# RSS Reader Export\n\n';
    content += `Generated: ${new Date().toLocaleString()}\n\n`;

    clusters.forEach((cluster, idx) => {
      content += `## Cluster ${idx + 1}\n\n`;
      cluster.forEach(article => {
        content += `### ${article.title}\n\n`;
        content += `**Source:** ${article.feed_source}\n`;
        content += `**Published:** ${article.published}\n`;
        content += `**Link:** ${article.link}\n\n`;
        if (article.summary && article.summary.summary) {
          content += `${article.summary.summary}\n\n`;
        }
        content += '---\n\n';
      });
    });
  } else if (format === 'txt') {
    // Plain text format
    content = 'RSS Reader Export\n';
    content += `Generated: ${new Date().toLocaleString()}\n`;
    content += '='.repeat(80) + '\n\n';

    clusters.forEach((cluster, idx) => {
      content += `CLUSTER ${idx + 1}\n`;
      content += '-'.repeat(80) + '\n\n';
      cluster.forEach(article => {
        content += `${article.title}\n`;
        content += `Source: ${article.feed_source}\n`;
        content += `Published: ${article.published}\n`;
        content += `Link: ${article.link}\n\n`;
        if (article.summary && article.summary.summary) {
          content += `${article.summary.summary}\n\n`;
        }
        content += '-'.repeat(80) + '\n\n';
      });
    });
  } else if (format === 'json') {
    // JSON format
    const exportData = {
      exported: new Date().toISOString(),
      clusters: clusters
    };
    content = JSON.stringify(exportData, null, 2);
  }

  // Write file
  fs.writeFileSync(filePath, content, 'utf8');

  // Show success message
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'Export Successful',
    message: 'Articles exported successfully!',
    detail: `Saved to: ${filePath}`,
    buttons: ['OK', 'Show File']
  }).then(result => {
    if (result.response === 1) {
      shell.showItemInFolder(filePath);
    }
  });
}

/**
 * IPC handlers for renderer process communication
 */
function setupIPC() {
  ipcMain.handle('get-app-version', () => {
    return app.getVersion();
  });

  ipcMain.handle('get-server-url', () => {
    return `http://127.0.0.1:${serverPort}`;
  });

  ipcMain.handle('open-external', async (_event, url) => {
    await shell.openExternal(url);
  });

  ipcMain.handle('show-item-in-folder', async (_event, path) => {
    shell.showItemInFolder(path);
  });

  // Badge management
  ipcMain.handle('update-badge', (_event, count) => {
    updateBadge(count);
    updateTrayMenu();
    return true;
  });

  // Recent articles
  ipcMain.handle('add-recent-article', (_event, article) => {
    addRecentArticle(article);
    return true;
  });

  ipcMain.handle('get-recent-articles', () => {
    return recentArticles;
  });

  // Export functionality
  ipcMain.handle('export-articles', async (_event, format, clusters) => {
    await exportArticles(format, clusters);
    return true;
  });

  // Python bridge optimized endpoints
  ipcMain.handle('python-get-status', async () => {
    if (!pythonBridge) return null;
    return await pythonBridge.getStatus();
  });

  ipcMain.handle('python-summarize-url', async (_event, url, style) => {
    if (!pythonBridge) throw new Error('Python bridge not initialized');
    return await pythonBridge.summarizeUrl(url, style);
  });

  ipcMain.handle('python-get-bookmarks', async (_event, filters) => {
    if (!pythonBridge) return null;
    return await pythonBridge.getBookmarks(filters);
  });

  ipcMain.handle('python-add-bookmark', async (_event, bookmarkData) => {
    if (!pythonBridge) throw new Error('Python bridge not initialized');
    return await pythonBridge.addBookmark(bookmarkData);
  });

  // Settings management
  ipcMain.handle('get-settings', (_event, key) => {
    if (key) {
      return store.get(key);
    }
    // Return all settings
    return store.store;
  });

  ipcMain.handle('set-setting', (_event, key, value) => {
    store.set(key, value);
    return true;
  });

  ipcMain.handle('reset-settings', () => {
    store.clear();
    return true;
  });
}

/**
 * App lifecycle event handlers
 */
app.whenReady().then(async () => {
  try {
    // Start Python server
    await startPythonServer();

    // Initialize Python bridge for optimized communication
    pythonBridge = new PythonBridge(serverPort);
    pythonBridge.startHealthCheck();
    log.info('Python bridge initialized');

    // Create window and menu
    createWindow();
    createMenu();
    setupIPC();

    // Create system tray
    createTray();

    // Set dark mode if enabled
    if (store.get('darkMode')) {
      nativeTheme.themeSource = 'dark';
    }

    log.info('Application ready');
  } catch (error) {
    log.error('Failed to start application:', error);
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start the application:\n\n${error.message}\n\nPlease make sure Python 3 and all dependencies are installed.`
    );
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (mainWindow === null) {
    createWindow();
  } else {
    mainWindow.show();
    if (process.platform === 'darwin') {
      app.dock.show();
    }
  }
});

app.on('before-quit', () => {
  isQuitting = true;
});

app.on('will-quit', async (event) => {
  event.preventDefault();

  // Cleanup Python bridge
  if (pythonBridge) {
    pythonBridge.destroy();
    pythonBridge = null;
    log.info('Python bridge cleaned up');
  }

  await stopPythonServer();
  app.exit(0);
});

app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    isQuitting = true;
    app.quit();
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  log.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (error) => {
  log.error('Unhandled rejection:', error);
});
