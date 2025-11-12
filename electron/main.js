/**
 * Data Points AI RSS Reader - Main Electron Process
 *
 * This file manages the application lifecycle, window creation,
 * Python server integration, and native Mac features.
 */

const { app, BrowserWindow, Menu, shell, dialog, ipcMain, nativeTheme } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const log = require('electron-log');
const Store = require('electron-store');

// Configure logging
log.transports.file.level = 'info';
log.transports.console.level = 'debug';

// Initialize persistent storage
const store = new Store({
  defaults: {
    windowBounds: { width: 1400, height: 900 },
    serverPort: 5005,
    darkMode: false,
    firstRun: true
  }
});

// Global references
let mainWindow = null;
let pythonProcess = null;
let serverPort = store.get('serverPort');
let isQuitting = false;

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

  mainWindow = new BrowserWindow({
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
      allowRunningInsecureContent: false
    },
    show: false // Don't show until ready
  });

  // Set window title
  mainWindow.setTitle('Data Points AI RSS Reader');

  // Load the application
  const serverUrl = `http://127.0.0.1:${serverPort}`;
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

  // Save window bounds on close
  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();

      // On macOS, keep app running in dock
      if (process.platform === 'darwin') {
        app.dock.hide();
      }
      return false;
    }

    // Save window bounds
    const bounds = mainWindow.getBounds();
    store.set('windowBounds', bounds);
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
            mainWindow.webContents.send('search-focus');
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
 * IPC handlers for renderer process communication
 */
function setupIPC() {
  ipcMain.handle('get-app-version', () => {
    return app.getVersion();
  });

  ipcMain.handle('get-server-url', () => {
    return `http://127.0.0.1:${serverPort}`;
  });

  ipcMain.handle('open-external', async (event, url) => {
    await shell.openExternal(url);
  });

  ipcMain.handle('show-item-in-folder', async (event, path) => {
    shell.showItemInFolder(path);
  });
}

/**
 * App lifecycle event handlers
 */
app.whenReady().then(async () => {
  try {
    // Start Python server
    await startPythonServer();

    // Create window and menu
    createWindow();
    createMenu();
    setupIPC();

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
