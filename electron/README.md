# Data Points AI RSS Reader - Mac Application

A sophisticated native Mac application for intelligent RSS feed reading, powered by AI.

## Features

- **Native Mac Experience**: Built with Electron for a truly native macOS feel
- **AI-Powered Summarization**: Automatic article summarization using Claude AI
- **Intelligent Clustering**: Groups related articles by topic
- **Bookmark Management**: Save and organize articles for later reading
- **Dark Mode**: Full support for macOS dark mode
- **Keyboard Shortcuts**: Native Mac keyboard shortcuts throughout

## Requirements

- macOS 10.13 (High Sierra) or later
- Python 3.11 or higher
- Node.js 18 or higher
- Anthropic API key (for Claude AI)

## Development Setup

### 1. Install Dependencies

```bash
# Install Node.js dependencies
cd electron
npm install

# Install Python dependencies (from parent directory)
cd ..
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install Playwright browser (optional, for paywall bypass)
python -m playwright install chromium
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### 3. Run in Development Mode

```bash
cd electron
npm run dev
```

## Building the Application

### Build for Your Architecture

```bash
# Build for current architecture
npm run build

# Build Universal Binary (Apple Silicon + Intel)
npm run build:universal

# Build for Apple Silicon only
npm run build:arm64

# Build for Intel only
npm run build:x64
```

### Build Output

The built application will be in `electron/dist/`:

- `.dmg` - Installer disk image
- `.zip` - Portable application archive
- `.app` - Unsigned application bundle (in `mac` or `mac-universal` folder)

## Installation

### From DMG

1. Open the `.dmg` file
2. Drag "Data Points AI RSS Reader" to Applications
3. Launch from Applications or Spotlight

### First Run

On first run, you may see a security warning. To allow the app:

1. Go to System Preferences → Security & Privacy
2. Click "Open Anyway" for Data Points AI RSS Reader
3. Or right-click the app and select "Open"

## Keyboard Shortcuts

### Application

- `Cmd+,` - Preferences
- `Cmd+Q` - Quit application
- `Cmd+W` - Close window
- `Cmd+H` - Hide window
- `Cmd+M` - Minimize window

### Navigation

- `Cmd+R` - Process feeds / Refresh
- `Cmd+N` - Summarize new URL
- `Cmd+B` - View bookmarks
- `Cmd+F` - Search/Find in page

### View

- `Cmd+Shift+D` - Toggle dark mode
- `Cmd++` - Zoom in
- `Cmd+-` - Zoom out
- `Cmd+0` - Reset zoom
- `Cmd+Ctrl+F` - Toggle fullscreen

### Editing

- `Cmd+Z` - Undo
- `Cmd+Shift+Z` - Redo
- `Cmd+X` - Cut
- `Cmd+C` - Copy
- `Cmd+V` - Paste
- `Cmd+A` - Select all

## Features Guide

### Processing RSS Feeds

1. Click "Home / Settings" in the navigation bar
2. Either:
   - Add custom feed URLs in the "Custom Feeds" tab
   - Use default feeds from `rss_feeds.txt`
3. Click "Process Feeds"
4. Wait for articles to be fetched, summarized, and clustered

### Summarizing Individual URLs

1. Click "Summarize URL" in the navigation
2. Enter the article URL
3. Click "Summarize"
4. View the AI-generated summary

### Managing Bookmarks

1. Click the bookmark icon on any article
2. View all bookmarks by clicking "Bookmarks" in navigation
3. Search, filter, and organize your saved articles

### Customizing Settings

1. Open Preferences (Cmd+,)
2. Configure:
   - Feed processing settings
   - Clustering parameters
   - Summary style preferences
   - Paywall bypass (use responsibly)

## Troubleshooting

### Application Won't Start

- Check that Python 3.11+ is installed: `python3 --version`
- Ensure all dependencies are installed
- Check the logs: Menu → Help → View Logs

### Python Server Errors

- Verify ANTHROPIC_API_KEY is set in `.env`
- Check that all Python packages are installed
- Try running the server manually: `./run_server.sh`

### Build Errors

- Clear build cache: `rm -rf electron/dist electron/node_modules`
- Reinstall dependencies: `npm install`
- Try building for specific architecture instead of universal

### Performance Issues

- Reduce batch size in settings
- Disable paywall bypass if not needed
- Clear cache: Delete `./summary_cache` and `./cluster_cache`

## Architecture

### Application Structure

```
electron/
├── main.js              # Main Electron process
├── preload.js           # Preload script for security
├── package.json         # Electron configuration
├── entitlements.mac.plist  # macOS entitlements
└── assets/              # Application icons and resources

../ (parent directory)
├── server.py            # FastAPI backend
├── static/              # Web UI assets
├── templates/           # HTML templates
├── api/                 # API utilities
├── cache/               # Caching system
├── clustering/          # Article clustering
├── reader/              # RSS feed reader
├── summarization/       # AI summarization
└── services/            # Business logic
```

### Technology Stack

- **Frontend**: HTML, CSS, JavaScript (existing web UI)
- **Desktop**: Electron 28
- **Backend**: FastAPI (Python)
- **AI**: Anthropic Claude API
- **Database**: SQLite (for bookmarks)
- **Packaging**: electron-builder

### Security Features

- Context isolation enabled
- Node integration disabled in renderer
- Secure IPC communication via preload script
- Sandboxed renderer processes
- Entitlements for minimum required permissions

## Development

### Project Structure

The Electron app wraps the existing FastAPI backend:

1. **Main Process** (`main.js`): Manages app lifecycle and Python server
2. **Preload Script** (`preload.js`): Secure bridge to renderer
3. **Renderer**: Loads the FastAPI web UI in a BrowserWindow

### Adding Features

1. **Mac-native features**: Add to `main.js`
2. **UI enhancements**: Modify web templates or add to `preload.js`
3. **Backend features**: Modify Python files as normal

### Debugging

- Development mode includes Chrome DevTools
- Main process logs: Check console output
- Python server logs: Check electron logs
- Production logs: Menu → Help → View Logs

## Building for Distribution

### Code Signing (Optional)

For distribution outside the Mac App Store:

1. Get an Apple Developer account
2. Create a Developer ID certificate
3. Add signing configuration to `package.json`
4. Set environment variables:
   ```bash
   export CSC_IDENTITY_AUTO_DISCOVERY=true
   export CSC_LINK=/path/to/certificate.p12
   export CSC_KEY_PASSWORD=certificate_password
   ```

### Notarization (Optional)

For distribution to end users:

1. Build and sign the app
2. Submit for notarization:
   ```bash
   xcrun notarytool submit "dist/Data Points AI RSS Reader-1.0.0-mac.dmg" \
     --apple-id "your@email.com" \
     --password "app-specific-password" \
     --team-id "TEAM_ID" \
     --wait
   ```
3. Staple the notarization:
   ```bash
   xcrun stapler staple "dist/Data Points AI RSS Reader.app"
   ```

## License

MIT License - see parent directory LICENSE file

## Support

- **Documentation**: https://github.com/tcarmody/rss-reader
- **Issues**: https://github.com/tcarmody/rss-reader/issues
- **Discussions**: https://github.com/tcarmody/rss-reader/discussions

## Credits

- Built with [Electron](https://www.electronjs.org/)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- UI based on the FastAPI web interface
