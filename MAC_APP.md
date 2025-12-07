# Data Points AI RSS Reader - Native Mac Application

This document describes the native macOS application built with Electron.

## Overview

The Mac app provides a sophisticated, native desktop experience for the Data Points AI RSS Reader. It wraps the existing FastAPI web application in an Electron shell, providing:

- **Native macOS Integration**: Menu bar, keyboard shortcuts, dock integration
- **Sophisticated UI**: Mac-native design with vibrancy effects and smooth animations
- **Python Backend**: Automatically starts and manages the FastAPI server
- **Offline Capable**: Works without an internet connection (for cached content)
- **Universal Binary**: Runs natively on both Apple Silicon and Intel Macs

## Quick Start

### Development

```bash
# 1. Install Node.js dependencies
cd electron
npm install

# 2. Ensure Python environment is set up
cd ..
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 4. Run the app
cd electron
npm run dev
```

### Building

```bash
cd electron

# Build universal binary (recommended)
make build-universal

# Or use npm scripts directly
npm run build:universal

# Output will be in electron/dist/
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│         Electron Main Process           │
│  - Window management                    │
│  - Python server lifecycle              │
│  - Menu and shortcuts                   │
│  - IPC communication                    │
└─────────────┬───────────────────────────┘
              │
              │ spawns
              │
┌─────────────▼───────────────────────────┐
│        Python FastAPI Server            │
│  - RSS feed processing                  │
│  - AI summarization                     │
│  - Article clustering                   │
│  - Bookmark management                  │
└─────────────┬───────────────────────────┘
              │
              │ serves
              │
┌─────────────▼───────────────────────────┐
│      Electron Renderer Process          │
│  - Loads web UI via localhost           │
│  - Enhanced with Mac-native styles      │
│  - Secure IPC via preload script        │
└─────────────────────────────────────────┘
```

### Key Files

- **`electron/main.js`**: Main process - manages app lifecycle and Python server
- **`electron/preload.js`**: Preload script - secure IPC bridge
- **`electron/package.json`**: Configuration and build settings
- **`static/css/electron.css`**: Mac-native UI enhancements
- **`templates/base.html`**: Updated to load Electron styles

## Features

### Native macOS Integration

1. **Menu Bar**
   - Application menu with About, Preferences, Services
   - File menu for common actions
   - Edit menu with standard shortcuts
   - View menu with zoom and dark mode
   - Window menu for window management
   - Help menu with documentation links

2. **Keyboard Shortcuts**
   - `Cmd+,` - Preferences
   - `Cmd+N` - New URL summary
   - `Cmd+R` - Refresh/Process feeds
   - `Cmd+B` - Bookmarks
   - `Cmd+F` - Find
   - `Cmd+Q` - Quit
   - And all standard macOS shortcuts

3. **Window Management**
   - Traffic light buttons (red, yellow, green)
   - Remembers window size and position
   - Hide on close (stays in dock)
   - Full screen support
   - Multiple displays support

4. **System Integration**
   - Dock icon and badge
   - Dark mode support (automatic)
   - Native notifications (future)
   - Touch Bar support (future)

### UI Enhancements

The Electron version includes sophisticated Mac-native styling:

- **Vibrancy Effects**: Translucent windows with backdrop blur
- **Smooth Animations**: 60fps animations with hardware acceleration
- **Native Typography**: SF Pro Display and SF Mono fonts
- **Depth and Shadows**: Layered shadows for visual hierarchy
- **Hover States**: Subtle interactive feedback
- **Focus Management**: Clear focus indicators for accessibility

### Security Features

- **Context Isolation**: Renderer process is isolated from Node.js
- **No Node Integration**: Renderer can't access Node.js APIs directly
- **Secure IPC**: Communication via preload script only
- **Content Security**: Prevents XSS and code injection
- **Sandboxing**: Renderer processes are sandboxed
- **Entitlements**: Minimal permissions requested

## Build System

### Build Targets

```bash
# Universal Binary (Apple Silicon + Intel)
make build-universal
npm run build:universal

# Apple Silicon only
make build-arm64
npm run build:arm64

# Intel only
make build-x64
npm run build:x64

# All architectures
make build
npm run build
```

### Build Output

After building, you'll find in `electron/dist/`:

- **DMG**: `Data Points AI RSS Reader-1.0.0-mac-universal.dmg`
  - Installer disk image for distribution
  - Includes Applications folder shortcut
  - Branded with app icon

- **ZIP**: `Data Points AI RSS Reader-1.0.0-mac-universal.zip`
  - Portable archive
  - Extract and run
  - Good for testing

- **APP**: `mac-universal/Data Points AI RSS Reader.app`
  - Unsigned application bundle
  - Can be moved to Applications
  - Requires Gatekeeper approval on first run

### Build Configuration

Key settings in `electron/package.json`:

```json
{
  "build": {
    "appId": "ai.datapoints.rss-reader",
    "category": "public.app-category.news",
    "mac": {
      "target": ["dmg", "zip"],
      "icon": "assets/icon.icns",
      "darkModeSupport": true,
      "minimumSystemVersion": "10.13.0"
    }
  }
}
```

## Icon Creation

### Automatic Icon Generation

```bash
cd electron

# Create placeholder icon (blue square with "DP")
./create-icons.sh

# Or provide custom icon
./create-icons.sh path/to/icon-1024x1024.png
```

### Manual Icon Creation

1. Create a 1024x1024px PNG image
2. Save as `electron/icon-source.png`
3. Run: `make icon`
4. Output: `electron/assets/icon.icns`

### Icon Requirements

- **Format**: PNG, 1024x1024px minimum
- **Design**: Simple, recognizable at small sizes
- **Style**: Flat design, no shadows (macOS adds them)
- **Colors**: Vibrant but not too bright
- **Export**: Use iconutil (included with macOS)

## Distribution

### For Personal Use

1. Build the app: `make build-universal`
2. Open the DMG in `dist/`
3. Drag to Applications
4. On first run:
   - Right-click → Open
   - Or System Preferences → Security → Open Anyway

### For Public Distribution

#### Code Signing

Required for distribution to prevent Gatekeeper warnings:

```bash
# 1. Get Apple Developer ID certificate
# 2. Export certificate to .p12 file
# 3. Set environment variables
export CSC_LINK=/path/to/certificate.p12
export CSC_KEY_PASSWORD=certificate_password

# 4. Build with signing
npm run build:universal
```

#### Notarization

Required for macOS 10.15+ to avoid warnings:

```bash
# After building and signing
xcrun notarytool submit \
  "dist/Data Points AI RSS Reader-1.0.0-mac.dmg" \
  --apple-id "your@email.com" \
  --password "app-specific-password" \
  --team-id "TEAM_ID" \
  --wait

# Staple the ticket
xcrun stapler staple "dist/Data Points AI RSS Reader.app"
```

#### App Store Distribution

Not currently configured, but possible with:

1. Mac App Store entitlements
2. Sandbox compliance
3. App Store Connect setup
4. Review process

## Development

### Project Structure

```
electron/
├── main.js                    # Main process
├── preload.js                 # Preload script
├── package.json              # Electron config
├── entitlements.mac.plist    # macOS permissions
├── build.sh                  # Build script
├── create-icons.sh           # Icon generator
├── Makefile                  # Build system
├── README.md                 # Documentation
└── assets/                   # Icons and resources
    └── icon.icns

../ (parent directory - Python app)
├── server.py                 # FastAPI backend
├── static/
│   ├── css/
│   │   ├── styles.css       # Base styles
│   │   └── electron.css     # Mac-native styles
│   └── js/
├── templates/                # HTML templates
└── [other Python files]
```

### Adding Features

#### Main Process Features

Add to `electron/main.js`:

```javascript
// Example: Add menu item
{
  label: 'My Feature',
  accelerator: 'Cmd+Shift+F',
  click: () => {
    // Your code here
  }
}
```

#### UI Enhancements

Add to `static/css/electron.css`:

```css
.electron-app .my-feature {
  /* Mac-native styling */
  backdrop-filter: blur(20px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
```

#### IPC Communication

In `electron/main.js`:

```javascript
ipcMain.handle('my-feature', async (event, data) => {
  // Handle IPC call
  return result;
});
```

In `electron/preload.js`:

```javascript
contextBridge.exposeInMainWorld('electronAPI', {
  myFeature: (data) => ipcRenderer.invoke('my-feature', data)
});
```

### Debugging

#### Development Tools

- **Main Process**: Console logs in terminal
- **Renderer Process**: Chrome DevTools (auto-opens in dev mode)
- **Python Server**: Logs in Electron console

#### Production Logs

```bash
# View logs
open ~/Library/Logs/Data\ Points\ AI\ RSS\ Reader/

# Or from app
Menu → Help → View Logs
```

#### Common Issues

1. **Python server won't start**
   - Check Python 3.11+ is installed
   - Verify virtual environment exists
   - Check API key in .env

2. **Build fails**
   - Clear cache: `make dist-clean && make install`
   - Check Node.js version (18+)
   - Verify Python dependencies

3. **App won't open**
   - Check Gatekeeper: Right-click → Open
   - Verify architecture matches system
   - Check Console.app for crash logs

## Performance

### Optimization Strategies

1. **Lazy Loading**
   - CSS loaded only when needed
   - Features initialized on demand
   - Cache warming for frequently accessed data

2. **Hardware Acceleration**
   - GPU-accelerated animations
   - Composited layers for smooth scrolling
   - Optimized rendering pipeline

3. **Memory Management**
   - Tiered caching (memory → disk)
   - Automatic cleanup of old data
   - Lazy loading of article content

4. **Startup Time**
   - Parallel initialization
   - Deferred non-critical tasks
   - Persistent state between launches

### Benchmarks

- **App Launch**: ~2-3 seconds
- **Python Server Start**: ~5-10 seconds
- **Article Processing**: 50-100 articles/minute
- **Memory Usage**: ~200-300 MB (idle)
- **Package Size**: ~150-200 MB (universal binary)

## Future Enhancements

### Planned Features

- [ ] Menu bar app mode (persistent in menu bar)
- [ ] Native notifications for new articles
- [ ] Touch Bar support
- [ ] Siri Shortcuts integration
- [ ] iCloud sync for bookmarks
- [ ] Share extension
- [ ] Quick Look plugin
- [ ] Spotlight integration
- [ ] Auto-update support (electron-updater)

### Potential Improvements

- [ ] Native article reader view
- [ ] Offline article saving
- [ ] Multiple account support
- [ ] Export to various formats
- [ ] Integration with read-later services
- [ ] Custom themes
- [ ] Plugin system
- [ ] RSS feed discovery

## Technical Details

### Requirements

- **macOS**: 10.13 (High Sierra) or later
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 500 MB for app + cache
- **Network**: Required for API calls and feed fetching

### Dependencies

- **Electron**: 28.x
- **Node.js**: 18.x or higher
- **Python**: 3.11 or higher
- **Anthropic API**: Claude 3.5 Sonnet/Haiku

### Technologies

- **Framework**: Electron
- **Backend**: FastAPI (Python)
- **AI**: Anthropic Claude API
- **Database**: SQLite
- **Cache**: Tiered (memory + disk)
- **Packaging**: electron-builder

## Support

### Documentation

- **Main README**: `/README.md` - Project overview
- **Claude Guide**: `/CLAUDE.md` - Development guide
- **Electron README**: `/electron/README.md` - Detailed app docs
- **This Document**: `/MAC_APP.md` - Mac app specifics

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions
- **Logs**: Check app logs for debugging

### Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (especially Mac app features)
5. Submit a pull request

## License

MIT License - see LICENSE file

## Credits

- **Electron**: Cross-platform desktop apps
- **Anthropic**: Claude AI models
- **FastAPI**: Modern Python web framework
- **electron-builder**: Build and packaging
- **Icons**: macOS system icons and custom artwork

---

Built with ❤️ for Mac users who love AI-powered RSS reading
