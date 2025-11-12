# Quick Start Guide - Mac App

Get your Data Points AI RSS Reader Mac app running in 5 minutes!

## Prerequisites Check

```bash
# Check Node.js (need 18+)
node --version

# Check Python (need 3.11+)
python3 --version

# Check npm
npm --version
```

Don't have these? Install:
- **Node.js**: Download from [nodejs.org](https://nodejs.org/)
- **Python 3.11**: Download from [python.org](https://www.python.org/)

## Installation Steps

### 1. Install Node Dependencies

```bash
cd rss-reader/electron
npm install
```

This installs Electron and build tools (~150 MB, takes 1-2 minutes).

### 2. Set Up Python Environment

```bash
cd ..  # Back to project root
./run_server.sh --help
```

This creates a virtual environment and installs Python dependencies (~500 MB, takes 3-5 minutes).

### 3. Configure API Key

Create `.env` file in project root:

```bash
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

Get your API key from: https://console.anthropic.com/

### 4. Run the App

```bash
cd electron
npm run dev
```

The app will:
1. Start the Python server
2. Open a window
3. Load the web interface

## First Time Setup

When the app opens:

1. **Configure Settings**
   - Set batch size (default: 25)
   - Set per-feed limit (default: 25)
   - Enable/disable paywall bypass

2. **Add Feeds**
   - Use default feeds (in `rss_feeds.txt`)
   - Or add custom feeds

3. **Process Articles**
   - Click "Process Default Feeds" or "Process Custom Feeds"
   - Wait for articles to be fetched and summarized
   - View clustered results

## Building the App

### Quick Build (Universal)

```bash
cd electron
make build-universal
```

Or:

```bash
npm run build:universal
```

Output: `electron/dist/Data Points AI RSS Reader-1.0.0-mac-universal.dmg`

### Install Built App

1. Open the `.dmg` file
2. Drag app to Applications folder
3. Launch from Applications or Spotlight
4. On first run: Right-click → Open (to bypass Gatekeeper)

## Common Issues

### "Python not found"

**Solution**: Install Python 3.11+ from python.org

### "Module not found"

**Solution**: Install Python dependencies:
```bash
cd rss-reader  # Project root
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### "API key invalid"

**Solution**: Check your `.env` file has correct key:
```bash
cat .env
# Should show: ANTHROPIC_API_KEY=sk-...
```

### "Port already in use"

**Solution**: Change port in `electron/main.js`:
```javascript
serverPort: 5006  // Change from 5005
```

### "Build fails"

**Solution**: Clean and rebuild:
```bash
make dist-clean
make install
make build-universal
```

## Development Workflow

### Edit Python Code

1. Edit any `.py` file
2. Restart the app (`Cmd+Q`, then relaunch)
3. Changes will be reflected

### Edit Templates/CSS

1. Edit HTML/CSS files
2. Reload page (`Cmd+R`)
3. Changes appear immediately

### Edit Electron Code

1. Edit `main.js` or `preload.js`
2. Restart the app
3. Changes take effect

### Debug Mode

```bash
npm run dev
```

This opens Chrome DevTools automatically.

## Useful Commands

```bash
# Development
npm run dev              # Run with DevTools
npm start               # Run without DevTools

# Building
make build-universal    # Universal binary (recommended)
make build-arm64       # Apple Silicon only
make build-x64         # Intel only

# Maintenance
make clean             # Remove build artifacts
make dist-clean        # Remove everything
make icon              # Generate app icon

# Help
make                   # Show all commands
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+,` | Preferences |
| `Cmd+N` | New URL Summary |
| `Cmd+R` | Refresh/Process Feeds |
| `Cmd+B` | Bookmarks |
| `Cmd+F` | Find |
| `Cmd+W` | Close Window |
| `Cmd+Q` | Quit App |
| `Cmd+Shift+D` | Toggle Dark Mode |

## Next Steps

1. **Customize Icon**: Place `icon-source.png` (1024x1024) in `electron/`, run `make icon`
2. **Add Feeds**: Edit `rss_feeds.txt` or add via UI
3. **Configure Clustering**: Adjust settings in Home page
4. **Explore Bookmarks**: Save interesting articles
5. **Share**: Build and share the `.dmg` with others

## Getting Help

- **Documentation**: See `MAC_APP.md` for detailed info
- **Issues**: https://github.com/tcarmody/rss-reader/issues
- **Logs**: Menu → Help → View Logs

## Success Checklist

- [ ] Node.js 18+ installed
- [ ] Python 3.11+ installed
- [ ] Dependencies installed (`npm install` in electron/)
- [ ] Python packages installed (`./run_server.sh`)
- [ ] API key configured (`.env` file)
- [ ] App runs in dev mode (`npm run dev`)
- [ ] Articles process successfully
- [ ] Build creates `.dmg` file

If all checked, you're ready to use and build the app! 🎉

---

**Still stuck?** Open an issue on GitHub with:
- Your operating system and version
- Node.js and Python versions
- Error messages or logs
- What you've tried so far

We're here to help!
