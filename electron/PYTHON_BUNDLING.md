# Python Bundling Guide for Mac App

This guide explains how to bundle Python with the Electron app for truly standalone distribution.

## Current Architecture

Currently, the app assumes Python 3.11+ is installed on the user's system. For better user experience, we can bundle Python directly with the app.

## Option 1: Standalone Python (Recommended)

Use `python-build-standalone` to include a complete Python runtime.

### Setup

```bash
cd electron

# Download standalone Python for macOS
curl -L https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.11.10%2B20241016-aarch64-apple-darwin-install_only.tar.gz -o python-macos-arm64.tar.gz

# For Intel Macs
curl -L https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.11.10%2B20241016-x86_64-apple-darwin-install_only.tar.gz -o python-macos-x64.tar.gz

# Extract
mkdir -p resources/python-arm64
mkdir -p resources/python-x64
tar -xzf python-macos-arm64.tar.gz -C resources/python-arm64
tar -xzf python-macos-x64.tar.gz -C resources/python-x64
```

### Update package.json

Add to `extraResources`:

```json
{
  "from": "resources/python-${arch}",
  "to": "python"
}
```

### Update main.js

```javascript
function getPythonCommand() {
  const appPath = getAppPath();

  if (app.isPackaged) {
    // Use bundled Python
    const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
    const pythonPath = path.join(process.resourcesPath, 'python', 'bin', 'python3');

    if (fs.existsSync(pythonPath)) {
      return pythonPath;
    }
  }

  // Fallback to system Python
  return 'python3';
}
```

### Install Dependencies at Build Time

Create `electron/scripts/install-python-deps.js`:

```javascript
const { execSync } = require('child_process');
const path = require('path');

module.exports = async function(context) {
  const pythonPath = path.join(context.appOutDir, 'Contents', 'Resources', 'python', 'bin', 'python3');
  const requirementsPath = path.join(context.appOutDir, 'Contents', 'Resources', 'app', 'requirements.txt');

  if (fs.existsSync(pythonPath) && fs.existsSync(requirementsPath)) {
    console.log('Installing Python dependencies...');
    execSync(`${pythonPath} -m pip install -r ${requirementsPath}`, {
      stdio: 'inherit'
    });
  }
};
```

Add to package.json:

```json
{
  "build": {
    "afterPack": "./scripts/after-pack.js",
    "afterSign": "./scripts/install-python-deps.js"
  }
}
```

## Option 2: PyInstaller (Alternative)

Bundle Python app as a standalone executable.

### Setup

```bash
cd ..  # Project root
pip install pyinstaller

# Create spec file
pyinstaller --name rss-server \
  --onefile \
  --hidden-import anthropic \
  --hidden-import fastapi \
  --hidden-import uvicorn \
  --add-data "templates:templates" \
  --add-data "static:static" \
  server.py
```

### Update package.json

```json
{
  "extraResources": [
    {
      "from": "../dist/rss-server",
      "to": "app/rss-server"
    },
    {
      "from": "../templates",
      "to": "app/templates"
    },
    {
      "from": "../static",
      "to": "app/static"
    }
  ]
}
```

### Update main.js

```javascript
function startPythonServer() {
  const appPath = getAppPath();

  if (app.isPackaged) {
    // Use bundled executable
    const serverPath = path.join(process.resourcesPath, 'app', 'rss-server');

    pythonProcess = spawn(serverPath, [
      '--host', '127.0.0.1',
      '--port', serverPort.toString()
    ], {
      cwd: path.join(process.resourcesPath, 'app'),
      stdio: ['pipe', 'pipe', 'pipe']
    });
  } else {
    // Development mode
    const pythonCmd = getPythonCommand();
    const serverScript = path.join(appPath, 'server.py');

    pythonProcess = spawn(pythonCmd, [
      '-m', 'uvicorn',
      'server:app',
      '--host', '127.0.0.1',
      '--port', serverPort.toString()
    ], {
      cwd: appPath,
      stdio: ['pipe', 'pipe', 'pipe']
    });
  }

  // ... rest of the code
}
```

## Option 3: Virtual Environment (Current)

The simplest option - include a pre-built virtual environment.

### Setup

```bash
cd ..  # Project root

# Create clean venv
python3 -m venv build_venv
source build_venv/bin/activate

# Install minimal dependencies
pip install -r essential_requirements.txt

# Deactivate
deactivate
```

### Update package.json

```json
{
  "extraResources": [
    {
      "from": "../build_venv",
      "to": "app/venv",
      "filter": [
        "**/*",
        "!**/__pycache__",
        "!**/*.pyc",
        "!**/*.pyo"
      ]
    }
  ]
}
```

### Update main.js

```javascript
function getPythonCommand() {
  const appPath = getAppPath();

  if (app.isPackaged) {
    const venvPython = path.join(process.resourcesPath, 'app', 'venv', 'bin', 'python');

    if (fs.existsSync(venvPython)) {
      return venvPython;
    }
  }

  // Fallback
  return 'python3';
}
```

## Comparison

| Method | Pros | Cons | Size Impact |
|--------|------|------|-------------|
| **Standalone Python** | Complete control, portable | Large download, complex setup | +150-200 MB |
| **PyInstaller** | Single executable, fast | Build complexity, debugging harder | +100-150 MB |
| **Virtual Environment** | Simple, reliable | Architecture-specific | +80-120 MB |

## Recommended Approach

For the Mac app, I recommend **Option 1 (Standalone Python)**:

1. Best user experience - no Python installation required
2. Reproducible builds
3. Version control
4. Universal binary support (separate bundles for ARM/Intel)

## Build Size Optimizations

### 1. Exclude Large Libraries

If using sentence-transformers, models can be huge. Options:

```bash
# Use minimal dependencies
pip install -r essential_requirements.txt

# Download models on first run instead of bundling
```

### 2. Compress Python Files

```bash
# In after-pack.js
const { execSync } = require('child_process');

// Compile Python files to bytecode
execSync(`find "${resourcesPath}/app" -name "*.py" -exec python3 -m py_compile {} \\;`);

// Remove source files (optional)
execSync(`find "${resourcesPath}/app" -name "*.py" -delete`);
```

### 3. Strip Unnecessary Files

Already implemented in `after-pack.js`:
- Remove `__pycache__`
- Remove `.pyc` files
- Remove documentation
- Remove tests

### 4. Use ASAR Archive

Enabled by default in package.json:
```json
{
  "asar": true
}
```

This compresses the Electron app files into a single archive.

## Expected Final Sizes

With optimizations:

- **Minimal** (current, no Python): ~80 MB
- **With venv**: ~150 MB
- **With standalone Python**: ~220 MB
- **With PyInstaller**: ~180 MB

## Installation Script

Create `electron/scripts/prepare-python.sh`:

```bash
#!/bin/bash
set -e

echo "Preparing Python for bundling..."

# Choose your option here
OPTION="standalone"  # or "pyinstaller" or "venv"

case $OPTION in
  standalone)
    echo "Downloading standalone Python..."
    # Download and extract as shown above
    ;;

  pyinstaller)
    echo "Building PyInstaller executable..."
    cd ..
    pyinstaller rss-server.spec
    ;;

  venv)
    echo "Creating virtual environment..."
    cd ..
    python3 -m venv build_venv
    source build_venv/bin/activate
    pip install -r essential_requirements.txt
    deactivate
    ;;
esac

echo "Python preparation complete!"
```

Make it executable:

```bash
chmod +x electron/scripts/prepare-python.sh
```

Run before building:

```bash
cd electron
./scripts/prepare-python.sh
npm run build:universal
```

## Testing

```bash
# Build the app
npm run build:universal

# Open the built app
open dist/mac-universal/Data\ Points\ AI\ RSS\ Reader.app

# Check that Python runs correctly
# Look for the Python server startup in logs
```

## Troubleshooting

### Python Not Found

```javascript
// Add to getPythonCommand()
const paths = [
  path.join(process.resourcesPath, 'python', 'bin', 'python3'),
  path.join(process.resourcesPath, 'app', 'venv', 'bin', 'python'),
  '/usr/local/bin/python3',
  '/opt/homebrew/bin/python3',
  'python3'
];

for (const pythonPath of paths) {
  if (fs.existsSync(pythonPath)) {
    return pythonPath;
  }
}

// Try command line
return 'python3';
```

### Dependencies Missing

Install at runtime if needed:

```javascript
async function ensureDependencies() {
  const pythonCmd = getPythonCommand();
  const requirementsPath = path.join(getAppPath(), 'requirements.txt');

  try {
    // Check if dependencies are installed
    const result = execSync(`${pythonCmd} -c "import anthropic"`);
  } catch (error) {
    // Install dependencies
    log.info('Installing Python dependencies...');
    execSync(`${pythonCmd} -m pip install -r ${requirementsPath}`, {
      stdio: 'inherit'
    });
  }
}
```

### Size Too Large

1. Use `essential_requirements.txt` instead of `requirements.txt`
2. Remove unused models/data
3. Compress with UPX (for executables)
4. Use lazy loading for large libraries

## Future Improvements

1. **Auto-updater for Python**: Update Python dependencies without reinstalling app
2. **Model downloading**: Download ML models on first run instead of bundling
3. **Differential updates**: Only update changed Python files
4. **Cloud dependencies**: Keep heavy processing server-side

---

For questions or issues, see the main README or open an issue.
