# Xcode Project Setup - Step by Step

This guide walks you through creating the Xcode project from scratch.

## Quick Setup (5 minutes)

### 1. Open Xcode and Create Project

```bash
# Open Xcode
open -a Xcode
```

**In Xcode:**
1. Choose "Create New Project" (or File → New → Project)
2. Select **macOS** tab
3. Choose **App** template
4. Click **Next**

### 2. Configure Project

Fill in the details:

- **Product Name:** `Data Points AI`
- **Team:** (select your team or "None")
- **Organization Identifier:** `com.datapointsai`
- **Bundle Identifier:** (auto-fills as `com.datapointsai.Data-Points-AI`)
- **Interface:** SwiftUI
- **Language:** Swift
- **Storage:** None (uncheck Core Data)
- **Include Tests:** (optional, can uncheck)

Click **Next**

### 3. Choose Location

- Navigate to: `/Users/timcarmody/workspace/rss-reader/macos-app`
- **Important:** Uncheck "Create Git repository" (project already has one)
- Click **Create**

### 4. Delete Auto-Generated Files

Xcode will create some default files we don't need:

**In Project Navigator (left sidebar):**
1. Right-click `ContentView.swift` → Delete → Move to Trash
2. Right-click `Data_Points_AIApp.swift` → Delete → Move to Trash

### 5. Add Our Swift Files

**Drag and drop ALL .swift files into Xcode:**

1. Open Finder and navigate to `/Users/timcarmody/workspace/rss-reader/macos-app`
2. Select all `.swift` files:
   - `DataPointsAIApp.swift`
   - `AppDelegate.swift`
   - `PythonServerManager.swift`
   - `ContentView.swift`
   - `AppState.swift`
   - `SettingsView.swift`

3. Drag them into Xcode's Project Navigator
4. In the dialog that appears:
   - ✅ Check "Copy items if needed"
   - ✅ Check "Create groups"
   - ✅ Check the app target
   - Click **Finish**

### 6. Add Info.plist and Entitlements

**Add Info.plist:**
1. Drag `Info.plist` into Xcode
2. Same settings as above (copy if needed, create groups, add to target)

**Add Entitlements:**
1. Drag `DataPointsAI.entitlements` into Xcode
2. Same settings

### 7. Configure Project Settings

**Click on the project name** (blue icon at top of Project Navigator)

#### General Tab

1. **Identity:**
   - Display Name: `Data Points AI`
   - Bundle Identifier: `com.datapointsai.rssreader`

2. **Deployment Info:**
   - Minimum Deployment: macOS 13.0
   - Category: News

3. **App Icon:**
   - Click on App Icon → Add (optional, can do later)

#### Signing & Capabilities Tab

1. **Signing:**
   - Automatically manage signing: ✅ (or uncheck if you have certificates)
   - Team: Select your team (or "Sign to Run Locally")

2. **Add Capabilities:**
   - Click "+" button
   - Add **"Outgoing Connections (Client)"**
   - Add **"Incoming Connections (Server)"**

3. **Hardened Runtime:**
   - Should appear automatically
   - Enable "Allow Unsigned Executable Memory" (for Python)
   - Enable "Allow DYLD Environment Variables"

#### Build Settings Tab

1. Search for "Info.plist File"
   - Set to: `Info.plist`

2. Search for "Code Signing Entitlements"
   - Set to: `DataPointsAI.entitlements`

3. Search for "Swift Language Version"
   - Should be: Swift 5

### 8. Build and Run!

**Press Cmd+R** (or click the Play button)

Xcode will:
1. Compile Swift code (~10 seconds)
2. Create the app bundle
3. Launch the app

You should see:
1. Status bar: "Starting server..."
2. After ~2 seconds: "Server running"
3. Web UI loads automatically

## Troubleshooting Setup

### "Cannot find 'PythonServerManager' in scope"

**Fix:** Make sure all Swift files are added to the target:
1. Click on the file in Project Navigator
2. Look at File Inspector (right sidebar)
3. Under "Target Membership", check the box for your app

### Build fails with "Multiple commands produce 'Info.plist'"

**Fix:** Remove duplicate Info.plist:
1. Click project → Build Phases
2. Expand "Copy Bundle Resources"
3. Remove `Info.plist` if it appears there
4. It should only be referenced in Build Settings

### App builds but won't run

**Fix:** Check entitlements and signing:
1. Product → Clean Build Folder (Cmd+Shift+K)
2. Try again with Cmd+R
3. Check Console.app for error messages

### Python server doesn't start

**Fix:** Verify Python environment:
```bash
# Check venv exists
ls -la /Users/timcarmody/workspace/rss-reader/rss_venv/bin/python

# Check uvicorn is installed
/Users/timcarmody/workspace/rss-reader/rss_venv/bin/python -m pip list | grep uvicorn
```

If missing, run:
```bash
cd /Users/timcarmody/workspace/rss-reader
./run_server.sh  # This sets up venv if needed
```

## Project Structure in Xcode

After setup, your Project Navigator should look like:

```
Data Points AI
├── 📁 Data Points AI
│   ├── DataPointsAIApp.swift
│   ├── AppDelegate.swift
│   ├── PythonServerManager.swift
│   ├── ContentView.swift
│   ├── AppState.swift
│   ├── SettingsView.swift
│   ├── Info.plist
│   ├── DataPointsAI.entitlements
│   └── Assets.xcassets
│       └── AppIcon.appiconset
└── 📁 Products
    └── Data Points AI.app
```

## Next Steps

### Development Workflow

1. **Edit Swift code** → Automatic rebuild on Cmd+R
2. **Edit Python code** → Just restart the app (Cmd+R)
3. **Edit web UI** → Reload in app (Cmd+R in running app)

### Adding an App Icon

1. **Create icon images** (1024x1024 PNG recommended):
   - Use existing icon from `electron/assets/icon.icns`
   - Or create new one

2. **Add to Xcode:**
   - Click Assets.xcassets in Project Navigator
   - Click AppIcon
   - Drag PNG files into appropriate size slots
   - Xcode auto-generates other sizes

### Debugging

**To see detailed logs:**
1. Run app from Xcode (Cmd+R)
2. Console pane shows all logs
3. Python output prefixed with `📝 Python:`
4. Swift output from `print()` statements

**To debug web UI:**
1. Right-click in web view
2. Choose "Inspect Element"
3. Safari Web Inspector opens

### Making a Release Build

1. **Product → Archive**
   - Wait for archive to complete
   - Organizer window opens

2. **Distribute App:**
   - Click "Distribute App"
   - Choose "Copy App"
   - Select destination folder
   - App is ready to share!

**To create a DMG:**
```bash
cd ~/Desktop  # or wherever you exported the app
hdiutil create -volname "Data Points AI" \
  -srcfolder "Data Points AI.app" \
  -ov -format UDZO \
  "DataPointsAI-1.0.0.dmg"
```

## Comparing to Electron

| Aspect | Xcode Setup | Electron Setup |
|--------|-------------|----------------|
| **Initial setup** | 5-10 minutes | 15-20 minutes |
| **Dependencies** | None (Xcode built-in) | npm install (1000+ packages) |
| **Build time** | 8-10 seconds | 30-60 seconds |
| **Build output** | Single .app bundle | .app + node_modules |
| **App size** | 10-15 MB | 150-250 MB |
| **Rebuild** | Instant (incremental) | 15-30 seconds |
| **Debugging** | Excellent (Xcode debugger) | Good (Chrome DevTools) |

## Common Xcode Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd+R | Build and run |
| Cmd+B | Build only |
| Cmd+. | Stop running |
| Cmd+Shift+K | Clean build folder |
| Cmd+0 | Toggle Navigator |
| Cmd+Option+0 | Toggle Inspector |
| Cmd+Shift+Y | Toggle Debug Console |
| Cmd+/ | Comment/uncomment |
| Cmd+Shift+O | Open quickly (file search) |

## Need Help?

- **Xcode documentation:** Help → Xcode Help
- **Swift documentation:** [swift.org](https://www.swift.org/documentation/)
- **SwiftUI tutorials:** [developer.apple.com](https://developer.apple.com/tutorials/swiftui)

---

**Ready to build?** Follow steps 1-8 above and you'll have a native Mac app running in 10 minutes!
