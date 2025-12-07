/**
 * After-pack script for optimizing the Electron build
 * Removes unnecessary files and reduces app size
 */

const fs = require('fs');
const path = require('path');

module.exports = async function(context) {
  const appPath = context.appOutDir;
  const resourcesPath = path.join(appPath, 'Contents', 'Resources');

  console.log('Running after-pack optimization...');
  console.log('App path:', appPath);

  // Files and directories to remove
  const toRemove = [
    // Python cache
    '**/__pycache__',
    '**/*.pyc',
    '**/*.pyo',
    '**/*.pyd',

    // macOS specific
    '**/.DS_Store',
    '**/.AppleDouble',
    '**/.LSOverride',

    // Git
    '**/.git',
    '**/.gitignore',
    '**/.gitattributes',

    // IDE
    '**/.vscode',
    '**/.idea',
    '**/*.swp',
    '**/*.swo',

    // Documentation
    '**/README.md',
    '**/CHANGELOG.md',
    '**/LICENSE.md',
    '**/docs/**',

    // Tests
    '**/test/**',
    '**/tests/**',
    '**/*_test.py',
    '**/*_test.js',

    // Source maps
    '**/*.map',

    // Backup files
    '**/*.bak',
    '**/*~',

    // Logs
    '**/*.log'
  ];

  let totalSaved = 0;
  let filesRemoved = 0;

  // Recursively remove matching files
  function removeMatching(dir, patterns) {
    if (!fs.existsSync(dir)) return;

    const items = fs.readdirSync(dir);

    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);

      // Check if matches any pattern
      const shouldRemove = patterns.some(pattern => {
        const regex = new RegExp(pattern.replace(/\*\*/g, '.*').replace(/\*/g, '[^/]*'));
        return regex.test(fullPath);
      });

      if (shouldRemove) {
        if (stat.isDirectory()) {
          // Get size before removing
          const size = getDirSize(fullPath);
          fs.rmSync(fullPath, { recursive: true, force: true });
          totalSaved += size;
          filesRemoved++;
          console.log(`  Removed directory: ${fullPath} (${formatBytes(size)})`);
        } else {
          const size = stat.size;
          fs.unlinkSync(fullPath);
          totalSaved += size;
          filesRemoved++;
        }
      } else if (stat.isDirectory()) {
        // Recurse into directory
        removeMatching(fullPath, patterns);
      }
    }
  }

  // Get directory size
  function getDirSize(dirPath) {
    let size = 0;

    if (!fs.existsSync(dirPath)) return size;

    const items = fs.readdirSync(dirPath);

    for (const item of items) {
      const fullPath = path.join(dirPath, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        size += getDirSize(fullPath);
      } else {
        size += stat.size;
      }
    }

    return size;
  }

  // Format bytes
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }

  console.log('Removing unnecessary files...');
  removeMatching(resourcesPath, toRemove);

  console.log(`\nOptimization complete!`);
  console.log(`  Files/directories removed: ${filesRemoved}`);
  console.log(`  Space saved: ${formatBytes(totalSaved)}`);

  // Report final size
  const finalSize = getDirSize(appPath);
  console.log(`  Final app size: ${formatBytes(finalSize)}`);
};
