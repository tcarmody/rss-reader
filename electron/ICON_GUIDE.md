# Icon Creation Guide

## Quick Start

The easiest way is to use the provided script:

```bash
# Create placeholder (blue square with "DP")
./create-icons.sh

# Or provide your own 1024x1024 PNG
./create-icons.sh path/to/my-icon.png
```

## Requirements

- Source image: **1024×1024 pixels** minimum
- Format: PNG (transparent or opaque)
- Tools: ImageMagick or sips (built into macOS)

## macOS Icon Design Guidelines

### Size

- Minimum: 1024×1024 px
- Color space: sRGB
- Transparency: Supported (optional)

### Design Principles

1. **Simplicity**
   - Clear and recognizable at small sizes
   - Avoid fine details that disappear when scaled down
   - Test at 16×16 px to ensure legibility

2. **Style**
   - Flat or slightly 3D (macOS adds depth automatically)
   - No drop shadows (system adds them)
   - Rounded corners optional (can be added later)
   - Consistent with macOS Big Sur+ style

3. **Colors**
   - Vibrant but not overpowering
   - Consider both light and dark mode
   - Use gradients sparingly
   - Ensure contrast for accessibility

4. **Shape**
   - Square canvas (1024×1024)
   - Icon can be any shape within canvas
   - Leave some padding (50-100px) from edges
   - Consider the rounded square mask macOS applies

### Examples of Good Icons

- **SF Symbols style**: Clean, minimal, recognizable
- **Apple's own apps**: Music, Safari, Mail
- **Popular Mac apps**: Slack, VS Code, Notion

### Icon Sizes Generated

The script creates these sizes (all required):

- 16×16 (1x) - Menu bar, lists
- 32×32 (2x) - Retina menu bar, lists
- 32×32 (1x) - Toolbars
- 64×64 (2x) - Retina toolbars
- 128×128 (1x) - Finder icon view
- 256×256 (2x) - Retina Finder
- 256×256 (1x) - Large Finder icons
- 512×512 (2x) - Retina large icons
- 512×512 (1x) - Very large icons
- 1024×1024 (2x) - Retina very large icons

## Creating Your Icon

### Option 1: Figma/Sketch (Recommended)

1. Create 1024×1024 artboard
2. Design your icon
3. Leave ~50px padding from edges
4. Export as PNG at 2x (2048×2048)
5. Scale down to 1024×1024 if needed
6. Run: `./create-icons.sh your-icon.png`

### Option 2: Photoshop

1. New file: 1024×1024 px, 72 DPI
2. Design on multiple layers
3. Flatten or keep transparency
4. Export: File → Save As → PNG
5. Run: `./create-icons.sh your-icon.png`

### Option 3: Online Tools

**Recommended**:
- [Canva](https://www.canva.com/) - Free, easy to use
- [Figma](https://www.figma.com/) - Professional, free tier
- [Pixelmator](https://www.pixelmator.com/) - Mac native ($40)

**Icon Generators**:
- [Icon Slate](https://www.kodlian.com/apps/icon-slate) - Mac app for .icns
- [Image2icon](https://img2icons.com/) - Online converter
- [Iconfinder](https://www.iconfinder.com/) - Icon marketplace

### Option 4: AI Generation

Use AI tools like:
- DALL-E 3
- Midjourney
- Stable Diffusion

**Prompt example**:
```
A modern, minimalist app icon for an RSS reader application.
Flat design with a blue gradient background.
Clean lines, professional, recognizable at small sizes.
Square format, 1024x1024 pixels.
Style: macOS Big Sur app icon.
```

Then process the output:
```bash
./create-icons.sh ai-generated-icon.png
```

## Testing Your Icon

### Visual Test

```bash
# Build with your icon
make build-universal

# Check in Finder
open dist/mac-universal/

# Right-click the .app → Get Info
# Icon should appear in top-left
```

### Size Test

View at different sizes:

```bash
# Quick Look
qlmanage -p assets/icon.icns

# Or open in Preview
open assets/icon.icns
```

Check if icon is clear at:
- 16×16 (menu bar)
- 32×32 (lists)
- 128×128 (Finder)
- 512×512 (large)

### Dark Mode Test

View in both light and dark mode:
- System Preferences → General → Appearance
- Switch between Light and Dark
- Icon should be visible in both

## Icon Template

Here's a simple template you can follow:

```
┌─────────────────────────────┐
│   ┌───────────────────┐     │
│   │                   │     │  ← 50px padding
│   │                   │     │
│   │   YOUR ICON       │     │
│   │   924×924 px      │     │
│   │   safe area       │     │
│   │                   │     │
│   │                   │     │
│   └───────────────────┘     │
└─────────────────────────────┘
    1024×1024 px canvas
```

## Color Palette Suggestions

### Option 1: Blue (Default)
- Primary: `#3b82f6` (Blue 500)
- Light: `#60a5fa` (Blue 400)
- Dark: `#2563eb` (Blue 600)

### Option 2: Gradient
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Option 3: Monochrome
- For dark backgrounds: White icon with subtle shadow
- For light backgrounds: Dark gray icon

## Automated Icon Generation

If you have ImageMagick installed:

```bash
# Create from text
convert -size 1024x1024 xc:'#3b82f6' \
  -gravity center \
  -pointsize 400 \
  -fill white \
  -font "Helvetica-Bold" \
  -annotate +0+0 "DP" \
  icon-source.png

# Then convert to .icns
./create-icons.sh icon-source.png
```

## Placeholder Icon

The default placeholder shows:
- Blue background (`#3b82f6`)
- White "DP" text (Data Points)
- Bold Helvetica font
- Centered

To customize colors:

```bash
# Edit create-icons.sh, line with xc:'#3b82f6'
# Change the hex color to your preference
```

## Icon in Build

The icon is automatically included when you:

```bash
make build-universal
# or
npm run build:universal
```

It's embedded in:
- The .app bundle
- The .dmg installer
- Finder displays
- Dock when running

## Troubleshooting

### Icon doesn't appear

1. Check `assets/icon.icns` exists
2. Verify file is valid: `file assets/icon.icns`
3. Rebuild: `make clean && make build-universal`
4. Clear icon cache: `sudo rm -rf /Library/Caches/com.apple.iconservices.store`

### Icon looks blurry

- Source image too small (need 1024×1024 minimum)
- Use PNG, not JPEG
- Ensure proper DPI (72 or higher)

### Icon wrong colors

- Check color space is sRGB
- Test in both light and dark mode
- Adjust contrast if needed

## Resources

### Design Tools
- [SF Symbols](https://developer.apple.com/sf-symbols/) - Apple's icon library
- [Figma Community](https://www.figma.com/community) - Free templates
- [Icon8](https://icons8.com/) - Icon library

### Guidelines
- [Apple HIG - App Icons](https://developer.apple.com/design/human-interface-guidelines/app-icons)
- [macOS Icon Template](https://applypixels.com/template/macos-11-big-sur)

### Tutorials
- [Designing macOS App Icons](https://blog.prototypr.io/designing-macos-app-icons-5e7fe0e0c4f)
- [Icon Design Best Practices](https://uxdesign.cc/icon-design-best-practices-cd85b0c6adb1)

## Getting Help

If you're stuck with icon creation:

1. Use the default placeholder (it works!)
2. Hire a designer on [Fiverr](https://www.fiverr.com/) ($5-50)
3. Ask in GitHub Discussions
4. Use an AI generator (DALL-E, Midjourney)

Remember: **A placeholder icon is fine for personal use!** You can always update it later.
