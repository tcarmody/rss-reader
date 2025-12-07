#!/bin/bash
#
# Build script for Data Points AI RSS Reader Mac App
#

set -e  # Exit on error

echo "🚀 Building Data Points AI RSS Reader for macOS..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the electron directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}Error: package.json not found. Please run this script from the electron directory.${NC}"
    exit 1
fi

# Check Node.js
echo -e "${BLUE}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    echo "Please install Node.js 18 or higher from https://nodejs.org/"
    exit 1
fi
NODE_VERSION=$(node -v)
echo -e "${GREEN}✓ Node.js ${NODE_VERSION}${NC}"

# Check Python
echo -e "${BLUE}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ ${PYTHON_VERSION}${NC}"

# Check ANTHROPIC_API_KEY
echo -e "${BLUE}Checking API key...${NC}"
if [ -f "../.env" ]; then
    if grep -q "ANTHROPIC_API_KEY" "../.env"; then
        echo -e "${GREEN}✓ API key configured${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: ANTHROPIC_API_KEY not found in .env file${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Warning: .env file not found${NC}"
fi

# Install npm dependencies
echo ""
echo -e "${BLUE}Installing Node.js dependencies...${NC}"
npm install

# Check Python dependencies
echo ""
echo -e "${BLUE}Checking Python dependencies...${NC}"
cd ..
if [ -d "rss_venv" ]; then
    echo -e "${GREEN}✓ Virtual environment found${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Virtual environment not found${NC}"
    echo "Run ./run_server.sh to set up Python environment"
fi

# Return to electron directory
cd electron

# Create assets directory if it doesn't exist
if [ ! -d "assets" ]; then
    echo ""
    echo -e "${BLUE}Creating assets directory...${NC}"
    mkdir -p assets
fi

# Check for icon
if [ ! -f "assets/icon.icns" ]; then
    echo -e "${YELLOW}⚠ Warning: assets/icon.icns not found${NC}"
    echo "The app will build without a custom icon"
fi

# Parse command line arguments
BUILD_TYPE="${1:-universal}"

# Build the application
echo ""
echo -e "${BLUE}Building application (${BUILD_TYPE})...${NC}"
case "$BUILD_TYPE" in
    "universal")
        npm run build:universal
        ;;
    "arm64")
        npm run build:arm64
        ;;
    "x64")
        npm run build:x64
        ;;
    "all")
        echo "Building all architectures..."
        npm run build
        ;;
    *)
        echo -e "${RED}Invalid build type: ${BUILD_TYPE}${NC}"
        echo "Usage: $0 [universal|arm64|x64|all]"
        exit 1
        ;;
esac

# Check build output
echo ""
if [ -d "dist" ]; then
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo ""
    echo "Build artifacts:"
    ls -lh dist/ | grep -E '\.(dmg|zip|app)$' || echo "  (none found)"
    echo ""
    echo -e "${BLUE}Installation:${NC}"
    echo "  1. Open the .dmg file in the dist/ folder"
    echo "  2. Drag Data Points AI RSS Reader to Applications"
    echo "  3. Launch from Applications or Spotlight"
    echo ""
    echo -e "${YELLOW}First Launch:${NC}"
    echo "  macOS may show a security warning. To allow:"
    echo "  - System Preferences → Security & Privacy → Open Anyway"
    echo "  - Or right-click the app → Open"
else
    echo -e "${RED}✗ Build failed - no dist directory found${NC}"
    exit 1
fi
