#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please create it first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment with python3. Trying python3.11..."
        python3.11 -m venv venv
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment. Please ensure Python 3 is installed."
            exit 1
        fi
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first (the ones needed for the server to run)
echo "Installing core dependencies..."
pip install flask feedparser requests beautifulsoup4 python-dotenv

# Try to install the rest of the dependencies but don't fail if some can't be installed
echo "Installing additional dependencies..."
# Create a temporary file with only the essential packages
cat > essential_requirements.txt << EOF
anthropic
requests
beautifulsoup4
feedparser
flask
python-dotenv
EOF

# Install essential packages first
pip install -r essential_requirements.txt

# Now try to install the rest but don't fail if they can't be installed
echo "Attempting to install additional dependencies (some may fail)..."
pip install -r requirements.txt || echo "Some dependencies could not be installed, but we'll continue anyway."

# Verify Flask is installed (the most critical dependency)
echo "Verifying Flask installation..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask is not installed correctly. Please check your Python environment."
    exit 1
fi

# Run the server
echo "Starting Data Points AI server..."
python server.py
