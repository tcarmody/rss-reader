#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script settings - easily modifiable variables
PYTHON_VERSION="3.11"
PORT=5005

# Color codes for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

# Function definitions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    command -v "$1" &> /dev/null
}

check_dependencies() {
    # Fix: Use specific import names that match how the packages are imported in the code
    log_info "Checking critical dependencies..."
    
    # For beautifulsoup4, the import is 'bs4'
    python -c "import bs4" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "beautifulsoup4 package missing - try installing it manually with: pip install beautifulsoup4"
        return 1
    fi
    
    # Check for python-dotenv using the actual import name
    python -c "import dotenv" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "python-dotenv package missing - try installing it manually with: pip install python-dotenv"
        return 1
    fi
    
    # Check other critical packages
    python -c "import flask" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "flask package missing - try installing it manually with: pip install flask"
        return 1
    fi
    
    python -c "import anthropic" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "anthropic package missing - try installing it manually with: pip install anthropic"
        return 1
    fi
    
    python -c "import feedparser" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "feedparser package missing - try installing it manually with: pip install feedparser"
        return 1
    fi
    
    python -c "import requests" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "requests package missing - try installing it manually with: pip install requests"
        return 1
    fi

    python -c "import spacy" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_warning "spacy package missing - entity recognition will use fallback method"
    fi
    
    log_success "All critical dependencies installed successfully!"
    return 0
}

setup_environment() {
    # Change to the script directory
    cd "$SCRIPT_DIR"

    # Check if requirements.txt exists - use the updated one from the artifacts
    if [ ! -f "requirements.txt" ]; then
        log_warning "requirements.txt not found. Creating a new one..."
        cat > requirements.txt << 'EOF'
# Core dependencies
anthropic>=0.7.0
beautifulsoup4>=4.11.0
feedparser>=6.0.0
flask>=2.0.0
python-dotenv>=0.19.0
requests>=2.27.0
tqdm>=4.64.0
psutil>=5.9.0

# NLP and Processing
spacy>=3.0.0
fasttext>=0.9.3
numpy>=1.20.0
python-dateutil>=2.9.0
scikit-learn>=1.0.0
langdetect>=1.0.9

# Caching and optimization
ratelimit>=2.2.0

# ML Dependencies
torch>=1.10.0
safetensors>=0.3.0
sentence-transformers>=2.2.0
transformers>=4.30.0

# Clustering Dependencies
hdbscan>=0.8.40
umap-learn>=0.5.1

# Topic modeling
bertopic>=0.16.0

# HTML/XML parsing
lxml>=4.9.0
html5lib>=1.1

# Required by dependencies
scipy>=1.3.1
joblib>=1.4.2
threadpoolctl>=3.1.0
pynndescent>=0.5
numba>=0.51.2
llvmlite>=0.44.0
EOF
    fi

    # Check if essential_requirements.txt exists, create if not
    if [ ! -f "essential_requirements.txt" ]; then
        log_info "Creating essential_requirements.txt"
        cat > essential_requirements.txt << 'EOF'
# Core dependencies required for basic functionality
anthropic>=0.7.0
beautifulsoup4>=4.11.0
feedparser>=6.0.0
flask>=2.0.0
python-dotenv>=0.19.0
requests>=2.27.0
spacy>=3.0.0
tqdm>=4.64.0
psutil>=5.9.0
EOF
    fi

    # Check if Python version is available
    PYTHON_CMD="python${PYTHON_VERSION}"
    if ! check_command "$PYTHON_CMD"; then
        log_error "Python ${PYTHON_VERSION} was not found. Please make sure it's installed."
        exit 1
    fi
    log_success "Found Python ${PYTHON_VERSION}!"

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Virtual environment not found. Creating one with Python ${PYTHON_VERSION}..."
        "$PYTHON_CMD" -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment with Python ${PYTHON_VERSION}."
            exit 1
        }
    fi

    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate" || {
        log_error "Failed to activate virtual environment."
        exit 1
    }

    # Verify Python version in virtual environment
    VENV_PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Using Python $VENV_PYTHON_VERSION in virtual environment"

    if [[ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]]; then
        log_warning "Virtual environment is not using Python ${PYTHON_VERSION}. Recreating..."
        deactivate
        rm -rf "$VENV_DIR"
        "$PYTHON_CMD" -m venv "$VENV_DIR" || {
            log_error "Failed to recreate virtual environment."
            exit 1
        }
        source "$VENV_DIR/bin/activate"
    fi

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    # Install dependencies
    install_dependencies
}

install_dependencies() {
    # Install core dependencies first with verbose output to debug issues
    log_info "Installing core dependencies..."
    pip install -v -r essential_requirements.txt
    
    # Install all dependencies with more verbose output
    log_info "Installing additional dependencies..."
    pip install -v -r requirements.txt || {
        log_warning "Some dependencies could not be installed, but we'll continue if critical ones are present."
    }
    
    # Install SpaCy model for NER (Named Entity Recognition)
    log_info "Installing SpaCy language model for entity recognition..."
    python -m spacy download en_core_web_sm || {
        log_warning "Could not install SpaCy language model. Entity recognition will use fallback method."
    }
    
    # Give the system a moment to finalize installations
    sleep 2
    
    # Verify critical dependencies
    check_dependencies || {
        log_error "Critical dependencies are missing. Please fix the issues above."
        exit 1
    }
}

check_environment_variables() {
    # Check for API key in environment or .env file
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        if [ -f .env ] && grep -q "ANTHROPIC_API_KEY" .env; then
            log_success "Found Anthropic API key in .env file"
        else
            log_warning "No Anthropic API key found in environment or .env file"
            log_info "Checking if .env file exists..."
            
            if [ ! -f .env ]; then
                log_info "Creating .env file template - you'll need to add your API key"
                echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
                echo "# Add other environment variables below as needed" >> .env
            fi
            
            log_info "You will need to add your Anthropic API key to the .env file"
        fi
    else
        log_success "Found Anthropic API key in environment"
    fi
}

run_server() {
    log_info "Starting Data Points AI server on port $PORT..."
    log_info "Press Ctrl+C to stop the server"
    echo "---------------------------------------------------------------"
    python server.py
}

# Main execution
log_info "Setting up RSS Reader environment..."
setup_environment
check_environment_variables
run_server