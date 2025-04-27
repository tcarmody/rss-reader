#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script settings - easily modifiable variables
PYTHON_VERSION="3.11"
PORT=5005
REQUIRED_PKGS=("flask" "feedparser" "requests" "beautifulsoup4" "python-dotenv" "anthropic")

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

check_dependency() {
    python -c "import $1" 2> /dev/null
    return $?
}

setup_environment() {
    # Change to the script directory
    cd "$SCRIPT_DIR"

    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found. Please create it first."
        exit 1
    fi

    # Check if essential_requirements.txt exists, create if not
    if [ ! -f "essential_requirements.txt" ]; then
        log_info "Creating essential_requirements.txt"
        cat > essential_requirements.txt << EOF
anthropic
requests
beautifulsoup4
feedparser
flask
python-dotenv
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
    pip install --upgrade pip &> /dev/null

    # Install dependencies
    install_dependencies
}

install_dependencies() {
    # Install core dependencies first
    log_info "Installing core dependencies..."
    pip install -r essential_requirements.txt &> /dev/null || {
        log_error "Failed to install essential dependencies."
        exit 1
    }

    # Install all dependencies
    log_info "Installing additional dependencies..."
    pip install -r requirements.txt &> /dev/null || {
        log_warning "Some dependencies could not be installed, but we'll continue anyway."
    }

    # Verify critical dependencies
    local missing_deps=()
    for pkg in "${REQUIRED_PKGS[@]}"; do
        if ! check_dependency "$pkg"; then
            missing_deps+=("$pkg")
        fi
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Critical dependencies missing: ${missing_deps[*]}"
        log_error "Please install these packages and try again."
        exit 1
    fi
    
    log_success "All critical dependencies installed successfully!"
}

check_environment_variables() {
    # Check for API key in environment or .env file
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        if [ -f .env ] && grep -q "ANTHROPIC_API_KEY" .env; then
            log_success "Found Anthropic API key in .env file"
        else
            log_warning "No Anthropic API key found in environment or .env file"
            log_info "You will need to provide your API key to use the application"
        fi
    else
        log_success "Found Anthropic API key in environment"
    fi
}

run_server() {
    log_info "Starting Data Points AI server on port $PORT..."
    log_info "Press Ctrl+C to stop the server"
    echo "---------------------------------------------------------------"
    python server.py --port $PORT
}

# Main execution
log_info "Setting up RSS Reader environment..."
setup_environment
check_environment_variables
run_server