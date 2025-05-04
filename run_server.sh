#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script settings
VENV_NAME="rss_venv"
PORT=5005
HOST="127.0.0.1"  # Default to localhost for security

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/$VENV_NAME"

# Log functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --public)
                HOST="0.0.0.0"
                log_warning "Running with public access (0.0.0.0). Make sure this is intended and secured."
                shift
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --reload)
                RELOAD_FLAG="--reload"
                log_info "Development mode: auto-reload enabled"
                shift
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            *)
                log_warning "Unknown argument: $1"
                shift
                ;;
        esac
    done
}

# Main function
main() {
    log_info "Setting up RSS Reader environment..."
    
    # Parse command line arguments
    parse_args "$@"
    
    # Make sure we're in the script directory
    cd "$SCRIPT_DIR"
    log_info "Working in directory: $(pwd)"
    
    # Check for Python 3.11 specifically
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        # Check if python3 is version 3.11
        PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ "$PY_VERSION" == "3.11" ]]; then
            PYTHON_CMD="python3"
        else
            log_warning "Python 3.11 not found, but found Python $PY_VERSION"
            log_warning "It's recommended to use Python 3.11 for this project"
            read -p "Continue with Python $PY_VERSION? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_error "Please install Python 3.11 and try again"
                exit 1
            fi
            PYTHON_CMD="python3"
        fi
    elif command -v python &> /dev/null; then
        # Check if python is version 3.11
        PY_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ "$PY_VERSION" == "3.11" ]]; then
            PYTHON_CMD="python"
        else
            log_warning "Python 3.11 not found, but found Python $PY_VERSION"
            log_warning "It's recommended to use Python 3.11 for this project"
            read -p "Continue with Python $PY_VERSION? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_error "Please install Python 3.11 and try again"
                exit 1
            fi
            PYTHON_CMD="python"
        fi
    else
        log_error "No Python interpreter found. Please install Python 3.11."
        exit 1
    fi
    
    log_info "Using Python: $($PYTHON_CMD --version)"
    
    # Create or activate virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment with Python 3.11..."
        # Check if venv module is available
        if ! $PYTHON_CMD -c "import venv" &> /dev/null; then
            log_error "Python venv module not available. Make sure python3.11-venv is installed."
            log_info "On Ubuntu/Debian, try: sudo apt-get install python3.11-venv"
            log_info "On CentOS/RHEL, try: sudo yum install python3.11-venv"
            log_info "On macOS, try: pip3 install virtualenv"
            exit 1
        fi
        
        $PYTHON_CMD -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment."
            exit 1
        }
        log_success "Virtual environment created with Python 3.11."
    else
        log_info "Using existing virtual environment."
    fi
    
    # Activate virtual environment
    if [ -f "$VENV_DIR/bin/activate" ]; then
        log_info "Activating virtual environment..."
        source "$VENV_DIR/bin/activate" || {
            log_error "Failed to activate virtual environment."
            exit 1
        }
        log_success "Virtual environment activated."
    else
        log_error "Activation script not found at $VENV_DIR/bin/activate"
        exit 1
    fi
    
    # Check if we need to install requirements
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        log_info "Installing dependencies from requirements.txt..."
        pip install -r "$SCRIPT_DIR/requirements.txt" || {
            log_warning "Some dependencies failed to install. The server may not work correctly."
        }
        
        # Download spaCy model after dependencies are installed
        log_info "Checking for spaCy model..."
        if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
            log_info "Downloading and installing spaCy model (en_core_web_sm)..."
            python -m spacy download en_core_web_sm || {
                log_warning "Failed to download spaCy model. Enhanced clustering may not work correctly."
            }
            log_success "spaCy model installed successfully."
        else
            log_info "spaCy model already installed."
        fi
    else
        log_info "No requirements.txt found. Installing minimal dependencies..."
        # Install FastAPI and Uvicorn instead of Flask
        pip install fastapi uvicorn[standard] jinja2 python-multipart || {
            log_error "Failed to install essential dependencies."
            exit 1
        }
    fi
    
    # Check if uvicorn is installed
    if ! command -v uvicorn &> /dev/null; then
        log_error "Uvicorn not found. Please make sure it's installed in the virtual environment."
        exit 1
    fi
    
    # Build uvicorn command
    UVICORN_CMD="uvicorn server:app --host $HOST --port $PORT"
    
    # Add optional flags
    if [ ! -z "$RELOAD_FLAG" ]; then
        UVICORN_CMD="$UVICORN_CMD $RELOAD_FLAG"
    fi
    
    if [ ! -z "$WORKERS" ]; then
        # Note: --reload and --workers are mutually exclusive
        if [ -z "$RELOAD_FLAG" ]; then
            UVICORN_CMD="$UVICORN_CMD --workers $WORKERS"
        else
            log_warning "Cannot use --reload with --workers. Ignoring --workers flag."
        fi
    fi
    
    # Start the server
    if [ -f "$SCRIPT_DIR/server.py" ]; then
        log_info "Starting FastAPI server on $HOST:$PORT..."
        log_info "Server command: $UVICORN_CMD"
        log_info "Using server.py from: $SCRIPT_DIR/server.py"
        
        # Execute the uvicorn command
        eval $UVICORN_CMD || {
            log_error "Failed to start FastAPI server. Check server.py for errors."
            exit 1
        }
    else
        log_error "server.py not found at: $SCRIPT_DIR/server.py"
        log_error "Current directory: $(pwd)"
        log_error "Directory contents:"
        ls -la "$SCRIPT_DIR"
        exit 1
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --public        Listen on all interfaces (0.0.0.0)"
    echo "  --port PORT     Specify port number (default: 5005)"
    echo "  --reload        Enable auto-reload for development"
    echo "  --workers N     Number of worker processes (not compatible with --reload)"
    echo "  --help          Show this help message"
}

# Check for help flag
if [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Run the main function with all arguments
main "$@"