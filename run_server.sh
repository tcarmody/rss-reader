#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script settings - easily modifiable variables
PYTHON_VERSION="3.11"
PORT=5005
VENV_NAME="rss_venv"

# Color codes for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/$VENV_NAME"

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

check_python_version() {
    # Find the appropriate Python command
    if check_command "python$PYTHON_VERSION"; then
        PYTHON_CMD="python$PYTHON_VERSION"
    elif check_command "python3"; then
        PYTHON_CMD="python3"
    elif check_command "python"; then
        PYTHON_CMD="python"
    else
        log_error "No Python interpreter found. Please install Python $PYTHON_VERSION."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION_FOUND=$(${PYTHON_CMD} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Found Python $PYTHON_VERSION_FOUND"
    
    # Return the command
    echo $PYTHON_CMD
}

setup_virtual_env() {
    local python_cmd=$1
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment with $python_cmd..."
        ${python_cmd} -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment."
            log_info "Trying with the venv module..."
            ${python_cmd} -m venv "$VENV_DIR" || {
                log_error "Failed to create virtual environment with venv module."
                exit 1
            }
        }
        log_success "Virtual environment created successfully."
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
        log_error "Virtual environment activation script not found."
        exit 1
    fi
}

install_dependencies() {
    # Upgrade pip first
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install required packages from requirements.txt
    if [ -f "$SCRIPT_DIR/essential_requirements.txt" ]; then
        log_info "Installing essential dependencies..."
        pip install -r "$SCRIPT_DIR/essential_requirements.txt" || {
            log_warning "Failed to install some essential dependencies. This might cause issues."
        }
    fi
    
    # Install required packages from requirements.txt
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        log_info "Installing dependencies from requirements.txt..."
        pip install -r "$SCRIPT_DIR/requirements.txt" || {
            log_warning "Failed to install some dependencies from requirements.txt. This might cause issues."
        }
    else
        log_warning "requirements.txt not found. Creating a minimal one..."
        cat > "$SCRIPT_DIR/requirements.txt" << 'EOF'
# Core dependencies
anthropic>=0.7.0
beautifulsoup4>=4.11.0
feedparser>=6.0.0
flask>=2.0.0
python-dotenv>=0.19.0
requests>=2.27.0
tqdm>=4.64.0
psutil>=5.9.0
spacy>=3.0.0
EOF
    fi

    # Install Spacy language model
    log_info "Installing Spacy language model..."
    python -m spacy download en_core_web_sm || {
        log_warning "Failed to install Spacy language model. Entity recognition will use fallback method."
    }

    # Create required directories
    log_info "Creating required directories..."
    mkdir -p "$SCRIPT_DIR/output" "$SCRIPT_DIR/templates" "$SCRIPT_DIR/static" "$SCRIPT_DIR/.cache"
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

run_diagnostics() {
    log_info "Running diagnostics to check setup..."
    python diagnostics.py || {
        log_warning "Diagnostics script failed or not found."
        log_info "This is not critical, but may help with troubleshooting if issues occur."
    }
}

run_server() {
    log_info "Starting RSS Reader server on port $PORT..."
    log_info "Press Ctrl+C to stop the server"
    echo "---------------------------------------------------------------"
    
    # Try to run the server with different methods
    if [ -f "server.py" ]; then
        python server.py --public --port $PORT || {
            log_warning "Failed to start with server.py directly."
            log_info "Trying flask run..."
            flask run --host=0.0.0.0 --port=$PORT || {
                log_error "Failed to start server. Please check logs for details."
                exit 1
            }
        }
    else
        log_error "server.py not found. Please make sure it exists in the project directory."
        exit 1
    fi
}

# Main execution flow
main() {
    log_info "Setting up RSS Reader environment..."
    
    # Move to the script directory
    cd "$SCRIPT_DIR"
    
    # Create the diagnostics script if it doesn't exist yet
    if [ ! -f "diagnostics.py" ]; then
        log_info "Creating diagnostics script..."
        cat > diagnostics.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics script for the RSS Reader project.
This script checks various components of the system to identify issues.
"""

import os
import sys
import importlib
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check Python environment and paths."""
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Current Working Directory: {os.getcwd()}")
    logger.info(f"Python Path: {sys.path}")
    
    # Check for .env file
    if os.path.exists('.env'):
        logger.info("Found .env file")
    else:
        logger.warning("No .env file found in current directory")

def check_dependencies():
    """Check if key dependencies are installed."""
    dependencies = [
        "anthropic",
        "flask",
        "feedparser",
        "beautifulsoup4",
        "requests",
        "python-dotenv",
        "spacy",
        "tqdm",
        "psutil"
    ]
    
    logger.info("Checking dependencies...")
    
    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✅ {dep}: {version}")
        except ImportError:
            logger.error(f"❌ {dep}: Not installed")
        except Exception as e:
            logger.error(f"⚠️ {dep}: Error checking - {str(e)}")

def check_project_modules():
    """Check if project modules can be imported."""
    project_modules = [
        "reader",
        "summarizer",
        "clustering",
        "cache",
        "fast_summarizer",
        "enhanced_clustering",
        "lm_cluster_analyzer",
        "utils.config",
        "utils.http",
        "utils.archive",
        "utils.performance"
    ]
    
    logger.info("Checking project modules...")
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            logger.info(f"✅ {module}: Successfully imported")
        except ImportError as e:
            logger.error(f"❌ {module}: Import failed - {str(e)}")
        except Exception as e:
            logger.error(f"⚠️ {module}: Error importing - {str(e)}")

def check_api_key():
    """Check if API key is properly configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            masked_key = f"{api_key[:5]}...{api_key[-5:]}" if len(api_key) > 10 else "***"
            logger.info(f"✅ ANTHROPIC_API_KEY found: {masked_key}")
        else:
            logger.warning("❌ ANTHROPIC_API_KEY not found in environment")
    except Exception as e:
        logger.error(f"Error checking API key: {str(e)}")

def validate_summarizer():
    """Test the summarizer module in isolation."""
    try:
        from summarizer import ArticleSummarizer
        
        logger.info("Initializing ArticleSummarizer...")
        summarizer = ArticleSummarizer()
        
        logger.info("✅ ArticleSummarizer initialized successfully")
        
        # Check if models are available
        logger.info(f"Available models: {summarizer.AVAILABLE_MODELS}")
        logger.info(f"Default model: {summarizer.DEFAULT_MODEL}")
        
        # Check cache directory
        cache_dir = getattr(summarizer.summary_cache, 'cache_dir', 'Unknown')
        logger.info(f"Cache directory: {cache_dir}")
        if not os.path.exists(cache_dir):
            logger.warning(f"Cache directory does not exist: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {cache_dir}")
            
    except Exception as e:
        logger.error(f"Error validating summarizer: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def check_required_files():
    """Check if required files exist."""
    required_files = [
        "main.py",
        "reader.py",
        "summarizer.py",
        "clustering.py",
        "cache.py",
        "server.py",
        "requirements.txt"
    ]
    
    logger.info("Checking required files...")
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"✅ {file}: Found")
        else:
            logger.warning(f"❌ {file}: Not found")

def check_templates():
    """Check if template directories and files exist."""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(".")), 'templates')
    if not os.path.exists(templates_dir):
        templates_dir = os.path.join(os.getcwd(), 'templates')
    
    if os.path.exists(templates_dir):
        logger.info(f"✅ Templates directory found: {templates_dir}")
        
        # Check required templates
        templates = [
            "feed-summary.html",
            "welcome.html",
            "error.html"
        ]
        
        for template in templates:
            template_path = os.path.join(templates_dir, template)
            if os.path.exists(template_path):
                logger.info(f"✅ Template found: {template}")
            else:
                logger.warning(f"❌ Template not found: {template}")
    else:
        logger.warning(f"❌ Templates directory not found: {templates_dir}")

def run_diagnostics():
    """Run all diagnostic checks."""
    print("="*60)
    print("Running RSS Reader Diagnostics")
    print("="*60)
    
    try:
        check_environment()
        print("-"*60)
        
        check_dependencies()
        print("-"*60)
        
        check_required_files()
        print("-"*60)
        
        check_project_modules()
        print("-"*60)
        
        check_api_key()
        print("-"*60)
        
        check_templates()
        print("-"*60)
        
        validate_summarizer()
        print("-"*60)
        
        print("Diagnostics complete!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_diagnostics()
EOF
        chmod +x diagnostics.py
        log_success "Created diagnostics script"
    fi
    
    # Check Python version and get appropriate command
    PYTHON_CMD=$(check_python_version)
    
    # Setup virtual environment
    setup_virtual_env "$PYTHON_CMD"
    
    # Install dependencies
    install_dependencies
    
    # Check environment variables
    check_environment_variables
    
    # Run diagnostics
    run_diagnostics
    
    # Run server
    run_server
}

# Execute main function
main