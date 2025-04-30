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

# Add parent directory to path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    # Also check the test modules
    test_modules = [
        "tests.lm_cluster_analyzer",
    ]
    
    logger.info("Checking test modules...")
    
    for module in test_modules:
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
    
    # Get the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    logger.info("Checking required files...")
    
    for file in required_files:
        file_path = os.path.join(parent_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✅ {file}: Found")
        else:
            logger.warning(f"❌ {file}: Not found")

def check_templates():
    """Check if template directories and files exist."""
    # Get the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(parent_dir, 'templates')
    
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