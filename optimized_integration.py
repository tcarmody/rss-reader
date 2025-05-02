#!/usr/bin/env python3
"""
Integration script for the optimized RSS reader clustering system.

This script provides a simple way to use the optimized clustering pipeline
while maintaining backward compatibility with the existing codebase.
"""

import os
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimized_integration")


def create_anthropic_client(api_key):
    """Create an Anthropic client handling API version differences."""
    try:
        # Try current API format
        return anthropic.Anthropic(api_key=api_key)
    except TypeError as e:
        if 'proxies' in str(e):
            # Try older API format
            import inspect
            if 'proxies' in inspect.signature(anthropic.Anthropic.__init__).parameters:
                return anthropic.Anthropic(api_key=api_key, proxies=None)
        # Re-raise if we can't handle the error
        raise

def apply_optimization_patch():
    """
    Apply optimization patches to the RSS reader clustering system.
    """
    try:
        logger.info("Applying optimization patches to clustering system...")
        
        # Set environment variables for optimized configuration
        os.environ["CLUSTER_CONFIDENCE_THRESHOLD"] = "0.85"
        os.environ["MAX_LM_COMPARISONS"] = "50"
        os.environ["MAX_LM_API_CALLS"] = "50"
        os.environ["CLUSTERING_BATCH_SIZE"] = "100"
        os.environ["USE_INCREMENTAL_CLUSTERING"] = "true"
        
        # Ensure we're using the optimized versions by patching the imports
        patch_imports()
        
        logger.info("Successfully applied optimization patches")
        return True
    except Exception as e:
        logger.error(f"Failed to apply optimization patches: {e}")
        return False

def patch_imports():
    """
    Patch the import system to use our optimized modules.
    This avoids having to modify the existing codebase.
    """
    import sys
    import types
    import builtins
    
    # Store original import function
    original_import = builtins.__import__
    
    # Map of modules to redirect
    module_redirects = {
        'enhanced_clustering': 'optimized_enhanced_clustering',
        'lm_cluster_analyzer': 'optimized_lm_cluster_analyzer'
    }
    
    # Define custom import function
    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Check if this is a module we want to redirect
        if name in module_redirects:
            try:
                # Try to import the optimized version first
                optimized_name = module_redirects[name]
                logger.debug(f"Redirecting import: {name} -> {optimized_name}")
                
                # Import the optimized module
                optimized_module = original_import(optimized_name, globals, locals, fromlist, level)
                
                # If succeeds, use the optimized version
                return optimized_module
            except ImportError:
                # If optimized version not available, fall back to original
                logger.warning(f"Optimized module {module_redirects[name]} not found, using original {name}")
                return original_import(name, globals, locals, fromlist, level)
        
        # For all other modules, use the original import
        return original_import(name, globals, locals, fromlist, level)
    
    # Replace the import function
    builtins.__import__ = patched_import

def create_optimized_reader():
    """
    Create an optimized version of the RSS reader that uses the enhanced clustering.
    
    Returns:
        An instance of EnhancedRSSReader with optimized clustering
    """
    # Apply optimization patches first
    apply_optimization_patch()
    
    try:
        # Import the enhanced reader after patches are applied
        from main import EnhancedRSSReader
        
        # Configure parameters
        batch_size = int(os.environ.get("BATCH_SIZE", "25"))
        batch_delay = int(os.environ.get("BATCH_DELAY", "5"))  # Reduced delay
        max_workers = int(os.environ.get("MAX_WORKERS", "3"))
        
        # Create the optimized reader
        reader = EnhancedRSSReader(
            feeds=None,  # Use default feeds
            batch_size=batch_size,
            batch_delay=batch_delay,
            max_workers=max_workers
        )
        
        # Ensure we're using the optimized clusterer
        from optimized_enhanced_clustering import create_optimized_clusterer
        reader.reader.clusterer = create_optimized_clusterer(summarizer=reader.reader.summarizer)
        
        logger.info("Created optimized RSS reader with enhanced clustering")
        return reader
    except Exception as e:
        logger.error(f"Failed to create optimized reader: {e}")
        # Fall back to original implementation
        logger.warning("Falling back to original implementation")
        from main import EnhancedRSSReader
        return EnhancedRSSReader()

async def run_optimized_reader(feed_urls=None):
    """
    Run the optimized RSS reader with performance monitoring.
    
    Args:
        feed_urls: Optional list of feed URLs to process
        
    Returns:
        str: Path to the generated HTML file or None if processing failed
    """
    try:
        # Create the optimized reader
        reader = create_optimized_reader()
        
        # If feed URLs are provided, override the default
        if feed_urls:
            reader.reader.feeds = feed_urls
            
        # Measure performance
        start_time = time.time()
        
        # Process feeds using the enhanced reader
        output_file = await reader.process_feeds()
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if output_file:
            logger.info(f"Successfully processed feeds in {elapsed_time:.2f}s")
            logger.info(f"Output file: {output_file}")
        else:
            logger.error(f"Failed to process feeds after {elapsed_time:.2f}s")
            
        return output_file
        
    except Exception as e:
        logger.error(f"Error running optimized reader: {e}")
        return None

def main():
    """
    Main function to run the optimized RSS reader from the command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the optimized RSS reader")
    parser.add_argument("--feeds", nargs="+", help="List of feed URLs to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of feeds to process in a batch")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of worker processes")
    
    args = parser.parse_args()
    
    # Set environment variables from arguments
    os.environ["BATCH_SIZE"] = str(args.batch_size)
    os.environ["MAX_WORKERS"] = str(args.max_workers)
    
    # Run the optimized reader
    asyncio.run(run_optimized_reader(args.feeds))

if __name__ == "__main__":
    main()
