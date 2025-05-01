#!/usr/bin/env python3
"""
Script to apply the EnhancedBatchProcessor fix.

This script demonstrates how to integrate the batch processor fix into the
existing codebase with minimal changes.
"""

import os
import sys
import logging
import time
import importlib
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_processor_fix.log")
    ]
)
logger = logging.getLogger("batch_processor_fix")

def apply_fix():
    """Apply the batch processor fix to the codebase."""
    logger.info("Starting batch processor fix application")
    
    try:
        # First, check if we can import the required modules
        logger.info("Checking imports...")
        from enhanced_batch_processor import EnhancedBatchProcessor, WorkerProcess
        from batch_processor_fix import patch_enhanced_batch_processor, create_fixed_batch_processor
        
        logger.info("Imports successful")
        
        # Apply the patches
        logger.info("Applying patches to the EnhancedBatchProcessor class")
        patched_class = patch_enhanced_batch_processor()
        
        logger.info("Testing the patched processor...")
        
        # Create a test processor
        processor = create_fixed_batch_processor(max_workers=2)
        
        # Create a simple test article
        test_article = {
            'text': 'This is a test article for batch processing.',
            'title': 'Test Article',
            'url': 'https://example.com/test'
        }
        
        # Test with a batch context
        with processor.batch_context():
            logger.info("Successfully initialized the processor with batch_context")
            
            # Test the synchronous process_batch_sync method
            logger.info("Testing synchronous batch processing...")
            results = processor.process_batch_sync([test_article])
            
            if results and len(results) > 0:
                logger.info("Synchronous batch processing successful!")
            else:
                logger.warning("Synchronous batch processing returned no results")
        
        logger.info("Patch testing complete")
        return True
        
    except Exception as e:
        logger.error(f"Error applying fix: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_async_processor():
    """Test the async batch processing functionality."""
    logger.info("Testing async batch processing...")
    
    try:
        from batch_processor_fix import create_fixed_batch_processor
        
        # Create a test processor
        processor = create_fixed_batch_processor(max_workers=2)
        
        # Create test articles
        test_articles = [
            {
                'text': 'This is the first test article for batch processing.',
                'title': 'Test Article 1',
                'url': 'https://example.com/test1'
            },
            {
                'text': 'This is the second test article for batch processing.',
                'title': 'Test Article 2',
                'url': 'https://example.com/test2'
            }
        ]
        
        # Use the processor with async batch processing
        with processor.batch_context():
            results = await processor.process_batch_async(test_articles)
            
            if results and len(results) > 0:
                logger.info(f"Async batch processing successful! Got {len(results)} results.")
                return True
            else:
                logger.warning("Async batch processing returned no results")
                return False
                
    except Exception as e:
        logger.error(f"Error in async test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def patch_fast_summarizer():
    """
    Patch the FastArticleSummarizer to use the fixed batch processor.
    
    This is the main integration point with the existing codebase.
    """
    logger.info("Patching FastArticleSummarizer to use fixed batch processor...")
    
    try:
        # Import the necessary modules
        from fast_summarizer import FastArticleSummarizer
        from batch_processor_fix import create_fixed_batch_processor
        import types
        
        # Create a new batch_summarize method that uses the fixed processor
        async def patched_batch_summarize(
            self,
            articles,
            max_concurrent=3,
            model=None,
            auto_select_model=True,
            temperature=0.3,
        ):
            """
            Process multiple articles in parallel batches using the fixed worker processes.
            
            This is a patched version that uses the fixed batch processor.
            """
            self.logger.info(f"Using patched batch_summarize with fixed processor")
            
            # Create a fixed batch processor
            fixed_processor = create_fixed_batch_processor(max_workers=max_concurrent)
            
            # Process the articles
            with fixed_processor.batch_context():
                self.logger.info(f"Processing {len(articles)} articles with fixed batch processor")
                results = await fixed_processor.process_batch_async(
                    articles=articles,
                    model=model,
                    temperature=temperature
                )
                
                self.logger.info(f"Fixed batch processing completed for {len(results)} articles")
                return results
        
        # Apply the patch to all existing instances of FastArticleSummarizer
        # We can't directly modify the class but we can patch instances when they're created
        
        # Save the original __init__ method
        original_init = FastArticleSummarizer.__init__
        
        # Create a patched __init__ method
        def patched_init(self, *args, **kwargs):
            # Call the original __init__
            original_init(self, *args, **kwargs)
            
            # Apply our patch to this instance
            self.batch_summarize = types.MethodType(patched_batch_summarize, self)
            self.logger.info("Applied fixed batch_summarize method to FastArticleSummarizer instance")
        
        # Replace the __init__ method
        FastArticleSummarizer.__init__ = patched_init
        
        logger.info("Successfully patched FastArticleSummarizer")
        return True
        
    except Exception as e:
        logger.error(f"Error patching FastArticleSummarizer: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def full_integration():
    """Perform full integration of the fix into the existing codebase."""
    logger.info("Starting full integration of the batch processor fix")
    
    # Step 1: Apply the core fix
    if not apply_fix():
        logger.error("Failed to apply core fix, aborting integration")
        return False
    
    # Step 2: Patch the FastArticleSummarizer
    if not patch_fast_summarizer():
        logger.warning("Failed to patch FastArticleSummarizer, continuing with partial integration")
    
    # Step 3: Test the async processor
    asyncio.run(test_async_processor())
    
    logger.info("Integration complete")
    return True