#!/usr/bin/env python3
"""
Main application entry point for the refactored RSS reader.
"""

# Set this environment variable to avoid HuggingFace tokenizers warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import logging
from typing import List, Dict, Optional

from common.logging import configure_logging
from summarization.article_summarizer import ArticleSummarizer
from summarization.fast_summarizer import FastSummarizer
from reader.enhanced_reader import EnhancedRSSReader

# Configure logging
logger = configure_logging(
    level=logging.INFO,
    log_file="./app.log",
    console=True
)

def setup_summarization_engine(api_key=None):
    """Set up the summarization engine with appropriate components."""
    # Create the base summarizer
    summarizer = ArticleSummarizer(api_key=api_key)
    
    # Create the optimized summarizer with enhanced features
    return FastSummarizer(
        api_key=api_key,
        rpm_limit=50,
        cache_size=256,
        max_batch_workers=3
    )

async def process_articles(articles, api_key=None, max_workers=3):
    """
    Process a list of articles using the fast summarizer.
    
    Args:
        articles: List of article dictionaries
        api_key: Optional API key (defaults to environment variable)
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of processed articles with summaries
    """
    # Create the summarizer
    summarizer = FastSummarizer(
        api_key=api_key,
        rpm_limit=50,
        cache_size=256,
        max_batch_workers=max_workers
    )
    
    # Process articles in batch
    logger.info(f"Processing {len(articles)} articles with {max_workers} workers")
    
    results = await summarizer.batch_summarize(
        articles=articles,
        max_concurrent=max_workers,
        auto_select_model=True
    )
    
    return results

async def main():
    """Main application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process articles with Claude API")
    parser.add_argument("--input", help="Input file with articles (JSON)")
    parser.add_argument("--output", help="Output file for summaries (JSON)")
    parser.add_argument("--workers", type=int, default=3, help="Maximum number of workers")
    
    args = parser.parse_args()
    
    # Load articles from input file
    import json
    
    if args.input:
        try:
            with open(args.input, 'r') as f:
                articles = json.load(f)
                
            # Process articles
            results = await process_articles(
                articles=articles,
                max_workers=args.workers
            )
            
            # Save results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                logger.info(f"Results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
                
        except Exception as e:
            logger.error(f"Error processing articles: {str(e)}")
    else:
        logger.error("No input file specified")

if __name__ == "__main__":
    asyncio.run(main())