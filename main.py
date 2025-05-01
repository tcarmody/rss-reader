#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RSS reader with improved clustering and summarization.
This script integrates the multi-article clustering system for better content organization.
"""

# Apply batch processor fixes
import apply_batch_fix
apply_batch_fix.apply()

import os
import sys
import logging
import time
import traceback
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)

def configure_logging():
    """Set up logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler("rss_reader_enhanced.log")
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Avoid duplicate log messages
    root_logger.propagate = False
    
    return root_logger

def setup_summarization_engine():
    """
    Set up the enhanced summarization engine with improved batch processing.
    
    Returns:
        FastArticleSummarizer: Configured with enhanced batch processing
    """
    logger.info("Setting up enhanced summarization engine...")
    
    try:
        # Import required modules directly (not the entire module)
        from summarizer import ArticleSummarizer
        # Import only the specific function needed to avoid circular imports
        from fast_summarizer import create_fast_summarizer
        
        # Create the original summarizer
        original_summarizer = ArticleSummarizer()
        
        # Configure environment variables for rate limiting and process management
        rpm_limit = int(os.environ.get('API_RPM_LIMIT', '50'))
        cache_size = int(os.environ.get('CACHE_SIZE', '256'))
        cache_dir = os.environ.get('CACHE_DIR', './summary_cache')
        ttl_days = int(os.environ.get('CACHE_TTL_DAYS', '30'))
        max_workers = int(os.environ.get('MAX_BATCH_WORKERS', '3'))
        
        # Create fast summarizer with enhanced batch processing
        fast_summarizer = create_fast_summarizer(
            original_summarizer=original_summarizer,
            rpm_limit=rpm_limit,
            cache_size=cache_size,
            cache_dir=cache_dir,
            ttl_days=ttl_days,
            max_batch_workers=max_workers
        )
        
        logger.info(f"Summarization engine successfully configured with {max_workers} workers")
        logger.info(f"Rate limit: {rpm_limit} RPM, Cache size: {cache_size}, TTL: {ttl_days} days")
        
        return fast_summarizer
        
    except Exception as e:
        logger.error(f"Error setting up summarization engine: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to initialize summarization engine: {e}")


def setup_clustering_engine(summarizer=None):
    """
    Set up the enhanced clustering engine with multi-article comparison capabilities.
    
    Args:
        summarizer: The summarizer instance that provides access to LLM
        
    Returns:
        EnhancedArticleClusterer: Configured clustering engine
    """
    logger.info("Setting up enhanced clustering engine...")
    
    try:
        # Import modules when needed, not at the top level
        from enhanced_clustering import create_enhanced_clusterer
        
        # Create the enhanced clusterer that uses LM-based multi-article clustering
        clusterer = create_enhanced_clusterer(summarizer=summarizer)
        
        # Try to create a cluster analyzer for advanced operations
        try:
            from lm_cluster_analyzer import create_cluster_analyzer
            analyzer = create_cluster_analyzer(summarizer=summarizer)
            # Store the analyzer in the clusterer for convenience
            clusterer.analyzer = analyzer
        except ImportError:
            logger.warning("LM cluster analyzer not available, skipping advanced cluster analysis")
        
        logger.info("Clustering engine successfully configured with multi-article capabilities")
        
        return clusterer
        
    except Exception as e:
        logger.error(f"Error setting up clustering engine: {e}")
        logger.error(traceback.format_exc())
        # Fall back to the original clustering if enhanced fails
        from clustering import ArticleClusterer
        logger.warning("Using fallback clustering engine without multi-article capabilities")
        return ArticleClusterer()


class EnhancedRSSReader:
    """
    Enhanced RSS reader with improved parallel batch processing for article summarization
    and multi-article clustering for better content organization.
    """
    
    def __init__(self, feeds=None, batch_size=25, batch_delay=15, max_workers=3):
        """
        Initialize the enhanced RSS reader
        
        Args:
            feeds: List of feed URLs to process (or None to use default)
            batch_size: Number of feeds to process in a batch
            batch_delay: Delay between batches in seconds
            max_workers: Maximum number of worker processes for batch summarization
        """
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.max_workers = max_workers
        
        # Set up the original RSS reader
        from reader import RSSReader
        self.reader = RSSReader(feeds=feeds, batch_size=batch_size, batch_delay=batch_delay)
        
        # Replace the standard summarizer with our enhanced version
        self.reader.summarizer = setup_summarization_engine()
        
        # Replace the standard clusterer with our enhanced version
        self.reader.clusterer = setup_clustering_engine(summarizer=self.reader.summarizer)
        
        # Log initialization
        logger.info(f"Enhanced RSS Reader initialized with {max_workers} summarization workers and multi-article clustering")
        
    async def batch_summarize_articles(self, articles):
        """
        Summarize a batch of articles using enhanced parallel processing.
        
        Args:
            articles: List of articles to summarize
            
        Returns:
            list: Processed articles with summaries
        """
        logger.info(f"Processing batch of {len(articles)} articles with enhanced parallel summarization")
        start_time = time.time()
        
        # Prepare articles for summarization
        articles_to_process = []
        for article in articles:
            # Skip articles that already have summaries
            if article.get('summary'):
                continue
                
            # Prepare the article for processing
            articles_to_process.append({
                'text': article.get('content', ''),
                'title': article.get('title', 'No Title'),
                'url': article.get('link', '#')
            })
        
        # Skip if no articles need summarization
        if not articles_to_process:
            logger.info("No articles need summarization in this batch")
            return articles
            
        logger.info(f"Summarizing {len(articles_to_process)} articles with enhanced batch processing")
        
        try:
            # Process articles in batch
            summarizer = self.reader.summarizer
            
            # Check for batch_summarize method (added by enhanced batch processor)
            if not hasattr(summarizer, 'batch_summarize'):
                logger.warning("Batch processing not available, falling back to sequential processing")
                # Process articles sequentially as fallback
                for article in articles_to_process:
                    try:
                        summary = summarizer.summarize(
                            text=article['text'],
                            title=article['title'],
                            url=article['url'],
                            auto_select_model=True
                        )
                        # Find the matching original article and update it
                        for orig_article in articles:
                            if orig_article.get('link') == article['url']:
                                orig_article['summary'] = summary
                                break
                    except Exception as e:
                        logger.error(f"Error summarizing article {article['url']}: {e}")
                
                elapsed_time = time.time() - start_time
                logger.info(f"Sequential summarization completed in {elapsed_time:.2f}s")
                return articles
            
            # Use batch processing if available
            results = await summarizer.batch_summarize(
                articles=articles_to_process,
                max_concurrent=self.max_workers,
                auto_select_model=True,
                temperature=0.3
            )
            
            # Match results back to original articles
            url_to_summary = {r['original']['url']: r['summary'] for r in results if 'summary' in r}
            
            # Update articles with summaries
            for article in articles:
                if article.get('link') in url_to_summary and not article.get('summary'):
                    article['summary'] = url_to_summary[article['link']]
            
            elapsed_time = time.time() - start_time
            logger.info(f"Batch summarization completed in {elapsed_time:.2f}s")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")
            logger.error(traceback.format_exc())
            return articles
    
    async def process_feeds(self):
        """
        Process RSS feeds with enhanced summarization and multi-article clustering.
        This method extends the original RSSReader's process_feeds method
        with more efficient parallel summarization and improved clustering.
        
        Returns:
            str: Path to the output HTML file or None if processing failed
        """
        try:
            all_articles = []

            # Process feeds in batches (reusing the original implementation)
            for batch in self.reader._get_feed_batches():
                logger.info(f"\n🔄 Processing Batch {batch['current']}/{batch['total']}: "
                           f"Feeds {batch['start']} to {batch['end']}")

                # Process each feed in the batch in parallel
                with ThreadPoolExecutor(max_workers=min(len(batch['feeds']), 10)) as executor:
                    futures = [executor.submit(self.reader._process_feed, feed) for feed in batch['feeds']]
                    batch_articles = []
                    for future in as_completed(futures):
                        articles = future.result()
                        if articles:
                            batch_articles.extend(articles)
                            logger.info(f"Added {len(articles)} articles to batch")

                all_articles.extend(batch_articles)
                logger.info(f"Batch complete. Total articles so far: {len(all_articles)}")

                # Add delay between batches if there are more
                if batch['current'] < batch['total']:
                    time.sleep(self.batch_delay)

            logger.info(f"Total articles collected: {len(all_articles)}")

            if not all_articles:
                logger.error("No articles collected from any feeds")
                return None

            # Use the enhanced clustering with multi-article capabilities
            logger.info("Clustering similar articles with enhanced multi-article clustering...")
            
            # Check if the enhanced clustering method exists
            if hasattr(self.reader.clusterer, 'cluster_with_summaries'):
                clusters = self.reader.clusterer.cluster_with_summaries(all_articles)
            else:
                # Fallback to basic clustering
                logger.warning("Enhanced clustering not available, using basic clustering")
                clusters = self.reader.clusterer.cluster_articles(all_articles)

            if not clusters:
                logger.error("No clusters created")
                return None

            logger.info(f"Created {len(clusters)} clusters")
            
            # Extract topics for each cluster using the LM-based analyzer if available
            if hasattr(self.reader.clusterer, 'analyzer'):
                for cluster in clusters:
                    try:
                        if cluster and len(cluster) > 0:
                            topics = self.reader.clusterer.analyzer.extract_cluster_topics(cluster)
                            if topics:
                                # Add topics to each article in the cluster
                                for article in cluster:
                                    article['cluster_topics'] = topics
                    except Exception as e:
                        logger.warning(f"Error extracting topics for cluster: {str(e)}")
            
            # Now process each cluster with enhanced summarization
            import asyncio
            processed_clusters = []
            
            for i, cluster in enumerate(clusters, 1):
                try:
                    if not cluster:
                        logger.warning(f"Empty cluster {i}, skipping")
                        continue

                    logger.info(f"Processing cluster {i}/{len(clusters)} with {len(cluster)} articles")
                    
                    # First, try to fetch full content for articles that don't have it
                    for article in cluster:
                        if not article.get('content') or len(article.get('content', '')) < 200:
                            try:
                                from utils.archive import fetch_article_content
                                logger.info(f"Fetching full content for article: {article['title']}")
                                full_content = fetch_article_content(article['link'], self.reader.session)
                                if full_content and len(full_content) > 200:
                                    article['content'] = full_content
                                    logger.info(f"Successfully fetched full content for {article['title']}")
                                else:
                                    logger.warning(f"Could not fetch meaningful content for {article['title']}")
                            except Exception as e:
                                logger.warning(f"Error fetching content for {article['title']}: {str(e)}")

                    if len(cluster) > 1:
                        # For clusters with multiple articles, use the primary article for summarization
                        best_article = max(cluster, key=lambda a: len(a.get('content', '')))
                        
                        # If the best article still has limited content, combine all articles
                        if len(best_article.get('content', '')) < 500:
                            combined_text = "\n\n".join([
                                f"Title: {article['title']}\n{article.get('content', '')}"
                                for article in cluster
                            ])
                            
                            # Create a combined article for summarization
                            combined_article = {
                                'content': combined_text,
                                'title': f"Combined summary of {len(cluster)} related articles",
                                'link': cluster[0]['link']
                            }
                            
                            # Summarize the combined article
                            combined_result = await self.batch_summarize_articles([combined_article])
                            
                            if combined_result and combined_result[0].get('summary'):
                                # Add the combined summary to each article in the cluster
                                for article in cluster:
                                    article['summary'] = combined_result[0]['summary']
                                    article['cluster_size'] = len(cluster)
                        else:
                            # Summarize individual articles in the cluster
                            updated_cluster = await self.batch_summarize_articles(cluster)
                            
                            # Ensure all articles in the cluster have the same summary (use the best one)
                            best_summary = None
                            for article in updated_cluster:
                                if article.get('summary'):
                                    best_summary = article['summary']
                                    break
                                    
                            if best_summary:
                                for article in updated_cluster:
                                    article['summary'] = best_summary
                                    article['cluster_size'] = len(cluster)
                    else:
                        # Single article processing
                        updated_cluster = await self.batch_summarize_articles(cluster)
                        updated_cluster[0]['cluster_size'] = 1
                
                    # Verify summaries have correct structure
                    for article in cluster:
                        # Ensure summary has the correct structure
                        if not isinstance(article.get('summary', {}), dict):
                            article['summary'] = {
                                'headline': article['title'],
                                'summary': str(article.get('summary', 'No summary available.'))
                            }
                        elif 'headline' not in article.get('summary', {}) or 'summary' not in article.get('summary', {}):
                            # Fix incomplete summary structure
                            summary_text = article.get('summary', {}).get('summary', '')
                            if not summary_text and isinstance(article.get('summary', {}), str):
                                summary_text = article.get('summary', '')
                            
                            article['summary'] = {
                                'headline': article.get('summary', {}).get('headline', article['title']),
                                'summary': summary_text or 'No summary available.'
                            }
                    
                    processed_clusters.append(cluster)
                    logger.info(f"Successfully processed cluster {i}")
                    
                except Exception as cluster_error:
                    logger.error(f"Error processing cluster {i}: {str(cluster_error)}", exc_info=True)
                    continue

            # Store the processed clusters for web server access
            self.reader.last_processed_clusters = processed_clusters
            
            # Generate HTML output using the original method
            output_file = self.reader.generate_html_output(processed_clusters)
            if output_file:
                logger.info(f"Successfully generated HTML output: {output_file}")
            else:
                logger.error("Failed to generate HTML output")

            return output_file

        except Exception as e:
            logger.error(f"Error processing feeds: {str(e)}", exc_info=True)
            return None


def main():
    """
    Main function to run the enhanced RSS reader.
    
    Process command line arguments and run the reader.
    
    Example:
        # Run directly
        python -m main
    """
    # Configure logging
    root_logger = configure_logging()
    
    parser = argparse.ArgumentParser(description="Enhanced RSS Reader and Summarizer with Multi-Article Clustering")
    parser.add_argument("--feeds", nargs="+", help="List of feed URLs to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of feeds to process in a batch")
    parser.add_argument("--batch-delay", type=int, default=15, help="Delay between batches in seconds")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for summarization")
    parser.add_argument("--disable-multi-article", action="store_true", help="Disable multi-article clustering")
    
    args = parser.parse_args()
    
    # Apply multi-article clustering setting to environment if specified
    if args.disable_multi_article:
        os.environ['ENABLE_MULTI_ARTICLE_CLUSTERING'] = 'false'
        logger.info("Multi-article clustering disabled by command line argument")
    
    try:
        # Print welcome message
        print("\n===== Enhanced RSS Reader with Multi-Article Clustering =====")
        print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {args.workers} worker processes for summarization")
        print("========================================================\n")
        
        # Initialize and run enhanced RSS reader
        rss_reader = EnhancedRSSReader(
            feeds=args.feeds,  # Will use default if None
            batch_size=args.batch_size,
            batch_delay=args.batch_delay,
            max_workers=args.workers
        )
        
        # Run the async process_feeds method
        import asyncio
        output_file = asyncio.run(rss_reader.process_feeds())

        if output_file:
            logger.info(f"✅ Successfully generated RSS summary: {output_file}")
            print(f"\nSummary generated at: {output_file}")
            return 0
        else:
            logger.warning("⚠️ No articles found or processed")
            print("\nNo articles found or processed. Check the log for details.")
            return 1

    except Exception as e:
        logger.error(f"❌ Error in main: {str(e)}")
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())