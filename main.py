#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RSS reader with FastArticleSummarizer integration.
This script integrates the improved parallel batch processing for article summarization.
"""

import os
import sys
import logging
import time
import traceback
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rss_reader_enhanced.log"),
        logging.StreamHandler()
    ]
)

def setup_summarization_engine():
    """
    Set up the enhanced summarization engine with improved batch processing.
    
    Returns:
        FastArticleSummarizer: Configured with enhanced batch processing
    """
    logging.info("Setting up enhanced summarization engine...")
    
    try:
        # Import required modules
        from summarizer import ArticleSummarizer
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
        
        logging.info(f"Summarization engine successfully configured with {max_workers} workers")
        logging.info(f"Rate limit: {rpm_limit} RPM, Cache size: {cache_size}, TTL: {ttl_days} days")
        
        return fast_summarizer
        
    except Exception as e:
        logging.error(f"Error setting up summarization engine: {e}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"Failed to initialize summarization engine: {e}")


class EnhancedRSSReader:
    """
    Enhanced RSS reader with improved parallel batch processing for article summarization.
    Extends the core functionality with more efficient handling of article batches.
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
        
        # Log initialization
        logging.info(f"Enhanced RSS Reader initialized with {max_workers} summarization workers")
        
    async def batch_summarize_articles(self, articles):
        """
        Summarize a batch of articles using enhanced parallel processing.
        
        Args:
            articles: List of articles to summarize
            
        Returns:
            list: Processed articles with summaries
        """
        logging.info(f"Processing batch of {len(articles)} articles with enhanced parallel summarization")
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
            logging.info("No articles need summarization in this batch")
            return articles
            
        logging.info(f"Summarizing {len(articles_to_process)} articles with enhanced batch processing")
        
        try:
            # Process articles in batch
            summarizer = self.reader.summarizer
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
            logging.info(f"Batch summarization completed in {elapsed_time:.2f}s")
            
            return articles
            
        except Exception as e:
            logging.error(f"Error in batch summarization: {e}")
            logging.error(traceback.format_exc())
            return articles
    
    async def process_feeds(self):
        """
        Process RSS feeds with enhanced summarization.
        This method extends the original RSSReader's process_feeds method
        with more efficient parallel summarization.
        
        Returns:
            str: Path to the output HTML file or None if processing failed
        """
        try:
            all_articles = []

            # Process feeds in batches (reusing the original implementation)
            for batch in self.reader._get_feed_batches():
                logging.info(f"\n🔄 Processing Batch {batch['current']}/{batch['total']}: "
                           f"Feeds {batch['start']} to {batch['end']}")

                # Process each feed in the batch in parallel
                with ThreadPoolExecutor(max_workers=min(len(batch['feeds']), 10)) as executor:
                    futures = [executor.submit(self.reader._process_feed, feed) for feed in batch['feeds']]
                    batch_articles = []
                    for future in as_completed(futures):
                        articles = future.result()
                        if articles:
                            batch_articles.extend(articles)
                            logging.info(f"Added {len(articles)} articles to batch")

                all_articles.extend(batch_articles)
                logging.info(f"Batch complete. Total articles so far: {len(all_articles)}")

                # Add delay between batches if there are more
                if batch['current'] < batch['total']:
                    time.sleep(self.batch_delay)

            logging.info(f"Total articles collected: {len(all_articles)}")

            if not all_articles:
                logging.error("No articles collected from any feeds")
                return None

            # First cluster the articles (using the original implementation)
            logging.info("Clustering similar articles...")
            clusters = self.reader.clusterer.cluster_articles(all_articles)

            if not clusters:
                logging.error("No clusters created")
                return None

            logging.info(f"Created {len(clusters)} clusters")
            
            # Now process each cluster with enhanced summarization
            import asyncio
            processed_clusters = []
            
            for i, cluster in enumerate(clusters, 1):
                try:
                    if not cluster:
                        logging.warning(f"Empty cluster {i}, skipping")
                        continue

                    logging.info(f"Processing cluster {i}/{len(clusters)} with {len(cluster)} articles")
                    
                    # First, try to fetch full content for articles that don't have it
                    for article in cluster:
                        if not article.get('content') or len(article.get('content', '')) < 200:
                            try:
                                from utils.archive import fetch_article_content
                                logging.info(f"Fetching full content for article: {article['title']}")
                                full_content = fetch_article_content(article['link'], self.reader.session)
                                if full_content and len(full_content) > 200:
                                    article['content'] = full_content
                                    logging.info(f"Successfully fetched full content for {article['title']}")
                                else:
                                    logging.warning(f"Could not fetch meaningful content for {article['title']}")
                            except Exception as e:
                                logging.warning(f"Error fetching content for {article['title']}: {str(e)}")

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
                    logging.info(f"Successfully processed cluster {i}")
                    
                except Exception as cluster_error:
                    logging.error(f"Error processing cluster {i}: {str(cluster_error)}", exc_info=True)
                    continue

            # Store the processed clusters for web server access
            self.reader.last_processed_clusters = processed_clusters
            
            # Generate HTML output using the original method
            output_file = self.reader.generate_html_output(processed_clusters)
            if output_file:
                logging.info(f"Successfully generated HTML output: {output_file}")
            else:
                logging.error("Failed to generate HTML output")

            return output_file

        except Exception as e:
            logging.error(f"Error processing feeds: {str(e)}", exc_info=True)
            return None


def main():
    """
    Main function to run the enhanced RSS reader.
    
    Process command line arguments and run the reader.
    
    Example:
        # Run directly
        python -m updated_main
    """
    parser = argparse.ArgumentParser(description="Enhanced RSS Reader and Summarizer")
    parser.add_argument("--feeds", nargs="+", help="List of feed URLs to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of feeds to process in a batch")
    parser.add_argument("--batch-delay", type=int, default=15, help="Delay between batches in seconds")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker processes for summarization")
    
    args = parser.parse_args()
    
    try:
        # Print welcome message
        print("\n===== Enhanced RSS Reader with Parallel Processing =====")
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
            logging.info(f"✅ Successfully generated RSS summary: {output_file}")
            print(f"\nSummary generated at: {output_file}")
            return 0
        else:
            logging.warning("⚠️ No articles found or processed")
            print("\nNo articles found or processed. Check the log for details.")
            return 1

    except Exception as e:
        logging.error(f"❌ Error in main: {str(e)}")
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())