"""Main RSS Reader class for fetching and processing feeds."""

import os
import time
import logging
import anthropic
import feedparser
import traceback

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template

from utils.config import get_env_var
from utils.http import create_http_session
from utils.performance import track_performance
from utils.archive import fetch_article_content, is_paywalled
from batch import BatchProcessor
from summarizer import ArticleSummarizer
from clustering import ArticleClusterer


class RSSReader:
    """
    Main class that handles RSS feed processing, article summarization, and clustering.

    This class orchestrates the entire process of:
    1. Fetching and parsing RSS feeds
    2. Generating AI-powered summaries
    3. Clustering similar articles
    4. Generating HTML output

    The class uses Claude API for high-quality summaries and semantic similarity
    for clustering related articles. It implements caching to avoid redundant
    API calls and includes fallback options for summarization.
    
    Example:
        reader = RSSReader()
        output_file = reader.process_feeds()
        print(f"Generated output at: {output_file}")
    """

    def __init__(self, feeds=None, batch_size=25, batch_delay=15):
        """
        Initialize RSSReader with feeds and settings.
        
        Args:
            feeds: List of RSS feed URLs (optional)
            batch_size: Number of feeds to process in a batch
            batch_delay: Delay between batches in seconds
        """
        self.feeds = feeds or self._load_default_feeds()
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.session = create_http_session()
        self.client = anthropic.Anthropic(api_key=get_env_var('ANTHROPIC_API_KEY'))
        self.batch_processor = BatchProcessor(batch_size=5)  # Process 5 API calls at a time
        self.summarizer = ArticleSummarizer()
        self.clusterer = ArticleClusterer()

    def _load_default_feeds(self):
        """
        Load feed URLs from the default file.
        
        Returns:
            list: List of feed URLs
        """
        feeds = []
        try:
            # Check for the file in the package directory or in the current directory
            file_paths = [
                os.path.join(os.path.dirname(__file__), 'rss_feeds.txt'),
                'rss_feeds.txt'
            ]
            
            for path in file_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        for line in f:
                            url = line.strip()
                            if url and not url.startswith('#'):
                                # Remove any inline comments and clean the URL
                                url = url.split('#')[0].strip()
                                url = ''.join(c for c in url if ord(c) >= 32)  # Remove control characters
                                if url:
                                    feeds.append(url)
                    return feeds
            
            logging.error("No rss_feeds.txt file found")
            return []
        except Exception as e:
            logging.error(f"Error loading feed URLs: {str(e)}")
            return []

    @track_performance()
    def process_cluster_summaries(self, clusters):
        """
        Process and generate summaries for article clusters.
        
        Args:
            clusters: List of article clusters
            
        Returns:
            list: Processed clusters with summaries
        """
        processed_clusters = []
        for i, cluster in enumerate(clusters, 1):
            try:
                if not cluster:
                    logging.warning(f"Empty cluster {i}, skipping")
                    continue

                logging.info(f"Processing cluster {i}/{len(clusters)} with {len(cluster)} articles")

                if len(cluster) > 1:
                    # For clusters with multiple articles, generate a combined summary
                    combined_text = "\n\n".join([
                        f"Title: {article['title']}\n{article.get('content', '')[:1000]}"
                        for article in cluster
                    ])

                    logging.info(f"Generating summary for cluster {i} with {len(cluster)} articles")
                    cluster_summary = self._generate_summary(
                        combined_text,
                        f"Combined summary of {len(cluster)} related articles",
                        cluster[0]['link']
                    )

                    # Add the cluster summary to each article
                    for article in cluster:
                        article['summary'] = cluster_summary
                        article['cluster_size'] = len(cluster)
                else:
                    # Single article
                    article = cluster[0]
                    if not article.get('summary'):
                        logging.info(f"Generating summary for single article: {article['title']}")
                        article['summary'] = self._generate_summary(
                            article.get('content', ''),
                            article['title'],
                            article['link']
                        )
                    article['cluster_size'] = 1

                processed_clusters.append(cluster)
                logging.info(f"Successfully processed cluster {i}")

            except Exception as cluster_error:
                logging.error(f"Error processing cluster {i}: {str(cluster_error)}", exc_info=True)
                continue

        return processed_clusters

    def _parse_entry(self, entry, feed_title):
        """
        Parse a feed entry into an article dictionary.
        
        Args:
            entry: feedparser entry object
            feed_title: Title of the feed
            
        Returns:
            dict: Parsed article data or None if parsing failed
        """
        try:
            # Extract content
            content = self._extract_content_from_entry(entry)

            return {
                'title': getattr(entry, 'title', 'No Title'),
                'link': getattr(entry, 'link', '#'),
                'published': getattr(entry, 'published', 'Unknown date'),
                'content': content,
                'feed_source': feed_title
            }

        except Exception as e:
            logging.error(f"Error parsing entry: {str(e)}")
            return None
            
    def _extract_content_from_entry(self, entry):
        """
        Extract and clean content from a feed entry.
        
        Args:
            entry: feedparser entry object
            
        Returns:
            str: Cleaned content text
        """
        content = ''
        # First try to get content
        if hasattr(entry, 'content'):
            raw_content = entry.content
            if isinstance(raw_content, list) and raw_content:
                content = raw_content[0].get('value', '')
            elif isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, (list, tuple)):
                content = ' '.join(str(item) for item in raw_content)
            else:
                content = str(raw_content)

        # Fallback to summary
        if not content and hasattr(entry, 'summary'):
            content = entry.summary

        # Final fallback to title
        if not content:
            content = getattr(entry, 'title', '')
            logging.warning("Using title as content fallback")

        # Clean content
        content = content.strip()
        
        # Check if this is a paywalled article and try to bypass
        paywall_bypass_enabled = get_env_var('ENABLE_PAYWALL_BYPASS', 'false').lower() == 'true'
        
        if paywall_bypass_enabled and hasattr(entry, 'link') and entry.link:
            article_url = entry.link
            
            # Check if content is short (likely just a summary) and the article might be paywalled
            if len(content) < 1000 or is_paywalled(article_url):
                logging.info(f"Content appears truncated or paywalled, attempting to fetch full content for: {article_url}")
                
                try:
                    # Try to fetch full content using archive services
                    full_content = fetch_article_content(article_url, self.session)
                    
                    if full_content and len(full_content) > len(content):
                        logging.info(f"Successfully retrieved full content for: {article_url}")
                        return full_content
                except Exception as e:
                    logging.warning(f"Error fetching full content: {str(e)}")
        
        return content

    @track_performance()
    def process_feeds(self):
        """
        Process all RSS feeds and generate summaries.
        
        This is the main method that orchestrates the full process:
        1. Fetch and parse feeds
        2. Cluster articles
        3. Generate summaries
        4. Create HTML output
        
        Returns:
            str: Path to the generated HTML file or None if processing failed
        """
        try:
            all_articles = []

            # Process feeds in batches
            for batch in self._get_feed_batches():
                logging.info(f"\n🔄 Processing Batch {batch['current']}/{batch['total']}: "
                           f"Feeds {batch['start']} to {batch['end']}")

                # Process each feed in the batch in parallel
                with ThreadPoolExecutor(max_workers=min(len(batch['feeds']), 10)) as executor:
                    futures = [executor.submit(self._process_feed, feed) for feed in batch['feeds']]
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

            # First cluster the articles
            logging.info("Clustering similar articles...")
            clusters = self.clusterer.cluster_articles(all_articles)

            if not clusters:
                logging.error("No clusters created")
                return None

            logging.info(f"Created {len(clusters)} clusters")

            # Now generate summaries for each cluster
            logging.info("Generating summaries for article clusters...")
            processed_clusters = self.process_cluster_summaries(clusters)

            if not processed_clusters:
                logging.error("No clusters were successfully processed")
                return None

            logging.info(f"Successfully processed {len(processed_clusters)} clusters")

            # Generate HTML output
            output_file = self.generate_html_output(processed_clusters)
            if output_file:
                logging.info(f"Successfully generated HTML output: {output_file}")
            else:
                logging.error("Failed to generate HTML output")

            return output_file

        except Exception as e:
            logging.error(f"Error processing feeds: {str(e)}", exc_info=True)
            return None

    @track_performance()
    def _process_feed(self, feed_url):
        """
        Process a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            list: Processed articles from the feed
        """
        try:
            feed = feedparser.parse(feed_url)
            articles = []

            if feed.entries:
                feed_title = feed.feed.get('title', feed_url)
                logging.info(f"📰 Found {len(feed.entries)} articles in feed: {feed_url}")

                for entry in feed.entries[:self.batch_size]:
                    article = self._parse_entry(entry, feed_title)
                    if article:
                        articles.append(article)

            return articles

        except Exception as e:
            logging.error(f"Error processing feed {feed_url}: {str(e)}")
            return []

    def _generate_summary(self, article_text, title, url):
        """
        Generate a summary for an article using the Anthropic API.
        
        Args:
            article_text: Text of the article to summarize
            title: Title of the article
            url: URL of the article
            
        Returns:
            dict: Summary with headline and content
        """
        return self.summarizer.summarize_article(article_text, title, url)

    def _get_feed_batches(self):
        """
        Generate batches of feeds to process.
        
        Yields:
            dict: Batch information containing feeds and metadata
        """
        logging.info("🚀 Initializing RSS Reader...")
        logging.info(f"📊 Total Feeds: {len(self.feeds)}")
        logging.info(f"📦 Batch Size: {self.batch_size}")
        logging.info(f"⏱️  Batch Delay: {self.batch_delay} seconds")

        total_batches = (len(self.feeds) + self.batch_size - 1) // self.batch_size
        logging.info(f"🔄 Total Batches: {total_batches}")

        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(self.feeds))
            yield {
                'current': batch_num + 1,
                'total': total_batches,
                'start': start_idx + 1,
                'end': end_idx,
                'feeds': self.feeds[start_idx:end_idx]
            }
            
    @track_performance()
    def generate_html_output(self, clusters):
        """
        Generate HTML output from the processed clusters.
        
        Args:
            clusters: List of article clusters with summaries
            
        Returns:
            str: Path to generated HTML file or False if generation failed
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), 'output')
            os.makedirs(output_dir, exist_ok=True)

            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')

            app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

            with app.app_context():
                html_content = render_template(
                    'feed-summary.html',
                    clusters=clusters,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                logging.info(f"Successfully wrote HTML output to {output_file}")
                return output_file

        except Exception as e:
            logging.error(f"Error generating HTML output: {str(e)}", exc_info=True)
            return False
