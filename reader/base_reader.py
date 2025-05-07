"""Main RSS Reader class for fetching and processing feeds with enhanced clustering."""

import feedparser
import os
import time
import logging
import traceback

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.config import get_env_var
from common.http import create_http_session
from common.performance import track_performance
from common.archive import fetch_article_content, is_paywalled
from common.source_extractor import is_aggregator_link, extract_original_source_url
from common.batch_processing import BatchProcessor
from summarization.article_summarizer import ArticleSummarizer
from clustering.base import ArticleClusterer

# Import the enhanced clusterer - make sure this import works
try:
    from clustering.enhanced import create_enhanced_clusterer
    ENHANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    logging.warning("Enhanced clustering module not available. Using basic clustering.")
    ENHANCED_CLUSTERING_AVAILABLE = False

class RSSReader:
    """
    Main class that handles RSS feed processing, article summarization, and clustering.

    This class orchestrates the entire process of:
    1. Fetching and parsing RSS feeds
    2. Generating AI-powered summaries
    3. Clustering similar articles using advanced techniques
    4. Generating HTML output

    The class uses Claude API for high-quality summaries and a two-phase
    clustering approach for better grouping of related articles. It implements
    caching to avoid redundant API calls and includes fallback options for summarization.
    
    Example:
        reader = RSSReader()
        output_file = reader.process_feeds()
        print(f"Generated output at: {output_file}")
    """

    def __init__(self, feeds=None, batch_size=25, batch_delay=15):
        """
        Initialize RSSReader with feeds and settings.
        
        Args:
            feeds: List of RSS feed URLs (None for default feeds, [] for no feeds)
            batch_size: Number of feeds to process in a batch
            batch_delay: Delay between batches in seconds
        """
        # Explicitly handle None vs empty list
        if feeds is None:
            logging.info("No feeds provided, loading defaults")
            self.feeds = self._load_default_feeds()
        else:
            logging.info(f"Using {len(feeds)} provided feeds")
            self.feeds = feeds
            
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.session = create_http_session()
        self.batch_processor = BatchProcessor(batch_size=5)  # Process 5 API calls at a time
        self.summarizer = ArticleSummarizer()
        
        # Use the enhanced clusterer if available, otherwise fall back to basic
        if ENHANCED_CLUSTERING_AVAILABLE:
            self.clusterer = create_enhanced_clusterer(summarizer=self.summarizer)
            logging.info("Using enhanced clustering for better article grouping")
        else:
            self.clusterer = ArticleClusterer()
            logging.info("Using basic clustering for article grouping")
        
        self.last_processed_clusters = []  # Store the last processed clusters for web server access

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
                    logging.info(f"Loading default feeds from {path}")
                    with open(path, 'r') as f:
                        for line in f:
                            url = line.strip()
                            if url and not url.startswith('#'):
                                # Remove any inline comments and clean the URL
                                url = url.split('#')[0].strip()
                                url = ''.join(c for c in url if ord(c) >= 32)  # Remove control characters
                                if url:
                                    feeds.append(url)
                    logging.info(f"Loaded {len(feeds)} default feeds")
                    return feeds
            
            logging.error("No rss_feeds.txt file found")
            return []
        except Exception as e:
            logging.error(f"Error loading feed URLs: {str(e)}")
            return []

    def _parse_date(self, date_str):
        """
        Parse date string with multiple formats.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            datetime object
        """
        if not date_str:
            return datetime.now()
        
        try:
            # First, try feedparser's date parsing
            parsed_date = feedparser._parse_date(date_str)
            if parsed_date:
                return datetime(*parsed_date[:6])
        except:
            pass
        
        # Try common date formats
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 822
            '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 with timezone name
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 with Z
            '%Y-%m-%d %H:%M:%S',         # Common format
            '%Y-%m-%d',                  # Date only
            '%d %b %Y %H:%M:%S %z',      # Another common format
            '%d %b %Y %H:%M:%S %Z',      # Another common format
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If all parsing attempts fail, return current time
        logging.warning(f"Could not parse date: {date_str}")
        return datetime.now()

    def _filter_articles_by_date(self, articles, hours):
        """
        Filter articles to include only those from the specified time range.
        
        Args:
            articles: List of articles to filter
            hours: Number of hours in the past to include
            
        Returns:
            List of filtered articles
        """
        if hours <= 0:  # No filtering if hours is 0 or negative
            return articles
            
        cutoff_date = datetime.now() - timedelta(hours=hours)
        filtered_articles = []
        
        for article in articles:
            try:
                article_date = self._parse_date(article.get('published', ''))
                # Remove timezone for comparison
                if article_date.replace(tzinfo=None) >= cutoff_date:
                    filtered_articles.append(article)
            except Exception as e:
                logging.debug(f"Could not parse date for article: {article.get('title')}. Including it anyway.")
                filtered_articles.append(article)  # Include articles with unparseable dates
        
        return filtered_articles

    @track_performance
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
                
                # First, try to fetch full content for articles that don't have it
                for article in cluster:
                    if not article.get('content') or len(article.get('content', '')) < 200:
                        try:
                            logging.info(f"Fetching full content for article: {article['title']}")
                            full_content = fetch_article_content(article['link'], self.session)
                            if full_content and len(full_content) > 200:
                                article['content'] = full_content
                                logging.info(f"Successfully fetched full content for {article['title']}")
                            else:
                                logging.warning(f"Could not fetch meaningful content for {article['title']}")
                        except Exception as e:
                            logging.warning(f"Error fetching content for {article['title']}: {str(e)}")

                if len(cluster) > 1:
                    # For clusters with multiple articles, generate a combined summary
                    # Use the article with the most content for summarization
                    best_article = max(cluster, key=lambda a: len(a.get('content', '')))
                    
                    # If the best article still has limited content, combine all articles
                    if len(best_article.get('content', '')) < 500:
                        combined_text = "\n\n".join([
                            f"Title: {article['title']}\n{article.get('content', '')}"
                            for article in cluster
                        ])
                    else:
                        # Use the best article for summarization
                        combined_text = f"Title: {best_article['title']}\n{best_article.get('content', '')}"

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
                    
                    # Ensure summary has the correct structure
                    if not isinstance(article['summary'], dict):
                        article['summary'] = {
                            'headline': article['title'],
                            'summary': str(article['summary'])
                        }
                    elif 'headline' not in article['summary'] or 'summary' not in article['summary']:
                        # Fix incomplete summary structure
                        summary_text = article['summary'].get('summary', '')
                        if not summary_text and isinstance(article['summary'], str):
                            summary_text = article['summary']
                        
                        article['summary'] = {
                            'headline': article['summary'].get('headline', article['title']),
                            'summary': summary_text or 'No summary available.'
                        }

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
        
        # Always try to fetch full content for articles, with special handling for aggregators
        if hasattr(entry, 'link') and entry.link:
            article_url = entry.link
            
            # Check if content is short (likely just a summary) or if we should always try to fetch full content
            paywall_bypass_enabled = get_env_var('ENABLE_PAYWALL_BYPASS', 'false').lower() == 'true'
            is_short_content = len(content) < 1000
            
            # For Techmeme and Google News links, always try to extract the original source
            # For other links, only try if paywall bypass is enabled and content is short
            if is_aggregator_link(article_url) or (paywall_bypass_enabled and (is_short_content or is_paywalled(article_url))):
                logging.info(f"Attempting to fetch full content for: {article_url}")
                
                try:
                    # fetch_article_content will automatically handle aggregator links and paywalls
                    full_content = fetch_article_content(article_url, self.session)
                    
                    if full_content and len(full_content) > len(content):
                        logging.info(f"Successfully retrieved full content for: {article_url}")
                        return full_content
                except Exception as e:
                    logging.warning(f"Error fetching full content: {str(e)}")
        
        return content

    @track_performance
    def process_feeds(self):
        """
        Process all RSS feeds and generate summaries.
        
        This is the main method that orchestrates the full process:
        1. Fetch and parse feeds
        2. Cluster articles using enhanced clustering
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

            # Apply time range filtering if enabled
            time_range_hours = int(os.environ.get('TIME_RANGE_HOURS', '0'))
            if time_range_hours > 0:
                filtered_articles = self._filter_articles_by_date(all_articles, time_range_hours)
                logging.info(f"Filtered articles from {len(all_articles)} to {len(filtered_articles)} using {time_range_hours} hour time range")
                all_articles = filtered_articles
                
                if not all_articles:
                    logging.warning("No articles remaining after time range filtering")
                    return None

            if not all_articles:
                logging.error("No articles collected from any feeds")
                return None

            # Cluster the articles using the improved clusterer
            logging.info("Clustering similar articles...")
            
            # Use a different clustering method depending on which clusterer we have
            if ENHANCED_CLUSTERING_AVAILABLE and hasattr(self.clusterer, 'cluster_with_summaries'):
                # Use the enhanced clustering with summaries if available
                clusters = self.clusterer.cluster_with_summaries(all_articles)
                logging.info(f"Created {len(clusters)} clusters with enhanced clustering")
            else:
                # Fall back to basic clustering
                clusters = self.clusterer.cluster_articles(all_articles)
                logging.info(f"Created {len(clusters)} clusters with basic clustering")

            if not clusters:
                logging.error("No clusters created")
                return None

            # Generate summaries for each cluster
            logging.info("Generating summaries for article clusters...")
            processed_clusters = self.process_cluster_summaries(clusters)

            if not processed_clusters:
                logging.error("No clusters were successfully processed")
                return None

            logging.info(f"Successfully processed {len(processed_clusters)} clusters")

            # Store the processed clusters for web server access
            self.last_processed_clusters = processed_clusters
            
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

    @track_performance
    def _process_feed(self, feed_url):
        """
        Process a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            list: Processed articles from the feed
        """
        try:
            logging.info(f"Processing feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            articles = []

            if feed.entries:
                feed_title = feed.feed.get('title', feed_url)
                logging.info(f"📰 Found {len(feed.entries)} articles in feed: {feed_url}")

                for entry in feed.entries[:self.batch_size]:
                    article = self._parse_entry(entry, feed_title)
                    if article:
                        articles.append(article)

            logging.info(f"Successfully processed {len(articles)} articles from {feed_url}")
            return articles

        except Exception as e:
            logging.error(f"Error processing feed {feed_url}: {str(e)}")
            return []

    def _generate_summary(self, article_text, title, url):
        """
        Generate a summary for an article using the ArticleSummarizer.
        
        Args:
            article_text: Text of the article to summarize
            title: Title of the article
            url: URL of the article
            
        Returns:
            dict: Summary with headline and content
        """
        try:
            summary = self.summarizer.summarize_article(article_text, title, url)
            
            # Ensure the summary has the correct structure
            if not isinstance(summary, dict):
                summary = {
                    'headline': title,
                    'summary': str(summary)
                }
            elif 'headline' not in summary or 'summary' not in summary:
                # Fix incomplete summary structure
                summary_text = summary.get('summary', '')
                if not summary_text and isinstance(summary, str):
                    summary_text = summary
                    
                summary = {
                    'headline': summary.get('headline', title),
                    'summary': summary_text or 'No summary available.'
                }
                
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {
                'headline': title,
                'summary': "Summary generation failed. Please try again later."
            }

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
            
    @track_performance
    def generate_html_output(self, clusters):
        """
        Generate HTML output from the processed clusters without Flask dependency.
        
        Args:
            clusters: List of article clusters with summaries
            
        Returns:
            str: Path to generated HTML file or None if generation failed
        """
        try:
            # Create output directory if it doesn't exist (using absolute path)
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f'rss_summary_{timestamp}.html')
            
            # Check if the template file exists
            template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
            template_file = os.path.join(template_dir, 'feed-summary.html')
            
            if not os.path.exists(template_file):
                logging.error(f"Template file not found: {template_file}")
                # Create a basic fallback template if the template is missing
                return self._generate_fallback_html(clusters, output_file)
            
            try:
                # Use Jinja2 directly instead of Flask
                from jinja2 import Environment, FileSystemLoader, select_autoescape
                
                # Create Jinja2 environment
                env = Environment(
                    loader=FileSystemLoader(template_dir),
                    autoescape=select_autoescape(['html', 'xml'])
                )
                
                # Load the template
                template = env.get_template('feed-summary.html')
                
                # Render the template
                html_content = template.render(
                    clusters=clusters,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                # Write to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logging.info(f"Successfully wrote HTML output to {output_file}")
                return output_file
                
            except Exception as render_error:
                logging.error(f"Error rendering template: {str(render_error)}")
                # Try fallback method if template rendering fails
                return self._generate_fallback_html(clusters, output_file)
                
        except Exception as e:
            logging.error(f"Error generating HTML output: {str(e)}", exc_info=True)
            return None
            
    def _generate_fallback_html(self, clusters, output_file):
        """
        Generate a basic HTML output without using templates as a fallback.
        
        Args:
            clusters: List of article clusters
            output_file: Path to output file
            
        Returns:
            str: Path to generated HTML file or None if generation failed
        """
        try:
            html = ['<!DOCTYPE html><html><head><title>RSS Summary</title>',
                    '<meta charset="UTF-8">',
                    '<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px}',
                    '.cluster{border:1px solid #ddd;margin-bottom:20px;padding:15px;border-radius:5px}',
                    '.article{border-bottom:1px solid #eee;padding:10px 0}',
                    '.article:last-child{border-bottom:none}',
                    '.article-title a{color:#2563eb;text-decoration:none}',
                    '.article-title a:hover{text-decoration:underline}',
                    '.timestamp{color:#666;font-style:italic}</style></head><body>',
                    f'<h1>RSS Summary</h1>',
                    f'<p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>']
            
            for cluster in clusters:
                html.append('<div class="cluster">')
                if cluster and len(cluster) > 0:
                    # Get headline from the first article's summary if available
                    headline = cluster[0].get('summary', {}).get('headline', cluster[0].get('title', 'Untitled'))
                    html.append(f'<h2>{headline}</h2>')
                    
                    # Add summary if available
                    summary = cluster[0].get('summary', {}).get('summary', '')
                    if summary:
                        html.append(f'<div class="summary">{summary}</div>')
                    
                    # Add all articles in the cluster
                    for article in cluster:
                        html.append('<div class="article">')
                        html.append(f'<h3 class="article-title"><a href="{article.get("link", "#")}" target="_blank">{article.get("title", "No title")}</a></h3>')
                        html.append(f'<p class="article-meta">Source: {article.get("feed_source", "Unknown")} | Published: {article.get("published", "Unknown date")}</p>')
                        html.append('</div>')
                html.append('</div>')
            
            html.append('</body></html>')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html))
            
            logging.info(f"Successfully wrote fallback HTML output to {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"Error generating fallback HTML: {str(e)}", exc_info=True)
            return None