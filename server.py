"""
Web server for RSS Reader that displays summarized articles in a browser.
Enhanced with controls for clustering settings.
"""

import os
import logging
import sys
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, render_template, redirect, url_for, request, jsonify, session

# Import in a try/except block to provide better error messages
try:
    from reader import RSSReader
    from utils.http import create_http_session
    from utils.archive import fetch_article_content
    from summarizer import ArticleSummarizer
    from lm_cluster_analyzer import create_cluster_analyzer
except ImportError:
    print("Error: Could not import required modules. Make sure all files are in the correct directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Set a secret key for session management
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Store the latest processed data
latest_data = {
    'clusters': [],
    'timestamp': None,
    'output_file': None,
    'raw_clusters': []  # Store the raw clusters for debugging
}

# Default clustering settings
DEFAULT_CLUSTERING_SETTINGS = {
    'enable_multi_article': True,
    'similarity_threshold': 0.7,
    'max_articles_per_batch': 5,
    'use_enhanced_clustering': True
}

def sort_clusters(clusters):
    """
    Sort clusters in the following order:
    1. Largest clusters first
    2. Stories from Techmeme
    3. Stories from reputable news sources
    4. Stories from technology companies
    5. Stories from everyone else
    """
    # Define lists of source domains for categorization
    techmeme_sources = ['techmeme.com']
    
    reputable_news_sources = [
        'nytimes.com', 'washingtonpost.com', 'wsj.com', 'reuters.com',
        'bloomberg.com', 'ft.com', 'economist.com', 'bbc.com', 'bbc.co.uk',
        'apnews.com', 'npr.org', 'cnn.com', 'cnbc.com', 'theverge.com',
        'wired.com', 'arstechnica.com', 'techcrunch.com', 'engadget.com'
    ]
    
    tech_company_sources = [
        'blog.google', 'blog.microsoft.com', 'apple.com', 'amazon.com', 
        'meta.com', 'facebook.com', 'engineering.fb.com', 'developer.apple.com',
        'azure.microsoft.com', 'aws.amazon.com', 'blog.twitter.com',
        'developer.android.com', 'developer.mozilla.org', 'netflix.com',
        'engineering.linkedin.com', 'github.blog', 'medium.engineering',
        'instagram-engineering.com', 'engineering.pinterest.com',
        'slack.engineering', 'dropbox.tech', 'spotify.engineering'
    ]
    
    # Define category scores (higher is more important)
    def get_source_score(article):
        if not article or 'feed_source' not in article:
            return 0
        
        source = article.get('feed_source', '').lower()
        
        for techmeme in techmeme_sources:
            if techmeme in source:
                return 4
        
        for reputable in reputable_news_sources:
            if reputable in source:
                return 3
        
        for tech_company in tech_company_sources:
            if tech_company in source:
                return 2
        
        return 1  # Everyone else
    
    # Create a sorting function
    def cluster_sort_key(cluster):
        if not cluster:
            return (-1, 0)  # Empty clusters go last
        
        # Primary sort by cluster size (descending)
        cluster_size = len(cluster)
        
        # Secondary sort by source category
        # Get the source from the first article in the cluster
        source_score = get_source_score(cluster[0])
        
        return (-cluster_size, -source_score)  # Negative to sort in descending order
    
    # Sort the clusters using the sorting function
    sorted_clusters = sorted(clusters, key=cluster_sort_key)
    
    # Log the sorting results for debugging
    logging.info(f"Sorted {len(sorted_clusters)} clusters")
    if sorted_clusters:
        for i, cluster in enumerate(sorted_clusters[:5]):  # Log first 5 clusters
            if cluster:
                source = cluster[0].get('feed_source', 'Unknown')
                logging.info(f"Cluster {i}: size={len(cluster)}, source={source}")
    
    return sorted_clusters

def get_clustering_settings():
    """Get current clustering settings from session or use defaults."""
    if 'clustering_settings' not in session:
        session['clustering_settings'] = DEFAULT_CLUSTERING_SETTINGS.copy()
    return session['clustering_settings']

@app.route('/')
def index():
    """Render the main page with the latest summaries or a welcome page if none exist."""
    # Initialize paywall_bypass status in session if not already set
    if 'paywall_bypass_enabled' not in session:
        session['paywall_bypass_enabled'] = False
    
    # Initialize feeds_list and use_default in session if not already set
    if 'feeds_list' not in session:
        session['feeds_list'] = None
    if 'use_default' not in session:
        session['use_default'] = True
    
    # Initialize clustering settings if not already set
    clustering_settings = get_clustering_settings()
    
    if latest_data['clusters']:
        # Sort the clusters before passing to the template
        sorted_clusters = sort_clusters(latest_data['clusters'])
        
        return render_template(
            'feed-summary.html',
            clusters=sorted_clusters,
            timestamp=latest_data['timestamp'],
            paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
            clustering_settings=clustering_settings
        )
    else:
        return render_template(
            'welcome.html',
            has_default_feeds=os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
            paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
            clustering_settings=clustering_settings
        )

# Route for the welcome page
@app.route('/welcome')
def welcome():
    """Explicitly render the welcome page regardless of data state."""
    # Get current clustering settings
    clustering_settings = get_clustering_settings()
    
    return render_template(
        'welcome.html',
        has_default_feeds=os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
        paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
        clustering_settings=clustering_settings
    )

@app.route('/toggle_paywall_bypass', methods=['POST'])
def toggle_paywall_bypass():
    """Toggle the paywall bypass setting."""
    current_status = session.get('paywall_bypass_enabled', False)
    session['paywall_bypass_enabled'] = not current_status
    
    # Log the change
    if session['paywall_bypass_enabled']:
        logging.info("Paywall bypass enabled by user")
    else:
        logging.info("Paywall bypass disabled by user")
    
    return redirect(request.referrer or url_for('index'))

@app.route('/update_clustering_settings', methods=['POST'])
def update_clustering_settings():
    """Update clustering settings from form submission."""
    # Get current settings
    clustering_settings = get_clustering_settings()
    
    # Update settings from form
    clustering_settings['enable_multi_article'] = request.form.get('enable_multi_article') == 'on'
    clustering_settings['use_enhanced_clustering'] = request.form.get('use_enhanced_clustering') == 'on'
    
    # Parse numeric values with error handling
    try:
        similarity = float(request.form.get('similarity_threshold', 0.7))
        clustering_settings['similarity_threshold'] = max(0.0, min(1.0, similarity))
    except (ValueError, TypeError):
        # Keep existing value on error
        pass
        
    try:
        max_articles = int(request.form.get('max_articles_per_batch', 5))
        clustering_settings['max_articles_per_batch'] = max(1, min(10, max_articles))
    except (ValueError, TypeError):
        # Keep existing value on error
        pass
    
    # Store updated settings in session
    session['clustering_settings'] = clustering_settings
    
    # Log the settings update
    logging.info(f"Updated clustering settings: {clustering_settings}")
    
    return redirect(request.referrer or url_for('index'))

@app.route('/reset_clustering_settings', methods=['POST'])
def reset_clustering_settings():
    """Reset clustering settings to defaults."""
    session['clustering_settings'] = DEFAULT_CLUSTERING_SETTINGS.copy()
    logging.info("Reset clustering settings to defaults")
    return redirect(request.referrer or url_for('index'))

@app.route('/refresh', methods=['POST'])
def refresh_feeds():
    """Process RSS feeds and update the latest data."""
    try:
        # Check if form has a new feed submission
        feeds_from_form = request.form.get('feeds', '').strip()
        use_default_from_form = request.form.get('use_default', 'false').lower() == 'true'
        
        # If there's form data for feeds, update the session
        if feeds_from_form or 'use_default' in request.form:
            # Store in session whether we're using default feeds or not
            session['use_default'] = use_default_from_form
            
            # If using custom feeds, store the feed list in session
            if not use_default_from_form and feeds_from_form:
                feeds_list = [url.strip() for url in feeds_from_form.split('\n') if url.strip()]
                session['feeds_list'] = feeds_list
                logging.info(f"Storing {len(feeds_list)} custom feeds in session")
            else:
                # If using default feeds, clear any stored custom feeds
                session['feeds_list'] = None
        
        # Now use the stored feeds from session for processing
        use_default = session.get('use_default', True)
        feeds_list = session.get('feeds_list')
        
        # Get optional parameters from the form or use defaults
        batch_size = request.form.get('batch_size', 25)
        try:
            batch_size = int(batch_size)
        except ValueError:
            batch_size = 25
            
        batch_delay = request.form.get('batch_delay', 15)
        try:
            batch_delay = int(batch_delay)
        except ValueError:
            batch_delay = 15
        
        # Get clustering settings
        clustering_settings = get_clustering_settings()
        
        # Set environment variables for clustering options
        os.environ['ENABLE_MULTI_ARTICLE_CLUSTERING'] = 'true' if clustering_settings['enable_multi_article'] else 'false'
        os.environ['MIN_SIMILARITY_THRESHOLD'] = str(clustering_settings['similarity_threshold'])
        os.environ['MAX_ARTICLES_PER_BATCH'] = str(clustering_settings['max_articles_per_batch'])
        os.environ['USE_ENHANCED_CLUSTERING'] = 'true' if clustering_settings['use_enhanced_clustering'] else 'false'
        
        # Log what we're doing
        if use_default:
            logging.info("Processing default feeds from rss_feeds.txt")
        elif feeds_list:
            logging.info(f"Processing {len(feeds_list)} custom feeds from session")
        else:
            logging.info("No feeds specified and no session data, will use default feeds")
            use_default = True
        
        # Set the environment variable for paywall bypass based on user preference
        if session.get('paywall_bypass_enabled', False):
            os.environ['ENABLE_PAYWALL_BYPASS'] = 'true'
            logging.info("Using paywall bypass as per user setting")
        else:
            os.environ['ENABLE_PAYWALL_BYPASS'] = 'false'
            logging.info("Paywall bypass disabled as per user setting")
        
        # Import the enhanced RSS reader with multi-article clustering
        from main import EnhancedRSSReader
        
        # Initialize and run RSS reader with clustering settings
        reader = EnhancedRSSReader(
            feeds=feeds_list if not use_default else None,
            batch_size=batch_size,
            batch_delay=batch_delay
        )
        
        # Process feeds and get output file
        import asyncio
        output_file = asyncio.run(reader.process_feeds())
        
        if output_file:
            # Get the clusters from the reader
            clusters = reader.reader.last_processed_clusters
            
            # Fix: Ensure every cluster has proper summaries attached to the first article
            for cluster in clusters:
                if cluster and len(cluster) > 0:
                    # Get the first article in the cluster
                    first_article = cluster[0]
                    
                    # Check if the article has a summary
                    if 'summary' not in first_article or first_article['summary'] is None:
                        # No summary exists, create a default one
                        logging.warning(f"No summary found for cluster with article: {first_article.get('title')}")
                        first_article['summary'] = {
                            'headline': first_article.get('title', 'News Article'),
                            'summary': f"This is a cluster of {len(cluster)} related articles about {first_article.get('title', 'various topics')}."
                        }
                    elif isinstance(first_article['summary'], str):
                        # Summary is a string, convert to proper dict format
                        summary_text = first_article['summary']
                        first_article['summary'] = {
                            'headline': first_article.get('title', 'News Article'),
                            'summary': summary_text
                        }
                    elif isinstance(first_article['summary'], dict):
                        # Summary is a dict, ensure it has the required fields
                        if 'headline' not in first_article['summary']:
                            first_article['summary']['headline'] = first_article.get('title', 'News Article')
                        if 'summary' not in first_article['summary']:
                            first_article['summary']['summary'] = f"This is a cluster of {len(cluster)} related articles."
                    
                    # Debug log to verify the summary structure
                    logging.info(f"Cluster summary: {first_article['summary']}")
            
            # Update latest data with the modified clusters
            latest_data['clusters'] = clusters
            latest_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            latest_data['output_file'] = output_file
            
            logging.info(f"Successfully refreshed feeds: {output_file}")
            return redirect(url_for('index'))
        else:
            logging.warning("No articles found or processed")
            return render_template('error.html', 
                                  message="No articles found or processed. Check the logs for details.",
                                  paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
                                  clustering_settings=clustering_settings)
    
    except Exception as e:
        logging.error(f"Error refreshing feeds: {str(e)}", exc_info=True)
        return render_template('error.html', 
                              message=f"Error: {str(e)}",
                              paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
                              clustering_settings=get_clustering_settings())

# Route to clear all data and return to welcome page
@app.route('/clear', methods=['GET', 'POST'])
def clear_data():
    """Clear all data and return to welcome page."""
    global latest_data
    latest_data = {
        'clusters': [],
        'timestamp': None,
        'output_file': None,
        'raw_clusters': []
    }
    logging.info("Data cleared by user")
    return redirect(url_for('welcome'))

# Route for summarizing a single URL
@app.route('/summarize', methods=['GET', 'POST'])
def summarize_single():
    """Handle summarization of a single URL."""
    # Get clustering settings
    clustering_settings = get_clustering_settings()
    
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        
        if not url:
            return render_template('error.html', 
                                  message="Please provide a valid URL to summarize.",
                                  paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
                                  clustering_settings=clustering_settings)
        
        # Add http:// if missing
        if not url.startswith('http'):
            url = 'https://' + url
        
        try:
            # Initialize summarizer (reusing existing code)
            summarizer = ArticleSummarizer()
            
            # Fetch the article content
            session_obj = create_http_session()
            
            # Set paywall bypass environment variable based on user preference
            if session.get('paywall_bypass_enabled', False):
                os.environ['ENABLE_PAYWALL_BYPASS'] = 'true'
                logging.info("Using paywall bypass as per user setting")
            else:
                os.environ['ENABLE_PAYWALL_BYPASS'] = 'false'
                logging.info("Paywall bypass disabled as per user setting")
            
            # Fetch article content (reusing existing code)
            content = fetch_article_content(url, session_obj)
            
            if not content or len(content) < 100:
                return render_template('error.html', 
                                      message="Could not extract sufficient content from the provided URL.",
                                      paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
                                      clustering_settings=clustering_settings)
            
            # Extract title from URL as a fallback
            domain = urlparse(url).netloc
            title = f"Article from {domain}"
            
            # Generate summary
            summary = summarizer.summarize_article(
                text=content,
                title=title,
                url=url
            )
            
            # Create a fake "cluster" for template compatibility
            fake_cluster = [{
                'title': summary.get('headline', title),
                'link': url,
                'feed_source': domain,
                'published': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'summary': summary
            }]
            
            # Render the result using existing template
            return render_template(
                'single-summary.html',
                url=url,
                cluster=fake_cluster,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
                clustering_settings=clustering_settings
            )
            
        except Exception as e:
            logging.error(f"Error summarizing URL: {str(e)}", exc_info=True)
            return render_template('error.html', 
                                  message=f"Error summarizing URL: {str(e)}",
                                  paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
                                  clustering_settings=clustering_settings)
    
    # GET request - show form
    return render_template(
        'summarize-form.html',
        paywall_bypass_enabled=session.get('paywall_bypass_enabled', False),
        clustering_settings=clustering_settings
    )

@app.route('/status')
def status():
    """Return the current status of the RSS reader."""
    clustering_settings = get_clustering_settings()
    
    return jsonify({
        'has_data': bool(latest_data['clusters']),
        'last_updated': latest_data['timestamp'],
        'article_count': sum(len(cluster) for cluster in latest_data['clusters']) if latest_data['clusters'] else 0,
        'cluster_count': len(latest_data['clusters']) if latest_data['clusters'] else 0,
        'paywall_bypass_enabled': session.get('paywall_bypass_enabled', False),
        'using_default_feeds': session.get('use_default', True),
        'custom_feed_count': len(session.get('feeds_list', [])) if session.get('feeds_list') else 0,
        'clustering_settings': clustering_settings
    })

@app.route('/debug')
def debug():
    """Return detailed debug information about the current data."""
    if not latest_data['clusters']:
        return jsonify({
            'status': 'No data available',
            'timestamp': None
        })
    
    # Create a simplified view of the data structure for debugging
    debug_info = {
        'timestamp': latest_data['timestamp'],
        'cluster_count': len(latest_data['clusters']),
        'paywall_bypass_enabled': session.get('paywall_bypass_enabled', False),
        'using_default_feeds': session.get('use_default', True),
        'custom_feed_count': len(session.get('feeds_list', [])) if session.get('feeds_list') else 0,
        'clustering_settings': get_clustering_settings(),
        'clusters': []
    }
    
    # Use sorted clusters for debugging too
    sorted_clusters = sort_clusters(latest_data['clusters'])
    
    for i, cluster in enumerate(sorted_clusters):
        cluster_info = {
            'id': i,
            'article_count': len(cluster),
            'sample_article': {}
        }
        
        if cluster:
            article = cluster[0]
            cluster_info['sample_article'] = {
                'title': article.get('title', 'Unknown'),
                'source': article.get('feed_source', 'Unknown'),
                'has_summary': 'summary' in article and bool(article['summary']),
                'summary_structure': str(type(article.get('summary', None)))
            }
            
            if 'summary' in article and article['summary']:
                cluster_info['sample_article']['headline'] = article['summary'].get('headline', 'None')
                # Truncate summary for display
                summary = article['summary'].get('summary', 'None')
                cluster_info['sample_article']['summary_preview'] = summary[:100] + '...' if len(summary) > 100 else summary
            
            # Add topics if available
            if 'cluster_topics' in article:
                cluster_info['topics'] = article.get('cluster_topics', [])
        
        debug_info['clusters'].append(cluster_info)
    
    return jsonify(debug_info)

@app.route('/test_route')
def test_route():
    """Test route to verify routing is working."""
    return "Routing is working correctly!"

def initialize_data():
    """Initialize the latest data by running the RSS reader once at startup."""
    try:
        # Default to paywall bypass disabled
        os.environ['ENABLE_PAYWALL_BYPASS'] = 'false'
        
        # Set default clustering settings
        os.environ['ENABLE_MULTI_ARTICLE_CLUSTERING'] = 'true'
        os.environ['MIN_SIMILARITY_THRESHOLD'] = '0.7'
        os.environ['MAX_ARTICLES_PER_BATCH'] = '5'
        os.environ['USE_ENHANCED_CLUSTERING'] = 'true'
        
        logging.info("Initializing RSS reader with enhanced multi-article clustering...")
        
        # Import the enhanced reader
        from main import EnhancedRSSReader
        reader = EnhancedRSSReader()
        
        # Process feeds asynchronously
        import asyncio
        output_file = asyncio.run(reader.process_feeds())
        
        if output_file:
            # Get the clusters from the reader
            clusters = reader.reader.last_processed_clusters
            
            # Fix: Ensure every cluster has proper summaries
            for cluster in clusters:
                if cluster and len(cluster) > 0:
                    # Get the first article in the cluster
                    first_article = cluster[0]
                    
                    # Check if the article has a summary
                    if 'summary' not in first_article or first_article['summary'] is None:
                        # No summary exists, create a default one
                        logging.warning(f"No summary found for cluster with article: {first_article.get('title')}")
                        first_article['summary'] = {
                            'headline': first_article.get('title', 'News Article'),
                            'summary': f"This is a cluster of {len(cluster)} related articles about {first_article.get('title', 'various topics')}."
                        }
                    elif isinstance(first_article['summary'], str):
                        # Summary is a string, convert to proper dict format
                        summary_text = first_article['summary']
                        first_article['summary'] = {
                            'headline': first_article.get('title', 'News Article'),
                            'summary': summary_text
                        }
                    elif isinstance(first_article['summary'], dict):
                        # Summary is a dict, ensure it has the required fields
                        if 'headline' not in first_article['summary']:
                            first_article['summary']['headline'] = first_article.get('title', 'News Article')
                        if 'summary' not in first_article['summary']:
                            first_article['summary']['summary'] = f"This is a cluster of {len(cluster)} related articles."
            
            # Update latest data with the modified clusters
            latest_data['clusters'] = clusters
            latest_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            latest_data['output_file'] = output_file
            
            logging.info(f"Successfully initialized RSS reader with data: {output_file}")
        else:
            logging.warning("No articles found or processed during initialization")
    
    except Exception as e:
        logging.error(f"Error initializing RSS reader: {str(e)}", exc_info=True)

# Helper function to determine if we're in a production environment
def is_production():
    """Check if we're running in a production environment."""
    # This is a simple check - you might want to use a more robust method
    # like checking for environment variables
    return not sys.flags.debug

if __name__ == '__main__':
    # Better handling of production vs development environment
    debug_mode = False  # Never run with debug=True in production
    
    # By default, only bind to localhost in production for security
    host = '127.0.0.1'  # Only accept connections from the local machine
    
    # Check if this is explicitly requested to be exposed externally
    if '--public' in sys.argv:
        host = '0.0.0.0'
        logging.warning("Running with public access (0.0.0.0). Make sure this is intended and secured.")
    
    # Check if initialization is requested
    if '--init' in sys.argv:
        initialize_data()
        
    # Use a safer configuration for the web server
    port = int(os.environ.get('PORT', 5005))
    logging.info(f"Starting server on {host}:{port} (debug={debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)