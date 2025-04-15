"""
Web server for RSS Reader that displays summarized articles in a browser.
"""

import os
import logging
import sys
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, jsonify

# Import in a try/except block to provide better error messages
try:
    from reader import RSSReader
except ImportError:
    print("Error: Could not import RSSReader. Make sure reader.py is in the same directory.")
    sys.exit(1)

# Set environment variable to enable paywall bypass
os.environ['ENABLE_PAYWALL_BYPASS'] = 'true'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Store the latest processed data
latest_data = {
    'clusters': [],
    'timestamp': None,
    'output_file': None,
    'raw_clusters': []  # Store the raw clusters for debugging
}

@app.route('/')
def index():
    """Render the main page with the latest summaries or a welcome page if none exist."""
    if latest_data['clusters']:
        return render_template(
            'feed-summary.html',
            clusters=latest_data['clusters'],
            timestamp=latest_data['timestamp']
        )
    else:
        return render_template(
            'welcome.html',
            has_default_feeds=os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt'))
        )

@app.route('/refresh', methods=['POST'])
def refresh_feeds():
    """Process RSS feeds and update the latest data."""
    try:
        # Check if we should use default feeds
        use_default = request.form.get('use_default', 'false').lower() == 'true'
        
        # Get optional parameters from the form
        feeds = request.form.get('feeds', '').strip()
        feeds_list = None
        
        # If use_default is true, leave feeds_list as None to use default feeds
        # Otherwise, parse the feeds from the textarea
        if not use_default and feeds:
            feeds_list = [url.strip() for url in feeds.split('\n') if url.strip()]
        
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
        
        # Log what we're doing
        if use_default:
            logging.info("Processing default feeds from rss_feeds.txt")
        elif feeds_list:
            logging.info(f"Processing {len(feeds_list)} custom feeds")
        else:
            logging.info("No feeds specified, will use default feeds")
        
        # Initialize and run RSS reader with paywall bypass enabled
        # Environment variable ENABLE_PAYWALL_BYPASS is already set at the top of the file
        reader = RSSReader(
            feeds=feeds_list,
            batch_size=batch_size,
            batch_delay=batch_delay
        )
        
        # Process feeds and get output file
        output_file = reader.process_feeds()
        
        if output_file:
            # Get the clusters from the reader
            clusters = reader.last_processed_clusters
            
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
            return render_template('error.html', message="No articles found or processed. Check the logs for details.")
    
    except Exception as e:
        logging.error(f"Error refreshing feeds: {str(e)}", exc_info=True)
        return render_template('error.html', message=f"Error: {str(e)}")

@app.route('/status')
def status():
    """Return the current status of the RSS reader."""
    return jsonify({
        'has_data': bool(latest_data['clusters']),
        'last_updated': latest_data['timestamp'],
        'article_count': sum(len(cluster) for cluster in latest_data['clusters']) if latest_data['clusters'] else 0,
        'cluster_count': len(latest_data['clusters']) if latest_data['clusters'] else 0
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
        'clusters': []
    }
    
    for i, cluster in enumerate(latest_data['clusters']):
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
        
        debug_info['clusters'].append(cluster_info)
    
    return jsonify(debug_info)

def initialize_data():
    """Initialize the latest data by running the RSS reader once at startup."""
    try:
        logging.info("Initializing RSS reader with paywall bypass enabled...")
        reader = RSSReader()
        output_file = reader.process_feeds()
        
        if output_file:
            # Get the clusters from the reader
            clusters = reader.last_processed_clusters
            
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