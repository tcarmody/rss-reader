"""
Web server for RSS Reader that displays summarized articles in a browser.
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, jsonify
from reader import RSSReader

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
            'welcome.html'
        )

@app.route('/refresh', methods=['POST'])
def refresh_feeds():
    """Process RSS feeds and update the latest data."""
    try:
        # Get optional parameters from the form
        feeds = request.form.get('feeds', '').strip()
        feeds_list = [url.strip() for url in feeds.split('\n') if url.strip()] if feeds else None
        
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
        
        # Initialize and run RSS reader
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
            
            # Very simple fix: Add a summary to the first article in each cluster
            for cluster in clusters:
                if cluster and len(cluster) > 0:
                    # Create a simple summary based on the cluster's first article title
                    first_article = cluster[0]
                    
                    # Create a summary object with the expected structure if it doesn't exist
                    if not hasattr(first_article, 'summary') or not first_article.get('summary'):
                        # Generate a simple summary from the article title
                        first_article['summary'] = {
                            'headline': first_article.get('title', 'News Article'),
                            'summary': f"This is a cluster of {len(cluster)} related articles about {first_article.get('title', 'various topics')}."
                        }
                    # If summary exists but isn't in the right format, fix it
                    elif isinstance(first_article.get('summary'), str) or not isinstance(first_article.get('summary'), dict):
                        summary_text = str(first_article.get('summary', ''))
                        first_article['summary'] = {
                            'headline': first_article.get('title', 'News Article'),
                            'summary': summary_text if summary_text else f"This is a cluster of {len(cluster)} related articles."
                        }
                    # If summary is a dict but missing required fields
                    elif isinstance(first_article.get('summary'), dict):
                        summary_dict = first_article.get('summary', {})
                        if 'headline' not in summary_dict:
                            summary_dict['headline'] = first_article.get('title', 'News Article')
                        if 'summary' not in summary_dict:
                            summary_dict['summary'] = f"This is a cluster of {len(cluster)} related articles."
                        first_article['summary'] = summary_dict
            
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
