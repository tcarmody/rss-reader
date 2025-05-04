#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web server for RSS Reader using FastAPI.
"""

# Apply streamlined batch processing fix
import batch_processing
batch_processing.apply()

import os
import logging
import sys
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(title="Data Points AI - RSS Reader")

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=os.environ.get('SECRET_KEY', os.urandom(24)))

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static')), name="static")

# Configure templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))

# Store the latest processed data
latest_data = {
    'clusters': [],
    'timestamp': None,
    'output_file': None,
    'raw_clusters': []
}

# Default clustering settings
DEFAULT_CLUSTERING_SETTINGS = {
    'enable_multi_article': True,
    'similarity_threshold': 0.7,
    'max_articles_per_batch': 5,
    'use_enhanced_clustering': True,
    'time_range_enabled': True,
    'time_range_value': 168,
    'time_range_unit': 'hours'
}

# Helper functions for session management
def get_clustering_settings(request: Request):
    """Get current clustering settings from session or use defaults."""
    if 'clustering_settings' not in request.session:
        request.session['clustering_settings'] = DEFAULT_CLUSTERING_SETTINGS.copy()
    return request.session['clustering_settings']

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page with the latest summaries or a welcome page if none exist."""
    # Initialize session variables if not set
    if 'paywall_bypass_enabled' not in request.session:
        request.session['paywall_bypass_enabled'] = False
    
    if 'feeds_list' not in request.session:
        request.session['feeds_list'] = None
    if 'use_default' not in request.session:
        request.session['use_default'] = True
    
    clustering_settings = get_clustering_settings(request)
    
    if latest_data['clusters']:
        sorted_clusters = sort_clusters(latest_data['clusters'])
        
        return templates.TemplateResponse(
            "feed-summary.html",
            {
                "request": request,
                "clusters": sorted_clusters,
                "timestamp": latest_data['timestamp'],
                "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                "clustering_settings": clustering_settings
            }
        )
    else:
        return templates.TemplateResponse(
            "welcome.html",
            {
                "request": request,
                "has_default_feeds": os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
                "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                "clustering_settings": clustering_settings
            }
        )

@app.get("/welcome", response_class=HTMLResponse)
async def welcome(request: Request):
    """Explicitly render the welcome page regardless of data state."""
    clustering_settings = get_clustering_settings(request)
    
    return templates.TemplateResponse(
        "welcome.html",
        {
            "request": request,
            "has_default_feeds": os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
            "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
            "clustering_settings": clustering_settings
        }
    )

@app.post("/toggle_paywall_bypass")
async def toggle_paywall_bypass(request: Request):
    """Toggle the paywall bypass setting."""
    current_status = request.session.get('paywall_bypass_enabled', False)
    request.session['paywall_bypass_enabled'] = not current_status
    
    # Log the change
    if request.session['paywall_bypass_enabled']:
        logging.info("Paywall bypass enabled by user")
    else:
        logging.info("Paywall bypass disabled by user")
    
    referer = request.headers.get('referer', '/')
    return RedirectResponse(url=referer, status_code=303)

@app.post("/update_clustering_settings")
async def update_clustering_settings(
    request: Request,
    enable_multi_article: Optional[str] = Form(None),
    use_enhanced_clustering: Optional[str] = Form(None),
    time_range_enabled: Optional[str] = Form(None),
    similarity_threshold: float = Form(0.7),
    max_articles_per_batch: int = Form(5),
    time_range_value: int = Form(168),
    time_range_unit: str = Form('hours')
):
    """Update clustering settings from form submission."""
    clustering_settings = get_clustering_settings(request)
    
    # Update boolean settings
    clustering_settings['enable_multi_article'] = enable_multi_article == 'on'
    clustering_settings['use_enhanced_clustering'] = use_enhanced_clustering == 'on'
    clustering_settings['time_range_enabled'] = time_range_enabled == 'on'
    
    # Update numeric settings
    clustering_settings['similarity_threshold'] = max(0.0, min(1.0, similarity_threshold))
    clustering_settings['max_articles_per_batch'] = max(1, min(10, max_articles_per_batch))
    clustering_settings['time_range_value'] = max(1, time_range_value)
    
    # Update time unit
    if time_range_unit in ['hours', 'days', 'weeks', 'months']:
        clustering_settings['time_range_unit'] = time_range_unit
    
    # Store updated settings in session
    request.session['clustering_settings'] = clustering_settings
    
    # Log the settings update
    logging.info(f"Updated clustering settings: {clustering_settings}")
    
    referer = request.headers.get('referer', '/')
    return RedirectResponse(url=referer, status_code=303)

@app.post("/refresh")
async def refresh_feeds(
    request: Request,
    feeds: Optional[str] = Form(None),
    use_default: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(25),
    batch_delay: Optional[int] = Form(15)
):
    """Process RSS feeds and update the latest data."""
    try:
        # Handle form data
        feeds_from_form = feeds.strip() if feeds else ''
        use_default_from_form = use_default == 'true' if use_default else False
        
        # Update session
        if feeds_from_form or use_default is not None:
            request.session['use_default'] = use_default_from_form
            
            if not use_default_from_form and feeds_from_form:
                feeds_list = [url.strip() for url in feeds_from_form.split('\n') if url.strip()]
                request.session['feeds_list'] = feeds_list
                logging.info(f"Storing {len(feeds_list)} custom feeds in session")
            else:
                request.session['feeds_list'] = None
                logging.info("Clearing custom feeds, will use defaults")
        
        # Get feeds from session
        use_default = request.session.get('use_default', True)
        feeds_list = request.session.get('feeds_list')
        
        # Get clustering settings
        clustering_settings = get_clustering_settings(request)
        
        # Calculate time range in hours
        if clustering_settings.get('time_range_enabled', False):
            time_value = clustering_settings.get('time_range_value', 168)
            time_unit = clustering_settings.get('time_range_unit', 'hours')
            
            if time_unit == 'days':
                time_range_hours = time_value * 24
            elif time_unit == 'weeks':
                time_range_hours = time_value * 24 * 7
            elif time_unit == 'months':
                time_range_hours = time_value * 24 * 30
            else:
                time_range_hours = time_value
            
            os.environ['TIME_RANGE_HOURS'] = str(time_range_hours)
            logging.info(f"Time range filter enabled: {time_value} {time_unit} ({time_range_hours} hours)")
        else:
            os.environ['TIME_RANGE_HOURS'] = '0'
            logging.info("Time range filter disabled")
        
        # Set environment variables for clustering options
        os.environ['ENABLE_MULTI_ARTICLE_CLUSTERING'] = 'true' if clustering_settings['enable_multi_article'] else 'false'
        os.environ['MIN_SIMILARITY_THRESHOLD'] = str(clustering_settings['similarity_threshold'])
        os.environ['MAX_ARTICLES_PER_BATCH'] = str(clustering_settings['max_articles_per_batch'])
        os.environ['USE_ENHANCED_CLUSTERING'] = 'true' if clustering_settings['use_enhanced_clustering'] else 'false'
        
        # Set paywall bypass setting
        if request.session.get('paywall_bypass_enabled', False):
            os.environ['ENABLE_PAYWALL_BYPASS'] = 'true'
        else:
            os.environ['ENABLE_PAYWALL_BYPASS'] = 'false'
        
        # Process feeds
        max_workers = clustering_settings.get('max_articles_per_batch', 3)
        
        # Import the enhanced RSS reader
        from main import EnhancedRSSReader
        
        # Initialize and run RSS reader
        feeds_to_pass = None if use_default else feeds_list
        reader = EnhancedRSSReader(
            feeds=feeds_to_pass,
            batch_size=batch_size,
            batch_delay=batch_delay,
            max_workers=max_workers
        )
        
        # Process feeds and get output file
        import asyncio
        output_file = await reader.process_feeds()
        
        # Get the clusters from the reader
        clusters = reader.last_processed_clusters

        if output_file and clusters:
            # Update latest data
            latest_data['clusters'] = clusters
            latest_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            latest_data['output_file'] = output_file
            
            logging.info(f"Successfully refreshed feeds: {output_file}")
            return RedirectResponse(url="/", status_code=303)
        else:
            logging.warning("No articles found or processed")
            return templates.TemplateResponse(
                'error.html', 
                {
                    "request": request,
                    "message": "No articles found or processed. Check the logs for details.",
                    "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                    "clustering_settings": clustering_settings
                }
            )
    
    except Exception as e:
        logging.error(f"Error refreshing feeds: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            'error.html', 
            {
                "request": request,
                "message": f"Error: {str(e)}",
                "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                "clustering_settings": get_clustering_settings(request)
            }
        )

@app.get("/clear")
@app.post("/clear")
async def clear_data():
    """Clear all data and return to welcome page."""
    global latest_data
    latest_data = {
        'clusters': [],
        'timestamp': None,
        'output_file': None,
        'raw_clusters': []
    }
    logging.info("Data cleared by user")
    return RedirectResponse(url="/welcome", status_code=303)

@app.get("/summarize", response_class=HTMLResponse)
async def summarize_single_get(request: Request):
    """Handle GET request for single URL summarization."""
    clustering_settings = get_clustering_settings(request)
    
    return templates.TemplateResponse(
        "summarize-form.html",
        {
            "request": request,
            "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
            "clustering_settings": clustering_settings
        }
    )

@app.post("/summarize")
async def summarize_single_post(
    request: Request,
    url: str = Form(...)
):
    """Handle POST request for single URL summarization."""
    clustering_settings = get_clustering_settings(request)
    
    url = url.strip()
    
    if not url:
        return templates.TemplateResponse(
            'error.html', 
            {
                "request": request,
                "message": "Please provide a valid URL to summarize.",
                "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                "clustering_settings": clustering_settings
            }
        )
    
    # Add http:// if missing
    if not url.startswith('http'):
        url = 'https://' + url
    
    try:
        # Initialize summarizer with enhanced batch processing
        max_workers = clustering_settings.get('max_articles_per_batch', 3)
        summarizer = setup_summarization_engine(max_workers)
        
        # Fetch the article content
        from utils.http import create_http_session
        from utils.archive import fetch_article_content
        
        session_obj = create_http_session()
        
        # Set paywall bypass environment variable
        if request.session.get('paywall_bypass_enabled', False):
            os.environ['ENABLE_PAYWALL_BYPASS'] = 'true'
        else:
            os.environ['ENABLE_PAYWALL_BYPASS'] = 'false'
        
        # Fetch article content
        content = fetch_article_content(url, session_obj)
        
        if not content or len(content) < 100:
            return templates.TemplateResponse(
                'error.html', 
                {
                    "request": request,
                    "message": "Could not extract sufficient content from the provided URL.",
                    "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                    "clustering_settings": clustering_settings
                }
            )
        
        # Extract title from URL as a fallback
        domain = urlparse(url).netloc
        title = f"Article from {domain}"
        
        # Use the summarizer
        if hasattr(summarizer, 'summarize'):
            summary = summarizer.summarize(
                text=content,
                title=title,
                url=url,
                auto_select_model=True
            )
        else:
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
        
        return templates.TemplateResponse(
            'single-summary.html',
            {
                "request": request,
                "url": url,
                "cluster": fake_cluster,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                "clustering_settings": clustering_settings
            }
        )
        
    except Exception as e:
        logging.error(f"Error summarizing URL: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            'error.html', 
            {
                "request": request,
                "message": f"Error summarizing URL: {str(e)}",
                "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
                "clustering_settings": clustering_settings
            }
        )

@app.get("/status")
async def status(request: Request):
    """Return the current status of the RSS reader."""
    clustering_settings = get_clustering_settings(request)
    
    return JSONResponse({
        'has_data': bool(latest_data['clusters']),
        'last_updated': latest_data['timestamp'],
        'article_count': sum(len(cluster) for cluster in latest_data['clusters']) if latest_data['clusters'] else 0,
        'cluster_count': len(latest_data['clusters']) if latest_data['clusters'] else 0,
        'paywall_bypass_enabled': request.session.get('paywall_bypass_enabled', False),
        'using_default_feeds': request.session.get('use_default', True),
        'custom_feed_count': len(request.session.get('feeds_list', [])) if request.session.get('feeds_list') else 0,
        'clustering_settings': clustering_settings,
        'has_enhanced_clustering': has_enhanced_clustering,
        'has_optimized_clustering': has_optimized_clustering
    })

# ... rest of the routes ...

if __name__ == '__main__':
    import uvicorn
    
    # Get configuration from environment or command line
    port = int(os.environ.get('PORT', 5005))
    host = '127.0.0.1'  # Default to localhost
    
    # Check command line arguments
    if '--public' in sys.argv:
        host = '0.0.0.0'
        logging.warning("Running with public access (0.0.0.0). Make sure this is intended and secured.")
    
    # Get port from command line if specified
    for i, arg in enumerate(sys.argv):
        if arg == '--port' and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                logging.warning(f"Invalid port number: {sys.argv[i + 1]}. Using default: {port}")
    
    logging.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)