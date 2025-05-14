#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web server for RSS Reader using FastAPI.
"""

# Remove the problematic import line
# from common.batch_processing import apply
# apply()

# Instead, import the BatchProcessor class directly
from common.batch_processing import BatchProcessor

import os
import logging
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse, quote
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from typing import Optional, List

# Import from refactored packages
from reader.enhanced_reader import EnhancedRSSReader
from summarization.article_summarizer import ArticleSummarizer
from summarization.fast_summarizer import create_fast_summarizer
from common.http import create_http_session
from common.archive import fetch_article_content, is_paywalled
from common.logging import configure_logging

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

# Default global settings (for feed processing)
DEFAULT_GLOBAL_SETTINGS = {
    'batch_size': 25,
    'batch_delay': 15
}

# Default clustering settings
DEFAULT_CLUSTERING_SETTINGS = {
    'enable_multi_article': True,
    'similarity_threshold': 0.7,
    'max_articles_per_batch': 5,
    'use_enhanced_clustering': True,
    'time_range_enabled': True,
    'time_range_value': 168, # Default to 7 days in hours
    'time_range_unit': 'hours',
    'fast_summarization_enabled': True, # Added based on feed-summary.html
    'auto_select_model': True          # Added based on feed-summary.html
}

# Helper functions for session management
def get_global_settings(request: Request):
    """Get current global settings from session or use defaults."""
    if 'global_settings' not in request.session:
        request.session['global_settings'] = DEFAULT_GLOBAL_SETTINGS.copy()
    # Ensure all keys are present, even if defaults change later
    settings = request.session['global_settings']
    for key, value in DEFAULT_GLOBAL_SETTINGS.items():
        if key not in settings:
            settings[key] = value
    request.session['global_settings'] = settings
    return settings

def get_clustering_settings(request: Request):
    """Get current clustering settings from session or use defaults."""
    if 'clustering_settings' not in request.session:
        request.session['clustering_settings'] = DEFAULT_CLUSTERING_SETTINGS.copy()
    # Ensure all keys are present
    settings = request.session['clustering_settings']
    for key, value in DEFAULT_CLUSTERING_SETTINGS.items():
        if key not in settings:
            settings[key] = value
    request.session['clustering_settings'] = settings
    return settings

def get_common_template_vars(request: Request):
    """Helper to get common variables for template context."""
    now = datetime.now(timezone.utc)
    return {
        "request": request,
        "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
        "clustering_settings": get_clustering_settings(request),
        "global_settings": get_global_settings(request), # Add global_settings here
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp_iso_format": now.isoformat(),
    }

def sort_clusters(clusters):
    """Sort clusters by the number of articles (descending) and by date."""
    if not clusters:
        return []
    
    def get_cluster_key(cluster_item): # Renamed to avoid conflict
        cluster_content = cluster_item # Assuming cluster_item is the list of articles
        cluster_size = len(cluster_content)
        most_recent_date = None
        for article in cluster_content:
            try:
                published_val = article.get('published', '') # Renamed to avoid conflict
                if isinstance(published_val, str):
                    try:
                        from dateutil import parser as date_parser
                        article_date = date_parser.parse(published_val)
                        if article_date.tzinfo is None: # Make timezone-aware if naive
                            article_date = article_date.replace(tzinfo=timezone.utc)
                    except:
                        article_date = None
                elif isinstance(published_val, datetime):
                    article_date = published_val
                    if article_date.tzinfo is None: # Make timezone-aware if naive
                        article_date = article_date.replace(tzinfo=timezone.utc)
                else:
                    article_date = None
                
                if article_date and (most_recent_date is None or article_date > most_recent_date):
                    most_recent_date = article_date
            except Exception as e:
                logging.warning(f"Could not parse date for article in sort_clusters: {e}")
                continue
        
        if most_recent_date is None:
            most_recent_date = datetime.now(timezone.utc) # Use timezone-aware datetime
            
        return (-cluster_size, -most_recent_date.timestamp())
    
    return sorted(clusters, key=get_cluster_key)

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page with the latest summaries or a welcome page if none exist."""
    if 'paywall_bypass_enabled' not in request.session:
        request.session['paywall_bypass_enabled'] = False
    if 'feeds_list' not in request.session:
        request.session['feeds_list'] = None
    if 'use_default' not in request.session:
        request.session['use_default'] = True
    
    common_vars = get_common_template_vars(request)
    
    if latest_data['clusters']:
        sorted_clusters = sort_clusters(latest_data['clusters'])
        timestamp_str = latest_data['timestamp']
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if timestamp_str else common_vars["timestamp_dt"]
        
        return templates.TemplateResponse(
            "feed-summary.html",
            {
                **common_vars,
                "clusters": sorted_clusters,
                "timestamp": timestamp_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "timestamp_iso_format": timestamp_dt.isoformat(),
            }
        )
    else:
        return templates.TemplateResponse(
            "welcome.html",
            {
                **common_vars,
                "has_default_feeds": os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
                "initial_summaries_loaded": False, # Explicitly set for welcome page
            }
        )

@app.get("/welcome", response_class=HTMLResponse)
async def welcome(request: Request):
    """Explicitly render the welcome page regardless of data state."""
    common_vars = get_common_template_vars(request)
    return templates.TemplateResponse(
        "welcome.html",
        {
            **common_vars,
            "has_default_feeds": os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
            "initial_summaries_loaded": bool(latest_data['clusters']), # Check if data exists
        }
    )

@app.post("/toggle_paywall_bypass")
async def toggle_paywall_bypass(request: Request):
    """Toggle the paywall bypass setting."""
    current_status = request.session.get('paywall_bypass_enabled', False)
    request.session['paywall_bypass_enabled'] = not current_status
    
    logging.info(f"Paywall bypass {'enabled' if request.session['paywall_bypass_enabled'] else 'disabled'} by user")
    
    referer = request.headers.get('referer', '/')
    return RedirectResponse(url=referer, status_code=303)

@app.post("/update_clustering_settings")
async def update_clustering_settings(
    request: Request,
    enable_multi_article: Optional[str] = Form(None),
    use_enhanced_clustering: Optional[str] = Form(None),
    time_range_enabled: Optional[str] = Form(None),
    fast_summarization_enabled: Optional[str] = Form(None), # Added
    auto_select_model: Optional[str] = Form(None),         # Added
    similarity_threshold: float = Form(DEFAULT_CLUSTERING_SETTINGS['similarity_threshold']),
    max_articles_per_batch: int = Form(DEFAULT_CLUSTERING_SETTINGS['max_articles_per_batch']),
    time_range_value: int = Form(DEFAULT_CLUSTERING_SETTINGS['time_range_value']),
    time_range_unit: str = Form(DEFAULT_CLUSTERING_SETTINGS['time_range_unit'])
):
    """Update clustering settings from form submission."""
    clustering_settings = get_clustering_settings(request) # Ensures all keys are initialized
    
    clustering_settings['enable_multi_article'] = enable_multi_article == 'on'
    clustering_settings['use_enhanced_clustering'] = use_enhanced_clustering == 'on'
    clustering_settings['time_range_enabled'] = time_range_enabled == 'on'
    clustering_settings['fast_summarization_enabled'] = fast_summarization_enabled == 'on'
    clustering_settings['auto_select_model'] = auto_select_model == 'on'
    
    clustering_settings['similarity_threshold'] = max(0.0, min(1.0, similarity_threshold))
    clustering_settings['max_articles_per_batch'] = max(1, min(20, max_articles_per_batch)) # Increased max
    clustering_settings['time_range_value'] = max(1, time_range_value)
    
    if time_range_unit in ['hours', 'days', 'weeks', 'months']:
        clustering_settings['time_range_unit'] = time_range_unit
    
    request.session['clustering_settings'] = clustering_settings
    logging.info(f"Updated clustering settings: {clustering_settings}")
    
    referer = request.headers.get('referer', '/')
    return RedirectResponse(url=referer, status_code=303)

@app.post("/reset_clustering_settings") # Added for welcome.html button
async def reset_clustering_settings(request: Request):
    request.session['clustering_settings'] = DEFAULT_CLUSTERING_SETTINGS.copy()
    logging.info("Clustering settings reset to defaults.")
    referer = request.headers.get('referer', '/')
    return RedirectResponse(url=referer, status_code=303)


@app.post("/refresh")
async def refresh_feeds(
    request: Request,
    feeds: Optional[str] = Form(None),
    use_default: Optional[str] = Form(None)
    # batch_size and batch_delay will be taken from global_settings
):
    """Process RSS feeds and update the latest data."""
    common_vars = get_common_template_vars(request) # For error responses
    global_settings = common_vars['global_settings']
    batch_size = global_settings['batch_size']
    batch_delay = global_settings['batch_delay']

    try:
        feeds_from_form = feeds.strip() if feeds else ''
        use_default_from_form = use_default == 'true'

        request.session['use_default'] = use_default_from_form
        if not use_default_from_form and feeds_from_form:
            feeds_list_session = [url.strip() for url in feeds_from_form.split('\n') if url.strip()]
            request.session['feeds_list'] = feeds_list_session
            logging.info(f"Storing {len(feeds_list_session)} custom feeds in session")
        else:
            request.session['feeds_list'] = None # Use default from file
            logging.info("Using default feeds from rss_feeds.txt")
        
        current_use_default = request.session.get('use_default', True)
        current_feeds_list = request.session.get('feeds_list')
        
        clustering_settings = get_clustering_settings(request)
        
        if clustering_settings.get('time_range_enabled', False):
            time_value = clustering_settings.get('time_range_value', 168)
            time_unit = clustering_settings.get('time_range_unit', 'hours')
            multipliers = {'hours': 1, 'days': 24, 'weeks': 24 * 7, 'months': 24 * 30}
            time_range_hours = time_value * multipliers.get(time_unit, 1)
            os.environ['TIME_RANGE_HOURS'] = str(time_range_hours)
            logging.info(f"Time range filter enabled: {time_value} {time_unit} ({time_range_hours} hours)")
        else:
            os.environ['TIME_RANGE_HOURS'] = '0'
            logging.info("Time range filter disabled")
        
        os.environ['ENABLE_MULTI_ARTICLE_CLUSTERING'] = 'true' if clustering_settings['enable_multi_article'] else 'false'
        os.environ['MIN_SIMILARITY_THRESHOLD'] = str(clustering_settings['similarity_threshold'])
        os.environ['MAX_ARTICLES_PER_BATCH'] = str(clustering_settings['max_articles_per_batch'])
        os.environ['USE_ENHANCED_CLUSTERING'] = 'true' if clustering_settings['use_enhanced_clustering'] else 'false'
        os.environ['ENABLE_PAYWALL_BYPASS'] = 'true' if request.session.get('paywall_bypass_enabled', False) else 'false'
        
        max_workers = clustering_settings.get('max_articles_per_batch', 3)
        
        # Ensure main is not run if this script is imported elsewhere
        # For this to work, EnhancedRSSReader should be importable without running main.py's __main__ block
        from main import EnhancedRSSReader as MainEnhancedRSSReader # Avoid name clash

        feeds_to_pass = None if current_use_default else current_feeds_list
        reader = MainEnhancedRSSReader(
            feeds=feeds_to_pass, # Pass None to use default file, or the list
            batch_size=batch_size,
            batch_delay=batch_delay,
            max_workers=max_workers
        )
        
        output_file = await reader.process_feeds()
        clusters = reader.last_processed_clusters

        if output_file and clusters:
            latest_data['clusters'] = clusters
            latest_data['timestamp'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") # Store as UTC
            latest_data['output_file'] = output_file
            
            logging.info(f"Successfully refreshed feeds: {output_file}")
            return RedirectResponse(url="/", status_code=303)
        else:
            logging.warning("No articles found or processed")
            return templates.TemplateResponse('error.html', {**common_vars, "message": "No articles found or processed. Check logs."})
    
    except Exception as e:
        logging.error(f"Error refreshing feeds: {str(e)}", exc_info=True)
        return templates.TemplateResponse('error.html', {**common_vars, "message": f"Error: {str(e)}"})

@app.get("/clear") # Ensure GET method is supported if linked directly
@app.post("/clear")
async def clear_data(request: Request): # Added request for consistency if common_vars were needed
    """Clear all data and return to welcome page."""
    global latest_data # Make sure it's the global one
    latest_data = {
        'clusters': [], 'timestamp': None, 'output_file': None, 'raw_clusters': []
    }
    logging.info("Data cleared by user")
    return RedirectResponse(url="/welcome", status_code=303) # Redirect to /welcome

@app.get("/summarize", response_class=HTMLResponse)
async def summarize_single_get(request: Request):
    """Handle GET request for URL summarization."""
    common_vars = get_common_template_vars(request)
    return templates.TemplateResponse("summarize-form.html", {**common_vars})

@app.post("/summarize")
async def summarize_single_post(request: Request, url: str = Form(...), style: str = Form("default")):
    """Handle POST request for URL summarization."""
    common_vars = get_common_template_vars(request)
    clustering_settings = common_vars['clustering_settings']
    urls_input = [u.strip() for u in url.splitlines() if u.strip()] # Use splitlines for robustness
    
    if not urls_input:
        return templates.TemplateResponse('error.html', {**common_vars, "message": "Please provide at least one valid URL."})
    
    try:
        # It's better practice to initialize these within the function if they are not heavyweight
        # or manage them as app state or dependencies if they are.
        from main import setup_summarization_engine 
        from summarization.fast_summarizer import create_fast_summarizer as create_batch_summarizer # alias
        
        max_workers = clustering_settings.get('max_articles_per_batch', 3)
        summarizer_engine = setup_summarization_engine()
        fast_summarizer_instance = create_batch_summarizer(
            original_summarizer=summarizer_engine, max_batch_workers=max_workers
        )
        
        http_session = create_http_session()
        os.environ['ENABLE_PAYWALL_BYPASS'] = 'true' if request.session.get('paywall_bypass_enabled', False) else 'false'
        
        processed_articles = []
        skipped_urls_map = {} # Renamed
        
        for single_url in urls_input:
            full_url = single_url if single_url.startswith(('http://', 'https://')) else 'https://' + single_url
            try:
                content = fetch_article_content(full_url, http_session)
                if content and len(content) >= 100:
                    domain = urlparse(full_url).netloc
                    processed_articles.append({
                        'text': content, 'content': content,
                        'title': f"Article from {domain}", 'url': full_url, 'link': full_url,
                        'feed_source': domain, 'published_iso_format': datetime.now(timezone.utc).isoformat(), # Add for consistency
                        'published': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    })
                else:
                    skipped_urls_map[full_url] = "Could not extract sufficient content or content too short."
            except Exception as e:
                logging.error(f"Error fetching content for {full_url}: {str(e)}")
                skipped_urls_map[full_url] = f"Error fetching content: {str(e)}"
        
        summarized_clusters = []
        if processed_articles:
            batch_results = await fast_summarizer_instance.batch_summarize(
                articles=processed_articles, 
                max_concurrent=max_workers, 
                auto_select_model=True,
                style=style  # Add style parameter
            )
            for result in batch_results:
                if 'original' in result and 'summary' in result:
                    original_article = result['original']
                    summary_data = result['summary']
                    article_cluster_item = [{ # Ensure it's a list of one cluster
                        'title': summary_data.get('headline', original_article.get('title', "Summary")),
                        'link': original_article.get('link', '#'),
                        'feed_source': original_article.get('feed_source', urlparse(original_article.get('link', '#')).netloc),
                        'published': original_article.get('published', datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
                        'published_iso_format': original_article.get('published_iso_format', datetime.now(timezone.utc).isoformat()),
                        'summary': summary_data, # Pass the whole summary object
                        'model_used': result.get('model_used', 'N/A') # Add model_used if available
                    }]
                    summarized_clusters.append(article_cluster_item)
        
        for skipped_url, reason in skipped_urls_map.items():
            domain = urlparse(skipped_url).netloc
            now_dt = datetime.now(timezone.utc)
            error_summary_item = [{
                'title': f"Error processing {skipped_url}", 'link': skipped_url,
                'feed_source': domain, 'published': now_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                'published_iso_format': now_dt.isoformat(),
                'summary': {'headline': f"Error Processing Article", 'summary': reason},
                'model_used': 'N/A'
            }]
            summarized_clusters.append(error_summary_item)

        template_name = "single-summary.html" if len(urls_input) == 1 and summarized_clusters else "multiple-summaries.html"
        # For single-summary, the template expects 'cluster' not 'clusters' and specific structure.
        context_vars_for_summary = {
            **common_vars,
            "urls": urls_input, # List of original URLs requested
            "timestamp": common_vars["timestamp"], # Use common_vars timestamp
            "timestamp_iso_format": common_vars["timestamp_iso_format"],
        }
        if template_name == "single-summary.html":
            # The single-summary.html template expects 'url' (the single URL) and 'cluster' (a single cluster)
            context_vars_for_summary["url"] = urls_input[0] if urls_input else "#"
            context_vars_for_summary["cluster"] = summarized_clusters[0] if summarized_clusters else [{
                'title': "Error", 'link': urls_input[0] if urls_input else "#", 'feed_source': "N/A",
                'published': common_vars["timestamp"], 'published_iso_format': common_vars["timestamp_iso_format"],
                'summary': {'headline': 'Processing Error', 'summary': 'Could not generate summary.'}, 'model_used': 'N/A'
            }]
        else:
            context_vars_for_summary["clusters"] = summarized_clusters


        return templates.TemplateResponse(template_name, context_vars_for_summary)
            
    except Exception as e:
        logging.error(f"Error in batch summarization: {str(e)}", exc_info=True)
        return templates.TemplateResponse('error.html', {**common_vars, "message": f"Error processing URLs: {str(e)}"})

@app.get("/status")
async def status(request: Request):
    """Return the current status of the RSS reader."""
    common_vars = get_common_template_vars(request) # Includes clustering_settings
    
    has_enhanced_clustering, has_optimized_clustering = False, False
    try:
        from clustering.enhanced import create_enhanced_clusterer
        has_enhanced_clustering = True
    except ImportError: pass
    try:
        from models.lm_analyzer import create_cluster_analyzer
        has_optimized_clustering = True
    except ImportError: pass
    
    return JSONResponse({
        'has_data': bool(latest_data['clusters']),
        'last_updated': latest_data['timestamp'],
        'article_count': sum(len(c) for c in latest_data['clusters'] if isinstance(c, list)) if latest_data['clusters'] else 0,
        'cluster_count': len(latest_data['clusters']) if latest_data['clusters'] else 0,
        'paywall_bypass_enabled': common_vars['paywall_bypass_enabled'],
        'using_default_feeds': request.session.get('use_default', True),
        'custom_feed_count': len(request.session.get('feeds_list', [])) if request.session.get('feeds_list') else 0,
        'clustering_settings': common_vars['clustering_settings'],
        'global_settings': common_vars['global_settings'], # Added global_settings
        'has_enhanced_clustering': has_enhanced_clustering,
        'has_optimized_clustering': has_optimized_clustering
    })

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5005))
    host = '127.0.0.1'
    if '--public' in sys.argv: host = '0.0.0.0'
    for i, arg in enumerate(sys.argv):
        if arg == '--port' and i + 1 < len(sys.argv):
            try: port = int(sys.argv[i + 1])
            except ValueError: logging.warning(f"Invalid port: {sys.argv[i+1]}. Using {port}.")
    logging.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=1) # Added workers=1 for development simplicity