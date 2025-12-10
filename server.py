#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web server for RSS Reader using FastAPI.
"""

# Set this environment variable to avoid HuggingFace tokenizers warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Remove the problematic import line
# from common.batch_processing import apply
# apply()

# Instead, import the BatchProcessor class directly
from common.batch_processing import BatchProcessor

import logging
import sys
import asyncio
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse, quote
from fastapi import FastAPI, Request, Form, HTTPException, Depends, Response
from bs4 import BeautifulSoup
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
# Updated imports to use new content modules
from content.archive.paywall import is_paywalled
from content.archive.providers import default_provider_manager
from common.logging import configure_logging
from cache.tiered_cache import TieredCache

# Import bookmark functionality
from services.bookmark_manager import BookmarkManager

# Import image prompt functionality
from services.image_prompt_generator import get_image_prompt_generator

# Import authentication functionality
from services.auth_manager import get_auth_manager
from services.user_data_manager import get_user_data_manager
from middleware.auth import (
    get_current_user, set_user_session, clear_user_session,
    require_login, require_admin, get_user_data,
    SESSION_USER_ID, COOKIE_REMEMBER_TOKEN
)

# Import for dependency injection
from main import setup_summarization_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(title="Data Points AI - RSS Reader")

# Add session middleware with secure configuration
SECRET_KEY = os.environ.get('SECRET_KEY')
IS_PRODUCTION = os.environ.get('ENVIRONMENT', '').lower() == 'production'

if not SECRET_KEY:
    if IS_PRODUCTION:
        logging.error("SECRET_KEY environment variable is required in production!")
        raise RuntimeError("SECRET_KEY must be set in production. Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'")
    # Generate a consistent secret key if none provided (for development only)
    import hashlib
    SECRET_KEY = hashlib.sha256(b'rss-reader-dev-key').hexdigest()
    logging.warning("Using development SECRET_KEY. Set SECRET_KEY environment variable for production!")

app.add_middleware(
    SessionMiddleware, 
    secret_key=SECRET_KEY,
    max_age=3600,  # 1 hour session timeout
    https_only=False,  # Set to True in production with HTTPS
    same_site='lax'  # CSRF protection
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static')), name="static")

# Configure templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))

# Security helper: Validate redirect URLs to prevent open redirect attacks
def is_safe_redirect_url(url: str) -> bool:
    """
    Check if a URL is safe for redirecting.
    Only allows relative URLs or same-origin URLs to prevent open redirects.
    """
    if not url:
        return False

    # Allow relative URLs
    if url.startswith('/'):
        # But not protocol-relative URLs (//example.com)
        if url.startswith('//'):
            return False
        return True

    # Parse the URL to check if it's same-origin
    try:
        parsed = urlparse(url)
        # If there's no netloc (network location/domain), it's a relative URL
        if not parsed.netloc:
            return True
        # Reject absolute URLs to other domains
        return False
    except Exception:
        return False

def get_safe_redirect_url(request_url: str, default: str = '/') -> str:
    """
    Get a safe redirect URL from an untrusted source.
    Returns the URL if safe, otherwise returns the default.
    """
    if is_safe_redirect_url(request_url):
        return request_url
    return default

# Initialize bookmark manager
bookmark_manager = BookmarkManager()

# Initialize cache for cluster data storage (replaces global latest_data)
cluster_data_cache = TieredCache(
    memory_size=10,  # Store up to 10 users' data in memory
    disk_path="./cluster_cache",
    ttl_days=1  # 1 day TTL for cluster data
)

# DEPRECATED: Global latest_data kept for backward compatibility only
# New code should use cluster_data_cache with user-specific keys
latest_data = {
    'clusters': [],
    'timestamp': None,
    'output_file': None,
    'raw_clusters': []
}

# Default global settings (for feed processing)
DEFAULT_GLOBAL_SETTINGS = {
    'batch_size': 25,
    'batch_delay': 15,
    'per_feed_limit': 25  # Maximum articles per feed
}

# Helper function to maintain compatibility with old fetch_article_content
def fetch_article_content(url, session=None):
    """
    Fetch article content using the new archive system, with PDF support.

    Args:
        url: The article URL to fetch
        session: Optional requests session

    Returns:
        str: Article content or empty string if failed
    """
    try:
        # Check if this is a PDF URL
        from content.extractors.pdf import get_pdf_extractor
        pdf_extractor = get_pdf_extractor()

        if pdf_extractor.is_pdf_url(url):
            logging.info(f"Detected PDF URL: {url}")
            result = pdf_extractor.extract_from_url(url, session)
            if result['success']:
                logging.info(f"Successfully extracted text from PDF: {len(result['content'])} characters")
                return result['content']
            else:
                logging.warning(f"Failed to extract PDF content: {result.get('error', 'Unknown error')}")
                return ""

        # Check if it's paywalled
        if is_paywalled(url):
            logging.info(f"Detected paywall for {url}, trying archive services")

            # Try archive services
            result = default_provider_manager.get_archived_content(url)
            if result.success and result.content:
                return result.content

        # If not paywalled or archive failed, try direct access
        if not session:
            session = create_http_session()

        response = session.get(url, timeout=15)
        if response.status_code == 200:
            # Check if response is actually a PDF (content-type detection)
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                logging.info(f"Detected PDF content-type for {url}")
                result = pdf_extractor.extract_from_bytes(response.content)
                if result['success']:
                    logging.info(f"Successfully extracted text from PDF response: {len(result['content'])} characters")
                    return result['content']
                else:
                    logging.warning(f"Failed to extract PDF content: {result.get('error', 'Unknown error')}")
                    return ""

            # Simple content extraction for HTML/XML
            parser = 'xml' if 'xml' in content_type or url.endswith('.xml') else 'html.parser'
            soup = BeautifulSoup(response.text, parser)

            # Remove unwanted elements
            for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments'):
                unwanted.decompose()

            # Try to find main content
            for selector in ['article', '.article', '.content', '.post-content', 'main']:
                elements = soup.select(selector)
                if elements:
                    paragraphs = elements[0].find_all('p')
                    content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 40)
                    if content:
                        return content

            # Fallback to body text
            if soup.body:
                return soup.body.get_text()

    except Exception as e:
        logging.warning(f"Error fetching article content: {e}")

    return ""

# Default clustering settings
DEFAULT_CLUSTERING_SETTINGS = {
    'enable_multi_article': True,
    'similarity_threshold': 0.3,  # Updated for simple clustering
    'max_articles_per_batch': 5,
    'use_simple_clustering': True,  # New setting for simple clustering
    'use_enhanced_clustering': False,  # Optional heavy clustering
    'time_range_enabled': True,
    'time_range_value': 72, # Default to 3 days in hours
    'time_range_unit': 'hours',
    'fast_summarization_enabled': True, # Added based on feed-summary.html
    'auto_select_model': True,          # Added based on feed-summary.html
    'default_summary_style': 'default'  # Added for style selection
}

# Helper functions for user session and cluster data management
def get_or_create_user_id(request: Request) -> str:
    """
    Get or create a unique user ID for this session.
    This ID is used as a cache key for user-specific cluster data.
    """
    if 'user_id' not in request.session:
        request.session['user_id'] = str(uuid.uuid4())
    return request.session['user_id']

def get_user_cluster_data(request: Request) -> dict:
    """
    Get cluster data for the current user from cache.
    Falls back to global latest_data if no user-specific data exists.
    """
    user_id = get_or_create_user_id(request)
    cache_key = f"clusters:{user_id}"

    # Try to get user-specific data from cache
    cached_data = cluster_data_cache.get(cache_key)
    if cached_data:
        return cached_data

    # Fall back to global latest_data (for backward compatibility)
    # Also try the global "latest" key
    global_cached = cluster_data_cache.get("clusters:latest")
    if global_cached:
        return global_cached

    # Final fallback to in-memory global state
    if latest_data['clusters']:
        return latest_data

    # Return empty data structure
    return {
        'clusters': [],
        'timestamp': None,
        'output_file': None,
        'raw_clusters': []
    }

def set_user_cluster_data(request: Request, data: dict) -> None:
    """
    Store cluster data for the current user in cache.
    Also updates the global "latest" key for backward compatibility.
    """
    user_id = get_or_create_user_id(request)
    cache_key = f"clusters:{user_id}"

    # Store user-specific data with 1 day TTL
    cluster_data_cache.set(cache_key, data, ttl=86400)

    # Also update global "latest" key (for status endpoint and backward compat)
    cluster_data_cache.set("clusters:latest", data, ttl=86400)

    # Update in-memory global state for maximum backward compatibility
    latest_data['clusters'] = data.get('clusters', [])
    latest_data['timestamp'] = data.get('timestamp')
    latest_data['output_file'] = data.get('output_file')
    latest_data['raw_clusters'] = data.get('raw_clusters', [])

# Dependency injection for summarizers (cached instances)
_summarizer_cache = {}

def get_summarizer_engine():
    """
    Dependency injection for ArticleSummarizer.
    Reuses cached instance across requests for better performance.
    """
    if 'engine' not in _summarizer_cache:
        _summarizer_cache['engine'] = setup_summarization_engine()
        logging.info("Created new ArticleSummarizer instance (cached)")
    return _summarizer_cache['engine']

def get_fast_summarizer(request: Request, max_workers: Optional[int] = None):
    """
    Dependency injection for FastSummarizer.
    Creates instances with request-specific settings.

    Args:
        request: FastAPI request object for accessing session settings
        max_workers: Optional override for max workers (uses clustering settings if None)

    Returns:
        FastSummarizer instance configured for this request
    """
    # Get max_workers from clustering settings if not provided
    if max_workers is None:
        clustering_settings = get_clustering_settings(request)
        max_workers = clustering_settings.get('max_articles_per_batch', 3)

    # Create cache key based on max_workers
    cache_key = f'fast_summarizer_{max_workers}'

    # Reuse cached instance if available with same config
    if cache_key not in _summarizer_cache:
        engine = get_summarizer_engine()
        _summarizer_cache[cache_key] = create_fast_summarizer(
            original_summarizer=engine,
            max_batch_workers=max_workers
        )
        logging.info(f"Created new FastSummarizer instance with max_workers={max_workers} (cached)")

    return _summarizer_cache[cache_key]

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
    user = get_current_user(request)
    return {
        "request": request,
        "user": user,  # None if not logged in, dict with id/username/is_admin if logged in
        "is_authenticated": user is not None,
        "paywall_bypass_enabled": request.session.get('paywall_bypass_enabled', False),
        "clustering_settings": get_clustering_settings(request),
        "global_settings": get_global_settings(request),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp_iso_format": now.isoformat(),
        "show_paywall_toggle": True,  # Can be overridden per route
        "has_default_feeds": os.path.exists(os.path.join(os.path.dirname(__file__), 'rss_feeds.txt')),
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
    # Redirect to login if not authenticated
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url='/login', status_code=303)

    if 'paywall_bypass_enabled' not in request.session:
        request.session['paywall_bypass_enabled'] = False

    common_vars = get_common_template_vars(request)

    # Get user-specific cluster data from cache
    user_data = get_user_cluster_data(request)

    if user_data['clusters']:
        sorted_clusters = sort_clusters(user_data['clusters'])
        timestamp_str = user_data['timestamp']
        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if timestamp_str else common_vars.get("timestamp_dt", datetime.now(timezone.utc))

        return templates.TemplateResponse(
            "feed_summary.html",
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
                "initial_summaries_loaded": False,
            }
        )

@app.get("/welcome", response_class=HTMLResponse)
async def welcome(request: Request):
    """Explicitly render the welcome page regardless of data state."""
    common_vars = get_common_template_vars(request)
    # Get user-specific cluster data to check if data exists
    user_data = get_user_cluster_data(request)
    return templates.TemplateResponse(
        "welcome.html",
        {
            **common_vars,
            "initial_summaries_loaded": bool(user_data['clusters']), # Check if data exists
        }
    )

@app.post("/toggle_paywall_bypass")
async def toggle_paywall_bypass(request: Request):
    """Toggle the paywall bypass setting."""
    current_status = request.session.get('paywall_bypass_enabled', False)
    request.session['paywall_bypass_enabled'] = not current_status
    
    logging.info(f"Paywall bypass {'enabled' if request.session['paywall_bypass_enabled'] else 'disabled'} by user")

    referer = request.headers.get('referer', '/')
    safe_url = get_safe_redirect_url(referer, default='/')
    return RedirectResponse(url=safe_url, status_code=303)

@app.post("/update_clustering_settings")
async def update_clustering_settings(
    request: Request,
    enable_multi_article: Optional[str] = Form(None),
    use_simple_clustering: Optional[str] = Form(None),     # Updated for simple clustering
    use_enhanced_clustering: Optional[str] = Form(None),   # Optional heavy clustering
    time_range_enabled: Optional[str] = Form(None),
    fast_summarization_enabled: Optional[str] = Form(None), # Added
    auto_select_model: Optional[str] = Form(None),         # Added
    default_summary_style: Optional[str] = Form(None),     # Added
    similarity_threshold: float = Form(DEFAULT_CLUSTERING_SETTINGS['similarity_threshold']),
    max_articles_per_batch: int = Form(DEFAULT_CLUSTERING_SETTINGS['max_articles_per_batch']),
    time_range_value: int = Form(DEFAULT_CLUSTERING_SETTINGS['time_range_value']),
    time_range_unit: str = Form(DEFAULT_CLUSTERING_SETTINGS['time_range_unit'])
):
    """Update clustering settings from form submission."""
    clustering_settings = get_clustering_settings(request) # Ensures all keys are initialized
    
    clustering_settings['enable_multi_article'] = enable_multi_article == 'on'
    clustering_settings['use_simple_clustering'] = use_simple_clustering == 'on'
    clustering_settings['use_enhanced_clustering'] = use_enhanced_clustering == 'on'
    clustering_settings['time_range_enabled'] = time_range_enabled == 'on'
    clustering_settings['fast_summarization_enabled'] = fast_summarization_enabled == 'on'
    clustering_settings['auto_select_model'] = auto_select_model == 'on'
    
    # Handle default summary style
    if default_summary_style and default_summary_style in ['default', 'bullet', 'newswire']:
        clustering_settings['default_summary_style'] = default_summary_style
    
    clustering_settings['similarity_threshold'] = max(0.0, min(1.0, similarity_threshold))
    clustering_settings['max_articles_per_batch'] = max(1, min(20, max_articles_per_batch)) # Increased max
    clustering_settings['time_range_value'] = max(1, time_range_value)
    
    if time_range_unit in ['hours', 'days', 'weeks', 'months']:
        clustering_settings['time_range_unit'] = time_range_unit
    
    request.session['clustering_settings'] = clustering_settings
    logging.info(f"Updated clustering settings: {clustering_settings}")

    referer = request.headers.get('referer', '/')
    safe_url = get_safe_redirect_url(referer, default='/')
    return RedirectResponse(url=safe_url, status_code=303)

@app.post("/reset_clustering_settings") # Added for welcome.html button
async def reset_clustering_settings(request: Request):
    request.session['clustering_settings'] = DEFAULT_CLUSTERING_SETTINGS.copy()
    logging.info("Clustering settings reset to defaults.")
    referer = request.headers.get('referer', '/')
    safe_url = get_safe_redirect_url(referer, default='/')
    return RedirectResponse(url=safe_url, status_code=303)


@app.post("/refresh")
@require_login()
async def refresh_feeds(
    request: Request,
    feeds: Optional[str] = Form(None),
    use_default: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(None),
    batch_delay: Optional[int] = Form(None),
    per_feed_limit: Optional[int] = Form(None)
):
    """Process RSS feeds and update the latest data."""
    common_vars = get_common_template_vars(request)  # For error responses
    global_settings = common_vars['global_settings']

    # Update global settings from form values if provided
    if batch_size is not None:
        global_settings['batch_size'] = batch_size
    if batch_delay is not None:
        global_settings['batch_delay'] = batch_delay
    if per_feed_limit is not None:
        global_settings['per_feed_limit'] = per_feed_limit

    # Save updated settings to session
    request.session['global_settings'] = global_settings

    # Get values to use for processing
    batch_size = global_settings['batch_size']
    batch_delay = global_settings['batch_delay']
    per_feed_limit = global_settings['per_feed_limit']

    try:
        # Get user's feeds from their database
        user = get_current_user(request)
        user_feeds = get_user_feeds(request)

        # Check if form provides custom feeds
        feeds_from_form = feeds.strip() if feeds else ''
        use_default_from_form = use_default == 'true'

        if not use_default_from_form and feeds_from_form:
            # Use feeds from form (custom one-time feeds)
            feeds_list = [url.strip() for url in feeds_from_form.split('\n') if url.strip()]
            logging.info(f"Using {len(feeds_list)} custom feeds from form")
        elif user_feeds:
            # Use user's saved feeds from database
            feeds_list = user_feeds
            logging.info(f"Using {len(feeds_list)} feeds from user's database")
        else:
            # No feeds available
            return templates.TemplateResponse('error.html', {
                **common_vars,
                "message": "No feeds configured. Please add some feeds first."
            })

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
        os.environ['USE_SIMPLE_CLUSTERING'] = 'true' if clustering_settings.get('use_simple_clustering', True) else 'false'
        os.environ['USE_ENHANCED_CLUSTERING'] = 'true' if clustering_settings['use_enhanced_clustering'] else 'false'
        os.environ['ENABLE_PAYWALL_BYPASS'] = 'true' if request.session.get('paywall_bypass_enabled', False) else 'false'

        max_workers = clustering_settings.get('max_articles_per_batch', 3)

        from reader.enhanced_reader import EnhancedRSSReader as MainEnhancedRSSReader

        reader = MainEnhancedRSSReader(
            feeds=feeds_list,  # Pass the user's feed list
            batch_size=batch_size,
            batch_delay=batch_delay,
            max_workers=max_workers,
            per_feed_limit=per_feed_limit
        )

        output_file = await reader.process_feeds()
        clusters = reader.last_processed_clusters

        if output_file and clusters:
            # Store cluster data in user-specific cache
            cluster_data = {
                'clusters': clusters,
                'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                'output_file': output_file,
                'raw_clusters': clusters
            }
            set_user_cluster_data(request, cluster_data)

            logging.info(f"Successfully refreshed feeds for user {user['username']}: {output_file}")
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
    return templates.TemplateResponse("summarize_form.html", {**common_vars})  # Fixed filename

@app.post("/summarize")
async def summarize_single_post(request: Request, url: str = Form(...), style: str = Form("default")):
    """Handle POST request for URL summarization."""
    common_vars = get_common_template_vars(request)
    clustering_settings = common_vars['clustering_settings']
    urls_input = [u.strip() for u in url.splitlines() if u.strip()] # Use splitlines for robustness
    
    if not urls_input:
        return templates.TemplateResponse('error.html', {**common_vars, "message": "Please provide at least one valid URL."})
    
    try:
        # Use dependency injection to get cached summarizer instance
        max_workers = clustering_settings.get('max_articles_per_batch', 3)
        fast_summarizer_instance = get_fast_summarizer(request, max_workers=max_workers)
        
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
                style=style  # Ensure style parameter is passed
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
                        'content': original_article.get('content', ''), # Include original content for toggle
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

        template_name = "single_summary.html" if len(urls_input) == 1 and summarized_clusters else "multiple_summaries.html"  # Fixed filenames
        # For single-summary, the template expects 'cluster' not 'clusters' and specific structure.
        context_vars_for_summary = {
            **common_vars,
            "urls": urls_input, # List of original URLs requested
            "timestamp": common_vars["timestamp"], # Use common_vars timestamp
            "timestamp_iso_format": common_vars["timestamp_iso_format"],
        }
        if template_name == "single_summary.html":
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
    
    # Get user-specific cluster data for status (falls back to global if needed)
    user_data = get_user_cluster_data(request)

    return JSONResponse({
        'has_data': bool(user_data['clusters']),
        'last_updated': user_data['timestamp'],
        'article_count': sum(len(c) for c in user_data['clusters'] if isinstance(c, list)) if user_data['clusters'] else 0,
        'cluster_count': len(user_data['clusters']) if user_data['clusters'] else 0,
        'paywall_bypass_enabled': common_vars['paywall_bypass_enabled'],
        'using_default_feeds': request.session.get('use_default', True),
        'custom_feed_count': len(request.session.get('feeds_list', [])) if request.session.get('feeds_list') else 0,
        'clustering_settings': common_vars['clustering_settings'],
        'global_settings': common_vars['global_settings'], # Added global_settings
        'has_enhanced_clustering': has_enhanced_clustering,
        'has_optimized_clustering': has_optimized_clustering
    })

# Summarization API endpoints
@app.post("/api/summarize")
async def summarize_article(request: Request):
    """API endpoint to summarize a single article URL."""
    try:
        data = await request.json()
        url = data.get('url')
        
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
            
        # Ensure URL has proper scheme
        full_url = url if url.startswith(('http://', 'https://')) else 'https://' + url

        # Use dependency injection to get cached summarizer instance
        fast_summarizer = get_fast_summarizer(request)
        
        # Create HTTP session
        http_session = create_http_session()
        
        # Enable paywall bypass if configured
        os.environ['ENABLE_PAYWALL_BYPASS'] = 'true' if request.session.get('paywall_bypass_enabled', True) else 'false'
        
        # Check if this is a Google News or other aggregator URL
        from content.extractors.aggregator import is_aggregator_link, extract_source_url
        
        original_url = full_url
        if is_aggregator_link(full_url):
            try:
                logging.info(f"Detected aggregator link in bookmark: {full_url}")
                result = extract_source_url(full_url, http_session)
                if result.success and result.extracted_url and result.extracted_url != full_url:
                    extracted_url = result.extracted_url
                    logging.info(f"Using extracted source URL: {extracted_url}")
                    original_url = extracted_url
            except Exception as extract_error:
                logging.warning(f"Error extracting source URL: {str(extract_error)}")
                # Continue with original URL if extraction fails
        
        # Fetch article content with multiple fallback attempts
        content = None
        urls_to_try = [original_url]
        if original_url != full_url:
            urls_to_try.append(full_url)  # Try original aggregator URL as fallback
        
        for attempt_url in urls_to_try:
            try:
                logging.info(f"Attempting to fetch content from: {attempt_url}")
                content = fetch_article_content(attempt_url, http_session)
                if content and len(content) >= 100:
                    logging.info(f"Successfully fetched content from: {attempt_url}")
                    break
                else:
                    logging.warning(f"Insufficient content from: {attempt_url}")
            except Exception as fetch_error:
                logging.warning(f"Error fetching content from {attempt_url}: {str(fetch_error)}")
        
        # If we couldn't get content from any URL, try a direct request as last resort
        if not content or len(content) < 100:
            try:
                logging.info(f"Attempting direct request to: {original_url}")
                response = http_session.get(original_url, timeout=10)
                if response.status_code == 200:
                    # Detect content type and use appropriate parser
                    content_type = response.headers.get('content-type', '').lower()
                    parser = 'xml' if 'xml' in content_type or original_url.endswith('.xml') else 'html.parser'
                    soup = BeautifulSoup(response.text, parser)
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    content = soup.get_text(separator=' ', strip=True)
                    if len(content) >= 100:
                        logging.info("Successfully fetched content with direct request")
                    else:
                        logging.warning("Direct request returned insufficient content")
            except Exception as direct_error:
                logging.warning(f"Direct request failed: {str(direct_error)}")
        
        # If we still don't have content, use the bookmark's stored content if available
        if (not content or len(content) < 100) and data.get('stored_content'):
            logging.info("Using stored content from bookmark")
            content = data.get('stored_content')
        
        # If we still don't have content, use the bookmark's stored summary if available
        if (not content or len(content) < 100) and data.get('stored_summary'):
            logging.info("Using stored summary from bookmark")
            content = data.get('stored_summary')
            
        # Final check if we have enough content
        if not content or len(content) < 50:  # Lowered threshold for final check
            raise HTTPException(status_code=400, detail="Could not extract sufficient content from the URL")
        
        # Prepare article data
        domain = urlparse(original_url).netloc
        article = {
            'text': content, 
            'content': content,
            'title': data.get('title', f"Article from {domain}"), 
            'url': original_url, 
            'link': original_url,
            'feed_source': domain, 
            'published_iso_format': datetime.now(timezone.utc).isoformat(),
            'published': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Generate summary
        try:
            # The summarize method is not async, so we don't use await
            result = fast_summarizer.summarize(
                text=article['content'],
                title=article['title'],
                url=article['url'],
                auto_select_model=True
            )
        except Exception as summarize_error:
            logging.error(f"Error in summarize call: {str(summarize_error)}")
            raise HTTPException(status_code=500, detail=f"Error generating summary: {str(summarize_error)}")
            
        if not isinstance(result, dict):
            logging.error(f"Unexpected result type from summarizer: {type(result)}")
            raise HTTPException(status_code=500, detail="Unexpected result from summarizer")
        
        if not result or 'summary' not in result:
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        
        # Extract summary data
        # The result structure from FastSummarizer is {'headline': '...', 'summary': '...', 'style': '...'}
        title = result.get('headline', article.get('title', "Article Summary"))
        summary = result.get('summary', "No summary available")
        
        return {
            "title": title,
            "summary": summary,
            "url": original_url,
            "model_used": result.get('model_used', 'N/A')
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error summarizing article: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error summarizing article: {str(e)}")

@app.post("/api/summarize-batch")
async def summarize_articles_batch(request: Request):
    """API endpoint to summarize multiple article URLs in a batch."""
    try:
        data = await request.json()
        urls = data.get('urls', [])
        bookmark_ids = data.get('bookmark_ids', [])
        
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        # Import aggregator utilities (keep these here as they're less commonly used)
        from content.extractors.aggregator import is_aggregator_link, extract_source_url

        common_vars = get_common_template_vars(request)
        clustering_settings = common_vars['clustering_settings']
        max_workers = clustering_settings.get('max_articles_per_batch', 3)

        # Use dependency injection to get cached summarizer instance
        fast_summarizer = get_fast_summarizer(request, max_workers=max_workers)
        
        # Create HTTP session
        http_session = create_http_session()
        
        # Enable paywall bypass if configured
        os.environ['ENABLE_PAYWALL_BYPASS'] = 'true' if request.session.get('paywall_bypass_enabled', True) else 'false'
        
        # Fetch content for all URLs
        processed_articles = []
        failed_urls = {}
        
        # Get stored bookmark data if available
        stored_bookmark_data = {}
        if bookmark_ids and len(bookmark_ids) == len(urls):
            for i, bookmark_id in enumerate(bookmark_ids):
                try:
                    bookmark = bookmark_manager.get_bookmark(int(bookmark_id))
                    if bookmark:
                        stored_bookmark_data[urls[i]] = {
                            'title': bookmark.get('title', ''),
                            'content': bookmark.get('content', ''),
                            'summary': bookmark.get('summary', '')
                        }
                except Exception as e:
                    logging.warning(f"Error fetching bookmark {bookmark_id}: {str(e)}")
        
        for i, url in enumerate(urls):
            full_url = url if url.startswith(('http://', 'https://')) else 'https://' + url
            
            # Check if this is an aggregator URL
            original_url = full_url
            if is_aggregator_link(full_url):
                try:
                    logging.info(f"Detected aggregator link in batch: {full_url}")
                    result = extract_source_url(full_url, http_session)
                    if result.success and result.extracted_url and result.extracted_url != full_url:
                        extracted_url = result.extracted_url
                        logging.info(f"Using extracted source URL: {extracted_url}")
                        original_url = extracted_url
                except Exception as extract_error:
                    logging.warning(f"Error extracting source URL: {str(extract_error)}")
            
            # Try multiple content sources with fallbacks
            content = None
            urls_to_try = [original_url]
            if original_url != full_url:
                urls_to_try.append(full_url)  # Try original aggregator URL as fallback
            
            for attempt_url in urls_to_try:
                try:
                    logging.info(f"Attempting to fetch content from: {attempt_url}")
                    content = fetch_article_content(attempt_url, http_session)
                    if content and len(content) >= 100:
                        logging.info(f"Successfully fetched content from: {attempt_url}")
                        break
                    else:
                        logging.warning(f"Insufficient content from: {attempt_url}")
                except Exception as fetch_error:
                    logging.warning(f"Error fetching content from {attempt_url}: {str(fetch_error)}")
            
            # If we couldn't get content from any URL, try a direct request as last resort
            if not content or len(content) < 100:
                try:
                    logging.info(f"Attempting direct request to: {original_url}")
                    response = http_session.get(original_url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.extract()
                        content = soup.get_text(separator=' ', strip=True)
                        if len(content) >= 100:
                            logging.info("Successfully fetched content with direct request")
                        else:
                            logging.warning("Direct request returned insufficient content")
                except Exception as direct_error:
                    logging.warning(f"Direct request failed: {str(direct_error)}")
            
            # If we still don't have content, use the bookmark's stored content if available
            if (not content or len(content) < 100) and url in stored_bookmark_data and stored_bookmark_data[url].get('content'):
                logging.info(f"Using stored content from bookmark for {url}")
                content = stored_bookmark_data[url]['content']
            
            # If we still don't have content, use the bookmark's stored summary if available
            if (not content or len(content) < 100) and url in stored_bookmark_data and stored_bookmark_data[url].get('summary'):
                logging.info(f"Using stored summary from bookmark for {url}")
                content = stored_bookmark_data[url]['summary']
            
            # If we have enough content, add to processed articles
            if content and len(content) >= 50:  # Lower threshold for final check
                domain = urlparse(original_url).netloc
                title = stored_bookmark_data.get(url, {}).get('title', f"Article from {domain}")
                
                processed_articles.append({
                    'text': content, 
                    'content': content,
                    'title': title, 
                    'url': original_url, 
                    'link': original_url,
                    'feed_source': domain, 
                    'published_iso_format': datetime.now(timezone.utc).isoformat(),
                    'published': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                })
            else:
                failed_urls[full_url] = "Could not extract sufficient content from any source."
        
        if not processed_articles:
            raise HTTPException(status_code=400, detail="Could not process any of the provided URLs")
        
        # Generate summaries for all articles
        # The batch_summarize method expects articles with 'text', 'title', and 'url' keys
        # Make sure our processed articles have these keys properly set
        for article in processed_articles:
            # Ensure text key is set (FastSummarizer expects this)
            if 'text' not in article and 'content' in article:
                article['text'] = article['content']
                
            # Make sure all required fields are present
            if 'title' not in article:
                article['title'] = f"Article from {urlparse(article.get('url', '')).netloc}"
            if 'url' not in article and 'link' in article:
                article['url'] = article['link']
        
        try:
            # batch_summarize is always async (FastSummarizer.batch_summarize is async def)
            batch_results = await fast_summarizer.batch_summarize(
                articles=processed_articles,
                max_concurrent=max_workers,
                auto_select_model=True
            )
            
            if not isinstance(batch_results, list):
                logging.error(f"Unexpected batch result type: {type(batch_results)}")
                raise HTTPException(status_code=500, detail="Unexpected result type from batch summarizer")
                
        except Exception as batch_error:
            logging.error(f"Error in batch summarize call: {str(batch_error)}")
            raise HTTPException(status_code=500, detail=f"Error generating batch summaries: {str(batch_error)}")
        
        # Format results
        summaries = []
        for result in batch_results:
            # The batch_summarize method returns a list of dicts with the structure:
            # {'original': article_dict, 'summary': summary_dict, 'model_used': model_name}
            # where summary_dict is {'headline': '...', 'summary': '...', 'style': '...'}
            if isinstance(result, dict):
                if 'original' in result and 'summary' in result:
                    original_article = result['original']
                    summary_data = result['summary']
                    
                    # Make sure summary_data is a dictionary
                    if isinstance(summary_data, dict):
                        summaries.append({
                            'title': summary_data.get('headline', original_article.get('title', "Summary")),
                            'summary': summary_data.get('summary', "No summary available"),
                            'url': original_article.get('link', original_article.get('url', '#')),
                            'model_used': result.get('model_used', 'N/A')
                        })
                    else:
                        # Handle the case where summary_data is not a dictionary
                        logging.warning(f"Unexpected summary_data type: {type(summary_data)}")
                        summaries.append({
                            'title': original_article.get('title', "Summary"),
                            'summary': str(summary_data) if summary_data else "No summary available",
                            'url': original_article.get('link', original_article.get('url', '#')),
                            'model_used': result.get('model_used', 'N/A')
                        })
            else:
                logging.warning(f"Unexpected result type in batch_results: {type(result)}")
                # Try to extract useful information if possible
                if hasattr(result, 'get'):
                    summaries.append({
                        'title': "Summary",
                        'summary': str(result),
                        'url': '#',
                        'model_used': 'N/A'
                    })
        
        return {
            "summaries": summaries,
            "failed_urls": failed_urls
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error summarizing articles batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error summarizing articles batch: {str(e)}")

# Bookmark API endpoints
def get_user_bookmark_manager(request: Request):
    """Get the bookmark manager for the current user."""
    user = get_current_user(request)
    if not user:
        return None
    user_data = get_user_data_manager(user['id'])
    return user_data.get_bookmark_manager()


@app.post("/api/bookmarks")
@require_login(redirect_to_login=False)
async def create_bookmark(
    request: Request,
    title: str = Form(...),
    url: str = Form(...),
    summary: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Add a new bookmark."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    tag_list = tags.split(',') if tags else []
    bookmark_id = user_bookmark_manager.add_bookmark(
        title=title,
        url=url,
        summary=summary,
        content=content,
        tags=tag_list
    )
    return {"id": bookmark_id, "status": "success"}


@app.get("/api/bookmarks")
@require_login(redirect_to_login=False)
async def get_bookmarks(
    request: Request,
    read: Optional[bool] = None,
    tags: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get all bookmarks with optional filtering."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    tag_list = tags.split(',') if tags else None
    bookmarks = user_bookmark_manager.get_bookmarks(
        filter_read=read,
        tags=tag_list,
        limit=limit,
        offset=offset
    )
    return {"bookmarks": bookmarks}


@app.get("/api/bookmarks/{bookmark_id}")
@require_login(redirect_to_login=False)
async def get_bookmark(
    request: Request,
    bookmark_id: int
):
    """Get a single bookmark by ID."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    bookmark = user_bookmark_manager.get_bookmark(bookmark_id)
    if not bookmark:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return bookmark


@app.put("/api/bookmarks/{bookmark_id}")
@require_login(redirect_to_login=False)
async def update_bookmark(
    request: Request,
    bookmark_id: int,
    title: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    read_status: Optional[bool] = Form(None)
):
    """Update a bookmark."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    update_data = {}
    if title is not None:
        update_data['title'] = title
    if url is not None:
        update_data['url'] = url
    if summary is not None:
        update_data['summary'] = summary
    if content is not None:
        update_data['content'] = content
    if tags is not None:
        update_data['tags'] = tags.split(',') if tags else []
    if read_status is not None:
        update_data['read_status'] = read_status

    success = user_bookmark_manager.update_bookmark(bookmark_id, **update_data)
    if not success:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"status": "success"}


@app.put("/api/bookmarks/{bookmark_id}/read")
@require_login(redirect_to_login=False)
async def update_read_status(
    request: Request,
    bookmark_id: int,
    status: bool = True
):
    """Mark a bookmark as read/unread."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    success = user_bookmark_manager.mark_as_read(bookmark_id, status)
    if not success:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"status": "success"}


@app.delete("/api/bookmarks/{bookmark_id}")
@require_login(redirect_to_login=False)
async def delete_bookmark(
    request: Request,
    bookmark_id: int
):
    """Delete a bookmark."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    success = user_bookmark_manager.delete_bookmark(bookmark_id)
    if not success:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"status": "success"}

@app.get("/api/bookmarks/export/{format_type}")
@require_login(redirect_to_login=False)
async def export_bookmarks(
    request: Request,
    format_type: str,
    read: Optional[bool] = None,
    tags: Optional[str] = None
):
    """Export bookmarks to JSON or CSV format."""
    if format_type.lower() not in ['json', 'csv']:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

    user_bookmark_manager = get_user_bookmark_manager(request)
    tag_list = tags.split(',') if tags else None
    try:
        data = user_bookmark_manager.export_bookmarks(
            format_type=format_type.lower(),
            filter_read=read,
            tags=tag_list
        )

        # Set appropriate content type and filename
        content_type = "application/json" if format_type.lower() == 'json' else "text/csv"
        filename = f"bookmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type.lower()}"

        response = Response(content=data)
        response.headers["Content-Type"] = content_type
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bookmarks/import")
@require_login(redirect_to_login=False)
async def import_bookmarks(
    request: Request,
    json_data: str = Form(...)
):
    """Import bookmarks from JSON data."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    try:
        count = user_bookmark_manager.import_from_json(json_data)
        return {"status": "success", "imported": count}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Bookmark UI routes
@app.get("/bookmarks", response_class=HTMLResponse)
@require_login()
async def view_bookmarks(request: Request):
    """Render the bookmarks page."""
    user_bookmark_manager = get_user_bookmark_manager(request)
    bookmarks = user_bookmark_manager.get_bookmarks()
    common_vars = get_common_template_vars(request)
    # Override paywall toggle for bookmarks page
    common_vars["show_paywall_toggle"] = False

    return templates.TemplateResponse(
        "bookmarks.html",
        {
            **common_vars,
            "bookmarks": bookmarks,
        }
    )


# Helper to get user's feeds
def get_user_feeds(request: Request) -> list:
    """Get the feed list for the current user."""
    user = get_current_user(request)
    if not user:
        return []
    user_data = get_user_data_manager(user['id'])
    return user_data.get_feed_urls()


@app.get("/feeds", response_class=HTMLResponse)
@require_login()
async def manage_feeds(request: Request):
    """Render the feed management page."""
    common_vars = get_common_template_vars(request)

    # Get feeds from user's database
    feeds = get_user_feeds(request)

    return templates.TemplateResponse(
        "feeds.html",
        {
            **common_vars,
            "feeds": feeds,
            "feeds_count": len(feeds),
        }
    )


@app.post("/feeds/add")
@require_login()
async def add_feed(request: Request, feed_url: str = Form(...)):
    """Add a new RSS feed."""
    feed_url = feed_url.strip()

    if not feed_url:
        raise HTTPException(status_code=400, detail="Feed URL cannot be empty")

    user = get_current_user(request)
    user_data = get_user_data_manager(user['id'])

    # Try to add the feed
    feed = user_data.add_feed(feed_url)
    if not feed:
        raise HTTPException(status_code=400, detail="Feed already exists")

    return RedirectResponse(url="/feeds", status_code=303)


@app.post("/feeds/delete")
@require_login()
async def delete_feed(request: Request, feed_url: str = Form(...)):
    """Delete an RSS feed."""
    user = get_current_user(request)
    user_data = get_user_data_manager(user['id'])

    success = user_data.remove_feed(feed_url.strip())

    if not success:
        raise HTTPException(status_code=404, detail="Feed not found")

    return RedirectResponse(url="/feeds", status_code=303)

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Render the settings page with clustering settings."""
    common_vars = get_common_template_vars(request)

    return templates.TemplateResponse(
        "settings.html",
        {
            **common_vars,
        }
    )

# Image Prompt Generation API
@app.post("/api/generate-image-prompt")
async def generate_image_prompt(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    style: str = Form(default="editorial")
):
    """
    Generate an AI image prompt from article content.

    Args:
        title: Article title
        content: Original article content (not summary)
        style: Always generates editorial illustrations

    Returns:
        JSON response with generated prompt and metadata
    """
    try:
        # Input validation
        if not title.strip():
            raise HTTPException(status_code=400, detail="Title is required")
        
        if not content.strip():
            raise HTTPException(status_code=400, detail="Content is required")
        
        # Get image prompt generator
        generator = get_image_prompt_generator()
        
        # Generate prompt
        result = await generator.generate_prompt(
            title=title.strip(),
            content=content.strip(),
            style=style
        )
        
        # Add success status
        result["status"] = "success"
        
        logging.info(f"Generated editorial illustration prompt for '{title[:50]}...'")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Error in image prompt generation API: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate image prompt: {str(e)}"
        )

@app.get("/api/image-prompt-styles")
async def get_image_prompt_styles():
    """Get available image prompt styles with descriptions."""
    try:
        generator = get_image_prompt_generator()
        styles = generator.get_available_styles()
        
        return JSONResponse(content={
            "status": "success",
            "styles": styles
        })
        
    except Exception as e:
        logging.error(f"Error getting image prompt styles: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get image prompt styles: {str(e)}"
        )

# =============================================================================
# Authentication Routes
# =============================================================================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page."""
    # If already logged in, redirect to home
    if get_current_user(request):
        return RedirectResponse(url='/', status_code=303)

    common_vars = get_common_template_vars(request)
    error = request.query_params.get('error', '')
    success = request.query_params.get('success', '')

    return templates.TemplateResponse(
        "auth/login.html",
        {
            **common_vars,
            "error": error,
            "success": success,
            "next_url": request.session.get('next_url', '/'),
        }
    )


@app.post("/login")
async def login_submit(
    request: Request,
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    remember_me: Optional[str] = Form(None)
):
    """Process login form submission."""
    auth_manager = get_auth_manager()
    user, error = auth_manager.authenticate_user(username, password)

    if not user:
        return RedirectResponse(
            url=f'/login?error={quote(error)}',
            status_code=303
        )

    # Set session
    set_user_session(request, user)

    # Handle "remember me" - create persistent token
    redirect_response = RedirectResponse(
        url=request.session.pop('next_url', '/'),
        status_code=303
    )

    if remember_me == 'on':
        token = auth_manager.create_session_token(user.id, expires_days=30)
        if token:
            redirect_response.set_cookie(
                key=COOKIE_REMEMBER_TOKEN,
                value=token,
                max_age=30 * 24 * 60 * 60,  # 30 days
                httponly=True,
                samesite='lax'
            )

    logging.info(f"User logged in: {user.username}")
    return redirect_response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render the registration page."""
    # If already logged in, redirect to home
    if get_current_user(request):
        return RedirectResponse(url='/', status_code=303)

    common_vars = get_common_template_vars(request)
    error = request.query_params.get('error', '')

    return templates.TemplateResponse(
        "auth/register.html",
        {
            **common_vars,
            "error": error,
        }
    )


@app.post("/register")
async def register_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...)
):
    """Process registration form submission."""
    # Validate password confirmation
    if password != password_confirm:
        return RedirectResponse(
            url='/register?error=Passwords+do+not+match',
            status_code=303
        )

    auth_manager = get_auth_manager()
    user, error = auth_manager.register_user(username, email, password)

    if not user:
        return RedirectResponse(
            url=f'/register?error={quote(error)}',
            status_code=303
        )

    # Initialize user's data directory and import default feeds
    user_data = get_user_data_manager(user.id)
    feeds_imported = user_data.import_default_feeds()
    logging.info(f"New user {user.username} registered, imported {feeds_imported} default feeds")

    # Log the user in automatically
    set_user_session(request, user)

    return RedirectResponse(url='/', status_code=303)


@app.get("/logout")
async def logout(request: Request, response: Response):
    """Log out the current user."""
    user = get_current_user(request)

    # Invalidate remember token if present
    remember_token = request.cookies.get(COOKIE_REMEMBER_TOKEN)
    if remember_token:
        auth_manager = get_auth_manager()
        auth_manager.invalidate_session_token(remember_token)

    # Clear session
    clear_user_session(request)

    if user:
        logging.info(f"User logged out: {user['username']}")

    # Create redirect response and clear cookie
    redirect_response = RedirectResponse(url='/login', status_code=303)
    redirect_response.delete_cookie(key=COOKIE_REMEMBER_TOKEN)

    return redirect_response


@app.get("/profile", response_class=HTMLResponse)
@require_login()
async def profile_page(request: Request):
    """Render the user profile page."""
    common_vars = get_common_template_vars(request)
    user = get_current_user(request)

    # Get user's data manager for stats
    user_data = get_user_data_manager(user['id'])
    bookmark_manager = user_data.get_bookmark_manager()

    # Get stats
    feeds = user_data.get_feeds()
    bookmarks = bookmark_manager.get_bookmarks()

    return templates.TemplateResponse(
        "auth/profile.html",
        {
            **common_vars,
            "feed_count": len(feeds),
            "bookmark_count": len(bookmarks),
        }
    )


@app.post("/profile/change-password")
@require_login()
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    new_password_confirm: str = Form(...)
):
    """Change the user's password."""
    if new_password != new_password_confirm:
        return RedirectResponse(
            url='/profile?error=New+passwords+do+not+match',
            status_code=303
        )

    user = get_current_user(request)
    auth_manager = get_auth_manager()

    # Verify current password
    db_user, _ = auth_manager.authenticate_user(user['username'], current_password)
    if not db_user:
        return RedirectResponse(
            url='/profile?error=Current+password+is+incorrect',
            status_code=303
        )

    # Update password (need to get fresh user object and update)
    session = auth_manager._get_session()
    try:
        from models.user import User
        user_obj = session.query(User).filter(User.id == user['id']).first()
        if user_obj:
            user_obj.set_password(new_password)
            session.commit()
            logging.info(f"Password changed for user: {user['username']}")
    finally:
        session.close()

    return RedirectResponse(
        url='/profile?success=Password+changed+successfully',
        status_code=303
    )


# =============================================================================
# Main Entry Point
# =============================================================================

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