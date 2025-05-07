"""HTTP utilities for the RSS reader."""

import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_http_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    """
    Create a requests session with retry capability.
    
    Args:
        retries: Number of retries to attempt
        backoff_factor: Backoff factor for retries
        status_forcelist: HTTP status codes to retry on
        
    Returns:
        requests.Session: Configured session object
    """
    session = requests.Session()
    
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Set reasonable timeouts
    session.timeout = (10, 30)  # (connect, read) timeouts in seconds
    
    logging.debug("Created HTTP session with retry capability")
    return session
