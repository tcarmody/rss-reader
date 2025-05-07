"""Configuration utilities for the RSS reader."""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_var(name, default=None):
    """
    Get an environment variable or return a default value.
    
    Args:
        name: Name of the environment variable
        default: Default value if not found
        
    Returns:
        Value of the environment variable or default
    """
    value = os.environ.get(name, default)
    if value is None:
        logging.warning(f"Environment variable {name} not found")
    return value
