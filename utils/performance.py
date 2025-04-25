"""Performance tracking utilities for the RSS reader."""

import time
import logging
import functools
import psutil
from datetime import datetime


def track_performance(func):
    """
    Decorator to track function performance.
    
    Logs execution time and memory usage for the decorated function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            logging.info(
                f"Performance: {func.__name__} - "
                f"Time: {execution_time:.2f}s, "
                f"Memory: {memory_used:.2f}MB"
            )
            
            # Log to performance log file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"{timestamp} | {func.__name__} | "
                f"Time: {execution_time:.2f}s | "
                f"Memory: {memory_used:.2f}MB\n"
            )
            
            try:
                with open("performance_logs.txt", "a") as f:
                    f.write(log_entry)
            except Exception as e:
                logging.error(f"Error writing to performance log: {e}")
            
    return wrapper