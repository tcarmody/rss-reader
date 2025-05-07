"""
Unified batch processing module for parallel task execution.
"""

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Process items in batches with rate limiting and thread safety.
    
    Features:
    - Thread-based parallelism with configurable worker count
    - Rate limiting with adjustable delay between tasks
    - Thread safety with proper locking
    - Detailed logging and error reporting
    """
    
    def __init__(self, max_workers=3, rate_limit_delay=0.2):
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Delay between tasks to avoid rate limiting (seconds)
        """
        self.max_workers = max(1, max_workers)
        self.rate_limit_delay = rate_limit_delay
        self.lock = threading.RLock()
        self.last_task_time = 0
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")
        self.logger.info(f"Initialized BatchProcessor with {max_workers} workers and {rate_limit_delay}s delay")
    
    def process_batch(self, items, task_func, timeout=None):
        """
        Process a batch of items using the provided function.
        
        Args:
            items: List of items to process
            task_func: Function to call for each item (item) -> result
            timeout: Optional timeout in seconds
            
        Returns:
            List of results in order of completion
        """
        if not items:
            return []
            
        self.logger.info(f"Processing batch of {len(items)} items")
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all tasks
            for item in items:
                # Apply rate limiting
                with self.lock:
                    current_time = time.time()
                    time_since_last = current_time - self.last_task_time
                    
                    if time_since_last < self.rate_limit_delay:
                        sleep_time = self.rate_limit_delay - time_since_last
                        time.sleep(sleep_time)
                    
                    self.last_task_time = time.time()
                
                # Submit the task
                future = executor.submit(self._execute_task, task_func, item)
                futures.append(future)
            
            # Wait for completion or timeout
            for future in futures:
                try:
                    if timeout and time.time() - start_time > timeout:
                        break
                        
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing batch item: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e)
                    })
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.get('success', False))
        self.logger.info(f"Batch completed in {elapsed:.2f}s: {len(results)}/{len(items)} processed, {success_count} successful")
        
        return results
    
    def _execute_task(self, task_func, item):
        """
        Execute a single task with error handling.
        
        Args:
            task_func: Function to execute
            item: Item to process
            
        Returns:
            Task result with success indicator
        """
        try:
            start_time = time.time()
            result = task_func(item)
            elapsed = time.time() - start_time
            
            # Ensure result has the right format
            if isinstance(result, dict):
                if 'success' not in result:
                    result['success'] = True
                result['elapsed'] = elapsed
                return result
            else:
                return {
                    'success': True,
                    'result': result,
                    'elapsed': elapsed
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'elapsed': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    async def process_batch_async(self, items, task_func, timeout=None):
        """
        Process a batch of items asynchronously.
        
        This is a wrapper around process_batch for use with async/await.
        
        Args:
            items: List of items to process
            task_func: Function to call for each item
            timeout: Optional timeout in seconds
            
        Returns:
            List of results
        """
        import asyncio
        
        # Run the synchronous method in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.process_batch(
                items=items,
                task_func=task_func,
                timeout=timeout
            )
        )
        
        return result