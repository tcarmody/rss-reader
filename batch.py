"""Batch processing utilities for handling multiple items efficiently."""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor


class BatchProcessor:
    """
    Process items in batches with rate limiting and thread safety.
    
    This class helps manage processing multiple items in parallel while respecting
    rate limits and ensuring thread safety.
    
    Example:
        processor = BatchProcessor(batch_size=5)
        processor.add({'func': my_function, 'args': [arg1, arg2], 'kwargs': {'key': 'value'}})
        results = processor.get_results()
    """

    def __init__(self, batch_size=5, requests_per_second=5):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of items to process in a batch
            requests_per_second: Maximum API calls per second
        """
        self.batch_size = batch_size
        self.delay = 1.0 / requests_per_second  # Time between requests
        self.queue = []
        self.results = []
        self.last_request_time = 0
        self.lock = threading.Lock()  # Add lock for thread safety

    def add(self, item):
        """
        Add an item to the processing queue.
        
        Args:
            item: Dict containing 'func', 'args', and 'kwargs' keys
        """
        with self.lock:
            self.queue.append(item)
            if len(self.queue) >= self.batch_size:
                self._process_batch()

    def _process_batch(self):
        """Process a batch of items with rate limiting."""
        with self.lock:
            if not self.queue:
                return
            
            batch_to_process = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
        
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            for item in batch_to_process:
                # Ensure rate limiting
                now = time.time()
                time_since_last = now - self.last_request_time
                if time_since_last < self.delay:
                    time.sleep(self.delay - time_since_last)

                future = executor.submit(
                    item['func'],
                    *item['args'],
                    **item['kwargs']
                )
                futures.append(future)
                self.last_request_time = time.time()

            # Wait for all futures to complete
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        with self.lock:
                            self.results.append(result)
                except Exception as e:
                    logging.error(f"Error in batch processing: {str(e)}")

    def get_results(self):
        """
        Process remaining items and return all results.
        
        Returns:
            list: Results from all processed items
        """
        while True:
            with self.lock:
                if not self.queue:
                    break
            self._process_batch()
        
        with self.lock:
            return self.results
