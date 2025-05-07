"""
Tests for batch processing functionality.
"""

import asyncio
import time
import unittest
from unittest.mock import Mock, patch

from common.batch_processing import BatchProcessor

class TestBatchProcessor(unittest.TestCase):
    """Test the BatchProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor(max_workers=3, rate_limit_delay=0.1)
    
    def test_process_batch_empty(self):
        """Test processing an empty batch."""
        results = self.processor.process_batch(items=[], task_func=lambda x: x)
        self.assertEqual(results, [])
    
    def test_process_batch_simple(self):
        """Test processing a simple batch."""
        items = [1, 2, 3, 4, 5]
        results = self.processor.process_batch(
            items=items,
            task_func=lambda x: x * 2
        )
        
        self.assertEqual(len(results), 5)
        self.assertEqual(sorted([r['result'] for r in results]), [2, 4, 6, 8, 10])
    
    def test_process_batch_with_errors(self):
        """Test processing a batch with errors."""
        def task_func(x):
            if x % 2 == 0:
                raise ValueError(f"Error processing {x}")
            return x * 2
        
        items = [1, 2, 3, 4, 5]
        results = self.processor.process_batch(
            items=items,
            task_func=task_func
        )
        
        self.assertEqual(len(results), 5)
        
        # Check success items
        success_items = [r for r in results if r.get('success', False)]
        self.assertEqual(len(success_items), 3)
        self.assertEqual(sorted([r['result'] for r in success_items]), [2, 6, 10])
        
        # Check error items
        error_items = [r for r in results if not r.get('success', False)]
        self.assertEqual(len(error_items), 2)
        self.assertTrue(all('error' in r for r in error_items))
    
    def test_rate_limiting(self):
        """Test that rate limiting is applied."""
        start_time = time.time()
        
        # Process items with a noticeable delay
        items = list(range(5))
        self.processor.process_batch(
            items=items,
            task_func=lambda x: x
        )
        
        elapsed = time.time() - start_time
        
        # With 5 items and 0.1s delay, should take at least 0.4s
        # (first item has no delay, subsequent 4 items have delay)
        self.assertTrue(elapsed >= 0.4, f"Elapsed time {elapsed} < 0.4s")
    
    async def test_process_batch_async(self):
        """Test asynchronous batch processing."""
        items = [1, 2, 3, 4, 5]
        results = await self.processor.process_batch_async(
            items=items,
            task_func=lambda x: x * 2
        )
        
        self.assertEqual(len(results), 5)
        self.assertEqual(sorted([r['result'] for r in results]), [2, 4, 6, 8, 10])
    
    def test_timeout(self):
        """Test that timeout works correctly."""
        def slow_task(x):
            time.sleep(0.2)
            return x
        
        items = list(range(10))
        
        # With timeout 0.5s, should only process 2-3 items
        results = self.processor.process_batch(
            items=items,
            task_func=slow_task,
            timeout=0.5
        )
        
        # Check that we got fewer results than items
        self.assertLess(len(results), len(items))

# Run the tests
if __name__ == "__main__":
    unittest.main()