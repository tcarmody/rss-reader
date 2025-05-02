"""
Streamlined batch processing module for RSS reader.

This module provides a clean, reliable implementation of batch processing
for article summarization, combining both the processor itself and
the application logic in one place.
"""

import os
import sys
import time
import logging
import threading
import queue
import types
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Union
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    A reliable thread-based batch processor for handling multiple tasks in parallel.
    
    Features:
    - Thread-based parallelism for reliability
    - Simple task queue with clear timeout handling
    - Explicit rate limiting with adjustable delay
    - Robust error handling and task isolation
    - Progress tracking and detailed logging
    """
    
    def __init__(self, max_workers=3, rate_limit_delay=0.2):
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Delay between tasks to avoid rate limiting (seconds)
        """
        self.max_workers = max(1, max_workers)  # Ensure at least 1 worker
        self.rate_limit_delay = rate_limit_delay
        self.logger = logging.getLogger("BatchProcessor")
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.RLock()  # For thread-safe operations
        self.last_task_time = 0
        
        self.logger.info(f"Initialized with {self.max_workers} workers and {rate_limit_delay}s rate limit delay")
    
    def add_task(self, func, *args, **kwargs):
        """
        Add a task to the processing queue.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        """
        task = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'id': f"task_{time.time()}_{id(func)}"
        }
        
        self.task_queue.put(task)
        self.logger.debug(f"Added task {task['id']} to queue")
    
    def _worker(self, task):
        """
        Worker function to process a single task.
        
        Args:
            task: Task dictionary containing function and arguments
            
        Returns:
            Task result or exception information
        """
        result = {
            'task_id': task['id'],
            'success': False,
            'result': None,
            'error': None,
            'elapsed': 0
        }
        
        try:
            # Apply rate limiting
            with self.lock:
                current_time = time.time()
                time_since_last = current_time - self.last_task_time
                
                if time_since_last < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - time_since_last
                    self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                
                self.last_task_time = time.time()
            
            # Execute the task
            self.logger.debug(f"Processing task {task['id']}")
            start_time = time.time()
            
            # Extract function and arguments
            func = task['func']
            args = task['args']
            kwargs = task['kwargs']
            
            # Execute the function
            task_result = func(*args, **kwargs)
            
            # Record success
            elapsed = time.time() - start_time
            result['success'] = True
            result['result'] = task_result
            result['elapsed'] = elapsed
            
            self.logger.debug(f"Task {task['id']} completed in {elapsed:.2f}s")
            
        except Exception as e:
            # Record failure with error information
            import traceback
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            
            result['success'] = False
            result['error'] = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            result['elapsed'] = elapsed
            
            self.logger.error(f"Task {task['id']} failed after {elapsed:.2f}s: {str(e)}")
        
        return result
    
    def process_batch(self, tasks=None, timeout=None):
        """
        Process a batch of tasks.
        
        This can either process tasks that were previously added with add_task(),
        or process a new list of tasks provided directly to this method.
        
        Args:
            tasks: Optional list of task dictionaries with 'func', 'args', and 'kwargs'
            timeout: Optional timeout in seconds for the entire batch
            
        Returns:
            List of results in the order tasks were processed
        """
        # Add provided tasks to the queue if any
        if tasks:
            for task in tasks:
                if callable(task.get('func')):
                    self.add_task(task['func'], *task.get('args', []), **task.get('kwargs', {}))
                else:
                    self.logger.warning(f"Skipping invalid task: {task}")
        
        # Get the number of tasks to process
        queue_size = self.task_queue.qsize()
        if queue_size == 0:
            self.logger.warning("No tasks to process")
            return []
        
        self.logger.info(f"Processing batch of {queue_size} tasks with {self.max_workers} workers")
        start_time = time.time()
        results = []
        
        # Process tasks using thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            tasks_in_progress = 0
            
            # Keep processing until the queue is empty or timeout
            while not self.task_queue.empty() or futures:
                # Check timeout
                if timeout and time.time() - start_time > timeout:
                    self.logger.warning(f"Batch processing timed out after {timeout}s")
                    break
                
                # Submit tasks to the thread pool up to max_workers
                while not self.task_queue.empty() and tasks_in_progress < self.max_workers:
                    try:
                        task = self.task_queue.get_nowait()
                        future = executor.submit(self._worker, task)
                        futures.append(future)
                        tasks_in_progress += 1
                    except queue.Empty:
                        break
                
                # Get any completed futures
                completed = []
                for future in futures:
                    if future.done():
                        completed.append(future)
                
                # Process completed futures
                for future in completed:
                    futures.remove(future)
                    tasks_in_progress -= 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing future: {str(e)}")
                        results.append({
                            'task_id': 'unknown',
                            'success': False,
                            'error': {'type': type(e).__name__, 'message': str(e)},
                            'elapsed': 0
                        })
                
                # Small sleep to avoid CPU spinning
                if futures:
                    time.sleep(0.01)
        
        # Process time statistics
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        error_count = len(results) - success_count
        
        self.logger.info(
            f"Batch completed in {elapsed:.2f}s: {len(results)} tasks processed, "
            f"{success_count} successful, {error_count} failed"
        )
        
        return results
    
    async def process_batch_async(self, tasks=None, timeout=None):
        """
        Process a batch of tasks asynchronously.
        
        This is a wrapper around process_batch for use with async/await.
        
        Args:
            tasks: Optional list of task dictionaries with 'func', 'args', and 'kwargs'
            timeout: Optional timeout in seconds for the entire batch
            
        Returns:
            List of results in the order tasks were processed
        """
        import asyncio
        
        # Run the synchronous method in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.process_batch(tasks=tasks, timeout=timeout)
        )
        
        return result


class ArticleBatchProcessor:
    """
    Specialized batch processor for article summarization.
    
    This class provides a simple and reliable interface specifically for
    processing article summaries in batches.
    """
    
    def __init__(self, summarizer, max_workers=3, rate_limit_delay=0.5):
        """
        Initialize the article batch processor.
        
        Args:
            summarizer: The article summarizer instance
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Delay between tasks to avoid rate limiting
        """
        self.summarizer = summarizer
        self.processor = BatchProcessor(max_workers=max_workers, 
                                       rate_limit_delay=rate_limit_delay)
        self.logger = logging.getLogger("ArticleBatchProcessor")
    
    def _process_article(self, article, model=None, temperature=0.3):
        """
        Process a single article.
        
        Args:
            article: Article dictionary with 'text', 'title', and 'url'
            model: Optional model to use
            temperature: Temperature setting
            
        Returns:
            Dictionary with original article and summary
        """
        try:
            # Get article data
            text = article.get('text', article.get('content', ''))
            title = article.get('title', 'No Title')
            url = article.get('url', article.get('link', '#'))
            
            # Generate summary
            summary = self.summarizer.summarize_article(
                text=text,
                title=title,
                url=url,
                model=model,
                temperature=temperature
            )
            
            return {
                'original': article,
                'summary': summary,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing article {article.get('title', 'Unknown')}: {str(e)}")
            return {
                'original': article,
                'error': str(e),
                'success': False,
                'summary': {
                    'headline': article.get('title', 'Error'),
                    'summary': f"Summary generation failed: {str(e)}. Please try again later."
                }
            }
    
    def process_articles(self, articles, model=None, temperature=0.3, timeout=None):
        """
        Process a batch of articles synchronously.
        
        Args:
            articles: List of article dictionaries
            model: Optional model to use for all articles
            temperature: Temperature setting
            timeout: Optional timeout in seconds
            
        Returns:
            List of processed articles with summaries
        """
        if not articles:
            return []
            
        self.logger.info(f"Processing batch of {len(articles)} articles")
        
        # Create tasks for all articles
        tasks = []
        for article in articles:
            tasks.append({
                'func': self._process_article,
                'args': [article],
                'kwargs': {'model': model, 'temperature': temperature}
            })
        
        # Process the batch
        results = self.processor.process_batch(tasks=tasks, timeout=timeout)
        
        # Convert results to the expected format
        formatted_results = []
        for result in results:
            if result['success'] and result['result']:
                # The result is the dictionary returned by _process_article
                article_result = result['result']
                formatted_results.append(article_result)
            else:
                # Create an error entry if something went wrong
                error_message = "Unknown error"
                if result.get('error'):
                    error_message = result['error'].get('message', "Unknown error")
                
                # Create a minimal error result
                formatted_results.append({
                    'original': {'title': 'Unknown Article', 'url': '#'},
                    'error': error_message,
                    'success': False,
                    'summary': {
                        'headline': 'Error',
                        'summary': f"Summary generation failed: {error_message}. Please try again later."
                    }
                })
        
        return formatted_results
    
    async def process_articles_async(self, articles, model=None, temperature=0.3, timeout=None):
        """
        Process a batch of articles asynchronously.
        
        Args:
            articles: List of article dictionaries
            model: Optional model to use for all articles
            temperature: Temperature setting
            timeout: Optional timeout in seconds
            
        Returns:
            List of processed articles with summaries
        """
        import asyncio
        
        # Run the synchronous method in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.process_articles(
                articles=articles,
                model=model,
                temperature=temperature,
                timeout=timeout
            )
        )
        
        return result


# Integration Functions
def apply_fix_to_fast_summarizer(fast_summarizer, max_workers=3):
    """
    Apply the batch processing fix to a FastArticleSummarizer instance.
    
    Args:
        fast_summarizer: An instance of FastArticleSummarizer
        max_workers: Maximum number of worker threads
        
    Returns:
        The modified FastArticleSummarizer instance
    """
    # Define the patched batch_summarize method
    async def patched_batch_summarize(self, articles, max_concurrent=3, 
                               model=None, auto_select_model=True, 
                               temperature=0.3):
        """
        Process a batch of articles using the improved batch processor.
        
        Args:
            articles: List of article dictionaries
            max_concurrent: Maximum number of concurrent workers
            model: Optional model identifier
            auto_select_model: Whether to automatically select model based on content
            temperature: Temperature setting
            
        Returns:
            List of processed articles with summaries
        """
        # Create a batch processor with the original summarizer
        batch_processor = ArticleBatchProcessor(
            summarizer=self.original,
            max_workers=max_concurrent,
            rate_limit_delay=0.5  # Adjust as needed
        )
        
        # Process articles in batch
        prepared_articles = []
        
        # Auto-select models if needed
        if auto_select_model and not model:
            # Group articles by selected model
            model_groups = {}
            
            for article in articles:
                # Clean text for complexity estimation
                text = article.get('text', article.get('content', ''))
                text = self.original.clean_text(text) if hasattr(self.original, 'clean_text') else text
                
                # Select model based on content length
                if len(text) < 2000:
                    selected_model = "haiku"  # Use fastest model for short articles
                elif len(text) > 10000:
                    selected_model = "sonnet-3.7"  # Use most capable model for long articles
                else:
                    selected_model = "sonnet"  # Use balanced model for medium articles
                
                # Add to the appropriate model group
                if selected_model not in model_groups:
                    model_groups[selected_model] = []
                
                # Create a copy with normalized fields
                prepared_article = {
                    'text': text,
                    'content': text,  # For compatibility
                    'title': article.get('title', 'No Title'),
                    'url': article.get('url', article.get('link', '#')),
                    'link': article.get('link', article.get('url', '#'))
                }
                
                model_groups[selected_model].append(prepared_article)
            
            # Process each model group
            all_results = []
            for selected_model, group_articles in model_groups.items():
                self.logger.info(f"Processing {len(group_articles)} articles with model {selected_model}")
                
                results = await batch_processor.process_articles_async(
                    articles=group_articles,
                    model=selected_model,
                    temperature=temperature
                )
                
                all_results.extend(results)
                
            return all_results
        else:
            # Process all with the same model - prepare articles first
            for article in articles:
                # Ensure text field exists
                text = article.get('text', article.get('content', ''))
                
                # Clean the text if possible
                if hasattr(self.original, 'clean_text'):
                    text = self.original.clean_text(text)
                
                prepared_article = {
                    'text': text,
                    'content': text,  # For compatibility
                    'title': article.get('title', 'No Title'),
                    'url': article.get('url', article.get('link', '#')),
                    'link': article.get('link', article.get('url', '#'))
                }
                prepared_articles.append(prepared_article)
            
            # Process the batch
            return await batch_processor.process_articles_async(
                articles=prepared_articles,
                model=model,
                temperature=temperature
            )
    
    # Apply the patch to the FastArticleSummarizer
    fast_summarizer.batch_summarize = types.MethodType(patched_batch_summarize, fast_summarizer)
    
    logger.info(f"Applied batch processing fix to FastArticleSummarizer with {max_workers} workers")
    return fast_summarizer


# Compatibility layer for existing code that uses EnhancedBatchProcessor
class BatchProcessorLegacyAdapter:
    """
    Adapter class to provide backward compatibility with EnhancedBatchProcessor.
    """
    
    def __init__(self, max_workers=3):
        """Initialize with the recommended number of workers."""
        self.max_workers = max_workers
        self.logger = logging.getLogger("BatchProcessorLegacyAdapter")
    
    async def process_batch_async(self, summarizer, articles, model=None, temperature=0.3):
        """Process a batch of articles asynchronously."""
        processor = ArticleBatchProcessor(
            summarizer=summarizer,
            max_workers=self.max_workers,
            rate_limit_delay=0.5
        )
        
        return await processor.process_articles_async(
            articles=articles,
            model=model,
            temperature=temperature
        )
    
    def process_batch_sync(self, articles, model=None, temperature=0.3, timeout=None):
        """Synchronous version for backward compatibility."""
        from summarizer import ArticleSummarizer
        
        # Create a summarizer if not available
        summarizer = getattr(self, 'summarizer', ArticleSummarizer())
        
        processor = ArticleBatchProcessor(
            summarizer=summarizer,
            max_workers=self.max_workers,
            rate_limit_delay=0.5
        )
        
        return processor.process_articles(
            articles=articles,
            model=model,
            temperature=temperature,
            timeout=timeout
        )
    
    @contextmanager
    def batch_context(self):
        """Context manager for backward compatibility."""
        try:
            yield self
        finally:
            pass  # No cleanup needed


# Application function to apply the fix
def apply():
    """
    Apply the batch processing fix to the RSS reader.
    
    This function patches the FastArticleSummarizer and provides
    compatibility shims for code that uses EnhancedBatchProcessor.
    """
    logger.info("Applying batch processing fix...")
    
    try:
        # Find and patch FastArticleSummarizer
        try:
            from fast_summarizer import FastArticleSummarizer
            
            # Monkey-patch any existing instances in the global namespace
            import inspect
            import gc
            
            # Get all FastArticleSummarizer instances
            summarizers = [obj for obj in gc.get_objects() 
                          if isinstance(obj, FastArticleSummarizer)]
            
            if summarizers:
                logger.info(f"Found {len(summarizers)} FastArticleSummarizer instances to patch")
                for summarizer in summarizers:
                    apply_fix_to_fast_summarizer(summarizer)
            else:
                logger.info("No existing FastArticleSummarizer instances found")
            
            # Also patch the class for future instances
            original_init = FastArticleSummarizer.__init__
            
            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                apply_fix_to_fast_summarizer(self)
                
            FastArticleSummarizer.__init__ = patched_init
            logger.info("Patched FastArticleSummarizer.__init__ for future instances")
            
        except ImportError:
            logger.warning("FastArticleSummarizer not found, skipping patch")
        
        # Add compatibility layer for EnhancedBatchProcessor
        try:
            import sys
            
            class CompatibilityWrapper:
                """Module-like object that provides compatibility with the old API."""
                
                def __init__(self):
                    self.EnhancedBatchProcessor = BatchProcessorLegacyAdapter
                    self.WorkerProcess = object  # Dummy class
                    self.ProcessingTask = object  # Dummy class
                    self.ProcessingResult = object  # Dummy class
                    self.QueueHandler = object  # Dummy class
                
                def create_enhanced_batch_processor(self, max_workers=3):
                    """Create a batch processor with the old API."""
                    return BatchProcessorLegacyAdapter(max_workers)
                    
                def add_enhanced_batch_to_fast_summarizer(self, fast_summarizer, max_workers=3):
                    """Add the batch method to a fast summarizer instance."""
                    return apply_fix_to_fast_summarizer(fast_summarizer, max_workers)
            
            # Replace the enhanced_batch_processor module
            if 'enhanced_batch_processor' in sys.modules:
                # If it was already imported, replace it
                sys.modules['enhanced_batch_processor'] = CompatibilityWrapper()
                logger.info("Replaced existing enhanced_batch_processor module with compatibility wrapper")
            else:
                # If not yet imported, prepare to replace it on import
                sys.meta_path.insert(0, CompatibilityImportHook('enhanced_batch_processor', CompatibilityWrapper()))
                logger.info("Installed import hook for enhanced_batch_processor")
            
        except Exception as e:
            logger.warning(f"Failed to install compatibility layer: {e}")
        
        logger.info("✅ Batch processing fix successfully applied!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply batch processing fix: {e}")
        import traceback
        traceback.print_exc()
        return False


# Helper for module replacement
class CompatibilityImportHook:
    """Import hook to provide compatibility modules."""
    
    def __init__(self, module_name, replacement):
        self.module_name = module_name
        self.replacement = replacement
    
    def find_spec(self, fullname, path, target=None):
        if fullname == self.module_name:
            import importlib.machinery
            import importlib.util
            
            # Create a dummy spec
            loader = importlib.machinery.SourceFileLoader(fullname, f"{fullname}.py")
            spec = importlib.util.spec_from_loader(fullname, loader)
            spec.loader = self
            return spec
        return None
    
    def create_module(self, spec):
        # Return the replacement module instead
        return self.replacement()
    
    def exec_module(self, module):
        # Already initialized in create_module
        pass


# For direct usage
if __name__ == "__main__":
    # Configure logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Apply the fix
    result = apply()
    
    if result:
        print("\n✅ Batch processing fix successfully applied!")
        print()
        print("You can now use the improved batch processing functionality:")
        print("1. FastArticleSummarizer.batch_summarize() is now using the improved implementation")
        print("2. Existing code that uses EnhancedBatchProcessor will continue to work")
        print()
    else:
        print("\n❌ Failed to apply batch processing fix. Check the logs for details.")
        print()
