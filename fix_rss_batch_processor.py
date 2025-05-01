#!/usr/bin/env python3
"""
Integration script to fix the batch processing issue in the RSS reader application.

This script specifically addresses two issues:
1. The timeout issue with workers not being recognized
2. The issue with 0 workers being initialized
"""

import os
import sys
import logging
import importlib
import pickle
import time
import queue
import multiprocessing
import traceback
import signal
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rss_batch_processor_fix.log")
    ]
)
logger = logging.getLogger("rss_batch_processor_fix")

def patch_enhanced_batch_processor():
    """
    Direct patch to the EnhancedBatchProcessor's process_batch_sync method.
    This patch addresses the specific timeout issue in the worker communication.
    """
    try:
        # Import the original module
        from enhanced_batch_processor import EnhancedBatchProcessor, WorkerProcess, ProcessingTask, ProcessingResult
        
        # Store the original methods
        original_process_batch_sync = EnhancedBatchProcessor.process_batch_sync
        original_worker_main = WorkerProcess._worker_main
        original_init = EnhancedBatchProcessor.__init__
        original_start_workers = EnhancedBatchProcessor.start_workers
        
        # Define a patched __init__ method to ensure min_workers is at least 1
        def patched_init(self, max_workers=3, log_level=logging.INFO):
            """
            Patched initialization to ensure at least 1 worker is created.
            """
            # Ensure max_workers is at least 1
            max_workers = max(1, max_workers)
            logger.info(f"Initializing EnhancedBatchProcessor with {max_workers} workers (patched)")
            
            # Call original init with ensured min workers
            original_init(self, max_workers=max_workers, log_level=log_level)
        
        # Define a patched start_workers method
        def patched_start_workers(self):
            """
            Patched start_workers to ensure at least 1 worker is created
            and to improve worker initialization.
            """
            # Ensure max_workers is at least 1
            self.max_workers = max(1, self.max_workers)
            self.logger.info(f"Starting {self.max_workers} worker processes (patched)")
            
            # Create workers
            for i in range(self.max_workers):
                worker = WorkerProcess(
                    worker_id=i,
                    ready_queue=self.ready_queue,
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    log_queue=self.log_queue,
                    shutdown_event=self.shutdown_event
                )
                worker.start()
                self.workers.append(worker)
                
            # Wait for all workers to initialize
            self.logger.info("Waiting for workers to initialize...")
            initialization_timeout = 30.0  # seconds
            start_time = time.time()
            ready_workers = set()
            
            while len(ready_workers) < self.max_workers and (time.time() - start_time < initialization_timeout):
                # Check for initialization errors
                if not self.result_queue.empty():
                    try:
                        result = pickle.loads(self.result_queue.get_nowait())
                        if isinstance(result, dict) and result.get('type') == 'init_error':
                            worker_id = result.get('worker_id')
                            error = result.get('error')
                            tb = result.get('traceback')
                            self.logger.error(f"Worker {worker_id} failed to initialize: {error}\n{tb}")
                            # Remove the failed worker
                            self.workers = [w for w in self.workers if w.worker_id != worker_id]
                            self.max_workers -= 1
                    except Exception as e:
                        self.logger.error(f"Error checking for worker initialization: {e}")
                
                # Check for ready workers
                try:
                    worker_id = self.ready_queue.get(timeout=0.5)
                    # FIX: Put the worker ID back in the ready queue if already seen
                    if worker_id in ready_workers:
                        self.ready_queue.put(worker_id)
                    else:
                        ready_workers.add(worker_id)
                        self.logger.info(f"Worker {worker_id} ready ({len(ready_workers)}/{self.max_workers})")
                except queue.Empty:
                    # No worker ready yet, wait a bit
                    time.sleep(0.1)
                    
            # Check for timeout
            if len(ready_workers) < self.max_workers:
                self.logger.warning(f"Initialization timed out. Only {len(ready_workers)}/{self.max_workers} workers ready.")
                # Update max_workers to match actual ready workers
                self.max_workers = len(ready_workers)
                    
            if self.max_workers == 0:
                self.logger.error("No workers initialized successfully. Creating a single fallback worker...")
                
                # Create a single fallback worker with different settings
                try:
                    fallback_worker = WorkerProcess(
                        worker_id=999,  # Use a special ID for the fallback worker
                        ready_queue=self.ready_queue,
                        task_queue=self.task_queue,
                        result_queue=self.result_queue,
                        log_queue=self.log_queue,
                        shutdown_event=self.shutdown_event
                    )
                    fallback_worker.start()
                    self.workers.append(fallback_worker)
                    self.max_workers = 1
                    
                    # Wait for the fallback worker to initialize
                    self.logger.info("Waiting for fallback worker to initialize...")
                    try:
                        worker_id = self.ready_queue.get(timeout=10.0)
                        self.logger.info(f"Fallback worker {worker_id} ready")
                        # Put the worker ID back in the ready queue
                        self.ready_queue.put(worker_id)
                        return 1
                    except queue.Empty:
                        self.logger.error("Fallback worker failed to initialize")
                        from enhanced_batch_processor import ProcessInitError
                        raise ProcessInitError("All worker processes failed to initialize including fallback")
                except Exception as e:
                    self.logger.error(f"Error creating fallback worker: {e}")
                    from enhanced_batch_processor import ProcessInitError
                    raise ProcessInitError("All worker processes failed to initialize")
            
            self.logger.info(f"{self.max_workers} workers initialized and ready")
            return self.max_workers
        
        # Define the fixed version of process_batch_sync
        def fixed_process_batch_sync(self, articles, model=None, temperature=0.3, timeout=None):
            """
            Fixed version of process_batch_sync to prevent timeout issues.
            This specifically addresses the issue where the method times out
            waiting for a worker, even though workers are available.
            """
            if not articles:
                self.logger.info("No articles to process, returning empty list")
                return []
                
            start_time = time.time()
            self.logger.info(f"Processing batch of {len(articles)} articles (fixed method)")
            
            # Create tasks for all articles
            tasks = {}
            for i, article in enumerate(articles):
                task_id = f"task_{start_time}_{i}"
                
                # Handle different article formats (text vs content key)
                if 'text' not in article and 'content' in article:
                    # Create a copy with text key for compatibility
                    article_copy = dict(article)
                    article_copy['text'] = article['content']
                    task_article = article_copy
                else:
                    task_article = article
                
                # Handle different URL formats (url vs link key)
                if 'url' not in task_article and 'link' in task_article:
                    task_article['url'] = task_article['link']
                
                self.logger.debug(f"Created task {task_id} for article: {task_article.get('title', 'Untitled')}")
                
                task = ProcessingTask(
                    article=task_article,
                    model=model,
                    temperature=temperature,
                    task_id=task_id
                )
                tasks[task_id] = task
                
            # Start the workers if not already started
            workers_started = False
            if not self.workers:
                try:
                    self.start_workers()
                    workers_started = True
                except Exception as e:
                    self.logger.error(f"Failed to start workers: {e}")
                    return [{
                        'original': article,
                        'error': f"Failed to start worker processes: {str(e)}",
                        'summary': {
                            'headline': article.get('title', 'Error'),
                            'summary': f"Summary generation failed: {str(e)}. Please try again later."
                        }
                    } for article in articles]
                
            try:
                # FIX: Drain the ready queue to ensure we have all available workers
                worker_ids = []
                self.logger.info("Clearing ready queue to collect available workers")
                drain_start = time.time()
                
                # Try to collect ready workers for up to 5 seconds
                while time.time() - drain_start < 5.0:
                    try:
                        worker_id = self.ready_queue.get_nowait()
                        worker_ids.append(worker_id)
                        self.logger.debug(f"Found ready worker: {worker_id}")
                    except queue.Empty:
                        break
                
                # If no workers ready, wait a bit longer for at least one
                if not worker_ids:
                    self.logger.info("No workers ready yet, waiting for at least one...")
                    try:
                        worker_id = self.ready_queue.get(timeout=10.0)
                        worker_ids.append(worker_id)
                        self.logger.info(f"Successfully got worker {worker_id} ready")
                    except queue.Empty:
                        self.logger.warning("No workers became ready after extended wait")
                
                # Log how many workers we found
                self.logger.info(f"Found {len(worker_ids)} ready workers out of {self.max_workers}")
                
                # If still no workers, create a special fallback worker
                if not worker_ids and self.max_workers > 0:
                    self.logger.warning("No workers available from ready queue, creating fallback signal")
                    # Create a fake worker ID signal
                    worker_ids.append(0)  # Use ID 0 as a fallback
                
                # Submit tasks to workers
                submitted_tasks = set()
                
                for task_id, task in tasks.items():
                    # Get a worker ID from our saved list
                    if worker_ids:
                        worker_id = worker_ids.pop(0)
                        self.logger.debug(f"Using worker {worker_id} from pool")
                    else:
                        # Wait for a worker to be ready
                        try:
                            self.logger.info("Waiting for a ready worker...")
                            worker_id = self.ready_queue.get(timeout=5.0)
                            self.logger.info(f"Got worker {worker_id}")
                        except queue.Empty:
                            self.logger.warning("Timed out waiting for a ready worker")
                            break
                            
                    # Submit the task to the task queue
                    try:
                        self.logger.debug(f"Submitting task {task_id} to worker {worker_id}")
                        self.task_queue.put(pickle.dumps(task))
                        submitted_tasks.add(task_id)
                        self.logger.debug(f"Successfully submitted task {task_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to submit task: {e}")
                        # Put the worker ID back in the pool
                        worker_ids.append(worker_id)
                        continue
                
                # If no tasks were submitted, return empty results
                if not submitted_tasks:
                    self.logger.warning("No tasks could be submitted, returning empty results")
                    return [{
                        'original': article,
                        'error': "Failed to submit task for processing",
                        'summary': {
                            'headline': article.get('title', 'Error'),
                            'summary': "Summary generation failed: Could not submit task for processing."
                        }
                    } for article in articles]
                
                # Process results
                results = {}
                remaining_timeout = timeout or 60.0  # Default 60 seconds if no timeout specified
                start_wait_time = time.time()
                
                self.logger.info(f"Waiting for results from {len(submitted_tasks)} submitted tasks")
                
                # Wait for results with periodic logging
                last_log_time = time.time()
                
                while len(results) < len(submitted_tasks):
                    # Calculate remaining timeout
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0.1, (timeout or 60.0) - elapsed)
                    
                    if remaining_timeout <= 0.1:
                        self.logger.warning(f"Batch processing timed out after {elapsed:.2f}s")
                        break
                    
                    # Log progress periodically
                    if time.time() - last_log_time > 10.0:
                        self.logger.info(f"Still waiting for results: {len(results)}/{len(submitted_tasks)} "
                                         f"received after {elapsed:.2f}s")
                        last_log_time = time.time()
                    
                    # Wait for a result
                    try:
                        result_data = self.result_queue.get(timeout=min(remaining_timeout, 5.0))
                        result = pickle.loads(result_data)
                        
                        # Store the result
                        task_id = result.task_id
                        results[task_id] = result
                        
                        # Log progress
                        self.logger.info(f"Received result {len(results)}/{len(submitted_tasks)}")
                                
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing result: {e}")
                        continue
                
                # Convert results to the expected output format
                formatted_results = []
                for i, article in enumerate(articles):
                    task_id = f"task_{start_time}_{i}"
                    if task_id in results:
                        result = results[task_id]
                        if result.success:
                            formatted_results.append({
                                'original': result.original,
                                'summary': result.summary,
                                'elapsed': result.elapsed
                            })
                        else:
                            formatted_results.append({
                                'original': result.original,
                                'error': result.error,
                                'summary': {
                                    'headline': result.original.get('title', 'Error'),
                                    'summary': f"Summary generation failed: {result.error}. Please try again later."
                                }
                            })
                    else:
                        # Task was not processed
                        formatted_results.append({
                            'original': article,
                            'error': "Task not processed (timeout or worker failure)",
                            'summary': {
                                    'headline': article.get('title', 'Error'),
                                    'summary': "Summary generation failed: Task not processed. Please try again later."
                                }
                            })
                
                elapsed_time = time.time() - start_time
                self.logger.info(f"Batch processing completed: {len(results)}/{len(submitted_tasks)} " +
                               f"articles processed in {elapsed_time:.2f}s")
                
                return formatted_results
                
            finally:
                # Shutdown workers if we started them
                if workers_started:
                    self.shutdown()
        
        # Define the fixed worker_main function
        def fixed_worker_main(worker_id, ready_queue, task_queue, result_queue, log_queue, shutdown_event, 
                            summarizer_module='summarizer', summarizer_class='ArticleSummarizer'):
            """
            Fixed version of the worker main function to improve reliability in worker communication.
            """
            # Configure logging
            logger = WorkerProcess._configure_logging(worker_id, log_queue)
            logger.info(f"Worker {worker_id} starting (fixed version)")
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            
            # Initialize the summarizer
            try:
                # Import modules dynamically to avoid circular imports
                summarizer_module_obj = importlib.import_module(summarizer_module)
                SummarizerClass = getattr(summarizer_module_obj, summarizer_class)
                summarizer = SummarizerClass()
                logger.info(f"Worker {worker_id} initialized {summarizer_class}")
            except Exception as e:
                error_msg = f"Worker {worker_id} failed to initialize summarizer: {str(e)}"
                logger.error(error_msg)
                result_queue.put(pickle.dumps({
                    'type': 'init_error',
                    'worker_id': worker_id,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }))
                return
            
            # Signal that we're ready to accept tasks - critical for worker availability
            logger.info(f"Worker {worker_id} signaling ready")
            ready_queue.put(worker_id)
            logger.info(f"Worker {worker_id} ready for tasks")
            
            # Track task count for diagnostics
            task_count = 0
            
            # Process tasks until shutdown
            while not shutdown_event.is_set():
                try:
                    # Get a task from the task queue with timeout
                    try:
                        task_data = task_queue.get(timeout=0.5)
                        logger.debug(f"Worker {worker_id} received task data")
                    except queue.Empty:
                        # If no task, put worker_id back in ready_queue to signal availability
                        if not shutdown_event.is_set():
                            ready_queue.put(worker_id)
                            logger.debug(f"Worker {worker_id} signaling ready (no tasks)")
                        continue
                    
                    # Check if this is a shutdown signal
                    if task_data == 'SHUTDOWN':
                        logger.info(f"Worker {worker_id} received shutdown signal")
                        break
                    
                    # Process the task
                    try:
                        # Deserialize the task
                        task = pickle.loads(task_data)
                        task_count += 1
                        
                        # Fix: Handle different key formats for article fields
                        text = task.article.get('text', task.article.get('content', ''))
                        title = task.article.get('title', 'No Title')
                        url = task.article.get('url', task.article.get('link', '#'))
                        
                        logger.info(f"Worker {worker_id} processing article {task_count}: {url}")
                        
                        # Process the article
                        start_time = time.time()
                        summary = summarizer.summarize_article(
                            text=text,
                            title=title,
                            url=url,
                            model=task.model,
                            temperature=task.temperature
                        )
                        elapsed = time.time() - start_time
                        
                        # Create successful result
                        result = ProcessingResult(
                            task_id=task.task_id,
                            success=True,
                            original=task.article,
                            summary=summary,
                            elapsed=elapsed
                        )
                        
                        logger.info(f"Worker {worker_id} completed article in {elapsed:.2f}s")
                    
                    except Exception as e:
                        # Create error result
                        tb = traceback.format_exc()
                        logger.error(f"Worker {worker_id} error processing article: {str(e)}\n{tb}")
                        
                        result = ProcessingResult(
                            task_id=task.task_id,
                            success=False,
                            original=task.article,
                            error=f"{type(e).__name__}: {str(e)}",
                        )
                    
                    # Send the result back
                    try:
                        result_queue.put(pickle.dumps(result))
                        logger.debug(f"Worker {worker_id} sent result for task {task.task_id}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed to send result: {e}")
                    
                    # Signal that we're ready for another task
                    try:
                        if not shutdown_event.is_set():
                            ready_queue.put(worker_id)
                            logger.debug(f"Worker {worker_id} ready for next task")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed to signal ready: {e}")
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} unexpected error: {str(e)}\n{traceback.format_exc()}")
                    # Add worker back to ready queue after an error
                    try:
                        if not shutdown_event.is_set():
                            ready_queue.put(worker_id)
                    except:
                        pass
                    
            logger.info(f"Worker {worker_id} shutting down")
        
        # Apply the patches
        logger.info("Applying patches to EnhancedBatchProcessor...")
        EnhancedBatchProcessor.__init__ = patched_init
        EnhancedBatchProcessor.start_workers = patched_start_workers
        EnhancedBatchProcessor.process_batch_sync = fixed_process_batch_sync
        WorkerProcess._worker_main = fixed_worker_main
        
        logger.info("Successfully patched EnhancedBatchProcessor")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch EnhancedBatchProcessor: {e}")
        traceback.print_exc()
        return False

def patch_fast_summarizer():
    """
    Apply fixes to the FastArticleSummarizer class to ensure it uses the
    fixed batch processor.
    """
    try:
        # Import the necessary modules
        from fast_summarizer import FastArticleSummarizer, create_fast_summarizer
        import types
        
        logger.info("Patching FastArticleSummarizer...")
        
        # Store the original batch_summarize method
        original_batch_summarize = None
        if hasattr(FastArticleSummarizer, 'batch_summarize'):
            original_batch_summarize = FastArticleSummarizer.batch_summarize
        
        # Create a new batch_summarize method
        async def fixed_batch_summarize(
            self,
            articles,
            max_concurrent=3,
            model=None,
            auto_select_model=True,
            temperature=0.3,
        ):
            """
            Fixed batch processing method to resolve timeout issues.
            """
            self.logger.info(f"Using fixed batch_summarize method")
            
            # Apply the enhanced batch processor fix first
            if not hasattr(self, '_fixed_batch_processor_applied'):
                patch_enhanced_batch_processor()
                self._fixed_batch_processor_applied = True
            
            # Ensure we have at least one worker
            max_concurrent = max(1, max_concurrent)
            
            # Prepare articles for processing
            articles_to_process = []
            for article in articles:
                # Skip articles that already have summaries
                if article.get('summary'):
                    continue
                    
                # Clean text for compatibility
                text = article.get('content', article.get('text', ''))
                if hasattr(self.original, 'clean_text'):
                    text = self.original.clean_text(text)
                
                # Create a copy of the article with both 'text' and 'content' keys
                prepared_article = {
                    'text': text,
                    'content': text,
                    'title': article.get('title', 'No Title'),
                    'url': article.get('link', article.get('url', '#')),
                    'link': article.get('link', article.get('url', '#'))
                }
                
                articles_to_process.append(prepared_article)
            
            # Skip if no articles need summarization
            if not articles_to_process:
                self.logger.info("No articles need summarization in this batch")
                return articles
                
            self.logger.info(f"Processing {len(articles_to_process)} articles with fixed batch method")
            
            try:
                from enhanced_batch_processor import EnhancedBatchProcessor
                
                # Create a new batch processor with the fixes applied
                batch_processor = EnhancedBatchProcessor(max_workers=max_concurrent)
                
                with batch_processor.batch_context():
                    results = await batch_processor.process_batch_async(
                        articles=articles_to_process,
                        model=model,
                        temperature=temperature
                    )
                
                # Match results back to original articles
                url_map = {}
                for r in results:
                    if 'original' in r:
                        url = r['original'].get('url', r['original'].get('link', '#'))
                        url_map[url] = r.get('summary', {})
                
                # Update the original articles with summaries
                for article in articles:
                    url = article.get('link', article.get('url', '#'))
                    if url in url_map and not article.get('summary'):
                        article['summary'] = url_map[url]
                
                return articles
                
            except Exception as e:
                self.logger.error(f"Error in fixed batch_summarize: {e}")
                traceback.print_exc()
                
                # Fallback to sequential processing if batch fails
                self.logger.info("Falling back to sequential processing")
                for article in articles_to_process:
                    try:
                        summary = self.summarize(
                            text=article['text'],
                            title=article['title'],
                            url=article['url'],
                            auto_select_model=auto_select_model
                        )
                        # Find the matching original article and update it
                        for orig_article in articles:
                            if orig_article.get('link') == article['url'] or orig_article.get('url') == article['url']:
                                orig_article['summary'] = summary
                                break
                    except Exception as e:
                        self.logger.error(f"Error summarizing article: {e}")
                
                return articles
        
        # Apply the patch to all existing instances
        original_init = FastArticleSummarizer.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call the original __init__
            original_init(self, *args, **kwargs)
            
            # Apply our fixed batch_summarize method
            self.batch_summarize = types.MethodType(fixed_batch_summarize, self)
            
            # Apply the batch processor fix
            if not hasattr(self, '_fixed_batch_processor_applied'):
                patch_enhanced_batch_processor()
                self._fixed_batch_processor_applied = True
                
            self.logger.info("FastArticleSummarizer initialized with fixed batch processing")
        
        # Replace the __init__ method
        FastArticleSummarizer.__init__ = patched_init
        
        logger.info("Successfully patched FastArticleSummarizer")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch FastArticleSummarizer: {e}")
        traceback.print_exc()
        return False

def apply_all_fixes():
    """Apply all fixes to the RSS reader application."""
    logger.info("Applying all fixes to the RSS reader application")
    
    # First patch the EnhancedBatchProcessor
    if not patch_enhanced_batch_processor():
        logger.error("Failed to apply fixes")
        print("\n❌ Failed to apply batch processor fix")
        return 1

if __name__ == "__main__":
    sys.exit(main())
 patch EnhancedBatchProcessor")
        return False
    
    # Then patch the FastArticleSummarizer
    if not patch_fast_summarizer():
        logger.warning("Failed to patch FastArticleSummarizer, but continuing with basic fix")
    
    logger.info("All fixes applied successfully")
    return True

def create_runner():
    """Create a runner script that applies the fixes when the application starts."""
    runner_path = Path('apply_batch_fix.py')
    
    runner_content = '''#!/usr/bin/env python3
"""
Runner script to apply the batch processor fix.

Include this at the start of your main.py:
```
# Apply batch processor fixes
import apply_batch_fix
apply_batch_fix.apply()
```
"""

import sys
import logging
import importlib.util
import traceback

def apply():
    """Apply the batch processor fix."""
    try:
        # Try to import the fix module
        spec = importlib.util.spec_from_file_location("fix_rss_batch_processor", "fix_rss_batch_processor.py")
        if not spec or not spec.loader:
            print("Error: Could not find fix_rss_batch_processor.py")
            return False
            
        fix_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fix_module)
        
        # Apply the fixes
        if hasattr(fix_module, 'apply_all_fixes') and callable(fix_module.apply_all_fixes):
            result = fix_module.apply_all_fixes()
            if result:
                print("Successfully applied batch processor fix")
            else:
                print("Failed to apply batch processor fix")
            return result
        else:
            print("Error: apply_all_fixes function not found in fix module")
            return False
    except Exception as e:
        print(f"Error applying batch processor fix: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply()
'''
    
    try:
        with open(runner_path, 'w') as f:
            f.write(runner_content)
        logger.info(f"Created runner script at {runner_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create runner script: {e}")
        return False

def main():
    """Main function to apply all fixes."""
    logger.info("Starting RSS batch processor fix application")
    
    # Apply the fixes
    if apply_all_fixes():
        logger.info("Successfully applied all fixes")
        
        # Create the runner script
        if create_runner():
            logger.info("Created runner script")
        
        print("\n✅ Batch processor fix successfully applied!")
        print("To use this fix in your application, add the following at the start of your main.py:")
        print("```")
        print("# Apply batch processor fixes")
        print("import apply_batch_fix")
        print("apply_batch_fix.apply()")
        print("```\n")
        return 0
    else:
        logger.error("Failed to