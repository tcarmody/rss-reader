"""
Enhanced parallel batch processing for ArticleSummarizer.
Addresses issues with tokenizer parallelism and process management.
"""

import os
# Set this at the very beginning to prevent tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import concurrent.futures
import logging
import multiprocessing
import pickle
import queue
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple, Any, Union

# Custom exception for process initialization errors
class ProcessInitError(Exception):
    """Raised when a worker process fails to initialize."""
    pass

@dataclass
class ProcessingTask:
    """Data container for an article processing task."""
    article: Dict[str, str]
    model: str
    temperature: float
    task_id: str
    request_time: datetime = None
    
    def __post_init__(self):
        if self.request_time is None:
            self.request_time = datetime.now()


@dataclass
class ProcessingResult:
    """Data container for an article processing result."""
    task_id: str
    success: bool
    original: Dict[str, str]
    summary: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    elapsed: Optional[float] = None


class WorkerProcess:
    """
    A worker process that handles article summarization.
    This isolates the tokenizer and model in a separate process.
    """
    
    def __init__(self, worker_id: int, ready_queue: multiprocessing.Queue, 
                 result_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue,
                 shutdown_event: multiprocessing.Event):
        self.worker_id = worker_id
        self.ready_queue = ready_queue
        self.result_queue = result_queue
        self.log_queue = log_queue
        self.shutdown_event = shutdown_event
        self.process = None
        self.summarizer = None
        
    def start(self):
        """Start the worker process."""
        self.process = multiprocessing.Process(
            target=self._worker_main,
            args=(self.worker_id, self.ready_queue, self.result_queue, 
                  self.log_queue, self.shutdown_event)
        )
        self.process.daemon = True
        self.process.start()
        
    def terminate(self):
        """Terminate the worker process."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
            
    @staticmethod
    def _configure_logging(worker_id: int, log_queue: multiprocessing.Queue):
        """Configure logging for the worker process."""
        logger = logging.getLogger(f"WorkerProcess-{worker_id}")
        handler = QueueHandler(log_queue)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
    
    @staticmethod
    def _worker_main(worker_id: int, ready_queue: multiprocessing.Queue, 
                    result_queue: multiprocessing.Queue, log_queue: multiprocessing.Queue,
                    shutdown_event: multiprocessing.Event):
        """Main function for the worker process."""
        # Configure logging
        logger = WorkerProcess._configure_logging(worker_id, log_queue)
        logger.info(f"Worker {worker_id} starting")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        
        # Initialize the summarizer
        try:
            from summarizer import ArticleSummarizer
            summarizer = ArticleSummarizer()
            logger.info(f"Worker {worker_id} initialized ArticleSummarizer")
        except Exception as e:
            error_msg = f"Worker {worker_id} failed to initialize ArticleSummarizer: {str(e)}"
            logger.error(error_msg)
            result_queue.put(pickle.dumps({
                'type': 'init_error',
                'worker_id': worker_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            }))
            return
        
        # Signal that we're ready to accept tasks
        ready_queue.put(worker_id)
        logger.info(f"Worker {worker_id} ready for tasks")
        
        # Process tasks until shutdown
        while not shutdown_event.is_set():
            try:
                # Get a task from the queue with timeout
                try:
                    task_data = ready_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Check if this is a shutdown signal
                if task_data == 'SHUTDOWN':
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break
                
                # Process the task
                try:
                    # Deserialize the task
                    task = pickle.loads(task_data)
                    logger.info(f"Worker {worker_id} processing article: {task.article['url']}")
                    
                    # Process the article
                    start_time = time.time()
                    summary = summarizer.summarize_article(
                        text=task.article['text'],
                        title=task.article['title'],
                        url=task.article['url'],
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
                    
                    logger.info(f"Worker {worker_id} completed article: {task.article['url']} in {elapsed:.2f}s")
                
                except Exception as e:
                    # Create error result
                    tb = traceback.format_exc()
                    logger.error(f"Worker {worker_id} error processing {task.article['url']}: {str(e)}\n{tb}")
                    
                    result = ProcessingResult(
                        task_id=task.task_id,
                        success=False,
                        original=task.article,
                        error=f"{type(e).__name__}: {str(e)}",
                    )
                
                # Send the result back
                result_queue.put(pickle.dumps(result))
                
                # Signal that we're ready for another task
                ready_queue.put(worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {str(e)}\n{traceback.format_exc()}")
                # Don't exit, try to recover
                
        logger.info(f"Worker {worker_id} shutting down")


class QueueHandler(logging.Handler):
    """
    A logging handler that puts logs into a multiprocessing queue.
    """
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def emit(self, record):
        try:
            self.queue.put_nowait(pickle.dumps(record))
        except Exception:
            self.handleError(record)


class QueueListener:
    """
    A listener that processes log records from a queue.
    """
    def __init__(self, queue, *handlers):
        self.queue = queue
        self.handlers = handlers
        self._stop_event = threading.Event()
        self._thread = None
        
    def start(self):
        """Start the listener."""
        self._thread = threading.Thread(target=self._monitor)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self):
        """Stop the listener."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
    def _monitor(self):
        """Monitor the queue for log records."""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=0.2)
                if record is None:
                    break
                    
                record = pickle.loads(record)
                for handler in self.handlers:
                    if record.levelno >= handler.level:
                        handler.handle(record)
            except (queue.Empty, EOFError):
                continue
            except Exception:
                import sys
                sys.stderr.write('Error in log queue listener:\n')
                traceback.print_exc(file=sys.stderr)


class EnhancedBatchProcessor:
    """
    Enhanced batch processor for article summarization.
    Uses isolated worker processes to avoid tokenizer parallelism issues.
    """
    
    def __init__(self, max_workers=3, log_level=logging.INFO):
        self.max_workers = max_workers
        self.logger = self._setup_logger(log_level)
        self.ready_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.log_queue = multiprocessing.Queue()
        self.shutdown_event = multiprocessing.Event()
        self.workers = []
        self.log_listener = None
        self._initialize_log_listener()
        
    def _setup_logger(self, log_level):
        """Set up the logger for the batch processor."""
        logger = logging.getLogger("EnhancedBatchProcessor")
        logger.setLevel(log_level)
        
        # Check if logger already has handlers
        if not logger.handlers:
            # Add a console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Add a file handler
            try:
                file_handler = logging.FileHandler("batch_processor.log")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not set up file logging: {e}")
                
        return logger
    
    def _initialize_log_listener(self):
        """Initialize the log listener for worker process logs."""
        # Create handlers that match the main logger's handlers
        handlers = []
        for handler in self.logger.handlers:
            try:
                # For StreamHandler (usually to console)
                if isinstance(handler, logging.StreamHandler):
                    handler_copy = handler.__class__(handler.stream)
                # For FileHandler
                elif isinstance(handler, logging.FileHandler):
                    handler_copy = handler.__class__(handler.baseFilename, mode=handler.mode)
                # For other handler types - try a generic copy or skip
                else:
                    try:
                        handler_copy = handler.__class__()
                    except:
                        self.logger.warning(f"Couldn't copy handler of type {handler.__class__.__name__}, skipping")
                        continue
                        
                handler_copy.setFormatter(handler.formatter)
                handler_copy.setLevel(handler.level)
                handlers.append(handler_copy)
            except Exception as e:
                self.logger.warning(f"Error copying log handler: {e}")
        
        # Create and start the log listener
        self.log_listener = QueueListener(self.log_queue, *handlers)
        self.log_listener.start()
        self.logger.info("Log listener started")
        
    def start_workers(self):
        """Start the worker processes."""
        self.logger.info(f"Starting {self.max_workers} worker processes")
        
        for i in range(self.max_workers):
            worker = WorkerProcess(
                worker_id=i,
                ready_queue=self.ready_queue,
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
        
        while len(ready_workers) < self.max_workers:
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
                worker_id = self.ready_queue.get(timeout=0.1)
                ready_workers.add(worker_id)
                self.logger.info(f"Worker {worker_id} ready ({len(ready_workers)}/{self.max_workers})")
            except queue.Empty:
                pass
                
            # Check for timeout
            if time.time() - start_time > initialization_timeout:
                self.logger.warning(f"Initialization timed out. Only {len(ready_workers)}/{self.max_workers} workers ready.")
                # Update max_workers to match actual ready workers
                self.max_workers = len(ready_workers)
                break
                
        if self.max_workers == 0:
            self.logger.error("No workers initialized successfully.")
            raise ProcessInitError("All worker processes failed to initialize")
            
        self.logger.info(f"{self.max_workers} workers initialized and ready")
        return self.max_workers
        
    def shutdown(self):
        """Shutdown all worker processes."""
        self.logger.info("Shutting down worker processes")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Send explicit shutdown signals
        for _ in range(len(self.workers)):
            try:
                self.ready_queue.put('SHUTDOWN', timeout=0.5)
            except queue.Full:
                pass
                
        # Terminate workers
        for worker in self.workers:
            worker.terminate()
            
        # Clear the queues
        self._drain_queue(self.ready_queue)
        self._drain_queue(self.result_queue)
        self._drain_queue(self.log_queue)
        
        # Stop the log listener
        if self.log_listener:
            self.log_listener.stop()
            
        self.logger.info("All workers shut down")
        
    def _drain_queue(self, q):
        """Drain a queue to prevent hanging on join."""
        try:
            while True:
                q.get_nowait()
        except (queue.Empty, EOFError):
            pass
            
    @contextmanager
    def batch_context(self):
        """Context manager for batch processing."""
        try:
            self.start_workers()
            yield self
        finally:
            self.shutdown()
            
    def process_batch_sync(self, articles, model=None, temperature=0.3, timeout=None):
        """
        Process a batch of articles synchronously.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            model: Optional model to use for all articles
            temperature: Temperature setting
            timeout: Optional timeout in seconds for the entire batch
            
        Returns:
            List of processing results in the same order as input articles
        """
        if not articles:
            return []
            
        start_time = time.time()
        self.logger.info(f"Processing batch of {len(articles)} articles")
        
        # Create tasks for all articles
        tasks = {}
        for i, article in enumerate(articles):
            task_id = f"task_{start_time}_{i}"
            task = ProcessingTask(
                article=article,
                model=model,
                temperature=temperature,
                task_id=task_id
            )
            tasks[task_id] = task
            
        # Start the workers if not already started
        workers_started = False
        if not self.workers:
            self.start_workers()
            workers_started = True
            
        try:
            # Submit all tasks to ready workers
            submitted_tasks = set()
            for task_id, task in tasks.items():
                # Wait for a worker to be ready
                try:
                    worker_id = self.ready_queue.get(timeout=30.0)
                except queue.Empty:
                    self.logger.warning("Timed out waiting for a ready worker")
                    break
                    
                # Submit the task
                self.ready_queue.put(pickle.dumps(task))
                submitted_tasks.add(task_id)
                self.logger.debug(f"Submitted task {task_id} for {task.article['url']} to worker {worker_id}")
                
            # Process results
            results = {}
            remaining_timeout = timeout
            start_wait_time = time.time()
            
            while len(results) < len(submitted_tasks):
                # Calculate remaining timeout if specified
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0.1, timeout - elapsed)
                    if remaining_timeout <= 0.1:
                        self.logger.warning(f"Batch processing timed out after {elapsed:.2f}s")
                        break
                
                # Wait for a result
                try:
                    result_data = self.result_queue.get(timeout=remaining_timeout or 0.5)
                    result = pickle.loads(result_data)
                    
                    # Store the result
                    task_id = result.task_id
                    results[task_id] = result
                    
                    # Log progress
                    self.logger.info(f"Received result {len(results)}/{len(submitted_tasks)} - " +
                                    f"{'SUCCESS' if result.success else 'FAILURE'} for {result.original['url']}")
                                    
                except queue.Empty:
                    wait_time = time.time() - start_wait_time
                    self.logger.debug(f"Waiting for results... {len(results)}/{len(submitted_tasks)} received after {wait_time:.2f}s")
                    # Continue waiting
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
                                'headline': result.original['title'],
                                'summary': f"Summary generation failed: {result.error}. Please try again later."
                            }
                        })
                else:
                    # Task was not processed
                    formatted_results.append({
                        'original': article,
                        'error': "Task not processed (timeout or worker failure)",
                        'summary': {
                            'headline': article['title'],
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
                
    async def process_batch_async(self, articles, model=None, temperature=0.3, timeout=None):
        """
        Process a batch of articles asynchronously.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            model: Optional model to use for all articles
            temperature: Temperature setting
            timeout: Optional timeout in seconds for the entire batch
            
        Returns:
            List of processing results in the same order as input articles
        """
        # Run the synchronous processing in a thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                partial(self.process_batch_sync, 
                       articles=articles, 
                       model=model, 
                       temperature=temperature,
                       timeout=timeout)
            )
        return result


# Helper method to integrate with existing code
def create_enhanced_batch_processor(max_workers=3):
    """Create and return an EnhancedBatchProcessor instance."""
    return EnhancedBatchProcessor(max_workers=max_workers)


# Integrate with FastArticleSummarizer
def add_enhanced_batch_to_fast_summarizer(fast_summarizer, max_workers=3):
    """
    Add enhanced batch processing to an existing FastArticleSummarizer instance.
    
    Args:
        fast_summarizer: An instance of FastArticleSummarizer
        max_workers: Maximum number of worker processes
        
    Returns:
        The modified FastArticleSummarizer instance
    """
    # Create the batch processor
    batch_processor = create_enhanced_batch_processor(max_workers=max_workers)
    
    # Add batch_summarize method to the FastArticleSummarizer
    async def batch_summarize(
        self,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        model: Optional[str] = None,
        auto_select_model: bool = True,
        temperature: float = 0.3,
    ) -> List[Dict[str, Dict[str, str]]]:
        """
        Process multiple articles in parallel batches using isolated worker processes.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent processes
            model: Claude model to use (shorthand name or full identifier)
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting for generation
            
        Returns:
            List of article summaries with original metadata
        """
        # Auto-select models if needed
        if auto_select_model and not model:
            prepared_articles = []
            for article in articles:
                # Clean text for complexity estimation
                text = self.original.clean_text(article['text'])
                
                # Import here to avoid circular imports
                try:
                    # Dynamic import to prevent circular imports
                    import importlib
                    model_selection_module = importlib.import_module('model_selection')
                    select_model = model_selection_module.auto_select_model
                    
                    selected_model = select_model(
                        text,
                        self.original.AVAILABLE_MODELS,
                        self.original.DEFAULT_MODEL,
                        self.logger
                    )
                except ImportError:
                    # Fallback if import fails
                    self.logger.warning("Could not import model_selection, using default model")
                    selected_model = self.original.DEFAULT_MODEL
                
                # Create a copy of the article with cleaned text
                prepared_article = {
                    'text': text,
                    'title': article['title'],
                    'url': article['url']
                }
                
                prepared_articles.append((prepared_article, selected_model))
                
            # Group articles by model to process together
            model_groups = {}
            for article, selected_model in prepared_articles:
                if selected_model not in model_groups:
                    model_groups[selected_model] = []
                model_groups[selected_model].append(article)
                
            # Process each model group
            results = []
            for selected_model, group_articles in model_groups.items():
                self.logger.info(f"Processing {len(group_articles)} articles with model {selected_model}")
                with batch_processor.batch_context():
                    group_results = await batch_processor.process_batch_async(
                        articles=group_articles,
                        model=selected_model,
                        temperature=temperature
                    )
                results.extend(group_results)
                
            # Reorder results to match original input
            url_to_result = {r['original']['url']: r for r in results}
            ordered_results = [url_to_result[a['url']] for a in articles if a['url'] in url_to_result]
            
            return ordered_results
        else:
            # Process all articles with the same model
            with batch_processor.batch_context():
                results = await batch_processor.process_batch_async(
                    articles=articles,
                    model=model,
                    temperature=temperature
                )
            return results
    
    # Set the method on the FastArticleSummarizer instance
    import types
    fast_summarizer.batch_summarize = types.MethodType(batch_summarize, fast_summarizer)
    
    return fast_summarizer


# Usage example
if __name__ == "__main__":
    import threading
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test data
    test_articles = [
        {
            'text': 'This is a test article about artificial intelligence.',
            'title': 'AI Test Article',
            'url': 'https://example.com/ai-article'
        },
        {
            'text': 'Another test article about machine learning and big data.',
            'title': 'ML Test Article',
            'url': 'https://example.com/ml-article'
        }
    ]
    
    # Test synchronous processing
    print("Testing synchronous batch processing")
    processor = create_enhanced_batch_processor(max_workers=2)
    
    with processor.batch_context():
        results = processor.process_batch_sync(test_articles)
        
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Headline: {result['summary']['headline']}")
            print(f"  Summary length: {len(result['summary']['summary'])}")
            
    # Test async processing
    print("\nTesting asynchronous batch processing")
    
    async def test_async():
        processor = create_enhanced_batch_processor(max_workers=2)
        with processor.batch_context():
            results = await processor.process_batch_async(test_articles)
            
        for i, result in enumerate(results):
            print(f"Async Result {i+1}:")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Headline: {result['summary']['headline']}")
                print(f"  Summary length: {len(result['summary']['summary'])}")
    
    # Run the async test in a new event loop
    async_loop = asyncio.new_event_loop()
    async_loop.run_until_complete(test_async())
    async_loop.close()