"""
Fixes for the EnhancedBatchProcessor to resolve timeout and deadlock issues.

This module provides patched versions of methods from the EnhancedBatchProcessor
that fix synchronization issues between the main process and worker processes.
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
import importlib
from contextlib import contextmanager
from datetime import datetime

# Import the original classes to extend them
from enhanced_batch_processor import (
    EnhancedBatchProcessor, WorkerProcess, 
    ProcessingTask, ProcessingResult, QueueHandler
)

def patch_enhanced_batch_processor():
    """
    Apply patches to the EnhancedBatchProcessor class to fix the worker communication issues.
    
    Returns:
        The patched EnhancedBatchProcessor class
    """
    # Save original method references
    original_process_batch_sync = EnhancedBatchProcessor.process_batch_sync
    original_worker_main = WorkerProcess._worker_main
    
    # Apply patches
    EnhancedBatchProcessor.process_batch_sync = process_batch_sync_fixed
    WorkerProcess._worker_main = worker_main_fixed
    
    # Log the patching
    logging.info("Applied fixes to EnhancedBatchProcessor and WorkerProcess")
    
    return EnhancedBatchProcessor

def worker_main_fixed(worker_id, ready_queue, task_queue, result_queue, log_queue, shutdown_event, 
                    summarizer_module='summarizer', summarizer_class='ArticleSummarizer'):
    """
    Fixed version of the worker main function to improve ready state signaling.
    
    This patched version improves how workers signal their readiness and handles
    task dictionary structure variations to prevent KeyErrors.
    """
    # Configure logging
    logger = WorkerProcess._configure_logging(worker_id, log_queue)
    logger.info(f"Worker {worker_id} starting")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
    # Initialize the summarizer
    try:
        # Use importlib for dynamic imports
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
    
    # Signal that we're ready to accept tasks
    ready_queue.put(worker_id)
    logger.info(f"Worker {worker_id} ready for tasks")
    
    # For diagnostics, track task processing counts
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
                # FIX: Only put back in ready queue if shutdown event is not set
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
                
                # Fix: Safely access URL with fallback
                url = task.article.get('url', task.article.get('link', '#'))
                logger.info(f"Worker {worker_id} processing article {task_count}: {url}")
                
                # Fix: Handle key differences by extracting required fields with fallbacks
                text = task.article.get('text', task.article.get('content', ''))
                title = task.article.get('title', 'No Title')
                
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
                
                logger.info(f"Worker {worker_id} completed article {url} in {elapsed:.2f}s")
            
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
                logger.debug(f"Worker {worker_id} sent result")
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to put result in queue: {str(e)}")
            
            # Signal that we're ready for another task
            try:
                if not shutdown_event.is_set():
                    ready_queue.put(worker_id)
                    logger.debug(f"Worker {worker_id} signaling ready for next task")
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to signal ready: {str(e)}")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} unexpected error: {str(e)}\n{traceback.format_exc()}")
            # FIX: Add worker back to ready queue after an error
            try:
                if not shutdown_event.is_set():
                    ready_queue.put(worker_id)
                    logger.debug(f"Worker {worker_id} signaling ready after error")
            except Exception as re:
                logger.error(f"Worker {worker_id} failed to signal ready after error: {str(re)}")
            
    logger.info(f"Worker {worker_id} shutting down")


def process_batch_sync_fixed(self, articles, model=None, temperature=0.3, timeout=None):
    """
    Fixed version of process_batch_sync to prevent deadlocks and improve reliability.
    
    This patched version enhances queue management, adds better retry logic, and
    improves logging and diagnostics for the batch processing workflow.
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
        # FIX: Drain the ready queue to ensure we have all available workers
        worker_ids = []
        drain_start = time.time()
        drain_timeout = 5.0  # Wait up to 5 seconds to collect ready workers
        
        while time.time() - drain_start < drain_timeout:
            try:
                worker_id = self.ready_queue.get(timeout=0.5)
                worker_ids.append(worker_id)
                self.logger.debug(f"Found ready worker: {worker_id}")
                
                # If we've gotten all workers, we can stop draining
                if len(worker_ids) >= self.max_workers:
                    break
            except queue.Empty:
                # If queue is empty, we've gotten all currently ready workers
                break
                
        self.logger.info(f"Found {len(worker_ids)} ready workers out of {self.max_workers}")
        
        # If we have no ready workers, try to wait a bit longer for at least one
        if not worker_ids:
            self.logger.warning("No ready workers found initially, waiting for at least one...")
            try:
                worker_id = self.ready_queue.get(timeout=10.0)
                worker_ids.append(worker_id)
                self.logger.info(f"Found worker {worker_id} ready after extended wait")
            except queue.Empty:
                self.logger.error("No workers ready after extended wait - possible deadlock")
        
        # Submit all tasks to ready workers
        submitted_tasks = set()
        
        for task_id, task in tasks.items():
            # Get a worker ID from our saved list or wait for one
            if worker_ids:
                worker_id = worker_ids.pop(0)
                self.logger.debug(f"Using ready worker {worker_id} from pool")
            else:
                # Wait for a worker to be ready
                try:
                    worker_id = self.ready_queue.get(timeout=30.0)
                    self.logger.info(f"Got worker {worker_id} after waiting")
                except queue.Empty:
                    self.logger.warning("Timed out waiting for a ready worker")
                    break
                
            # Submit the task to the task queue
            try:
                self.task_queue.put(pickle.dumps(task), timeout=2.0)
                submitted_tasks.add(task_id)
                self.logger.debug(f"Submitted task {task_id} to worker {worker_id}")
            except queue.Full:
                # If queue is full, put the worker ID back and skip this task
                worker_ids.append(worker_id)
                self.logger.warning(f"Task queue full, could not submit task {task_id}")
                continue
        
        if not submitted_tasks:
            self.logger.error("No tasks could be submitted - possible system issue")
            # Return empty results for all articles
            return [{
                'original': article,
                'error': "Failed to submit task for processing (system issue)",
                'summary': {
                    'headline': article.get('title', 'Error'),
                    'summary': "Summary generation failed: System issue. Please try again later."
                }
            } for article in articles]
        
        # Process results
        results = {}
        remaining_timeout = timeout or 180.0  # Default 3 minutes if no timeout specified
        start_wait_time = time.time()
        
        # FIX: Use submitted_tasks to know how many tasks were actually submitted
        self.logger.info(f"Waiting for results from {len(submitted_tasks)} submitted tasks")
        
        while len(results) < len(submitted_tasks):
            # Calculate remaining timeout
            elapsed = time.time() - start_time
            remaining_timeout = max(0.1, (timeout or 180.0) - elapsed)
            
            if remaining_timeout <= 0.1:
                self.logger.warning(f"Batch processing timed out after {elapsed:.2f}s")
                break
            
            # Wait for a result
            try:
                result_data = self.result_queue.get(timeout=min(remaining_timeout, 5.0))
                result = pickle.loads(result_data)
                
                # Store the result
                task_id = result.task_id
                results[task_id] = result
                
                # Log progress
                status = 'SUCCESS' if getattr(result, 'success', False) else 'FAILURE'
                self.logger.info(f"Received result {len(results)}/{len(submitted_tasks)} - {status}")
                        
            except queue.Empty:
                wait_time = time.time() - start_wait_time
                self.logger.debug(f"Still waiting for results... {len(results)}/{len(submitted_tasks)} received after {wait_time:.2f}s")
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
                if getattr(result, 'success', False):
                    formatted_results.append({
                        'original': result.original,
                        'summary': result.summary,
                        'elapsed': getattr(result, 'elapsed', None)
                    })
                else:
                    formatted_results.append({
                        'original': result.original,
                        'error': getattr(result, 'error', "Unknown error"),
                        'summary': {
                            'headline': result.original.get('title', 'Error'),
                            'summary': f"Summary generation failed: {getattr(result, 'error', 'Unknown error')}. Please try again later."
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


# Function to create a fixed EnhancedBatchProcessor
def create_fixed_batch_processor(max_workers=3):
    """
    Create an EnhancedBatchProcessor with the fixes applied.
    
    Args:
        max_workers: Maximum number of worker processes
        
    Returns:
        EnhancedBatchProcessor: Instance with the fixed methods
    """
    # Apply patches first
    patched_class = patch_enhanced_batch_processor()
    
    # Create an instance
    processor = patched_class(max_workers=max_workers)
    
    return processor


# When the module is imported, apply the patches automatically
patch_enhanced_batch_processor()

# Module usage example
if __name__ == "__main__":
    print("This module applies fixes to the EnhancedBatchProcessor class.")
    print("Import it in your code and use patch_enhanced_batch_processor()")
    print("or create_fixed_batch_processor() to get a fixed instance.")
