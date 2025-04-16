"""
Parallel batch processing for ArticleSummarizer using the spawn method
to avoid tokenizer parallelism issues.
"""

import os
# Set this at the very top before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing
import logging
import time
import pickle
import traceback
import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Any, Union

# This will be used for compatibility with the FastArticleSummarizer
class SpawnBatchProcessor:
    """
    Handles batch processing of articles using multiprocessing with spawn method.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SpawnBatchProcessor")
    
    @staticmethod
    def process_article_worker(worker_data):
        """
        Worker function that runs in a separate process.
        This isolates each process to avoid tokenizer conflicts.
        
        Args:
            worker_data: Dictionary with all necessary data
            
        Returns:
            Processing result
        """
        article = worker_data['article']
        model = worker_data['model']
        temperature = worker_data['temperature']
        
        try:
            # Import inside worker to ensure clean environment
            from summarizer import ArticleSummarizer
            
            # Create a fresh summarizer instance
            summarizer = ArticleSummarizer()
            
            # Process the article
            start_time = time.time()
            summary = summarizer.summarize_article(
                text=article['text'],
                title=article['title'],
                url=article['url'],
                model=model,
                temperature=temperature
            )
            elapsed = time.time() - start_time
            
            return {
                'success': True,
                'original': article,
                'summary': summary,
                'elapsed': elapsed
            }
        except Exception as e:
            tb = traceback.format_exc()
            return {
                'success': False,
                'original': article,
                'error': str(e),
                'traceback': tb
            }
    
    def batch_process_sync(
        self,
        summarizer,
        articles: List[Dict[str, str]],
        model: Optional[str] = None,
        max_workers: int = 3,
        auto_select_model: bool = False,
        temperature: float = 0.3
    ) -> List[Dict]:
        """
        Process multiple articles in parallel using multiprocessing with spawn method.
        This is a synchronous version.
        
        Args:
            summarizer: Original summarizer instance (for model selection)
            articles: List of articles to process
            model: Model to use
            max_workers: Maximum number of parallel workers
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Processing batch of {len(articles)} articles with {max_workers} workers")
        start_time = time.time()
        
        # Prepare worker data with models selected if needed
        worker_data_list = []
        for article in articles:
            if auto_select_model and not model:
                # Import here to avoid circular imports
                from model_selection import auto_select_model as select_model
                selected_model = select_model(
                    article['text'],
                    summarizer.AVAILABLE_MODELS,
                    summarizer.DEFAULT_MODEL,
                    self.logger
                )
            else:
                selected_model = summarizer._get_model(model)
            
            worker_data = {
                'article': article,
                'model': selected_model,
                'temperature': temperature
            }
            worker_data_list.append(worker_data)
        
        # Initialize multiprocessing with spawn context
        ctx = multiprocessing.get_context('spawn')
        
        # Process articles in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx
        ) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self.process_article_worker, data): i
                for i, data in enumerate(worker_data_list)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    if result['success']:
                        self.logger.info(f"Processed article {idx+1}/{len(articles)}: {articles[idx]['url']}")
                    else:
                        self.logger.error(
                            f"Error processing article {idx+1}/{len(articles)}: "
                            f"{articles[idx]['url']} - {result['error']}"
                        )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Worker exception for article {idx+1}: {str(e)}")
                    results.append({
                        'success': False,
                        'original': articles[idx],
                        'error': str(e)
                    })
        
        # Sort results to match input order
        sorted_results = []
        for i in range(len(articles)):
            for result in results:
                if result['original'] == articles[i]:
                    sorted_results.append(result)
                    break
        
        elapsed = time.time() - start_time
        self.logger.info(f"Batch processing completed in {elapsed:.2f}s")
        
        return sorted_results
    
    async def batch_process_async(
        self,
        summarizer,
        articles: List[Dict[str, str]],
        model: Optional[str] = None,
        max_concurrent: int = 3,
        auto_select_model: bool = False,
        temperature: float = 0.3
    ) -> List[Dict]:
        """
        Async wrapper for batch processing.
        Uses a thread to run the synchronous batch processing.
        
        Args:
            summarizer: Original summarizer instance
            articles: List of articles to process
            model: Model to use
            max_concurrent: Maximum number of concurrent processes
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting
            
        Returns:
            List of processing results
        """
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: self.batch_process_sync(
                    summarizer=summarizer,
                    articles=articles,
                    model=model,
                    max_workers=max_concurrent,
                    auto_select_model=auto_select_model,
                    temperature=temperature
                )
            )
        
        return result


# Modified FastArticleSummarizer that uses the SpawnBatchProcessor
def add_parallel_batch_to_fast_summarizer(fast_summarizer):
    """
    Adds parallel batch processing capability to an existing FastArticleSummarizer.
    
    Args:
        fast_summarizer: Existing FastArticleSummarizer instance
        
    Returns:
        The modified FastArticleSummarizer instance
    """
    # Create batch processor
    batch_processor = SpawnBatchProcessor()
    
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
        Process multiple articles in parallel batches using spawn method.
        
        Args:
            articles: List of article dicts with 'text', 'title', and 'url' keys
            max_concurrent: Maximum number of concurrent processes
            model: Claude model to use (shorthand name or full identifier)
            auto_select_model: Whether to auto-select model based on content
            temperature: Temperature setting for generation
            
        Returns:
            List of article summaries with original metadata
        """
        results = await batch_processor.batch_process_async(
            summarizer=self.original,
            articles=articles,
            model=model,
            max_concurrent=max_concurrent,
            auto_select_model=auto_select_model,
            temperature=temperature
        )
        
        # Format results to match expected output
        formatted_results = []
        for result in results:
            if result['success']:
                formatted_results.append({
                    'original': result['original'],
                    'summary': result['summary']
                })
            else:
                formatted_results.append({
                    'original': result['original'],
                    'error': result.get('error', 'Unknown error'),
                    'summary': {
                        'headline': result['original']['title'],
                        'summary': f"Summary generation failed: {result.get('error', 'Unknown error')}. Please try again later."
                    }
                })
        
        return formatted_results
    
    # Set the method on the FastArticleSummarizer instance
    import types
    fast_summarizer.batch_summarize = types.MethodType(batch_summarize, fast_summarizer)
    
    return fast_summarizer