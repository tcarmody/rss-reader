#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script to compare performance between different summarization approaches.
This script compares the original and enhanced batch processing implementations.
"""

# Add main function to run the benchmark
def main():
    """Main function to run benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Benchmark different summarization methods")
    parser.add_argument("--articles", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--size", choices=['small', 'medium', 'large'], default='medium', 
                        help="Size of test articles")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker processes")
    parser.add_argument("--methods", nargs="+", choices=['sequential', 'async_batch', 'parallel_batch', 'enhanced_batch'],
                        default=None, help="Methods to benchmark (defaults to all)")
    args = parser.parse_args()
    
    print(f"\n===== Summarization Benchmark =====")
    print(f"Articles: {args.articles} {args.size}")
    print(f"Workers: {args.workers}")
    print(f"Methods: {args.methods or 'all'}")
    print("===================================\n")
    
    # Run benchmarks
    benchmark = SummarizationBenchmark(
        num_articles=args.articles,
        article_size=args.size,
        max_workers=args.workers
    )
    
    asyncio.run(benchmark.run_benchmarks(methods=args.methods))
    
    return 0

import os
import sys
import time
import logging
import asyncio
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)

class SummarizationBenchmark:
    """Benchmark different summarization approaches."""
    
    def __init__(self, num_articles=10, article_size='medium', max_workers=3):
        """
        Initialize the benchmark.
        
        Args:
            num_articles: Number of articles to summarize
            article_size: Size of test articles ('small', 'medium', 'large')
            max_workers: Maximum number of worker processes for batch processing
        """
        self.num_articles = num_articles
        self.article_size = article_size
        self.max_workers = max_workers
        self.results = {}
        
        # Set article content size
        self.content_sizes = {
            'small': 500,    # ~500 chars
            'medium': 2000,  # ~2000 chars
            'large': 8000    # ~8000 chars
        }
        
        # Ensure the selected size is valid
        if article_size not in self.content_sizes:
            raise ValueError(f"Invalid article size: {article_size}. Choose from: {list(self.content_sizes.keys())}")
        
        # Log benchmark configuration
        logging.info(f"Initializing benchmark with {num_articles} {article_size} articles and {max_workers} workers")
        
    def generate_test_articles(self):
        """
        Generate test articles for benchmarking.
        
        Returns:
            list: List of test articles
        """
        logging.info(f"Generating {self.num_articles} test articles of size {self.article_size}")
        
        # Get target content length
        target_length = self.content_sizes[self.article_size]
        
        # Generate articles
        articles = []
        for i in range(self.num_articles):
            # Create article with appropriate length
            paragraphs = []
            remaining_length = target_length
            
            while remaining_length > 0:
                # Create a paragraph (100-200 chars)
                paragraph_length = min(remaining_length, 100 + (i % 100))
                paragraph = f"This is paragraph {len(paragraphs) + 1} of test article {i + 1}. " + \
                           f"It contains sample text for benchmarking summarization performance. " + \
                           f"The content is designed to be {self.article_size} sized with {target_length} characters. " + \
                           "A" * max(0, paragraph_length - 180)  # Pad to desired length
                
                paragraphs.append(paragraph)
                remaining_length -= len(paragraph)
            
            # Combine paragraphs with newlines
            content = "\n\n".join(paragraphs)
            
            # Create the article
            article = {
                'text': content,
                'title': f"Test Article {i + 1}",
                'url': f"https://example.com/article-{i + 1}"
            }
            
            articles.append(article)
        
        return articles
        
    async def benchmark_original_sequential(self):
        """
        Benchmark the original ArticleSummarizer with sequential processing.
        
        Returns:
            dict: Benchmark results
        """
        logging.info("Starting benchmark: Original ArticleSummarizer (Sequential)")
        
        try:
            # Import and initialize summarizer
            from summarizer import ArticleSummarizer
            summarizer = ArticleSummarizer()
            
            # Generate test articles
            articles = self.generate_test_articles()
            
            # Process articles sequentially
            start_time = time.time()
            processed_articles = []
            
            for i, article in enumerate(articles, 1):
                try:
                    logging.info(f"Processing article {i}/{len(articles)}")
                    summary = summarizer.summarize_article(
                        text=article['text'],
                        title=article['title'],
                        url=article['url']
                    )
                    processed_articles.append({
                        'original': article,
                        'summary': summary
                    })
                except Exception as e:
                    logging.error(f"Error processing article {i}: {e}")
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            articles_per_second = len(processed_articles) / elapsed_time if elapsed_time > 0 else 0
            avg_time_per_article = elapsed_time / len(processed_articles) if processed_articles else 0
            
            # Store results
            result = {
                'method': 'original_sequential',
                'num_articles': len(articles),
                'article_size': self.article_size,
                'successful_articles': len(processed_articles),
                'elapsed_time': elapsed_time,
                'articles_per_second': articles_per_second,
                'avg_time_per_article': avg_time_per_article
            }
            
            self.results['original_sequential'] = result
            logging.info(f"Benchmark completed: Original Sequential - {elapsed_time:.2f}s total, {avg_time_per_article:.2f}s per article")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in original sequential benchmark: {e}")
            return {
                'method': 'original_sequential',
                'error': str(e)
            }
            
    async def benchmark_original_async_batch(self):
        """
        Benchmark the original ArticleSummarizer with its async batch processing.
        
        Returns:
            dict: Benchmark results
        """
        logging.info("Starting benchmark: Original async_batch")
        
        try:
            # Import and initialize
            from summarizer import ArticleSummarizer
            from async_batch import summarize_articles_batch
            
            summarizer = ArticleSummarizer()
            
            # Generate test articles
            articles = self.generate_test_articles()
            
            # Process articles with original batch method
            start_time = time.time()
            
            try:
                results = await summarize_articles_batch(
                    summarizer=summarizer,
                    articles=articles,
                    max_concurrent=self.max_workers
                )
                
                processed_articles = results
                
            except Exception as batch_error:
                logging.error(f"Error in original batch processing: {batch_error}")
                processed_articles = []
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            articles_per_second = len(processed_articles) / elapsed_time if elapsed_time > 0 else 0
            avg_time_per_article = elapsed_time / len(processed_articles) if processed_articles else 0
            
            # Store results
            result = {
                'method': 'original_async_batch',
                'num_articles': len(articles),
                'article_size': self.article_size,
                'successful_articles': len(processed_articles),
                'elapsed_time': elapsed_time,
                'articles_per_second': articles_per_second,
                'avg_time_per_article': avg_time_per_article
            }
            
            self.results['original_async_batch'] = result
            logging.info(f"Benchmark completed: Original Async Batch - {elapsed_time:.2f}s total, {avg_time_per_article:.2f}s per article")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in original async batch benchmark: {e}")
            return {
                'method': 'original_async_batch',
                'error': str(e)
            }
    
    async def benchmark_parallel_batch(self):
        """
        Benchmark the original spawn-based parallel batch processing.
        
        Returns:
            dict: Benchmark results
        """
        logging.info("Starting benchmark: Parallel Batch Processing (Spawn)")
        
        try:
            # Import and initialize
            from summarizer import ArticleSummarizer
            from parallel_batch_processor import SpawnBatchProcessor
            
            summarizer = ArticleSummarizer()
            batch_processor = SpawnBatchProcessor()
            
            # Generate test articles
            articles = self.generate_test_articles()
            
            # Process articles with spawn-based batch method
            start_time = time.time()
            
            try:
                results = await batch_processor.batch_process_async(
                    summarizer=summarizer,
                    articles=articles,
                    max_concurrent=self.max_workers
                )
                
                processed_articles = [r for r in results if r.get('success', False)]
                
            except Exception as batch_error:
                logging.error(f"Error in parallel batch processing: {batch_error}")
                processed_articles = []
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            articles_per_second = len(processed_articles) / elapsed_time if elapsed_time > 0 else 0
            avg_time_per_article = elapsed_time / len(processed_articles) if processed_articles else 0
            
            # Store results
            result = {
                'method': 'parallel_batch',
                'num_articles': len(articles),
                'article_size': self.article_size,
                'successful_articles': len(processed_articles),
                'elapsed_time': elapsed_time,
                'articles_per_second': articles_per_second,
                'avg_time_per_article': avg_time_per_article
            }
            
            self.results['parallel_batch'] = result
            logging.info(f"Benchmark completed: Parallel Batch - {elapsed_time:.2f}s total, {avg_time_per_article:.2f}s per article")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in parallel batch benchmark: {e}")
            return {
                'method': 'parallel_batch',
                'error': str(e)
            }
    
    async def benchmark_enhanced_batch(self):
        """
        Benchmark the enhanced batch processing.
        
        Returns:
            dict: Benchmark results
        """
        logging.info("Starting benchmark: Enhanced Batch Processing")
        
        try:
            # Import and initialize
            from summarizer import ArticleSummarizer
            from fast_summarizer import create_fast_summarizer
            
            original_summarizer = ArticleSummarizer()
            fast_summarizer = create_fast_summarizer(
                original_summarizer=original_summarizer,
                max_batch_workers=self.max_workers
            )
            
            # Generate test articles
            articles = self.generate_test_articles()
            
            # Process articles with enhanced batch method
            start_time = time.time()
            
            try:
                results = await fast_summarizer.batch_summarize(
                    articles=articles,
                    max_concurrent=self.max_workers
                )
                
                processed_articles = [r for r in results if 'summary' in r]
                
            except Exception as batch_error:
                logging.error(f"Error in enhanced batch processing: {batch_error}")
                processed_articles = []
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            articles_per_second = len(processed_articles) / elapsed_time if elapsed_time > 0 else 0
            avg_time_per_article = elapsed_time / len(processed_articles) if processed_articles else 0
            
            # Store results
            result = {
                'method': 'enhanced_batch',
                'num_articles': len(articles),
                'article_size': self.article_size,
                'successful_articles': len(processed_articles),
                'elapsed_time': elapsed_time,
                'articles_per_second': articles_per_second,
                'avg_time_per_article': avg_time_per_article
            }
            
            self.results['enhanced_batch'] = result
            logging.info(f"Benchmark completed: Enhanced Batch - {elapsed_time:.2f}s total, {avg_time_per_article:.2f}s per article")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in enhanced batch benchmark: {e}")
            return {
                'method': 'enhanced_batch',
                'error': str(e)
            }
    
    async def run_benchmarks(self, methods=None):
        """
        Run all specified benchmarks.
        
        Args:
            methods: List of benchmark methods to run or None for all
            
        Returns:
            dict: Benchmark results
        """
        # Define available benchmark methods
        available_methods = {
            'sequential': self.benchmark_original_sequential,
            'async_batch': self.benchmark_original_async_batch,
            'parallel_batch': self.benchmark_parallel_batch,
            'enhanced_batch': self.benchmark_enhanced_batch
        }
        
        # If no specific methods are specified, run all
        if methods is None:
            methods = list(available_methods.keys())
        
        # Validate methods
        for method in methods:
            if method not in available_methods:
                logging.warning(f"Invalid benchmark method: {method}. Skipping.")
        
        # Run requested benchmarks
        valid_methods = [m for m in methods if m in available_methods]
        logging.info(f"Running benchmarks: {', '.join(valid_methods)}")
        
        for method in valid_methods:
            await available_methods[method]()
        
        # Save results to file
        self.save_results()
        
        # Print comparison
        self.print_comparison()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to a file."""
        if not self.results:
            logging.warning("No benchmark results to save")
            return
            
        # Create results directory if it doesn't exist
        os.makedirs('benchmark_results', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"benchmark_results/benchmark_{timestamp}.json"
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            logging.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving benchmark results: {e}")
    
    def print_comparison(self):
        """Print a comparison of benchmark results."""
        if not self.results:
            print("\nNo benchmark results to compare")
            return
            
        print("\n" + "=" * 80)
        print(f"BENCHMARK COMPARISON: {self.num_articles} {self.article_size} articles, {self.max_workers} workers")
        print("=" * 80)
        
        # Table headers
        headers = ["Method", "Total Time (s)", "Articles/sec", "Avg Time/Article (s)", "Success Rate"]
        format_str = "{:<20} {:<15} {:<12} {:<20} {:<12}"
        
        print(format_str.format(*headers))
        print("-" * 80)
        
        # Print results for each method
        for method, result in self.results.items():
            if 'error' in result:
                print(f"{method}: ERROR - {result['error']}")
                continue
                
            # Format data
            total_time = f"{result['elapsed_time']:.2f}"
            articles_per_sec = f"{result['articles_per_second']:.2f}"
            avg_time = f"{result['avg_time_per_article']:.2f}"
            success_rate = f"{(result['successful_articles'] / result['num_articles']) * 100:.1f}%"
            
            print(format_str.format(method, total_time, articles_per_sec, avg_time, success_rate))
        
        print("=" * 80)
        
        # Calculate speedup compared to sequential processing
        if 'original_sequential' in self.results and 'enhanced_batch' in self.results:
            seq_time = self.results['original_sequential']['elapsed_time']
            enh_time = self.results['enhanced_batch']['elapsed_time']
            
            if seq_time > 0:
                speedup = seq_time / enh_time
                print(f"Enhanced batch processing is {speedup:.2f}x faster than sequential processing")
        
        print("=" * 80)