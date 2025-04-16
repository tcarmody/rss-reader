import asyncio
import concurrent.futures
import logging
from typing import List, Dict, Optional, Any

async def summarize_articles_batch(
    summarizer,
    articles: List[Dict[str, str]],
    model: Optional[str] = None,
    max_concurrent: int = 3,
    temperature: float = 0.3,
    auto_select_model: bool = False
):
    """
    Process multiple articles in parallel batches.
    
    Args:
        summarizer: ArticleSummarizer instance
        articles: List of article dicts with 'text', 'title', and 'url' keys
        model: Claude model to use
        max_concurrent: Maximum number of concurrent API calls
        temperature: Temperature setting
        auto_select_model: Whether to auto-select model based on complexity
        
    Returns:
        List of article summaries with original metadata
    """
    logger = getattr(summarizer, 'logger', logging.getLogger(__name__))
    logger.info(f"Processing batch of {len(articles)} articles (max_concurrent={max_concurrent})")
    
    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_article(article):
        async with semaphore:
            logger.debug(f"Processing article: {article['url']}")
            
            # Check cache first
            if auto_select_model and model is None:
                # Clean text for complexity estimation
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    cleaned_text = await loop.run_in_executor(
                        executor, summarizer.clean_text, article['text']
                    )
                
                # Auto-select model
                from model_selection import auto_select_model as select_model
                model_id = select_model(
                    cleaned_text, 
                    summarizer.AVAILABLE_MODELS, 
                    summarizer.DEFAULT_MODEL,
                    logger
                )
            else:
                model_id = summarizer._get_model(model)
                
                # Clean text
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    cleaned_text = await loop.run_in_executor(
                        executor, summarizer.clean_text, article['text']
                    )
            
            # Generate cache key
            cache_key = f"{cleaned_text}:{model_id}:{temperature}"
            
            # Check cache
            cached_summary = summarizer.summary_cache.get(cache_key)
            if cached_summary:
                logger.info(f"Cache hit for {article['url']}")
                return {
                    'original': article,
                    'summary': cached_summary
                }
            
            # Use ThreadPoolExecutor for API call (blocking operation)
            # Extract source from URL for attribution
            source_name = summarizer._extract_source_from_url(article['url'])
            
            # Create the prompt (uses the original prompt)
            prompt = summarizer._create_summary_prompt(cleaned_text, article['url'], source_name)
            
            try:
                # Use the synchronous API call in a thread pool
                loop = asyncio.get_event_loop()
                summary_text = await loop.run_in_executor(
                    None,
                    lambda: summarizer._call_claude_api(
                        model_id=model_id,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=400
                    )
                )
                
                # Parse the response
                result = summarizer._parse_summary_response(
                    summary_text, 
                    article['title'], 
                    article['url'], 
                    source_name
                )
                
                # Cache the result
                summarizer.summary_cache.set(cache_key, result)
                
                logger.info(f"Successfully summarized {article['url']}")
                
                return {
                    'original': article,
                    'summary': result
                }
            except Exception as e:
                logger.error(f"Error summarizing {article['url']}: {str(e)}")
                # Return error result
                return {
                    'original': article,
                    'error': str(e),
                    'summary': {
                        'headline': article['title'],
                        'summary': f"Summary generation failed: {str(e)}. Please try again later."
                    }
                }
    
    # Create tasks for all articles
    tasks = [process_article(article) for article in articles]
    
    # Execute all tasks and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results, handling any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception in batch processing for article {i}: {str(result)}")
            processed_results.append({
                'original': articles[i],
                'error': str(result),
                'summary': {
                    'headline': articles[i]['title'],
                    'summary': f"Summary generation failed: {str(result)}. Please try again later."
                }
            })
        else:
            processed_results.append(result)
    
    logger.info(f"Completed batch processing of {len(articles)} articles")
    return processed_results