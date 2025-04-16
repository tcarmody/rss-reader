import asyncio
import logging
import time
from summarizer import ArticleSummarizer
from fast_summarizer import FastArticleSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_single_article():
    """Example of using the optimized summarizer for a single article."""
    # Create original summarizer
    original = ArticleSummarizer()
    
    # Create optimized version
    fast = FastArticleSummarizer(
        original_summarizer=original,
        rpm_limit=40,  # Set safe API limits
        cache_size=100
    )
    
    # Sample article
    article_text = """
    [Your article text here]
    """
    
    article_title = "Sample Article Title"
    article_url = "https://example.com/article"
    
    # Measure time for original summarizer
    start_time = time.time()
    original_summary = original.summarize_article(
        text=article_text,
        title=article_title,
        url=article_url
    )
    original_time = time.time() - start_time
    
    print(f"Original summarizer took {original_time:.2f} seconds")
    print(f"Headline: {original_summary['headline']}")
    
    # Measure time for fast summarizer
    start_time = time.time()
    fast_summary = fast.summarize(
        text=article_text,
        title=article_title,
        url=article_url,
        auto_select_model=True
    )
    fast_time = time.time() - start_time
    
    print(f"Fast summarizer took {fast_time:.2f} seconds")
    print(f"Headline: {fast_summary['headline']}")
    
    # Show improvement
    improvement = (original_time - fast_time) / original_time * 100
    print(f"Performance improvement: {improvement:.1f}%")


async def example_batch_articles():
    """Example of batch processing multiple articles."""
    # Create original summarizer
    original = ArticleSummarizer()
    
    # Create optimized version
    fast = FastArticleSummarizer(
        original_summarizer=original,
        rpm_limit=40
    )
    
    # Sample articles
    articles = [
        {
            'text': "Article 1 text here...",
            'title': "Article 1 Title",
            'url': "https://example.com/article1"
        },
        {
            'text': "Article 2 text here...",
            'title': "Article 2 Title",
            'url': "https://example.com/article2"
        },
        {
            'text': "Article 3 text here...",
            'title': "Article 3 Title",
            'url': "https://example.com/article3"
        },
        {
            'text': "Article 4 text here...",
            'title': "Article 4 Title",
            'url': "https://example.com/article4"
        },
        {
            'text': "Article 5 text here...",
            'title': "Article 5 Title",
            'url': "https://example.com/article5"
        }
    ]
    
    # Process sequentially with original summarizer
    start_time = time.time()
    original_results = []
    
    for article in articles:
        try:
            result = original.summarize_article(
                text=article['text'],
                title=article['title'],
                url=article['url']
            )
            original_results.append(result)
        except Exception as e:
            print(f"Error processing {article['url']}: {str(e)}")
    
    original_time = time.time() - start_time
    print(f"Sequential processing took {original_time:.2f} seconds")
    
    # Process in parallel with batch summarizer
    start_time = time.time()
    batch_results = await fast.batch_summarize(
        articles=articles,
        max_concurrent=3,
        auto_select_model=True
    )
    batch_time = time.time() - start_time
    
    print(f"Batch processing took {batch_time:.2f} seconds")
    
    # Show improvement
    improvement = (original_time - batch_time) / original_time * 100
    print(f"Performance improvement: {improvement:.1f}%")
    
    # Display results
    for i, result in enumerate(batch_results):
        print(f"Article {i+1}: {result['summary']['headline']}")


def example_long_article():
    """Example of processing a long article."""
    # Create original summarizer
    original = ArticleSummarizer()
    
    # Create optimized version
    fast = FastArticleSummarizer(original_summarizer=original)
    
    # Very long article (just for demonstration)
    long_article = " ".join(["This is paragraph " + str(i) + ". " * 20 for i in range(100)])
    
    # Process with optimized summarizer
    summary = fast.summarize(
        text=long_article,
        title="Very Long Article",
        url="https://example.com/long-article",
        auto_select_model=True
    )
    
    print(f"Processed long article: {len(long_article)} characters")
    print(f"Headline: {summary['headline']}")
    print(f"Summary length: {len(summary['summary'])} characters")


if __name__ == "__main__":
    # Run examples
    print("\n=== Single Article Example ===")
    example_single_article()
    
    print("\n=== Long Article Example ===")
    example_long_article()
    
    print("\n=== Batch Processing Example ===")
    asyncio.run(example_batch_articles())