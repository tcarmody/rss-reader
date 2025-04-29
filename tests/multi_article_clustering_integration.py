"""
Integration script demonstrating how to use the enhanced clustering with multiple article comparison.

This script showcases how to:
1. Initialize the enhanced article clusterer
2. Process a batch of articles
3. Use the new multi-article clustering capabilities
4. Analyze and refine the resulting clusters
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from dotenv here to enable .env file support
from dotenv import load_dotenv
load_dotenv()

async def demonstrate_enhanced_clustering():
    """
    Demonstrate the enhanced clustering functionality with a sample dataset.
    """
    logger.info("Starting enhanced clustering demonstration")
    
    try:
        # Import required modules
        from summarizer import ArticleSummarizer
        from enhanced_clustering import create_enhanced_clusterer
        from lm_cluster_analyzer import create_cluster_analyzer
        
        # Initialize the summarizer first (needed for LM access)
        summarizer = ArticleSummarizer()
        logger.info("Initialized ArticleSummarizer successfully")
        
        # Create enhanced clusterer with the summarizer
        clusterer = create_enhanced_clusterer(summarizer=summarizer)
        logger.info("Created enhanced article clusterer")
        
        # Create a cluster analyzer with the same summarizer
        analyzer = create_cluster_analyzer(summarizer=summarizer)
        logger.info("Created LM cluster analyzer")
        
        # Sample articles for demonstration
        sample_articles = create_sample_articles()
        logger.info(f"Created {len(sample_articles)} sample articles")
        
        # Apply the enhanced clustering
        logger.info("Starting cluster generation...")
        start_time = datetime.now()
        
        # This uses the two-phase approach with embedding + LM
        clusters = clusterer.cluster_with_summaries(sample_articles)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Clustering completed in {elapsed:.2f} seconds")
        logger.info(f"Generated {len(clusters)} clusters")
        
        # Display cluster information
        await display_cluster_info(clusters, analyzer)
        
        return clusters
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}", exc_info=True)
        return []


async def display_cluster_info(clusters: List[List[Dict[str, Any]]], analyzer) -> None:
    """
    Display information about the generated clusters.
    
    Args:
        clusters: List of article clusters
        analyzer: LMClusterAnalyzer instance
    """
    if not clusters:
        logger.info("No clusters to display")
        return
        
    logger.info("\n==== Cluster Information ====")
    
    for i, cluster in enumerate(clusters, 1):
        if not cluster:
            continue
            
        # Get article titles
        titles = [article.get('title', 'No title') for article in cluster]
        
        logger.info(f"\nCluster {i} ({len(cluster)} articles):")
        logger.info("Articles:")
        for j, title in enumerate(titles, 1):
            logger.info(f"  {j}. {title}")
        
        # Extract topics for the cluster
        topics = analyzer.extract_cluster_topics(cluster)
        if topics:
            logger.info(f"Topics: {', '.join(topics)}")
        
        # If the cluster has a summary, show it
        if 'summary' in cluster[0] and cluster[0]['summary']:
            if isinstance(cluster[0]['summary'], dict):
                logger.info(f"Headline: {cluster[0]['summary'].get('headline', 'No headline')}")
            else:
                logger.info(f"Summary available: {type(cluster[0]['summary'])}")
        
        logger.info("-" * 50)


def create_sample_articles() -> List[Dict[str, Any]]:
    """
    Create a sample dataset of articles for demonstration.
    
    Returns:
        List of article dictionaries
    """
    # Sample articles on related technology topics
    sample_articles = [
        {
            'title': 'Apple Announces M3 Pro and M3 Max Chips for MacBook Pro',
            'content': 'Apple today announced its latest Apple Silicon chips, the M3 Pro and M3 Max. The new chips feature enhanced performance and power efficiency compared to the previous generation. The M3 Pro offers up to a 12-core CPU and 18-core GPU, while the M3 Max features up to a 16-core CPU and 40-core GPU. The chips will debut in the latest MacBook Pro models.',
            'link': 'https://example.com/apple-m3',
            'published': datetime.now().isoformat(),
            'feed_source': 'TechNews'
        },
        {
            'title': 'New MacBook Pro Models Feature M3 Chips and Mini-LED Displays',
            'content': 'Apple has revealed updated MacBook Pro models powered by the new M3 Pro and M3 Max chips. The laptops retain the same design as the previous generation but with significant performance improvements. The 14-inch and 16-inch models both feature mini-LED displays with ProMotion technology, supporting up to 120Hz refresh rates.',
            'link': 'https://example.com/macbook-m3',
            'published': datetime.now().isoformat(),
            'feed_source': 'AppleInsider'
        },
        {
            'title': 'Microsoft Launches Surface Laptop Studio 2 with Enhanced Performance',
            'content': 'Microsoft has unveiled the Surface Laptop Studio 2, featuring Intel Core i7 H-series processors and NVIDIA GeForce RTX 4060 graphics. The new model offers up to 2x faster performance than its predecessor, according to Microsoft. The unique hinged design allows the device to transform between laptop, stage, and studio modes.',
            'link': 'https://example.com/surface-studio',
            'published': datetime.now().isoformat(),
            'feed_source': 'Windows Central'
        },
        {
            'title': 'Surface Laptop Studio 2: Microsoft's Most Powerful Surface Yet',
            'content': 'The new Surface Laptop Studio 2 has been announced by Microsoft, featuring significant performance upgrades. The device now offers 13th Gen Intel processors, NVIDIA RTX 4000-series graphics, and up to 64GB of RAM. Microsoft claims it's the most powerful Surface device they've ever created, targeting creative professionals and developers.',
            'link': 'https://example.com/microsoft-surface',
            'published': datetime.now().isoformat(),
            'feed_source': 'The Verge'
        },
        {
            'title': 'Google Announces Pixel 8 and Pixel 8 Pro with Tensor G3',
            'content': 'Google has unveiled its latest flagship smartphones, the Pixel 8 and Pixel 8 Pro. Both devices are powered by the new Google Tensor G3 processor, which offers enhanced AI capabilities. The phones feature updated camera systems, with the Pro model offering a 48MP ultrawide lens. Google promises 7 years of OS updates for both devices.',
            'link': 'https://example.com/google-pixel-8',
            'published': datetime.now().isoformat(),
            'feed_source': 'Android Authority'
        },
        {
            'title': 'Pixel 8 Pro Camera Review: Google's Best Camera Yet',
            'content': 'The Pixel 8 Pro's camera system has received significant upgrades this year, including a new 48MP ultrawide camera and enhanced processing capabilities. The main 50MP sensor now captures more light than before, resulting in improved low-light performance. Google has also added new AI-powered features like Best Take and Magic Editor.',
            'link': 'https://example.com/pixel-camera-review',
            'published': datetime.now().isoformat(),
            'feed_source': 'DXOMARK'
        },
        {
            'title': 'Samsung Unveils New Foldable Display Technology',
            'content': 'Samsung Display has announced a new generation of foldable display technology that reduces the visibility of the crease by 60% compared to current models. The company says the new panels can withstand up to 400,000 folds without damage. The technology is expected to debut in next year's Galaxy Z Fold and Z Flip devices.',
            'link': 'https://example.com/samsung-foldable',
            'published': datetime.now().isoformat(),
            'feed_source': 'SamMobile'
        },
        {
            'title': 'NVIDIA Announces New AI Supercomputer',
            'content': 'NVIDIA today unveiled its latest AI supercomputer, designed specifically for training large language models. The system uses thousands of H100 GPUs connected via NVIDIA's NVLink technology. According to NVIDIA, the new supercomputer can train models with trillions of parameters in a fraction of the time required by previous systems.',
            'link': 'https://example.com/nvidia-ai',
            'published': datetime.now().isoformat(),
            'feed_source': 'TechCrunch'
        },
        {
            'title': 'Meta Releases New Open Source AI Model',
            'content': 'Meta (formerly Facebook) has released a new open-source large language model that the company claims matches the performance of proprietary alternatives. The model, named "MetaLLaMA," has been trained on over 8 trillion tokens of data. Meta says it's releasing the model to accelerate AI research and enable more transparent development.',
            'link': 'https://example.com/meta-ai-model',
            'published': datetime.now().isoformat(),
            'feed_source': 'The Information'
        },
        {
            'title': 'Amazon Introduces New Alexa Features Powered by Generative AI',
            'content': 'Amazon has announced a major update to Alexa, powered by generative AI technology. The new features include more natural conversations, better understanding of complex requests, and the ability to control multiple smart home devices with a single command. Amazon says the updates will roll out to Echo devices over the next few months.',
            'link': 'https://example.com/alexa-update',
            'published': datetime.now().isoformat(),
            'feed_source': 'CNET'
        }
    ]
    
    return sample_articles


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_enhanced_clustering())
