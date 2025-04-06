"""Main entry point for the RSS reader."""

import sys
import logging
import traceback
import argparse
from rss_reader.reader import RSSReader


def main():
    """
    Main function to run the RSS reader.
    
    Process command line arguments and run the reader.
    
    Example:
        # Run directly
        python -m rss_reader.main
        
        # Import and use in another script
        from rss_reader.reader import RSSReader
        reader = RSSReader()
        output_file = reader.process_feeds()
    """
    parser = argparse.ArgumentParser(description="RSS Reader and Summarizer")
    parser.add_argument("--feeds", nargs="+", help="List of feed URLs to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of feeds to process in a batch")
    parser.add_argument("--batch-delay", type=int, default=15, help="Delay between batches in seconds")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run RSS reader
        rss_reader = RSSReader(
            feeds=args.feeds,  # Will use default if None
            batch_size=args.batch_size,
            batch_delay=args.batch_delay
        )
        
        output_file = rss_reader.process_feeds()

        if output_file:
            logging.info(f"✅ Successfully generated RSS summary: {output_file}")
            print(f"\nSummary generated at: {output_file}")
            return 0
        else:
            logging.warning("⚠️ No articles found or processed")
            print("\nNo articles found or processed. Check the log for details.")
            return 1

    except Exception as e:
        logging.error(f"❌ Error in main: {str(e)}")
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
