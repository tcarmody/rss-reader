"""
Request coalescing for concurrent summarization requests.

Prevents duplicate API calls when multiple clients request
the same article simultaneously.
"""

import asyncio
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from summarization.fast_summarizer import FastSummarizer

logger = logging.getLogger(__name__)


class CoalescingSummarizer:
    """
    Wrapper around FastSummarizer that coalesces concurrent requests.

    When multiple requests for the same article arrive simultaneously,
    only one API call is made and all requesters receive the same result.
    """

    def __init__(
        self,
        base_summarizer: FastSummarizer,
        coalesce_window_seconds: float = 5.0
    ):
        """
        Initialize the coalescing summarizer.

        Args:
            base_summarizer: The underlying FastSummarizer instance
            coalesce_window_seconds: Time window for coalescing (seconds)
        """
        self.base = base_summarizer
        self.coalesce_window = coalesce_window_seconds

        # Track in-flight requests
        self.in_flight: Dict[str, asyncio.Future] = {}
        self.in_flight_timestamps: Dict[str, datetime] = {}

        # Lock for thread-safe access to in_flight dict
        self.lock = asyncio.Lock()

        logger.info(
            f"Initialized CoalescingSummarizer with "
            f"{coalesce_window_seconds}s coalesce window"
        )

    def _get_cache_key(
        self,
        text: str,
        title: str,
        model: Optional[str],
        style: str
    ) -> str:
        """
        Generate a cache key for coalescing.

        Args:
            text: Article text
            title: Article title
            model: Model identifier
            style: Summary style

        Returns:
            Cache key string
        """
        # Use first 1000 chars + title + model + style for key
        # This balances uniqueness with reasonable key length
        key_input = f"{text[:1000]}:{title}:{model or 'auto'}:{style}"
        return hashlib.md5(key_input.encode('utf-8')).hexdigest()

    async def summarize(
        self,
        text: str,
        title: str,
        url: str,
        model: Optional[str] = None,
        force_refresh: bool = False,
        auto_select_model: bool = True,
        temperature: float = 0.3,
        style: str = "default"
    ) -> Dict[str, str]:
        """
        Summarize with request coalescing.

        Args:
            text: Article text
            title: Article title
            url: Article URL
            model: Optional model override
            force_refresh: Force new summary (bypasses coalescing)
            auto_select_model: Auto-select model by complexity
            temperature: Generation temperature
            style: Summary style

        Returns:
            Summary dictionary
        """
        # If force_refresh, bypass coalescing
        if force_refresh:
            logger.debug(f"Bypassing coalescing for {url} (force_refresh=True)")
            return self.base.summarize(
                text=text,
                title=title,
                url=url,
                model=model,
                force_refresh=force_refresh,
                auto_select_model=auto_select_model,
                temperature=temperature,
                style=style
            )

        # Generate cache key
        cache_key = self._get_cache_key(text, title, model, style)

        async with self.lock:
            # Check if request is already in flight
            if cache_key in self.in_flight:
                # Check if it's within the coalesce window
                timestamp = self.in_flight_timestamps[cache_key]
                elapsed = (datetime.now() - timestamp).total_seconds()

                if elapsed < self.coalesce_window:
                    # Coalesce with existing request
                    logger.info(
                        f"Coalescing request for {url} "
                        f"(existing request {elapsed:.2f}s old)"
                    )
                    future = self.in_flight[cache_key]

                    # Release lock while waiting
                    # (don't hold lock during await)

            else:
                # No existing request - create new one
                future = None

        # If we're coalescing, wait for existing request
        if cache_key in self.in_flight and future is not None:
            try:
                return await future
            except Exception as e:
                logger.error(f"Coalesced request failed: {e}")
                # Fall through to create new request
                future = None

        # Create new request if not coalescing
        if future is None:
            async with self.lock:
                # Double-check it wasn't created while we waited for lock
                if cache_key in self.in_flight:
                    future = self.in_flight[cache_key]
                else:
                    # Create new future for this request
                    future = asyncio.Future()
                    self.in_flight[cache_key] = future
                    self.in_flight_timestamps[cache_key] = datetime.now()

                    logger.debug(f"Created new in-flight request for {url}")

            # Perform actual summarization
            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.base.summarize(
                        text=text,
                        title=title,
                        url=url,
                        model=model,
                        force_refresh=False,  # Already handled above
                        auto_select_model=auto_select_model,
                        temperature=temperature,
                        style=style
                    )
                )

                # Set result on future
                future.set_result(result)

                return result

            except Exception as e:
                # Set exception on future so waiting tasks get it
                logger.error(f"Summarization failed for {url}: {e}")
                future.set_exception(e)
                raise

            finally:
                # Clean up in-flight tracking
                async with self.lock:
                    if cache_key in self.in_flight:
                        del self.in_flight[cache_key]
                    if cache_key in self.in_flight_timestamps:
                        del self.in_flight_timestamps[cache_key]

        else:
            # Wait for existing request
            return await future

    async def batch_summarize(
        self,
        articles: list,
        max_concurrent: int = 3,
        model: Optional[str] = None,
        auto_select_model: bool = True,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
        style: str = "default"
    ) -> list:
        """
        Batch summarize with coalescing.

        Args:
            articles: List of article dicts
            max_concurrent: Max concurrent requests
            model: Optional model override
            auto_select_model: Auto-select models
            temperature: Generation temperature
            timeout: Optional timeout
            style: Summary style

        Returns:
            List of summary results
        """
        # Create tasks with coalescing
        semaphore = asyncio.Semaphore(max_concurrent)

        async def summarize_one(article: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                text = article.get('text', article.get('content', ''))
                title = article.get('title', '')
                url = article.get('url', article.get('link', ''))

                try:
                    summary = await self.summarize(
                        text=text,
                        title=title,
                        url=url,
                        model=model,
                        auto_select_model=auto_select_model,
                        temperature=temperature,
                        style=style
                    )

                    return {
                        'original': article,
                        'summary': summary
                    }

                except Exception as e:
                    logger.error(f"Error in batch summarize for {url}: {e}")
                    return {
                        'original': article,
                        'summary': {
                            'headline': title or 'Error',
                            'summary': f"Error: {str(e)}",
                            'style': style,
                            'error': True
                        }
                    }

        # Create all tasks
        tasks = [summarize_one(article) for article in articles]

        # Execute with optional timeout
        if timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Batch timed out after {timeout}s")
                # Return completed results
                results = []
                for task in tasks:
                    if task.done():
                        try:
                            results.append(task.result())
                        except Exception:
                            pass
                return results
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter exceptions
        successful = []
        for r in results:
            if not isinstance(r, Exception):
                successful.append(r)
            else:
                logger.error(f"Task exception: {r}")

        return successful

    def get_stats(self) -> Dict[str, Any]:
        """
        Get coalescing statistics.

        Returns:
            Dict with statistics
        """
        return {
            'in_flight_count': len(self.in_flight),
            'coalesce_window': self.coalesce_window
        }

    async def cleanup_stale_requests(self):
        """
        Clean up stale in-flight requests.

        Should be called periodically to prevent memory leaks.
        """
        async with self.lock:
            now = datetime.now()
            stale_keys = []

            for key, timestamp in self.in_flight_timestamps.items():
                age = (now - timestamp).total_seconds()
                if age > self.coalesce_window * 2:
                    stale_keys.append(key)

            for key in stale_keys:
                logger.warning(f"Cleaning up stale request (age={age:.1f}s)")
                if key in self.in_flight:
                    del self.in_flight[key]
                if key in self.in_flight_timestamps:
                    del self.in_flight_timestamps[key]

            if stale_keys:
                logger.info(f"Cleaned up {len(stale_keys)} stale requests")


def create_coalescing_summarizer(
    base_summarizer: FastSummarizer,
    coalesce_window_seconds: float = 5.0
) -> CoalescingSummarizer:
    """
    Factory function to create a coalescing summarizer.

    Args:
        base_summarizer: Base FastSummarizer instance
        coalesce_window_seconds: Coalesce window in seconds

    Returns:
        CoalescingSummarizer instance
    """
    return CoalescingSummarizer(
        base_summarizer=base_summarizer,
        coalesce_window_seconds=coalesce_window_seconds
    )
