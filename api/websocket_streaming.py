"""
WebSocket-based streaming for real-time summary delivery.

Provides streaming summaries as they complete, rather than waiting
for entire batch to finish.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    CONNECTED = "connected"
    SUMMARY = "summary"
    ERROR = "error"
    PROGRESS = "progress"
    COMPLETE = "complete"
    CLUSTER_UPDATE = "cluster_update"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamMessage:
    """Structured message for WebSocket streaming."""
    type: MessageType
    timestamp: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            'type': self.type.value,
            'timestamp': self.timestamp,
            'data': self.data
        })


class ConnectionManager:
    """
    Manages active WebSocket connections.

    Features:
    - Connection tracking
    - Broadcast to all connections
    - Connection-specific messaging
    - Heartbeat support
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self._connection_counter = 0
        logger.info("Initialized WebSocket ConnectionManager")

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept and track a new WebSocket connection.

        Args:
            websocket: WebSocket instance

        Returns:
            Connection ID
        """
        await websocket.accept()

        self._connection_counter += 1
        connection_id = f"conn_{self._connection_counter}_{datetime.now().strftime('%H%M%S')}"

        self.active_connections[connection_id] = websocket

        logger.info(f"WebSocket connected: {connection_id}")

        # Send connected message
        await self.send_message(
            connection_id,
            MessageType.CONNECTED,
            {'connection_id': connection_id}
        )

        return connection_id

    def disconnect(self, connection_id: str):
        """
        Remove a connection.

        Args:
            connection_id: Connection to remove
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_message(
        self,
        connection_id: str,
        message_type: MessageType,
        data: Dict[str, Any]
    ):
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection
            message_type: Type of message
            data: Message data
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Connection {connection_id} not found")
            return

        message = StreamMessage(
            type=message_type,
            timestamp=datetime.now().isoformat(),
            data=data
        )

        try:
            await self.active_connections[connection_id].send_text(message.to_json())
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.disconnect(connection_id)

    async def broadcast(self, message_type: MessageType, data: Dict[str, Any]):
        """
        Broadcast a message to all connections.

        Args:
            message_type: Type of message
            data: Message data
        """
        disconnected = []

        for connection_id in self.active_connections:
            try:
                await self.send_message(connection_id, message_type, data)
            except Exception:
                disconnected.append(connection_id)

        # Clean up disconnected
        for conn_id in disconnected:
            self.disconnect(conn_id)

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


class StreamingSummarizer:
    """
    Streaming summarization with WebSocket delivery.

    Processes articles and streams summaries as they complete,
    providing real-time progress updates.
    """

    def __init__(
        self,
        base_summarizer,
        connection_manager: ConnectionManager
    ):
        """
        Initialize streaming summarizer.

        Args:
            base_summarizer: Underlying summarizer (FastSummarizer)
            connection_manager: WebSocket connection manager
        """
        self.base_summarizer = base_summarizer
        self.connection_manager = connection_manager
        logger.info("Initialized StreamingSummarizer")

    async def stream_summaries(
        self,
        connection_id: str,
        articles: List[Dict[str, str]],
        max_concurrent: int = 3,
        model: Optional[str] = None,
        style: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Stream summaries to a WebSocket connection as they complete.

        Args:
            connection_id: WebSocket connection ID
            articles: List of articles to summarize
            max_concurrent: Maximum concurrent summarizations
            model: Optional model override
            style: Summary style

        Returns:
            List of all summaries (also streamed individually)
        """
        total = len(articles)
        completed = 0
        results = []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_article(index: int, article: Dict[str, str]):
            """Process a single article and stream result."""
            nonlocal completed

            async with semaphore:
                try:
                    text = article.get('text', article.get('content', ''))
                    title = article.get('title', '')
                    url = article.get('url', '')

                    # Run summarization in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    summary = await loop.run_in_executor(
                        None,
                        lambda: self.base_summarizer.summarize(
                            text=text,
                            title=title,
                            url=url,
                            model=model,
                            style=style
                        )
                    )

                    completed += 1
                    progress = completed / total

                    # Stream the summary
                    await self.connection_manager.send_message(
                        connection_id,
                        MessageType.SUMMARY,
                        {
                            'index': index,
                            'article_url': url,
                            'article_title': title,
                            'summary': summary,
                            'progress': progress,
                            'completed': completed,
                            'total': total
                        }
                    )

                    return {
                        'index': index,
                        'original': article,
                        'summary': summary,
                        'success': True
                    }

                except Exception as e:
                    completed += 1
                    progress = completed / total

                    logger.error(f"Error summarizing article {index}: {e}")

                    # Stream the error
                    await self.connection_manager.send_message(
                        connection_id,
                        MessageType.ERROR,
                        {
                            'index': index,
                            'article_url': article.get('url', ''),
                            'article_title': article.get('title', ''),
                            'error': str(e),
                            'progress': progress,
                            'completed': completed,
                            'total': total
                        }
                    )

                    return {
                        'index': index,
                        'original': article,
                        'error': str(e),
                        'success': False
                    }

        # Send initial progress
        await self.connection_manager.send_message(
            connection_id,
            MessageType.PROGRESS,
            {
                'progress': 0,
                'completed': 0,
                'total': total,
                'status': 'starting'
            }
        )

        # Process all articles
        tasks = [
            process_article(i, article)
            for i, article in enumerate(articles)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter results
        successful = []
        failed = []
        for result in results:
            if isinstance(result, Exception):
                failed.append(str(result))
            elif result.get('success'):
                successful.append(result)
            else:
                failed.append(result)

        # Send completion message
        await self.connection_manager.send_message(
            connection_id,
            MessageType.COMPLETE,
            {
                'total': total,
                'successful': len(successful),
                'failed': len(failed),
                'duration_estimate': 'completed'
            }
        )

        return successful

    async def stream_with_clustering(
        self,
        connection_id: str,
        articles: List[Dict[str, str]],
        clusterer,
        max_concurrent: int = 3,
        model: Optional[str] = None,
        style: str = "default"
    ) -> Dict[str, Any]:
        """
        Stream summaries with clustering updates.

        Provides both individual summary streaming and cluster updates
        as processing progresses.

        Args:
            connection_id: WebSocket connection ID
            articles: Articles to process
            clusterer: Clustering instance
            max_concurrent: Max concurrent operations
            model: Optional model override
            style: Summary style

        Returns:
            Dictionary with summaries and clusters
        """
        # First, do quick clustering
        await self.connection_manager.send_message(
            connection_id,
            MessageType.PROGRESS,
            {
                'status': 'clustering',
                'message': 'Organizing articles into topics...'
            }
        )

        # Run clustering
        loop = asyncio.get_event_loop()
        clusters = await loop.run_in_executor(
            None,
            lambda: clusterer.cluster_articles(articles)
        )

        # Send initial clusters
        await self.connection_manager.send_message(
            connection_id,
            MessageType.CLUSTER_UPDATE,
            {
                'clusters': [
                    {
                        'topic': getattr(cluster, 'topic', f'Cluster {i+1}'),
                        'article_count': len(cluster) if hasattr(cluster, '__len__') else len(cluster.all_articles) if hasattr(cluster, 'all_articles') else 0,
                        'articles': []  # Will be populated as summaries stream
                    }
                    for i, cluster in enumerate(clusters)
                ],
                'status': 'initial'
            }
        )

        # Now stream summaries
        summaries = await self.stream_summaries(
            connection_id=connection_id,
            articles=articles,
            max_concurrent=max_concurrent,
            model=model,
            style=style
        )

        return {
            'clusters': clusters,
            'summaries': summaries
        }


# WebSocket route handler
async def websocket_summarize_handler(
    websocket: WebSocket,
    connection_manager: ConnectionManager,
    streaming_summarizer: StreamingSummarizer
):
    """
    Handle WebSocket summarization requests.

    Protocol:
    1. Client connects
    2. Server sends 'connected' message with connection_id
    3. Client sends JSON with 'articles' array
    4. Server streams 'summary' messages as articles are processed
    5. Server sends 'complete' message when done
    6. Connection closes

    Args:
        websocket: WebSocket connection
        connection_manager: Connection manager instance
        streaming_summarizer: Streaming summarizer instance
    """
    connection_id = await connection_manager.connect(websocket)

    try:
        while True:
            # Wait for request from client
            data = await websocket.receive_json()

            articles = data.get('articles', [])
            if not articles:
                await connection_manager.send_message(
                    connection_id,
                    MessageType.ERROR,
                    {'error': 'No articles provided'}
                )
                continue

            # Process options
            options = {
                'max_concurrent': data.get('max_concurrent', 3),
                'model': data.get('model'),
                'style': data.get('style', 'default')
            }

            # Stream summaries
            await streaming_summarizer.stream_summaries(
                connection_id=connection_id,
                articles=articles,
                **options
            )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        await connection_manager.send_message(
            connection_id,
            MessageType.ERROR,
            {'error': str(e)}
        )
    finally:
        connection_manager.disconnect(connection_id)


class HeartbeatManager:
    """
    Manages WebSocket heartbeats to keep connections alive.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        interval: float = 30.0
    ):
        """
        Initialize heartbeat manager.

        Args:
            connection_manager: Connection manager
            interval: Heartbeat interval in seconds
        """
        self.connection_manager = connection_manager
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the heartbeat loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Started heartbeat manager (interval: {self.interval}s)")

    async def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped heartbeat manager")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to all connections."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)

                if self.connection_manager.connection_count > 0:
                    await self.connection_manager.broadcast(
                        MessageType.HEARTBEAT,
                        {'connections': self.connection_manager.connection_count}
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")


# Factory functions
def create_connection_manager() -> ConnectionManager:
    """Create a new connection manager."""
    return ConnectionManager()


def create_streaming_summarizer(
    base_summarizer,
    connection_manager: ConnectionManager
) -> StreamingSummarizer:
    """
    Create a streaming summarizer.

    Args:
        base_summarizer: Base summarizer instance
        connection_manager: Connection manager instance

    Returns:
        StreamingSummarizer instance
    """
    return StreamingSummarizer(base_summarizer, connection_manager)
