"""Event bus for real-time WebSocket push notifications.

Singleton that broadcasts JSON events to connected WebSocket clients.
Thread-safe: works from both async handlers and sync background threads
(e.g. Dream Engine) via asyncio.run_coroutine_threadsafe.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("cerebro-events")


class EventBus:
    """Lightweight pub/sub for WebSocket clients."""

    def __init__(self):
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running event loop (call from async startup)."""
        self._loop = loop

    def register(self, ws) -> None:
        self._clients.add(ws)
        logger.info(f"WS client registered ({len(self._clients)} total)")

    def unregister(self, ws) -> None:
        self._clients.discard(ws)
        logger.info(f"WS client unregistered ({len(self._clients)} total)")

    def emit(self, event_type: str, data: dict | None = None) -> None:
        """Broadcast an event to all connected clients.

        Safe to call from any thread. If called from a non-async thread
        (e.g. Dream Engine background thread), schedules the broadcast
        on the captured event loop.
        """
        message = json.dumps({
            "type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        })

        if self._loop is None:
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            # Already in the event loop — schedule directly
            asyncio.ensure_future(self._broadcast(message), loop=self._loop)
        else:
            # Called from a sync/background thread
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: str) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._clients.discard(ws)


# Module-level singleton
event_bus = EventBus()
