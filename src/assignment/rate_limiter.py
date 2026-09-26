"""
Assignment 11 — Rate Limiter.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""

from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 60,
    ):
        super().__init__(name="rate_limiter")

        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # Each user gets an independent queue of request timestamps.
        self.user_windows: dict[str, deque] = defaultdict(deque)

        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=message)
            ],
        )

    async def on_user_message_callback(
        self,
        *,
        invocation_context,
        user_message,
    ):
        """Return Content to block, or None to allow."""

        self.total_count += 1

        # Identify the user. Fall back to anonymous when ADK does not
        # provide a user ID.
        user_id = (
            getattr(invocation_context, "user_id", None)
            or "anonymous"
        )

        now = time.time()
        window = self.user_windows[user_id]

        # --------------------------------------------------------
        # 1. Remove timestamps outside the sliding window.
        # --------------------------------------------------------
        cutoff = now - self.window_seconds

        while window and window[0] <= cutoff:
            window.popleft()

        # --------------------------------------------------------
        # 2. Block if this user has reached the limit.
        # --------------------------------------------------------
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])

            # Avoid displaying "-0s" because of floating-point
            # rounding near the end of the window.
            wait = max(0, wait)

            self.blocked_count += 1

            return self._block_response(
                f"Rate limit exceeded. "
                f"Try again in {wait:.0f}s."
            )

        # --------------------------------------------------------
        # 3. Request is allowed: record its timestamp.
        # --------------------------------------------------------
        window.append(now)

        return None
