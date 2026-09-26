"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class AuditLogPlugin:
    """Framework-agnostic audit logger.

    Wire this into ADK callbacks or another application pipeline.
    The logger records events but never blocks requests itself.
    """

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []

        # request_id/user_id -> monotonic start time.
        # time.monotonic() is used for latency because it is not
        # affected by system clock adjustments.
        self._open: dict[str, float] = {}

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: str | None = None,
    ):
        """Store an input event and start its latency timer."""

        # If request_id is not supplied, use user_id as the key so
        # record_output() can still calculate latency.
        key = request_id or user_id

        self._open[key] = __import__("time").monotonic()

        self.logs.append(
            {
                "event": "input",
                "timestamp": utc_now_iso(),
                "user_id": user_id,
                "request_id": request_id,
                "text": text,
            }
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store an output event, decision, and request latency."""

        import time

        key = request_id or user_id

        start = self._open.pop(key, None)

        latency_ms = None
        if start is not None:
            latency_ms = round(
                (time.monotonic() - start) * 1000,
                2,
            )

        self.logs.append(
            {
                "event": "output",
                "timestamp": utc_now_iso(),
                "user_id": user_id,
                "request_id": request_id,
                "text": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk as a JSON array.

        By default the file is written to:
            <repo>/outputs/audit_log.json
        """

        path = Path(
            filepath or default_audit_log_path()
        )

        # Make sure outputs/ or any explicitly supplied parent
        # directory exists.
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.logs,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return str(path)
