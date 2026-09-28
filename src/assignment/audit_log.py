"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store an input and return its correlation id."""
        request_id = request_id or uuid.uuid4().hex
        self._open[request_id] = time.perf_counter()
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "input": text,
                "started_at": utc_now_iso(),
            }
        )
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching input record with its outcome."""
        request_id = request_id or uuid.uuid4().hex
        started = self._open.pop(request_id, None)
        latency_ms = round((time.perf_counter() - started) * 1000, 3) if started else None
        entry = next(
            (item for item in reversed(self.logs) if item.get("request_id") == request_id),
            None,
        )
        if entry is None:
            entry = {"request_id": request_id, "user_id": user_id}
            self.logs.append(entry)
        entry.update(
            {
                "user_id": user_id,
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "finished_at": utc_now_iso(),
                "latency_ms": latency_ms,
            }
        )
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
