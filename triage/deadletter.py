"""The "dead-letter" file: a safety box for tickets that failed.

When the AI/system fails on a ticket, the ticket is appended to failed_tickets.jsonl (one JSON
object per line, with the reason). Later you can re-run just those tickets:
    python -m triage --retry-failed failed_tickets.jsonl
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from triage.privacy import redact


class DeadLetterWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def write(self, ticket_id: str, status: str, reason: str, payload: Any) -> None:
        record = {
            "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticket_id": ticket_id,
            "status": status,
            "reason": reason,
            "ticket": payload,
        }
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(redact(record), ensure_ascii=False) + "\n")


def load_dead_letter(path: Path) -> list[Any]:
    """Return the original ticket payloads, ready to be validated and re-run."""
    payloads: list[Any] = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                payloads.append(json.loads(line)["ticket"])
    return payloads
