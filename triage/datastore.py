"""Load local fixtures and validate tickets independently. Live backends require tool adapters and authentication."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from triage.schemas import Ticket
from triage.privacy import redact
from triage.tools.retriever import Article


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8-sig") as fh:  # utf-8-sig also accepts files with a BOM
        return json.load(fh)


def load_tickets(path: Path) -> list[Ticket]:
    return [Ticket.model_validate(item) for item in load_json(path)]


def load_articles(path: Path) -> list[Article]:
    return [
        Article(id=a["id"], title=a["title"], body=a["body"], tags=tuple(a.get("tags", [])))
        for a in load_json(path)
    ]


@dataclass(frozen=True)
class RejectedInput:
    index: int
    raw: Any
    error: str


def parse_tickets(items: Sequence[Any]) -> tuple[list[Ticket], list[RejectedInput]]:
    """Validate each entry on its own so one malformed ticket cannot block the rest."""
    tickets: list[Ticket] = []
    rejected: list[RejectedInput] = []
    for index, item in enumerate(items):
        try:
            tickets.append(Ticket.model_validate(item))
        except Exception as exc:  # noqa: BLE001 - any parse/validation problem rejects only this entry
            rejected.append(RejectedInput(index, redact(item), type(exc).__name__ + ": invalid ticket schema"))
    return tickets, rejected


def load_tickets_lenient(path: Path) -> tuple[list[Ticket], list[RejectedInput]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of tickets")
    return parse_tickets(data)
