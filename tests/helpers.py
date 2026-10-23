"""Shared test helpers: fixtures built from the sample data, plus the shared test doubles."""
from __future__ import annotations

from typing import Any

from triage.config import Settings
from triage.datastore import load_json, load_tickets
from triage.schemas import Ticket
from triage.sla import SlaPolicy
from triage.testing import (  # noqa: F401 - re-exported for the tests
    OutcomeLLM, ScriptedLLM, ScriptedSession, calls, final, make_analysis, tool_call,
)

SETTINGS = Settings()
TICKETS: dict[str, Ticket] = {t.id: t for t in load_tickets(SETTINGS.data_dir / "tickets.json")}
SLA = SlaPolicy.from_records(load_json(SETTINGS.data_dir / "sla.json"))


def clone_ticket(ticket_id: str, source: str = "T-003", **customer_overrides: Any) -> Ticket:
    base = TICKETS[source]
    customer = base.customer.__class__(**{**base.customer.model_dump(), **customer_overrides})
    return Ticket(id=ticket_id, customer=customer, messages=list(base.messages),
                  last_agent_reply_minutes_ago=base.last_agent_reply_minutes_ago)
