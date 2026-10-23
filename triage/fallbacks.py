"""Human-bound results for failed, bypassed or invalid input. Failure urgency is a review placeholder."""
from __future__ import annotations

from typing import Any, Sequence

from triage.schemas import (
    Action, ResultStatus, Team, Ticket, ToolTrace, TriageResult, Urgency,
)
from triage.sla import SlaPolicy
from triage.privacy import redact

_PREVIEW_CHARS = 300


def _shorten(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_digest(ticket: Ticket, traces: Sequence[ToolTrace] = ()) -> str:
    c = ticket.customer
    messages = ticket.chronological()
    latest = _shorten(redact(messages[-1].text)) if messages else "(no message)"
    tools = ", ".join(t.name for t in traces) or "none"
    return (
        f"Customer {c.customer_id} ({c.plan.value}, {c.region or 'region n/a'}), "
        f"waiting {ticket.waiting_minutes()} min. Latest message: \"{latest}\" "
        f"Tools already run: {tools}."
    )


def fallback_result(
    ticket: Ticket,
    *,
    status: ResultStatus,
    reason: str,
    urgency: Urgency,
    action: Action,
    route_to: Team | None = None,
    traces: Sequence[ToolTrace] = (),
    sla_policy: SlaPolicy | None = None,
) -> TriageResult:
    return TriageResult(
        ticket_id=ticket.id,
        urgency=urgency,
        action=action,
        route_to=route_to,
        analysis=None,
        status=status,
        status_reason=reason,
        digest=build_digest(ticket, traces),
        sla=sla_policy.evaluate(ticket, urgency) if sla_policy else None,
        tool_calls=list(traces),
    )


def degraded_result(
    ticket: Ticket, reason: str, traces: Sequence[ToolTrace] = (), sla_policy: SlaPolicy | None = None
) -> TriageResult:
    """The system failed, not the customer. Urgency HIGH is a placeholder so a human looks soon."""
    return fallback_result(
        ticket, status=ResultStatus.DEGRADED, reason=reason, urgency=Urgency.HIGH,
        action=Action.ESCALATE_TO_HUMAN, traces=traces, sla_policy=sla_policy,
    )


def rejected_result(index: int, raw: Any, error: str) -> TriageResult:
    """Input that could not even be parsed into a Ticket."""
    ticket_id = raw.get("id") if isinstance(raw, dict) and isinstance(raw.get("id"), str) else f"input-{index}"
    return TriageResult(
        ticket_id=ticket_id,
        urgency=Urgency.HIGH,
        action=Action.ESCALATE_TO_HUMAN,
        route_to=None,
        analysis=None,
        status=ResultStatus.REJECTED,
        status_reason=_shorten(error, 500),
        digest=f"Unparseable ticket input (entry {index}): {_shorten(str(redact(raw)))}",
    )
