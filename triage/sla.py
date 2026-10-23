"""Compute first-response SLA status from fixture targets and unanswered-message age."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triage.schemas import Plan, SlaStatus, Ticket, Urgency


@dataclass(frozen=True)
class SlaTier:
    name: str
    response_minutes: dict[Urgency, int]  # first-human-response target per urgency


class SlaPolicy:
    def __init__(self, tiers: dict[Plan, SlaTier]) -> None:
        self._tiers = tiers

    @classmethod
    def from_records(cls, records: dict[str, Any]) -> "SlaPolicy":
        return cls(
            {
                Plan(plan): SlaTier(
                    name=rec["tier"],
                    response_minutes={Urgency(u): int(m) for u, m in rec["response_minutes"].items()},
                )
                for plan, rec in records.items()
            }
        )

    def describe(self, plan: Plan) -> str | None:
        tier = self._tiers.get(plan)
        if tier is None:
            return None
        targets = ", ".join(f"{u.value} {m}m" for u, m in sorted(tier.response_minutes.items(), key=lambda kv: -kv[0].rank))
        return f"{tier.name} (first-response targets: {targets})"

    def evaluate(self, ticket: Ticket, urgency: Urgency) -> SlaStatus | None:
        """Waiting time vs. target. remaining < 0 means the promise was already broken."""
        tier = self._tiers.get(ticket.customer.plan)
        if tier is None:
            return None
        target = tier.response_minutes[urgency]
        waiting = ticket.waiting_minutes()
        remaining = target - waiting
        return SlaStatus(
            tier=tier.name, urgency=urgency, target_minutes=target,
            waiting_minutes=waiting, remaining_minutes=remaining, breached=remaining < 0,
        )
