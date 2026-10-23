"""Isolate each ticket's failures, plan actions, and retain retryable records."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from triage.actions import plan_actions
from triage.agent import TriageAgent
from triage.breaker import CircuitBreaker
from triage.datastore import RejectedInput
from triage.deadletter import DeadLetterWriter
from triage.fallbacks import degraded_result, fallback_result, rejected_result
from triage.preflight import Preflight
from triage.schemas import Action, ResultStatus, Team, Ticket, TriageResult, Urgency
from triage.sla import SlaPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Processed:
    """What the pipeline returns for one ticket."""

    result: TriageResult
    retryable: bool  # worth re-running later (a model/system problem, not a content problem)
    ticket: Ticket  # the version that was analysed (card numbers already masked)


class TicketPipeline:
    """Everything that happens to one ticket. Never raises."""

    def __init__(
        self,
        agent: TriageAgent,
        preflight: Preflight,
        breaker: CircuitBreaker,
        sla_policy: SlaPolicy | None = None,
    ) -> None:
        self._agent = agent
        self._preflight = preflight
        self._breaker = breaker
        self._sla_policy = sla_policy

    def process(self, ticket: Ticket) -> Processed:
        try:
            processed = self._process(ticket)
        except Exception as exc:  # noqa: BLE001 - last line of defence: even a bug here must not lose the ticket
            logger.warning("pipeline_failed error_type=%s", type(exc).__name__)
            failed = degraded_result(ticket, type(exc).__name__ + ": pipeline unavailable", sla_policy=self._sla_policy)
            processed = Processed(failed, retryable=True, ticket=ticket)
        # Final step for every outcome (success or fallback): decide what should happen next.
        processed.result.planned_actions = plan_actions(processed.result, processed.ticket.customer.customer_id)
        return processed

    def _process(self, ticket: Ticket) -> Processed:
        # Step A: clean the input (drop blanks, cap size, mask card numbers).
        pre = self._preflight.run(ticket)
        if pre.bypass_reason:
            # Nothing readable to analyse: do not call the model, ask a person to get details.
            result = fallback_result(
                ticket, status=ResultStatus.BYPASSED, reason=pre.bypass_reason, urgency=Urgency.LOW,
                action=Action.ROUTE_TO_SPECIALIST, route_to=Team.TECHNICAL_SUPPORT, sla_policy=self._sla_policy,
            )
            return Processed(result, retryable=False, ticket=ticket)

        # Step B: if the model has been failing repeatedly, stop calling it (circuit open).
        if self._breaker.is_open:
            reason = f"Model circuit open after {self._breaker.consecutive_failures} consecutive failures."
            result = fallback_result(
                pre.ticket, status=ResultStatus.BYPASSED, reason=reason, urgency=Urgency.HIGH,
                action=Action.ESCALATE_TO_HUMAN, sla_policy=self._sla_policy,
            )
            result.input_notes = pre.notes
            return Processed(result, retryable=True, ticket=pre.ticket)

        # Step C: normal path. Run the agent, then tell the breaker how it went.
        result = self._agent.triage(pre.ticket)
        result.input_notes = pre.notes
        if result.action is Action.AUTO_RESPOND and any(
                "truncated" in note or "messages kept" in note for note in pre.notes):
            result.action = Action.ROUTE_TO_SPECIALIST
            result.route_to = Team.TECHNICAL_SUPPORT
            result.input_notes.append("Incomplete conversation requires human review.")
        if result.status is ResultStatus.OK:
            self._breaker.record_success()
            return Processed(result, retryable=False, ticket=pre.ticket)
        self._breaker.record_failure()
        return Processed(result, retryable=True, ticket=pre.ticket)


@dataclass
class BatchReport:
    results: list[TriageResult] = field(default_factory=list)
    breaker_opened: bool = False
    persistence_failures: int = 0
    dead_lettered: int = 0
    failures: int = 0  # tickets lost to system problems: rejected input, model failure, open circuit

    def counts(self) -> Counter:
        """How many results have each status (ok / degraded / bypassed / rejected)."""
        return Counter(r.status.value for r in self.results)

    @property
    def failure_ratio(self) -> float:
        """Share of tickets lost to system problems (not content problems such as a blank message)."""
        return self.failures / len(self.results) if self.results else 0.0

    def planned_actions(self):
        """Every action planned across all results (ready for the OutboxExecutor)."""
        return [a for r in self.results for a in r.planned_actions]


class BatchRunner:
    def __init__(self, pipeline: TicketPipeline, breaker: CircuitBreaker, dead_letter: DeadLetterWriter | None = None) -> None:
        self._pipeline = pipeline
        self._breaker = breaker
        self._dead_letter = dead_letter

    def run(self, tickets: Sequence[Ticket], rejected: Sequence[RejectedInput] = ()) -> BatchReport:
        report = BatchReport()

        # Entries that could not even be parsed into tickets: report and save them.
        for item in rejected:
            result = rejected_result(item.index, item.raw, item.error)
            result.planned_actions = plan_actions(result, None)
            report.results.append(result)
            report.failures += 1
            self._dead_letter_write(report, result, item.raw)

        # Normal tickets: one by one, each isolated from the others.
        for ticket in tickets:
            processed = self._pipeline.process(ticket)
            report.results.append(processed.result)
            if processed.retryable:
                report.failures += 1
                self._dead_letter_write(report, processed.result, processed.ticket.model_dump(mode="json"))

        report.breaker_opened = self._breaker.is_open
        return report

    def _dead_letter_write(self, report: BatchReport, result: TriageResult, payload: Any) -> None:
        if self._dead_letter is None:
            return
        try:
            self._dead_letter.write(result.ticket_id, result.status.value, result.status_reason or "", payload)
            report.dead_lettered += 1
        except OSError:
            report.persistence_failures += 1
            result.input_notes.append("Dead-letter persistence failed; retain this result for manual recovery.")
            logger.error("dead_letter_write_failed")
