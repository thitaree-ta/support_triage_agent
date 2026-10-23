"""Bounded model/tool orchestration with deterministic policy and evidence review.

Failures produce human-bound results; this component does not deliver replies."""
from __future__ import annotations

import logging
import time
from typing import Callable

from triage.fallbacks import degraded_result
from triage.llm import LLMClient
from triage.policy import PolicyEngine
from triage.decision_review import review
from triage.prompts import render_ticket
from triage.schemas import Ticket, ToolTrace, TriageAnalysis, TriageResult
from triage.sla import SlaPolicy
from triage.tools import ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """The model did not produce a usable analysis (or ran out of steps or time)."""


class TriageAgent:
    def __init__(
        self,
        llm: LLMClient,  # the AI behind an interface (OpenAI in production, a fake in tests)
        registry: ToolRegistry,  # the tools the AI may call
        policy: PolicyEngine,  # the safety rules applied afterwards
        system_prompt: str,  # the written instructions given to the AI
        max_steps: int = 6,  # hard limit on AI <-> tool rounds
        sla_policy: SlaPolicy | None = None,  # promised response times
        ticket_timeout_seconds: float | None = None,  # time budget per ticket (None = unlimited)
        max_tool_calls: int = 12,
        clock: Callable[[], float] = time.monotonic,  # injectable so tests can fake time
    ) -> None:
        if max_steps < 1 or max_tool_calls < 1:
            raise ValueError("Execution budgets must be positive")
        self._max_tool_calls = max_tool_calls
        self._llm = llm
        self._registry = registry
        self._policy = policy
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._sla_policy = sla_policy
        self._timeout = ticket_timeout_seconds
        self._clock = clock

    def triage(self, ticket: Ticket) -> TriageResult:
        """Analyse one ticket. Never raises: failures become a human-bound result."""
        traces: list[ToolTrace] = []  # a record of every tool call, kept even if we fail later
        try:
            # Step 1: let the model investigate and give its opinion.
            analysis = self._run_loop(ticket, traces)
            # Step 2: let our safety rules check (and possibly tighten) that opinion.
            outcome = self._policy.apply(ticket, analysis, traces, self._sla_policy)
            reviewed = review(ticket, analysis, traces, outcome.urgency, outcome.action, outcome.route_to)
        except Exception as exc:  # noqa: BLE001 - a ticket must never be dropped on failure
            logger.warning("triage_failed error_type=%s", type(exc).__name__)
            return degraded_result(ticket, str(exc) if isinstance(exc, AgentError) else type(exc).__name__ + ": analysis unavailable", traces, self._sla_policy)

        # Step 3: package everything into the final result.
        return TriageResult(
            ticket_id=ticket.id,
            urgency=reviewed.urgency,
            action=reviewed.action,
            route_to=reviewed.team,
            resolution_status=reviewed.resolution,
            evidence=reviewed.evidence,
            limitations=reviewed.limitations,
            analysis=analysis,
            adjustments=outcome.adjustments + reviewed.adjustments,
            reply_issues=outcome.reply_issues,
            sla=self._sla_policy.evaluate(ticket, reviewed.urgency) if self._sla_policy else None,
            # The model's deadline is relative to the latest message, not processing time.
            customer_deadline_remaining_minutes=(
                analysis.customer_deadline_minutes - min(m.minutes_ago for m in ticket.messages)
                if analysis.customer_deadline_minutes is not None and ticket.messages else None
            ),
            tool_calls=traces,
        )

    def _run_loop(self, ticket: Ticket, traces: list[ToolTrace]) -> TriageAnalysis:
        """The model <-> tools conversation. Returns the model's final analysis."""
        ctx = ToolContext(ticket)  # tools learn WHICH customer from here, never from the model
        deadline = None if self._timeout is None else self._clock() + self._timeout
        session = self._llm.start_session(
            self._system_prompt, render_ticket(ticket, self._sla_policy), self._registry.specs()
        )
        tool_results: list[ToolResult] = []  # answers to hand back on the next round

        for step in range(1, self._max_steps + 1):
            # Check the clock between rounds (a request already in flight is limited separately).
            if deadline is not None and self._clock() > deadline:
                raise AgentError(f"Ticket time budget of {self._timeout:g}s exceeded.")

            # On the LAST round we switch tools off to force a final answer.
            allow_tools = step < self._max_steps
            turn = session.next(tool_results, allow_tools=allow_tools)
            if deadline is not None and self._clock() > deadline:
                raise AgentError("Ticket time budget exceeded after model response.")

            # Case 1: the model gave its final answer -> done.
            if not turn.tool_calls:
                if turn.analysis is None:
                    raise AgentError("Model returned neither tool calls nor a valid analysis.")
                return TriageAnalysis.model_validate(turn.analysis)
            if not allow_tools:
                raise AgentError("Model requested tools on the final step.")

            # Case 2: the model wants tools. Run each, remember the result, loop again.
            if len(traces) + len(turn.tool_calls) > self._max_tool_calls:
                raise AgentError("Tool call budget exceeded.")
            tool_results = []
            for call in turn.tool_calls:
                if deadline is not None and self._clock() > deadline:
                    raise AgentError("Ticket time budget exceeded before tool call.")
                result, trace = self._registry.execute(call, ctx)  # never raises
                tool_results.append(result)
                traces.append(trace)

        raise AgentError(f"No final answer after {self._max_steps} steps.")
