"""Console rendering of triage results."""
from __future__ import annotations

import textwrap
from typing import Any

from triage.schemas import ResultStatus, TriageResult


def result_to_dict(result: TriageResult, include_traces: bool = False) -> dict[str, Any]:
    """JSON-ready dict. Raw tool outputs are large and may hold customer data, so opt-in only."""
    data = result.model_dump(mode="json")
    if not include_traces:
        for call in data["tool_calls"]:
            call.pop("output", None)
    return data


def render(result: TriageResult) -> str:
    lines = [f"{'=' * 70}", f"Ticket {result.ticket_id}", f"{'=' * 70}"]
    a = result.analysis
    route = f" -> {result.route_to.value}" if result.route_to else ""
    if result.status is not ResultStatus.OK:
        lines.append(f"STATUS    : {result.status.value.upper()}: {result.status_reason} (safe fallback, no auto-reply)")
        if result.status is ResultStatus.DEGRADED:
            lines.append("            Urgency below is a placeholder, not an assessment of the ticket.")
    lines.append(f"Urgency   : {result.urgency.value.upper()}")
    lines.append(f"Resolution: {result.resolution_status}")
    for limitation in result.limitations:
        lines.append(f"Limitation: {limitation}")
    lines.append(f"Action    : {result.action.value}{route}")
    if a:
        lines.append(f"Product   : {a.product}")
        issues = ", ".join([a.primary_issue_type.value, *(i.value for i in a.secondary_issue_types)])
        lines.append(f"Issue     : {issues}")
        lines.append(f"Sentiment : {a.customer_sentiment.value}   Language: {a.customer_language}")
        lines.append(f"Confidence: {a.confidence:.2f} (uncalibrated)   KB: {', '.join(a.kb_article_ids) or '-'}")
        if a.customer_deadline_minutes is not None:
            lines.append(f"Deadline  : customer reported within {a.customer_deadline_minutes} min of their message")
            remaining = result.customer_deadline_remaining_minutes
            if remaining is not None:
                lines.append(f"Remaining : {remaining} min at processing (negative = deadline passed)")
        if a.feature_requests:
            lines.append(f"Feature requests: {'; '.join(a.feature_requests)}")
    if result.sla:
        s = result.sla
        state = f"BREACHED by {-s.remaining_minutes} min" if s.breached else f"{s.remaining_minutes} min left"
        lines.append(f"SLA       : {s.tier} / {s.urgency.value}: target {s.target_minutes} min, waiting {s.waiting_minutes} min -> {state}")
    tools = ", ".join(f"{t.name}{'' if t.ok else ' (FAILED)'}" for t in result.tool_calls)
    lines.append(f"Tools     : {tools or '-'}")
    if result.planned_actions:
        names = ", ".join(_describe_action(a) for a in result.planned_actions)
        lines.append(f"Actions   : {names}  (--execute writes local files only)")
        for planned in result.planned_actions:
            if planned.payload.get("attention_reason"):
                lines.append(f"Attention : {planned.payload['attention_reason']}")
    for note in result.input_notes:
        lines.append(f"Input note: {note}")
    if result.digest:
        lines.append(_block("Digest for the person picking this up", result.digest))
    for issue in result.reply_issues:
        lines.append(f"Reply check: FLAGGED [{issue.rule}] \"{issue.excerpt}\" - {issue.reason}")
    for adj in result.adjustments:
        lines.append(f"Policy    : [{adj.rule}] {adj.field}: {adj.before} -> {adj.after} ({adj.reason})")
    if a:
        lines.append(_block("Reasoning", a.reasoning))
        if a.handoff_summary:
            lines.append(_block("Handoff summary", a.handoff_summary))
        if a.draft_reply:
            label = "Reply (eligible for local outbox; not sent)" if result.auto_send else "Draft reply (NOT sent, needs human)"
            lines.append(_block(label, a.draft_reply))
    return "\n".join(lines)


def _describe_action(action) -> str:
    detail = action.payload.get("team") if action.kind.value == "create_handoff" else None
    return f"{action.kind.value}({detail})" if detail else action.kind.value


def _block(title: str, text: str) -> str:
    body = textwrap.fill(text, width=90, initial_indent="  ", subsequent_indent="  ")
    return f"{title}:\n{body}"


def summary_table(results: list[TriageResult]) -> str:
    rows = [f"{'Ticket':<10}{'Status':<10}{'Urgency':<10}{'Action':<22}{'Route'}"]
    for r in results:
        route = r.route_to.value if r.route_to else "-"
        rows.append(f"{r.ticket_id:<10}{r.status.value:<10}{r.urgency.value:<10}{r.action.value:<22}{route}")
    return "\n".join(rows)
