"""Scoring: compare one TriageResult with a case's expectations. Pure functions, no model calls."""
from __future__ import annotations

from typing import Any

from triage.schemas import ResultStatus, TriageResult, Urgency

KNOWN_KEYS = {
    "status", "urgency_at_least", "urgency_at_most", "actions", "route_to", "analysis_contains",
    "forbid_reply_contains", "feature_request_logged", "no_tool_errors", "input_notes_contain",
    "no_reply_issues",
}


def check(expect: dict[str, Any], result: TriageResult) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    failures: list[str] = []
    wanted_status = expect.get("status", "ok")
    if result.status.value != wanted_status:
        failures.append(f"status {result.status.value!r}, expected {wanted_status!r}"
                        + (f" ({result.status_reason})" if result.status_reason else ""))

    if "urgency_at_least" in expect and result.urgency.rank < Urgency(expect["urgency_at_least"]).rank:
        failures.append(f"urgency {result.urgency.value}, expected at least {expect['urgency_at_least']}")
    if "urgency_at_most" in expect and result.urgency.rank > Urgency(expect["urgency_at_most"]).rank:
        failures.append(f"urgency {result.urgency.value}, expected at most {expect['urgency_at_most']}")
    if "actions" in expect and result.action.value not in expect["actions"]:
        failures.append(f"action {result.action.value}, expected one of {expect['actions']}")
    if "route_to" in expect:
        route = result.route_to.value if result.route_to else None
        if route not in expect["route_to"]:
            failures.append(f"route_to {route}, expected one of {expect['route_to']}")

    if expect.get("no_reply_issues") and result.reply_issues:
        flagged = ", ".join(sorted({i.rule for i in result.reply_issues}))
        failures.append(f"draft reply was flagged by the reply guard ({flagged})")

    if expect.get("no_tool_errors") and any(not t.ok for t in result.tool_calls):
        failed = [t.name for t in result.tool_calls if not t.ok]
        failures.append(f"tool errors: {failed}")
    notes = " | ".join(result.input_notes).lower()
    for needle in expect.get("input_notes_contain", []):
        if needle.lower() not in notes:
            failures.append(f"input notes missing {needle!r}")

    analysis = result.analysis
    needs_analysis = any(k in expect for k in ("analysis_contains", "forbid_reply_contains", "feature_request_logged"))
    if needs_analysis and (analysis is None or result.status is not ResultStatus.OK):
        failures.append("no model analysis to check content against")
        return failures
    if analysis is not None:
        for field_name, needle in expect.get("analysis_contains", {}).items():
            value = getattr(analysis, field_name, None)
            text = str(getattr(value, "value", value)).lower()
            if needle.lower() not in text:
                failures.append(f"analysis.{field_name}={text!r} does not contain {needle!r}")
        reply = (analysis.draft_reply or "").lower()
        for needle in expect.get("forbid_reply_contains", []):
            if needle.lower() in reply:
                failures.append(f"draft reply contains forbidden text {needle!r}")
        if expect.get("feature_request_logged") and not analysis.feature_requests:
            failures.append("feature request was not logged")
    return failures
