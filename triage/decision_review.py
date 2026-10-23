"""Conservative evidence review after model policy; never treats advice as resolution."""
from dataclasses import dataclass
from triage.approved_replies import approved_reply
from triage.schemas import Action, Team, Urgency, PolicyAdjustment, Ticket, ToolTrace, TriageAnalysis

@dataclass(frozen=True)
class Review:
    """Final decision plus provenance and explicit uncertainty for the reviewer."""
    resolution: str
    evidence: list[dict[str, str]]
    limitations: list[str]
    urgency: Urgency
    action: Action
    team: Team | None
    adjustments: list[PolicyAdjustment]

def _resolution_status(text: str, latest: str) -> str:
    """Customer confirmation is evidence of a report, never independent verification."""
    confirmed = any(phrase in latest for phrase in (
        "it works now", "problem is resolved", "ใช้งานได้แล้ว", "แก้ได้แล้ว",
    ))
    persists = any(phrase in latest for phrase in (
        "still not", "still doesn't", "still fails", "ยังไม่", "ยังเข้าไม่ได้",
        "ยังขึ้นว่าหมดอายุ", "still stuck", "remains stuck",
    ))
    malfunction = any(phrase in text for phrase in (
        "still shows light", "does not work", "doesn't work", "not working",
        "cannot access", "error 500", "เข้าไม่ได้", "ไม่ทำงาน",
        "has been stuck", "is stuck", "remains stuck", "still stuck",
        "expired immediately", "expires immediately", "หมดอายุทันที", "ยังขึ้นว่าหมดอายุ",
    ))
    if confirmed and not persists:
        return "customer_reported_resolved"
    return "unresolved" if malfunction or persists else "not_established"


def _evidence(ticket: Ticket, traces: list[ToolTrace]) -> list[dict[str, str]]:
    """Keep source indices stable so a reviewer can inspect the actual observation."""
    messages = [
        {"source": f"message:{index}", "kind": "customer_report", "text": message.text}
        for index, message in enumerate(ticket.chronological())
    ]
    tools = [
        {
            "source": f"tool:{index}:{trace.name}",
            "kind": "tool_result" if trace.ok else "tool_failure",
            "text": "See corresponding tool_calls entry; local fixture data is synthetic.",
        }
        for index, trace in enumerate(traces)
    ]
    return messages + tools


def review(ticket: Ticket, analysis: TriageAnalysis, traces: list[ToolTrace],
           urgency: Urgency, action: Action, team: Team | None) -> Review:
    """Evidence references describe observations, not calibrated model truth."""
    ordered = ticket.chronological()
    text = " ".join(message.text for message in ordered).casefold()
    latest = ordered[-1].text.casefold() if ordered else ""
    evidence = _evidence(ticket, traces)
    resolution = _resolution_status(text, latest)
    limitations = ["Resolution is based on customer reports, not independent execution.",
                   "Tool fixtures are synthetic; no live payment or regional health verification.",
                   "Confidence is an uncalibrated model self-assessment, not a probability of correctness."]
    adjustments: list[PolicyAdjustment] = []

    def raise_decision(
        rule: str,
        reason: str,
        floor: Urgency,
        minimum_action: Action,
        owner: Team,
    ) -> None:
        """Record why human ownership or urgency increased; never downgrade a risk."""
        nonlocal urgency, action, team
        for field, old, new in (("urgency", urgency, floor), ("action", action, minimum_action)):
            if new.rank > old.rank:
                adjustments.append(PolicyAdjustment(rule=rule, field=field, before=old.value,
                    after=new.value, reason=reason))
                if field == "urgency":
                    urgency = floor
                else:
                    action = minimum_action
        if team != owner:
            adjustments.append(PolicyAdjustment(rule=rule, field="route_to",
                before=team.value if team else "none", after=owner.value, reason=reason))
            team = owner
    # Narrow independent signals; do not equate seat capacity with affected users.
    outage = ("error 500" in text or "cannot access" in text or "เข้าไม่ได้" in text)
    multiple = any(p in text for p in ("multiple users", "all users", "เพื่อนร่วมงาน", "หลายคน"))
    if outage and multiple and resolution != "customer_reported_resolved":
        raise_decision("reported_multi_user_outage", "Customer reports multi-user access failure.",
                       Urgency.CRITICAL, Action.ESCALATE_TO_HUMAN, Team.INCIDENT_RESPONSE)
    if any(p in latest for p in ("my api key was leaked", "my account was hacked", "บัญชีถูกแฮ็ก")):
        raise_decision("reported_security_incident", "Explicit customer report of compromise.",
                       Urgency.CRITICAL, Action.ESCALATE_TO_HUMAN, Team.SECURITY)
    if resolution == "unresolved" and action is Action.AUTO_RESPOND:
        raise_decision("unresolved_functional_issue",
                       "Documentation offers advice; the customer has not confirmed resolution.",
                       Urgency.MEDIUM, Action.ROUTE_TO_SPECIALIST, Team.TECHNICAL_SUPPORT)
    if action is Action.AUTO_RESPOND:
        # No unrestricted generated reply is approved merely because it cites an article.
        # Approved replies are stored separately and must match actually retrieved content.
        approved = approved_reply(ticket, traces)
        if approved is None:
            raise_decision("unverified_reply_claims", "No approved response matches this request and retrieved evidence.",
                           urgency, Action.ROUTE_TO_SPECIALIST, Team.TECHNICAL_SUPPORT)
        else:
            analysis.draft_reply = approved
    if resolution == "unresolved":
        # Replace unsupported resolution claims in internal prose, preserving other evidence in traces.
        for field in ("reasoning", "handoff_summary"):
            value = getattr(analysis, field) or ""
            if any(p in value.casefold() for p in ("no unresolved malfunction", "issue is resolved", "problem is resolved")):
                setattr(analysis, field, "Customer reports a functional issue. Documentation provides proposed troubleshooting; resolution remains unverified.")
                limitations.append("Unsupported model resolution claim replaced with evidence-based wording.")
    return Review(resolution, evidence, limitations, urgency, action, team, adjustments)
