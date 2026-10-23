"""Deterministic urgency and action floors with auditable adjustments.

Rules raise caution; they do not establish the truth of model-extracted facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from triage.reply_guard import ReplyGuard
from triage.schemas import (
    Action, IssueType, Plan, PolicyAdjustment, ReplyIssue, Sentiment, SlaStatus, Team, Ticket,
    ToolTrace, TriageAnalysis, Urgency,
)
from triage.sla import SlaPolicy
from triage.tools import knowledge_base, service_status


# ----------------------------------------------------------------- shared data types
@dataclass(frozen=True)
class PolicyContext:
    """Everything a rule is allowed to look at (rules never change it)."""

    ticket: Ticket
    analysis: TriageAnalysis  # what the model said
    urgency: Urgency  # urgency after the rules that already ran
    action: Action  # action after the rules that already ran
    tool_calls: Sequence[ToolTrace]  # what the tools returned (our evidence)
    sla: SlaStatus | None = None  # only known by the action rules (needs the final urgency)
    reply_issues: Sequence[ReplyIssue] = ()  # problems the ReplyGuard found in the draft


@dataclass(frozen=True)
class Verdict:
    """A rule's answer: 'this ticket needs at least this urgency / at least this action'."""

    reason: str
    min_urgency: Urgency | None = None
    min_action: Action | None = None


class Rule(Protocol):
    name: str

    def check(self, ctx: PolicyContext) -> Verdict | None:
        """Return a Verdict if the rule fires, otherwise None."""


@dataclass(frozen=True)
class PolicyOutcome:
    urgency: Urgency
    action: Action
    route_to: Team | None
    adjustments: list[PolicyAdjustment]
    sla: SlaStatus | None = None
    reply_issues: list[ReplyIssue] = field(default_factory=list)


# ------------------------------------------------------------------ helper functions
def issue_types(analysis: TriageAnalysis) -> set[IssueType]:
    """All topics the model found (the main one plus any extra ones)."""
    return {analysis.primary_issue_type, *analysis.secondary_issue_types}


def retrieved_kb_ids(tool_calls: Sequence[ToolTrace]) -> set[str]:
    """IDs of KB articles the knowledge-base tool REALLY returned (proof the model saw them)."""
    ids: set[str] = set()
    for trace in tool_calls:
        if trace.name == knowledge_base.TOOL_NAME and trace.ok and isinstance(trace.output, dict):
            ids.update(r["id"] for r in trace.output.get("results", []))
    return ids


def retrieved_kb_text(tool_calls: Sequence[ToolTrace]) -> str:
    """The text of those articles (used to allow facts that are quoted straight from the KB)."""
    parts: list[str] = []
    for trace in tool_calls:
        if trace.name == knowledge_base.TOOL_NAME and trace.ok and isinstance(trace.output, dict):
            parts.extend(r.get("content", "") for r in trace.output.get("results", []))
    return " ".join(parts)


# ======================================================================================
# URGENCY RULES: may raise urgency. They run first.
# ======================================================================================
class ReportedFinancialRisk:
    """Narrow explicit charge-dispute phrases, independent of model extraction."""
    name = "reported_financial_risk"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        text = " ".join(m.text for m in ctx.ticket.messages).casefold()
        if any(phrase in text for phrase in (
            "charged twice", "pending charges", "three charges", "charges aren't reversed",
            "หักซ้ำ", "ตัดเงินผมผิด", "ขอคืนเงินด่วน",
        )):
            return Verdict("Customer reports a charge dispute; settlement is unverified.",
                           min_urgency=Urgency.HIGH, min_action=Action.ESCALATE_TO_HUMAN)
        return None


class BillingEvidenceFloor:
    name = "billing_evidence_floor"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if any(t.ok and t.name == "get_billing_charges" and isinstance(t.output, dict)
               and (t.output.get("duplicate_charge_suspected") or t.output.get("billing_sync_error"))
               for t in ctx.tool_calls):
            return Verdict("Retrieved billing evidence requires investigation.",
                           min_urgency=Urgency.HIGH, min_action=Action.ESCALATE_TO_HUMAN)
        return None


class MoneyIssueFloor:
    """Charges, refunds and disputes are never casual: at least HIGH."""

    name = "money_issue_floor"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.analysis.involves_money:
            return Verdict("Charges/refunds/disputes are at least HIGH urgency.", min_urgency=Urgency.HIGH)
        return None


class RestrictedTopicFloor:
    """Security incidents are CRITICAL; legal/privacy matters are at least HIGH."""

    name = "restricted_topic_floor"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        types = issue_types(ctx.analysis)
        if IssueType.SECURITY in types:
            return Verdict("Security issue is CRITICAL.", min_urgency=Urgency.CRITICAL)
        if IssueType.LEGAL_PRIVACY in types:
            return Verdict("Legal/privacy matter is at least HIGH.", min_urgency=Urgency.HIGH)
        return None


class MultiUserOutageFloor:
    """An outage hitting more than one user is CRITICAL."""

    name = "multi_user_outage_floor"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        a = ctx.analysis
        if a.multiple_users_affected and IssueType.OUTAGE in issue_types(a):
            return Verdict("Outage affecting multiple users is CRITICAL.", min_urgency=Urgency.CRITICAL)
        return None


class OutageEvidenceFloor:
    """Safety net for a model that MISSES an outage.

    If monitoring (fetched by the status tool) shows the region is unhealthy and the ticket is
    about access or an outage, the ticket is at least HIGH, whatever the model concluded.
    """

    name = "outage_evidence_floor"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.analysis.primary_issue_type not in {IssueType.OUTAGE, IssueType.ACCOUNT_ACCESS}:
            return None
        for trace in ctx.tool_calls:
            if (
                trace.name == service_status.TOOL_NAME
                and trace.ok
                and isinstance(trace.output, dict)
                and trace.output.get("internal_degraded") is True
            ):
                return Verdict("Internal monitoring shows the region is degraded.", min_urgency=Urgency.HIGH)
        return None


class ImminentDeadlineFloor:
    """A customer deadline close enough to miss makes the ticket at least HIGH."""

    name = "imminent_deadline_floor"

    def __init__(self, threshold_minutes: int = 240) -> None:
        self._threshold = threshold_minutes

    def check(self, ctx: PolicyContext) -> Verdict | None:
        deadline = ctx.analysis.customer_deadline_minutes
        if deadline is not None and deadline <= self._threshold:
            return Verdict(
                f"Customer deadline in {deadline} min (<= {self._threshold}) is at least HIGH urgency.",
                min_urgency=Urgency.HIGH,
            )
        return None


# ======================================================================================
# ACTION RULES: may make the action safer. They run after urgency is final.
# ======================================================================================
class CriticalNeedsHuman:
    name = "critical_needs_human"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.urgency is Urgency.CRITICAL:
            return Verdict("Critical tickets always go to a human.", min_action=Action.ESCALATE_TO_HUMAN)
        return None


class RestrictedTopicNeedsHuman:
    name = "restricted_topic_needs_human"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if issue_types(ctx.analysis) & {IssueType.SECURITY, IssueType.LEGAL_PRIVACY}:
            return Verdict("Security and legal/privacy matters always go to a person.", min_action=Action.ESCALATE_TO_HUMAN)
        return None


class MoneyNeedsHuman:
    name = "money_needs_human"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.analysis.involves_money:
            return Verdict(
                "Refunds/charges/disputes need a person; the agent must not auto-handle money.",
                min_action=Action.ESCALATE_TO_HUMAN,
            )
        return None


class EnterpriseHighNeedsHuman:
    name = "enterprise_high_needs_human"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.ticket.customer.plan is Plan.ENTERPRISE and ctx.urgency.rank >= Urgency.HIGH.rank:
            return Verdict("Enterprise customer with HIGH+ urgency.", min_action=Action.ESCALATE_TO_HUMAN)
        return None


class LowConfidence:
    """If the model itself is unsure (or reports a nonsense number), ask a person."""

    name = "low_confidence"

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def check(self, ctx: PolicyContext) -> Verdict | None:
        c = ctx.analysis.confidence
        if not (0.0 <= c <= 1.0) or c < self._threshold:  # NaN fails the first test too
            return Verdict(
                f"Model confidence {c} is below {self._threshold} (or invalid).",
                min_action=Action.ESCALATE_TO_HUMAN,
            )
        return None


class ToolFailure:
    """If any tool failed, the model decided with incomplete evidence."""

    name = "tool_failure"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        failed = [t.name for t in ctx.tool_calls if not t.ok]
        if failed:
            return Verdict(f"Tool(s) failed, evidence is incomplete: {failed}.", min_action=Action.ESCALATE_TO_HUMAN)
        return None


class SlaBreach:
    """The customer has waited longer than their plan promises: a person must see it now."""

    name = "sla_breach"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        sla = ctx.sla
        if sla is None or not sla.breached:
            return None
        action = Action.ESCALATE_TO_HUMAN if ctx.urgency.rank >= Urgency.HIGH.rank else Action.ROUTE_TO_SPECIALIST
        return Verdict(
            f"{sla.tier} {sla.urgency.value} SLA breached: waiting {sla.waiting_minutes} min, "
            f"target {sla.target_minutes} min.",
            min_action=action,
        )


class SuspectedSpam:
    name = "suspected_spam"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if IssueType.SPAM in issue_types(ctx.analysis):
            return Verdict("Never auto-reply to suspected spam/abuse.", min_action=Action.ROUTE_TO_SPECIALIST)
        return None


class UpsetCustomer:
    name = "angry_customer_needs_human"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.analysis.customer_sentiment is Sentiment.ANGRY:
            return Verdict("Angry customers should not receive a bot-only reply.", min_action=Action.ROUTE_TO_SPECIALIST)
        return None


class GroundedAutoRespond:
    """An automatic answer must be backed by KB articles that were actually retrieved.

    This stops the model from "remembering" an answer or citing an article it never looked at.
    """

    name = "grounded_auto_respond"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.action is not Action.AUTO_RESPOND:
            return None
        a = ctx.analysis
        if ctx.urgency.rank >= Urgency.HIGH.rank:
            return Verdict("High-impact tickets require review.", min_action=Action.ROUTE_TO_SPECIALIST)
        if not (a.draft_reply and a.draft_reply.strip()):
            return Verdict("Auto-respond without a draft reply.", min_action=Action.ROUTE_TO_SPECIALIST)
        if not a.kb_article_ids:
            return Verdict("Auto-respond without KB grounding.", min_action=Action.ROUTE_TO_SPECIALIST)
        ungrounded = sorted(set(a.kb_article_ids) - retrieved_kb_ids(ctx.tool_calls))
        if ungrounded:
            return Verdict(
                f"Cites KB articles that were never retrieved: {ungrounded}.",
                min_action=Action.ROUTE_TO_SPECIALIST,
            )
        return None


class UnsafeAutoReply:
    """An automatic reply whose wording failed the ReplyGuard must be reviewed by a person."""

    name = "unsafe_auto_reply"

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if ctx.action is Action.AUTO_RESPOND and ctx.reply_issues:
            labels = ", ".join(sorted({i.rule for i in ctx.reply_issues}))
            return Verdict(
                f"Draft reply failed the wording checks ({labels}); a person must review it before sending.",
                min_action=Action.ROUTE_TO_SPECIALIST,
            )
        return None


class AutoRespondKillSwitch:
    """Operator switch (TRIAGE_AUTO_RESPOND=false): nothing is answered without a person."""

    name = "auto_respond_disabled"

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    def check(self, ctx: PolicyContext) -> Verdict | None:
        if not self._enabled and ctx.action is Action.AUTO_RESPOND:
            return Verdict("Auto-respond is switched off (TRIAGE_AUTO_RESPOND=false).", min_action=Action.ROUTE_TO_SPECIALIST)
        return None


# ======================================================================================
# THE ENGINE
# ======================================================================================
class PolicyEngine:
    def __init__(
        self,
        urgency_rules: Sequence[Rule],
        action_rules: Sequence[Rule],
        reply_guard: ReplyGuard | None = None,
    ) -> None:
        self._urgency_rules = list(urgency_rules)
        self._action_rules = list(action_rules)
        self._reply_guard = reply_guard

    @classmethod
    def default(
        cls, low_confidence_threshold: float = 0.6, auto_respond_enabled: bool = True
    ) -> "PolicyEngine":
        """Default assignment rules. Preserve order: later rules observe earlier urgency/action changes."""
        return cls(
            urgency_rules=[
                MoneyIssueFloor(), RestrictedTopicFloor(), MultiUserOutageFloor(),
                OutageEvidenceFloor(), ImminentDeadlineFloor(), BillingEvidenceFloor(), ReportedFinancialRisk(),
            ],
            action_rules=[
                CriticalNeedsHuman(),
                RestrictedTopicNeedsHuman(),
                MoneyNeedsHuman(),
                BillingEvidenceFloor(), ReportedFinancialRisk(),
                EnterpriseHighNeedsHuman(),
                LowConfidence(low_confidence_threshold),
                ToolFailure(),
                SlaBreach(),
                SuspectedSpam(),
                UpsetCustomer(),
                GroundedAutoRespond(),
                UnsafeAutoReply(),
                AutoRespondKillSwitch(auto_respond_enabled),
            ],
            reply_guard=ReplyGuard(),
        )

    def apply(
        self,
        ticket: Ticket,
        analysis: TriageAnalysis,
        tool_calls: Sequence[ToolTrace],
        sla_policy: SlaPolicy | None = None,
    ) -> PolicyOutcome:
        adjustments: list[PolicyAdjustment] = []
        urgency = analysis.urgency  # start from what the model said...
        action = analysis.recommended_action  # ...and only ever move toward "safer"

        # Step 1: check the wording of the draft reply (for every action, so a human reviewer
        # also sees problems in a holding message).
        reply_issues = (
            self._reply_guard.check(analysis.draft_reply, retrieved_kb_text(tool_calls)) if self._reply_guard else []
        )

        # Step 2: urgency rules. Each may raise urgency; none can lower it.
        for rule in self._urgency_rules:
            verdict = rule.check(PolicyContext(ticket, analysis, urgency, action, tool_calls, None, reply_issues))
            if verdict and verdict.min_urgency and verdict.min_urgency.rank > urgency.rank:
                adjustments.append(_adjustment(rule, "urgency", urgency, verdict.min_urgency, verdict))
                urgency = verdict.min_urgency

        # Step 3: the promised response time depends on the FINAL urgency, so compute it now.
        sla = sla_policy.evaluate(ticket, urgency) if sla_policy else None

        # Step 4: action rules. Each may make the action safer; none can make it less safe.
        for rule in self._action_rules:
            verdict = rule.check(PolicyContext(ticket, analysis, urgency, action, tool_calls, sla, reply_issues))
            if verdict and verdict.min_action and verdict.min_action.rank > action.rank:
                adjustments.append(_adjustment(rule, "action", action, verdict.min_action, verdict))
                action = verdict.min_action

        # Step 5: a ticket that goes to people needs an owner team; an auto-answer needs none.
        route_to = analysis.route_to if action is not Action.AUTO_RESPOND else None
        if action is not Action.AUTO_RESPOND and route_to is None:
            outage_evidence = any(a.rule == OutageEvidenceFloor.name for a in adjustments)
            route_to = Team.INCIDENT_RESPONSE if outage_evidence else default_team(analysis)
            adjustments.append(
                PolicyAdjustment(
                    rule="default_team", field="route_to", before="none", after=route_to.value,
                    reason="Action needs a destination team; inferred from the issue type.",
                )
            )
        # Enforce specialist ownership for mandatory risk categories, even if a model supplied a team.
        required = None
        types = issue_types(analysis)
        if IssueType.SECURITY in types:
            required = Team.SECURITY
        elif IssueType.LEGAL_PRIVACY in types:
            required = Team.LEGAL_COMPLIANCE
        elif analysis.involves_money or ReportedFinancialRisk().check(
                PolicyContext(ticket, analysis, urgency, action, tool_calls)) or BillingEvidenceFloor().check(
                PolicyContext(ticket, analysis, urgency, action, tool_calls)):
            required = Team.BILLING
        elif IssueType.OUTAGE in types and urgency is Urgency.CRITICAL:
            required = Team.INCIDENT_RESPONSE
        if required and action is not Action.AUTO_RESPOND and route_to != required:
            adjustments.append(PolicyAdjustment(rule="required_team", field="route_to",
                before=route_to.value if route_to else "none", after=required.value,
                reason="Mandatory risk ownership overrides model destination."))
            route_to = required
        return PolicyOutcome(urgency, action, route_to, adjustments, sla, list(reply_issues))


def default_team(analysis: TriageAnalysis) -> Team:
    """Pick an owner team from the topic when the model did not name one."""
    types = issue_types(analysis)
    if IssueType.SECURITY in types:
        return Team.SECURITY
    if IssueType.LEGAL_PRIVACY in types:
        return Team.LEGAL_COMPLIANCE
    if analysis.involves_money or analysis.primary_issue_type is IssueType.BILLING:
        return Team.BILLING
    if analysis.primary_issue_type is IssueType.OUTAGE:
        return Team.INCIDENT_RESPONSE
    if analysis.primary_issue_type is IssueType.FEATURE_REQUEST:
        return Team.PRODUCT
    return Team.TECHNICAL_SUPPORT


def _adjustment(rule: Rule, field: str, before: Urgency | Action, after: Urgency | Action, v: Verdict) -> PolicyAdjustment:
    return PolicyAdjustment(rule=rule.name, field=field, before=before.value, after=after.value, reason=v.reason)
