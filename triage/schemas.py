"""Validated ticket, model-analysis, tool-trace and final-result contracts."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class _Ordered(str, Enum):
    """An enum whose members can be compared: later in the list = higher `rank`.
    That lets policy rules say "at least HIGH" with `urgency.rank >= Urgency.HIGH.rank`."""

    @property
    def rank(self) -> int:
        return list(type(self)).index(self)


class Urgency(_Ordered):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(_Ordered):
    """Ordered from least to most cautious; policies may only move *up* this scale."""

    AUTO_RESPOND = "auto_respond"
    ROUTE_TO_SPECIALIST = "route_to_specialist"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class Sentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class IssueType(str, Enum):
    BILLING = "billing"
    OUTAGE = "outage"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    HOW_TO = "how_to"
    ACCOUNT_ACCESS = "account_access"
    SECURITY = "security"  # compromised credentials, suspicious logins, data exposure
    LEGAL_PRIVACY = "legal_privacy"  # PDPA/GDPR requests, legal threats
    SPAM = "spam"  # spam, phishing, abuse
    OTHER = "other"


class Team(str, Enum):
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    INCIDENT_RESPONSE = "incident_response"
    CUSTOMER_SUCCESS = "customer_success"
    PRODUCT = "product"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"


class Plan(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# --------------------------------------------------------------------------- input
class CustomerProfile(BaseModel):
    customer_id: str
    plan: Plan
    tenure_months: int = Field(ge=0)
    region: str | None = None
    seats: int | None = Field(default=None, ge=1)
    notes: str = ""


class Message(BaseModel):
    minutes_ago: int = Field(ge=0)
    text: str


class Ticket(BaseModel):
    id: str
    customer: CustomerProfile
    messages: list[Message]
    last_agent_reply_minutes_ago: int | None = None  # None = no human has replied yet

    def chronological(self) -> list[Message]:
        return sorted(self.messages, key=lambda m: m.minutes_ago, reverse=True)

    def waiting_minutes(self) -> int:
        """How long the oldest customer message that has no human reply has been waiting."""
        cutoff = self.last_agent_reply_minutes_ago
        pending = [m.minutes_ago for m in self.messages if cutoff is None or m.minutes_ago < cutoff]
        return max(pending, default=0)


# ------------------------------------------------------------------ LLM output
class TriageAnalysis(BaseModel):
    """What the AI must return. OpenAI is told to follow exactly this shape, so the answer is
    validated before policy evaluation. Every field is required; failed responses use a human fallback."""

    urgency: Urgency = Field(description="Urgency per the rubric in the system prompt.")
    reasoning: str = Field(description="1-3 sentences: why this urgency and this action.")
    product: str = Field(description="Product area affected, e.g. 'Billing / Pro subscription'.")
    primary_issue_type: IssueType
    secondary_issue_types: list[IssueType] = Field(
        description="Other intents present in the thread (can be empty)."
    )
    customer_sentiment: Sentiment = Field(description="Judged over the whole thread, incl. trend.")
    customer_language: str = Field(description="Language of the customer's latest messages.")
    involves_money: bool = Field(
        description="True if charges, refunds, chargebacks or payment disputes are mentioned."
    )
    multiple_users_affected: bool = Field(
        description="True only if the ticket or tool evidence shows more than one user affected."
    )
    customer_deadline_minutes: int | None = Field(
        description="Minutes from the customer's latest message until a deadline they state "
        "(e.g. 'presentation in 2 hours' = 120). Null if none or vague ('this afternoon')."
    )
    feature_requests: list[str] = Field(description="Distinct feature requests to log (can be empty).")
    kb_article_ids: list[str] = Field(
        description="IDs of KB articles returned by the search tool that support the answer."
    )
    recommended_action: Action
    route_to: Team | None = Field(
        description="Team for route/escalate actions; null for auto_respond."
    )
    confidence: float = Field(description="0.0-1.0, uncalibrated self-assessment; never a probability of correctness.")
    draft_reply: str | None = Field(
        description="Reply in the customer's language, grounded in KB/tool facts. Never promises "
        "refunds, ETAs or root causes."
    )
    handoff_summary: str | None = Field(
        description="2-4 sentences for the human/specialist; null when auto-responding."
    )


# ------------------------------------------------------------------ final result
class ToolTrace(BaseModel):
    """A record of one tool call: what was asked, what came back, whether it worked.
    It is the EVIDENCE the policy rules check (e.g. "did the model really search the KB?")."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    output: Any = None
    error: str | None = None
    duration_ms: int = 0


class PolicyAdjustment(BaseModel):
    """One change a policy rule made to the AI's opinion. The list of these is the audit trail:
    it answers "what did the AI say, what did we change, and why?"."""

    rule: str
    field: str  # "urgency" | "action" | "route_to"
    before: str
    after: str
    reason: str


class ReplyIssue(BaseModel):
    """One problem found in a draft reply by the reply guard (see reply_guard.py)."""

    rule: str  # which check fired, e.g. "claims_action"
    excerpt: str  # the exact words that triggered it
    reason: str  # why that is a problem, in plain language


class ActionKind(str, Enum):
    SEND_REPLY = "send_reply"  # send the customer an automatic answer
    CREATE_HANDOFF = "create_handoff"  # open a work item for a person / team
    LOG_FEATURE_REQUEST = "log_feature_request"  # record a product suggestion


class PlannedAction(BaseModel):
    """Something the system would DO as a result of the triage (see actions.py)."""

    kind: ActionKind
    ticket_id: str
    idempotency_key: str  # same content -> same key, so running twice never duplicates
    payload: dict[str, Any]


class SlaStatus(BaseModel):
    """Computed by code (never by the model) from plan, final urgency and time waiting."""

    tier: str
    urgency: Urgency
    target_minutes: int
    waiting_minutes: int
    remaining_minutes: int  # negative = breached
    breached: bool


class ResultStatus(str, Enum):
    OK = "ok"  # the model analysed the ticket and policy was applied
    DEGRADED = "degraded"  # model/system failure: safe fallback, urgency is a placeholder
    BYPASSED = "bypassed"  # model deliberately not used (nothing to analyse, or circuit open)
    REJECTED = "rejected"  # input could not be parsed into a ticket


class TriageResult(BaseModel):
    """The final answer for one ticket: decision + evidence + what should happen next."""

    ticket_id: str
    resolution_status: Literal["not_established", "unresolved", "customer_reported_resolved"] = "not_established"
    evidence: list[dict[str, str]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    urgency: Urgency
    action: Action
    route_to: Team | None
    analysis: TriageAnalysis | None  # None unless status == OK
    status: ResultStatus = ResultStatus.OK
    status_reason: str | None = None
    digest: str | None = None  # human-readable summary for results produced without the model
    input_notes: list[str] = Field(default_factory=list)  # e.g. "card number masked"
    adjustments: list[PolicyAdjustment] = Field(default_factory=list)
    reply_issues: list[ReplyIssue] = Field(default_factory=list)  # problems found in the draft reply
    planned_actions: list[PlannedAction] = Field(default_factory=list)
    sla: SlaStatus | None = None
    customer_deadline_remaining_minutes: int | None = None
    tool_calls: list[ToolTrace] = Field(default_factory=list)

    @property
    def auto_send(self) -> bool:
        """True only when a reply may go out with no human in the loop."""
        return (
            self.status is ResultStatus.OK
            and self.action is Action.AUTO_RESPOND
            and self.analysis is not None
            and bool(self.analysis.draft_reply and self.analysis.draft_reply.strip())
            and self.urgency.rank < Urgency.HIGH.rank
            and not self.reply_issues
        )
