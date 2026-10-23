"""Offline heuristic adapter. Exercises real tools; never substitutes for live-model evaluation."""
import json
from triage.llm import LLMTurn
from triage.schemas import TriageAnalysis, Urgency, Action, IssueType, Sentiment, Team
from triage.tools import ToolCall

class MockLLM:
    def start_session(self, system_prompt, user_message, tools):
        return MockSession(user_message)

class MockSession:
    def __init__(self, text):
        self.text = text.casefold()
        self.started = False

    def next(self, tool_results=(), allow_tools=True):
        billing = any(w in self.text for w in ("charges", "charged", "refund", "ตัดเงิน", "หักซ้ำ", "คืนเงิน"))
        outage = any(w in self.text for w in ("error 500", "ระบบเข้าไม่ได้", "cannot access"))
        theme = "dark mode" in self.text
        if not self.started and allow_tools:
            self.started = True
            calls = [
                ToolCall("history", "get_customer_history", '{"max_tickets": 3}'),
                ToolCall("kb", "search_knowledge_base", json.dumps({"query":
                    "duplicate pending charge refund" if billing else
                    "error 500 multiple users access" if outage else
                    "dark mode system default schedule" if theme else "unsupported question"})),
            ]
            if billing:
                calls.append(ToolCall("billing", "get_billing_charges", '{"lookback_days": 7}'))
            if outage:
                region = "Thailand" if "thailand" in self.text else "unknown"
                calls.append(ToolCall("status", "check_service_status", json.dumps({"region": region})))
            return LLMTurn(tool_calls=tuple(calls))
        outputs = {r.call_id: json.loads(r.output) for r in tool_results}
        docs = outputs.get("kb", {}).get("results", [])
        # Conservative mock routes functional questions; it does not compose feature claims.
        urgency = Urgency.HIGH if billing else Urgency.CRITICAL if outage else Urgency.MEDIUM if theme else Urgency.LOW
        analysis = TriageAnalysis(
            urgency=urgency, reasoning="Offline heuristic assessment of customer reports; inspect tool traces for synthetic evidence.",
            product="Billing / Pro upgrade" if billing else "Platform access" if outage else "Appearance" if theme else "Unknown",
            primary_issue_type=IssueType.BILLING if billing else IssueType.OUTAGE if outage else IssueType.BUG if theme else IssueType.OTHER,
            secondary_issue_types=[], customer_sentiment=Sentiment.FRUSTRATED if billing or outage else Sentiment.NEUTRAL,
            customer_language="Thai" if any("ก" <= c <= "๛" for c in self.text) else "English",
            involves_money=billing, multiple_users_affected=outage and any(w in self.text for w in ("เพื่อนร่วมงาน", "multiple users", "several users")),
            customer_deadline_minutes=120 if "2 hours" in self.text else None,
            feature_requests=["Scheduled dark mode"] if theme and "schedule" in self.text else [],
            kb_article_ids=[d["id"] for d in docs],
            recommended_action=Action.ESCALATE_TO_HUMAN if billing or outage else Action.ROUTE_TO_SPECIALIST,
            route_to=Team.BILLING if billing else Team.INCIDENT_RESPONSE if outage else Team.TECHNICAL_SUPPORT,
            confidence=0.7, draft_reply=None,
            handoff_summary="Review the customer timeline and attached tool evidence. No account change, refund, or delivery has been performed.")
        return LLMTurn(analysis=analysis)
