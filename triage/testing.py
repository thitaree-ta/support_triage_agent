"""Test doubles shared by the unit tests and the offline failure drills (not used in production)."""
from __future__ import annotations

import json
from typing import Any, Sequence

from triage.llm import LLMTurn
from triage.schemas import Action, IssueType, Sentiment, TriageAnalysis, Urgency
from triage.tools import ToolCall, ToolResult


def make_analysis(**overrides: Any) -> TriageAnalysis:
    """A benign, well-grounded low-urgency analysis; override fields per scenario."""
    base: dict[str, Any] = dict(
        urgency=Urgency.LOW,
        reasoning="test",
        product="Desktop app / Appearance",
        primary_issue_type=IssueType.HOW_TO,
        secondary_issue_types=[],
        customer_sentiment=Sentiment.NEUTRAL,
        customer_language="English",
        involves_money=False,
        multiple_users_affected=False,
        customer_deadline_minutes=None,
        feature_requests=[],
        kb_article_ids=["KB-301"],
        recommended_action=Action.AUTO_RESPOND,
        route_to=None,
        confidence=0.9,
        draft_reply="Here is how to enable dark mode.",
        handoff_summary=None,
    )
    base.update(overrides)
    return TriageAnalysis(**base)


def tool_call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=json.dumps(arguments))


def calls(*tool_calls: ToolCall) -> LLMTurn:
    return LLMTurn(tool_calls=tuple(tool_calls))


def final(analysis: TriageAnalysis | None) -> LLMTurn:
    return LLMTurn(analysis=analysis)


class ScriptedSession:
    def __init__(self, script: Sequence[Any]) -> None:
        self._script = list(script)
        self.received: list[tuple[list[ToolResult], bool]] = []

    def next(self, tool_results: Sequence[ToolResult] = (), allow_tools: bool = True) -> LLMTurn:
        self.received.append((list(tool_results), allow_tools))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedLLM:
    """Plays back a fixed list of turns (or raises an Exception instance placed in the list)."""

    def __init__(self, script: Sequence[Any]) -> None:
        self.session = ScriptedSession(script)
        self.user_message = ""
        self.tools: Sequence[Any] = ()

    def start_session(self, system_prompt: str, user_message: str, tools: Sequence[Any]) -> ScriptedSession:
        self.user_message, self.tools = user_message, tools
        return self.session


class OutcomeLLM:
    """Each started session yields the next outcome (an LLMTurn, or an Exception to raise).
    The last outcome repeats. `sessions` counts how many tickets actually reached the model."""

    def __init__(self, outcomes: Sequence[Any]) -> None:
        self._outcomes = list(outcomes)
        self.sessions = 0
        self.user_messages: list[str] = []

    def start_session(self, system_prompt: str, user_message: str, tools: Sequence[Any]) -> ScriptedSession:
        index = min(self.sessions, len(self._outcomes) - 1)
        self.sessions += 1
        self.user_messages.append(user_message)
        return ScriptedSession([self._outcomes[index]])
