"""Provider-neutral model/session interfaces used by the agent, OpenAI adapter and test doubles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from triage.schemas import TriageAnalysis
from triage.tools.base import ToolCall, ToolResult, ToolSpec


@dataclass(frozen=True)
class LLMTurn:
    """One model response: either tool calls to run, or the final structured analysis."""

    tool_calls: tuple[ToolCall, ...] = ()
    analysis: TriageAnalysis | None = None


class LLMSession(Protocol):
    """A single ticket's conversation. State (history) lives inside the session."""

    def next(self, tool_results: Sequence[ToolResult] = (), allow_tools: bool = True) -> LLMTurn:
        """First call starts the conversation; later calls pass the results of the previous
        turn's tool calls. allow_tools=False forces a final answer."""


class LLMClient(Protocol):
    def start_session(
        self, system_prompt: str, user_message: str, tools: Sequence[ToolSpec]
    ) -> LLMSession: ...
