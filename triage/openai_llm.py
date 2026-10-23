"""OpenAI adapter: translates between our interface (llm.py) and the OpenAI Responses API.

What happens on each call to `session.next(...)`:
  - First call:  send the ticket text as the user message.
  - Later calls: send ONLY the tool results, and tell OpenAI which earlier response they belong
                 to (previous_response_id), so we do not resend the whole conversation.
  - We always pass the instructions again (OpenAI does not remember them between calls).
  - OpenAI answers with either "please run these tools" or the final structured analysis.
"""
from __future__ import annotations

from typing import Any, Sequence

from triage.llm import LLMTurn
from triage.schemas import TriageAnalysis
from triage.tools.base import ToolCall, ToolResult, ToolSpec


class OpenAIClient:
    def __init__(
        self, model: str, reasoning_effort: str = "", client: Any = None, request_timeout: float = 60.0
    ) -> None:
        if client is None:
            from openai import OpenAI  # imported lazily so tests/other adapters need no SDK

            # SDK retries 429/5xx/timeouts with backoff
            client = OpenAI(max_retries=3, timeout=request_timeout)
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort

    def start_session(
        self, system_prompt: str, user_message: str, tools: Sequence[ToolSpec]
    ) -> "OpenAISession":
        return OpenAISession(
            self._client, self._model, self._reasoning_effort, system_prompt, user_message, tools
        )


class OpenAISession:
    def __init__(
        self,
        client: Any,
        model: str,
        reasoning_effort: str,
        system_prompt: str,
        user_message: str,
        tools: Sequence[ToolSpec],
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._instructions = system_prompt
        self._user_message = user_message
        self._tools = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": True,
            }
            for t in tools
        ]
        self._previous_response_id: str | None = None

    def next(self, tool_results: Sequence[ToolResult] = (), allow_tools: bool = True) -> LLMTurn:
        if self._previous_response_id is None:
            input_items: list[dict[str, Any]] = [{"role": "user", "content": self._user_message}]
        else:
            input_items = [
                {"type": "function_call_output", "call_id": r.call_id, "output": r.output}
                for r in tool_results
            ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "instructions": self._instructions,  # not carried over by previous_response_id
            "input": input_items,
            "tools": self._tools,
            "tool_choice": "auto" if allow_tools else "none",
            "text_format": TriageAnalysis,
            "max_output_tokens": 4000,
        }
        if self._previous_response_id:
            kwargs["previous_response_id"] = self._previous_response_id
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}

        response = self._client.responses.parse(**kwargs)
        self._previous_response_id = response.id
        calls = tuple(
            ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments)
            for item in response.output
            if item.type == "function_call"
        )
        return LLMTurn(tool_calls=calls, analysis=None if calls else response.output_parsed)
