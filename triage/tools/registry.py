"""The tool registry: the one place that knows every tool (used by agent.py).

It does two jobs:
  1. specs()    tells the AI which tools exist and what arguments each needs.
  2. execute()  runs a tool the AI asked for. It checks the arguments first, and it NEVER lets
                a tool problem crash the program: errors are returned to the AI as text so it
                can react, and also recorded so the policy layer knows evidence is incomplete.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Iterable

from pydantic import ValidationError

from triage.schemas import ToolTrace
from triage.tools.outputs import OUTPUT_MODELS
from triage.tools.base import Tool, ToolCall, ToolContext, ToolError, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, call: ToolCall, ctx: ToolContext) -> tuple[ToolResult, ToolTrace]:
        """Run one call. Never raises: failures are reported back to the model and traced."""
        # Steps: find the tool -> parse the arguments -> validate them -> run -> record the outcome.
        started = time.perf_counter()
        arguments: dict = {}
        output: object = None
        error: str | None = None
        try:
            tool = self._tools.get(call.name)
            if tool is None:
                raise ToolError(f"Unknown tool '{call.name}'. Available: {sorted(self._tools)}")
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ToolError(f"Arguments are not valid JSON: {exc}") from exc
            params = tool.params_model.model_validate(arguments)
            output = tool.run(params, ctx)
            if not isinstance(output, dict):
                raise ValueError("Tool output must be an object")
            if call.name in OUTPUT_MODELS:
                OUTPUT_MODELS[call.name].model_validate(output)
            json.dumps(output, ensure_ascii=False, allow_nan=False)
        except ToolError as exc:
            error = str(exc)
        except ValidationError as exc:  # model sent bad arguments: tell it, no stack trace needed
            output = None
            error = "Invalid arguments or tool output: schema validation failed"
        except Exception as exc:  # noqa: BLE001 - a broken tool must not crash triage
            logger.warning("tool_failed error_type=%s", type(exc).__name__)
            output = None
            error = type(exc).__name__ + ": tool unavailable"

        ok = error is None
        payload = output if ok else {"error": error}
        trace = ToolTrace(
            name=call.name,
            arguments=arguments if isinstance(arguments, dict) else {},
            ok=ok,
            output=output,
            error=error,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        result = ToolResult(call_id=call.call_id, output=json.dumps(payload, ensure_ascii=False))
        return result, trace
