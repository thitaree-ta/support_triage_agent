"""Read-only tool contracts. New tools need registration, output validation and tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel

from triage.schemas import Ticket


class ToolError(Exception):
    """An expected tool failure. The message is returned to the model so it can adapt."""


@dataclass(frozen=True)
class ToolContext:
    """Per-ticket scope. Customer-scoped tools read the customer from here, never from model
    arguments, so a prompt-injected ticket cannot make the agent look up another customer."""

    ticket: Ticket

    @property
    def customer_id(self) -> str:
        return self.ticket.customer.customer_id


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool definition (adapters convert it to the provider's format)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (strict-mode compatible)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str  # raw JSON string, exactly as produced by the model


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str  # JSON string handed back to the model


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema in the shape OpenAI strict function calling expects (all fields required)."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"])
    return schema


class Tool(ABC):
    """Implement a read-only tool; register its instance and output schema, then add contract tests."""

    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, params: Any, ctx: ToolContext) -> dict[str, Any]:
        """Execute with validated params. Must be JSON-serialisable and side-effect free."""

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, strict_json_schema(self.params_model))
