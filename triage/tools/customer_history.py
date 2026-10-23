"""Tool `get_customer_history`: past tickets, recent activity, SLA tier, account manager.

Gives the AI context (is this a new or long-time customer? do they have an account manager?).
It always looks up the customer of the CURRENT ticket; the AI cannot choose another customer.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from triage.tools.base import Tool, ToolContext

TOOL_NAME = "get_customer_history"


class CustomerHistoryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tickets: int = Field(ge=0, le=10, description="How many recent past tickets to include (0-10).")


class CustomerHistoryTool(Tool):
    name = TOOL_NAME
    description = (
        "Get the current customer's support history and account context: past tickets, login "
        "activity, SLA tier, account manager. Scoped automatically to the customer of this ticket."
    )
    params_model = CustomerHistoryParams

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    def run(self, params: CustomerHistoryParams, ctx: ToolContext) -> dict[str, Any]:
        record = self._records.get(ctx.customer_id)
        if record is None:
            return {"found": False, "note": "No history on file for this customer."}
        limit = max(0, min(params.max_tickets, 10))
        return {
            "found": True,
            **{k: v for k, v in record.items() if k != "recent_tickets"},
            "recent_tickets": record.get("recent_tickets", [])[:limit],
        }
