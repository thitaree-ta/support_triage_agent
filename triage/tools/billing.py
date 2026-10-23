"""Read synthetic charges and flag possible duplicates; matching amounts do not prove duplicate settlement."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from triage.tools.base import Tool, ToolContext

TOOL_NAME = "get_billing_charges"
_DAY_MINUTES = 24 * 60
_VISIBLE_CHARGE_FIELDS = ("id", "amount_usd", "status", "created_minutes_ago", "description")  # no card data


class BillingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback_days: int = Field(ge=1, le=90, description="How many days of charges to include (1-90).")


class BillingTool(Tool):
    name = TOOL_NAME
    description = (
        "Look up recent charges, refunds and subscription status for the current customer. Use for "
        "payment failures, duplicate/unexpected charges or plan-not-upgraded issues. Read-only."
    )
    params_model = BillingParams

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    def run(self, params: BillingParams, ctx: ToolContext) -> dict[str, Any]:
        record = self._records.get(ctx.customer_id)
        if record is None:
            return {"found": False, "note": "No billing record for this customer."}
        window = max(1, min(params.lookback_days, 90)) * _DAY_MINUTES
        charges = [c for c in record.get("charges", []) if c["created_minutes_ago"] <= window]
        by_status: dict[str, float] = {}
        for charge in charges:
            by_status[charge["status"]] = round(by_status.get(charge["status"], 0.0) + charge["amount_usd"], 2)
        return {
            "found": True,
            "subscription_status": record.get("subscription_status"),
            "billing_sync_error": record.get("billing_sync_error"),
            "charges": [{k: c[k] for k in _VISIBLE_CHARGE_FIELDS} for c in charges],
            "totals_usd_by_status": by_status,
            "duplicate_charge_suspected": _has_duplicates(charges),
            "refunds": record.get("refunds", []),
        }


def _has_duplicates(charges: list[dict[str, Any]]) -> bool:
    """Same amount charged more than once within 24h of each other."""
    last_seen: dict[float, int] = {}
    for charge in sorted(charges, key=lambda c: c["created_minutes_ago"]):
        if charge["status"] not in {"pending", "captured"}:
            continue
        amount, age = charge["amount_usd"], charge["created_minutes_ago"]
        if amount in last_seen and age - last_seen[amount] <= _DAY_MINUTES:
            return True
        last_seen[amount] = age
    return False
