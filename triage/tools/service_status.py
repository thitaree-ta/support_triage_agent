"""Tool `check_service_status`: is the service healthy in the customer's region?

Returns two views: what the PUBLIC status page says, and what our INTERNAL monitoring sees.
They can disagree (the public page is updated by hand and can lag), and the tool flags that.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from triage.tools.base import Tool, ToolContext, ToolError

TOOL_NAME = "check_service_status"
_ERROR_RATE_ALERT_PCT = 5.0

_REGION_ALIASES = {
    "thailand": "ap-southeast", "th": "ap-southeast", "asia": "ap-southeast",
    "apac": "ap-southeast", "singapore": "ap-southeast", "ap-southeast": "ap-southeast",
    "us": "us-east", "usa": "us-east", "united states": "us-east", "us-east": "us-east",
    "eu": "eu-west", "europe": "eu-west", "eu-west": "eu-west",
}


class ServiceStatusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str = Field(min_length=1, max_length=80, description="Customer region, e.g. 'Thailand', 'Asia', 'us-east'.")


class ServiceStatusTool(Tool):
    name = TOOL_NAME
    description = (
        "Check service health for a region: what the public status page says AND what internal "
        "monitoring shows. The status page can lag behind reality, so compare both."
    )
    params_model = ServiceStatusParams

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    def run(self, params: ServiceStatusParams, ctx: ToolContext) -> dict[str, Any]:
        region = _REGION_ALIASES.get(params.region.strip().lower(), params.region.strip().lower())
        record = self._records.get(region)
        if record is None:
            raise ToolError(f"Unknown region '{params.region}'. Known: {sorted(self._records)}")
        public, internal = record["public_status_page"], record["internal_monitoring"]
        degraded = internal["error_rate_5xx_pct"] >= _ERROR_RATE_ALERT_PCT or internal["alert_firing"]
        return {
            "region": region,
            "public_status_page": public,
            "internal_monitoring": internal,
            "internal_degraded": degraded,
            "status_page_may_be_stale": degraded and public["status"] == "operational",
        }
