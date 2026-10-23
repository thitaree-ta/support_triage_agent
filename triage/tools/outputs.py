"""Validated output contracts for built-in tools; additive metadata remains compatible."""
from pydantic import BaseModel, ConfigDict, Field

class Output(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

class ArticleResult(Output):
    id: str
    title: str
    content: str
    score: float

class SearchOutput(Output):
    results: list[ArticleResult]
    note: str | None = None

class CustomerOutput(Output):
    found: bool
    recent_tickets: list[dict] = Field(default_factory=list)

class Charge(Output):
    id: str
    amount_usd: float = Field(ge=0, allow_inf_nan=False)
    status: str
    created_minutes_ago: int = Field(ge=0)
    description: str

class BillingOutput(Output):
    found: bool
    charges: list[Charge] = Field(default_factory=list)
    duplicate_charge_suspected: bool = False
    subscription_status: str | None = None

class StatusOutput(Output):
    region: str
    public_status_page: dict
    internal_monitoring: dict
    internal_degraded: bool
    status_page_may_be_stale: bool

OUTPUT_MODELS = {
    "search_knowledge_base": SearchOutput,
    "get_customer_history": CustomerOutput,
    "get_billing_charges": BillingOutput,
    "check_service_status": StatusOutput,
}
