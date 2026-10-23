"""Fixture integrity and real offline-pipeline checks; targets are not forced outcomes."""
import json
from pathlib import Path
from triage.config import Settings
from triage.datastore import load_tickets
from triage.factory import build_agent
from triage.mock_llm import MockLLM
from triage.schemas import TriageResult, ResultStatus

ROOT = Path(__file__).resolve().parent.parent

def test_three_distinct_additional_tickets():
    tickets = load_tickets(ROOT / "data/additional_tickets.json")
    targets = json.loads((ROOT / "evals/additional_expected.json").read_text())
    original = load_tickets(ROOT / "data/tickets.json")
    assert len(tickets) == 3
    assert {t.id for t in tickets}.isdisjoint({t.id for t in original})
    assert set(targets) == {t.id for t in tickets}
    assert {v["urgency"] for v in targets.values()} == {"low", "medium", "high"}
    assert all("urgency" not in t.model_dump_json() for t in tickets)

def test_additional_tickets_use_real_tools_without_forcing_targets():
    for ticket in load_tickets(ROOT / "data/additional_tickets.json"):
        result = build_agent(Settings(), llm=MockLLM()).triage(ticket)
        assert result.status == ResultStatus.OK
        assert {"get_customer_history", "search_knowledge_base"} <= {t.name for t in result.tool_calls}
        assert TriageResult.model_validate_json(result.model_dump_json()) == result
        # Quality mismatches are evaluation findings, not reasons to alter runtime rules.
