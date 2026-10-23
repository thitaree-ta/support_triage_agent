import json
import unittest

from tests.helpers import SETTINGS, TICKETS, tool_call
from triage.factory import build_registry
from triage.tools import ToolContext, ToolRegistry
from triage.tools import knowledge_base as kb_tool
from triage.tools.base import Tool


def run(registry: ToolRegistry, ticket_id: str, name: str, **args):
    ctx = ToolContext(TICKETS[ticket_id])
    result, trace = registry.execute(tool_call(name, **args), ctx)
    return json.loads(result.output), trace


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry(SETTINGS)

    def top_ids(self, query: str) -> list[str]:
        out, trace = run(self.registry, "T-001", kb_tool.TOOL_NAME, query=query)
        self.assertTrue(trace.ok)
        return [r["id"] for r in out["results"]]

    def test_billing_queries(self):
        self.assertIn("KB-102", self.top_ids("duplicate pending charge refund dispute"))
        self.assertIn("KB-101", self.top_ids("payment failed upgrade to pro"))
        self.assertIn("KB-103", self.top_ids("paid for pro but plan still free"))

    def test_outage_queries(self):
        self.assertIn("KB-201", self.top_ids("error 500 multiple users cannot access"))
        self.assertIn("KB-202", self.top_ids("status page operational but outage in region"))

    def test_dark_mode_queries(self):
        self.assertIn("KB-301", self.top_ids("dark mode missing in appearance settings"))
        self.assertIn("KB-303", self.top_ids("system default theme not following macos"))
        self.assertIn("KB-302", self.top_ids("schedule dark mode automatic switch"))

    def test_irrelevant_query_returns_nothing_and_a_warning(self):
        out, _ = run(self.registry, "T-001", kb_tool.TOOL_NAME, query="chocolate cake recipe")
        self.assertEqual(out["results"], [])
        self.assertIn("No relevant articles", out["note"])


class DataToolTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry(SETTINGS)

    def test_billing_detects_duplicate_charges(self):
        out, trace = run(self.registry, "T-001", "get_billing_charges", lookback_days=7)
        self.assertTrue(trace.ok)
        self.assertEqual(len(out["charges"]), 3)
        self.assertTrue(out["duplicate_charge_suspected"])
        self.assertEqual(out["totals_usd_by_status"], {"captured": 29.99, "pending": 59.98})
        self.assertEqual(out["subscription_status"], "free")
        self.assertNotIn("card_last4", json.dumps(out))  # data minimisation: no card data to the model

    def test_billing_lookback_window_filters(self):
        short, _ = run(self.registry, "T-003", "get_billing_charges", lookback_days=1)
        long, _ = run(self.registry, "T-003", "get_billing_charges", lookback_days=30)
        self.assertEqual(short["charges"], [])
        self.assertEqual(len(long["charges"]), 1)
        self.assertFalse(long["duplicate_charge_suspected"])

    def test_tools_are_scoped_to_the_ticket_customer(self):
        # T-002's customer has no charges even though T-001's customer does.
        out, _ = run(self.registry, "T-002", "get_billing_charges", lookback_days=30)
        self.assertEqual(out["charges"], [])

    def test_service_status_flags_stale_status_page(self):
        out, _ = run(self.registry, "T-002", "check_service_status", region="Thailand")
        self.assertEqual(out["region"], "ap-southeast")
        self.assertTrue(out["status_page_may_be_stale"])
        healthy, _ = run(self.registry, "T-002", "check_service_status", region="us-east")
        self.assertFalse(healthy["status_page_may_be_stale"])

    def test_unknown_region_is_reported_as_tool_error(self):
        out, trace = run(self.registry, "T-002", "check_service_status", region="mars")
        self.assertFalse(trace.ok)
        self.assertIn("Unknown region", out["error"])

    def test_customer_history_respects_limit(self):
        out, _ = run(self.registry, "T-002", "get_customer_history", max_tickets=1)
        self.assertEqual(len(out["recent_tickets"]), 1)
        self.assertEqual(out["sla_tier"], "enterprise-premium")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry(SETTINGS)
        self.ctx = ToolContext(TICKETS["T-001"])

    def test_unknown_tool(self):
        result, trace = self.registry.execute(tool_call("delete_everything"), self.ctx)
        self.assertFalse(trace.ok)
        self.assertIn("Unknown tool", json.loads(result.output)["error"])

    def test_invalid_json_arguments(self):
        from triage.tools import ToolCall

        result, trace = self.registry.execute(ToolCall("c1", "get_billing_charges", "{not json"), self.ctx)
        self.assertFalse(trace.ok)
        self.assertIn("not valid JSON", json.loads(result.output)["error"])

    def test_missing_or_wrong_arguments(self):
        _, trace = self.registry.execute(tool_call("get_billing_charges"), self.ctx)
        self.assertFalse(trace.ok)
        _, trace = self.registry.execute(tool_call("get_billing_charges", lookback_days="many"), self.ctx)
        self.assertFalse(trace.ok)

    def test_crashing_tool_does_not_propagate(self):
        from pydantic import BaseModel

        class NoParams(BaseModel):
            pass

        class Boom(Tool):
            name = "boom"
            description = "always fails"
            params_model = NoParams

            def run(self, params, ctx):
                raise RuntimeError("kaboom")

        registry = ToolRegistry([Boom()])
        result, trace = registry.execute(tool_call("boom"), self.ctx)
        self.assertFalse(trace.ok)
        self.assertIn("RuntimeError", json.loads(result.output)["error"])
        self.assertNotIn("kaboom", json.loads(result.output)["error"])

    def test_duplicate_registration_rejected(self):
        tool = next(iter(self.registry._tools.values()))
        with self.assertRaises(ValueError):
            self.registry.register(tool)

    def test_specs_are_strict_schemas(self):
        specs = self.registry.specs()
        self.assertEqual(len(specs), 4)
        for spec in specs:
            self.assertFalse(spec.parameters["additionalProperties"])
            self.assertEqual(spec.parameters["required"], list(spec.parameters["properties"]))
            self.assertTrue(spec.description)


if __name__ == "__main__":
    unittest.main()
