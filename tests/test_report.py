import json
import unittest

from tests.helpers import SLA, TICKETS, make_analysis
from triage.policy import PolicyEngine
from triage.report import render, result_to_dict
from triage.schemas import ToolTrace, TriageResult, Urgency


def sample_result() -> TriageResult:
    ticket = TICKETS["T-002"]
    trace = ToolTrace(name="get_billing_charges", arguments={"lookback_days": 7}, ok=True, output={"secret": "x" * 50})
    analysis = make_analysis(urgency=Urgency.CRITICAL, customer_deadline_minutes=90)
    outcome = PolicyEngine.default().apply(ticket, analysis, [trace], SLA)
    return TriageResult(ticket_id=ticket.id, urgency=outcome.urgency, action=outcome.action, route_to=outcome.route_to,
                        analysis=analysis, adjustments=outcome.adjustments, sla=outcome.sla, tool_calls=[trace])


class ReportTests(unittest.TestCase):
    def test_tool_outputs_are_omitted_unless_requested(self):
        result = sample_result()
        slim = result_to_dict(result)
        self.assertNotIn("output", slim["tool_calls"][0])
        self.assertNotIn("secret", json.dumps(slim))
        full = result_to_dict(result, include_traces=True)
        self.assertEqual(full["tool_calls"][0]["output"], {"secret": "x" * 50})

    def test_text_report_shows_sla_and_deadline(self):
        text = render(sample_result())
        self.assertIn("BREACHED by 90 min", text)
        self.assertIn("within 90 min", text)


if __name__ == "__main__":
    unittest.main()


def test_report_distinguishes_reported_deadline_from_remaining_time():
    result = sample_result()
    result.customer_deadline_remaining_minutes = -10
    text = render(result)
    assert "within 90 min of their message" in text
    assert "-10 min at processing" in text
    assert "uncalibrated" in text
