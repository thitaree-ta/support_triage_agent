import unittest
from pathlib import Path

from evals.runner import EvalRunner, load_cases
from evals.scoring import check
from tests.helpers import make_analysis
from triage.schemas import Action, ResultStatus, Team, ToolTrace, TriageResult, Urgency

CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.json"


def result(**kw) -> TriageResult:
    analysis = kw.pop("analysis", make_analysis())
    base = dict(ticket_id="E", urgency=Urgency.LOW, action=Action.AUTO_RESPOND, route_to=None, analysis=analysis)
    base.update(kw)
    return TriageResult(**base)


class CaseFileTests(unittest.TestCase):
    def test_cases_load_and_are_well_formed(self):
        cases = load_cases(CASES)
        self.assertGreaterEqual(len(cases), 18)
        self.assertEqual(len({c.id for c in cases}), len(cases))
        self.assertGreaterEqual(sum("safety" in c.tags for c in cases), 5)
        for c in cases:
            self.assertTrue(c.expect, c.id)

    def test_repeat_expands_long_messages(self):
        long_case = next(c for c in load_cases(CASES) if c.id == "EV-15")
        self.assertGreater(len(long_case.ticket.messages[0].text), 20000)

    def test_unknown_expectation_key_is_rejected(self):
        import json, tempfile
        bad = [{"id": "X", "description": "d", "ticket": {"id": "t", "customer": {"customer_id": "c", "plan": "free", "tenure_months": 1},
                                                           "messages": [{"minutes_ago": 1, "text": "hi"}]},
                "expect": {"urgnecy_at_least": "high"}}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(bad, fh)
        with self.assertRaises(ValueError):
            load_cases(Path(fh.name))


class ScoringTests(unittest.TestCase):
    def test_passing_result(self):
        r = result(urgency=Urgency.HIGH, action=Action.ESCALATE_TO_HUMAN, route_to=Team.BILLING)
        expect = {"urgency_at_least": "high", "actions": ["escalate_to_human"], "route_to": ["billing"]}
        self.assertEqual(check(expect, r), [])

    def test_each_kind_of_failure_is_reported(self):
        r = result(urgency=Urgency.LOW, action=Action.AUTO_RESPOND,
                   analysis=make_analysis(draft_reply="Your REFUND APPROVED", customer_language="English"))
        expect = {"urgency_at_least": "high", "actions": ["escalate_to_human"], "route_to": ["billing"],
                  "forbid_reply_contains": ["refund approved"], "analysis_contains": {"customer_language": "thai"},
                  "feature_request_logged": True}
        failures = check(expect, r)
        self.assertEqual(len(failures), 6, failures)

    def test_urgency_at_most_and_tool_errors(self):
        r = result(urgency=Urgency.HIGH, tool_calls=[ToolTrace(name="t", arguments={}, ok=False, error="x")])
        failures = check({"urgency_at_most": "medium", "no_tool_errors": True}, r)
        self.assertEqual(len(failures), 2)

    def test_status_must_match_and_content_checks_need_an_analysis(self):
        degraded = result(analysis=None, status=ResultStatus.DEGRADED, status_reason="API down")
        failures = check({"forbid_reply_contains": ["x"]}, degraded)
        self.assertTrue(any("status" in f for f in failures))
        self.assertTrue(any("no model analysis" in f for f in failures))
        bypassed = result(analysis=None, status=ResultStatus.BYPASSED, action=Action.ROUTE_TO_SPECIALIST)
        self.assertEqual(check({"status": "bypassed", "actions": ["route_to_specialist"]}, bypassed), [])

    def test_reply_guard_flags_fail_a_case_that_demands_clean_replies(self):
        from triage.schemas import ReplyIssue
        flagged = result(reply_issues=[ReplyIssue(rule="claims_action", excerpt="We’ve noted", reason="x")])
        self.assertEqual(check({"no_reply_issues": True}, result()), [])
        failures = check({"no_reply_issues": True}, flagged)
        self.assertTrue(failures and "claims_action" in failures[0])

    def test_input_notes_are_checked(self):
        r = result(input_notes=["payment card number masked before analysis"])
        self.assertEqual(check({"input_notes_contain": ["card"]}, r), [])
        self.assertEqual(len(check({"input_notes_contain": ["truncated"]}, r)), 1)


class RunnerTests(unittest.TestCase):
    def test_aggregates_pass_rate_safety_and_consistency(self):
        cases = [c for c in load_cases(CASES) if c.id in {"EV-02", "EV-11"}]
        answers = iter([
            result(urgency=Urgency.HIGH, action=Action.ESCALATE_TO_HUMAN, route_to=Team.BILLING),   # EV-02 run 1: pass
            result(urgency=Urgency.LOW, action=Action.AUTO_RESPOND),                                 # EV-02 run 2: under-triage
            result(urgency=Urgency.LOW, action=Action.AUTO_RESPOND),                                 # EV-11 run 1: pass
            result(urgency=Urgency.LOW, action=Action.AUTO_RESPOND),                                 # EV-11 run 2: pass
        ])
        report = EvalRunner(lambda ticket: next(answers), runs=2, model="m").run(cases)
        by_id = {o.case.id: o for o in report.outcomes}
        self.assertFalse(by_id["EV-02"].passed)  # one bad run fails the case
        self.assertTrue(by_id["EV-11"].passed)
        self.assertEqual(by_id["EV-02"].consistency, 0.5)
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.safety, (0, 1))
        md = report.render_markdown()
        self.assertIn("**1/2**", md)
        self.assertIn("| EV-02 |", md)
        self.assertIn("## Failures", md)
        self.assertIn("EV-02", report.render_text())


if __name__ == "__main__":
    unittest.main()
