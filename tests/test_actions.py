import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_analysis
from triage.actions import OutboxExecutor, plan_actions
from triage.schemas import (
    Action, ActionKind, ReplyIssue, ResultStatus, Team, TriageResult, Urgency,
)


def result(**kw) -> TriageResult:
    base = dict(ticket_id="T-9", urgency=Urgency.LOW, action=Action.AUTO_RESPOND, route_to=None,
                analysis=make_analysis(feature_requests=["Scheduled dark mode"]))
    base.update(kw)
    return TriageResult(**base)


def kinds(actions):
    return [a.kind for a in actions]


class PlannerTests(unittest.TestCase):
    def test_clean_auto_response_sends_a_reply_and_logs_the_feature_request(self):
        actions = plan_actions(result(), "cust-1")
        self.assertEqual(kinds(actions), [ActionKind.SEND_REPLY, ActionKind.LOG_FEATURE_REQUEST])
        self.assertEqual(actions[0].payload["customer_id"], "cust-1")
        self.assertEqual(actions[1].payload["request"], "Scheduled dark mode")

    def test_a_flagged_reply_is_never_planned_for_sending(self):
        flagged = result(reply_issues=[ReplyIssue(rule="claims_action", excerpt="We’ve noted", reason="x")])
        self.assertNotIn(ActionKind.SEND_REPLY, kinds(plan_actions(flagged)))

    def test_escalation_creates_a_handoff_with_summary_team_and_review_draft(self):
        escalated = result(action=Action.ESCALATE_TO_HUMAN, route_to=Team.BILLING, urgency=Urgency.HIGH,
                           analysis=make_analysis(handoff_summary="3 charges, plan still Free", draft_reply="Thanks."))
        (handoff,) = plan_actions(escalated, "cust-1")
        self.assertEqual(handoff.kind, ActionKind.CREATE_HANDOFF)
        self.assertEqual(handoff.payload["team"], "billing")
        self.assertEqual(handoff.payload["priority"], "escalation")
        self.assertEqual(handoff.payload["summary"], "3 charges, plan still Free")
        self.assertEqual(handoff.payload["draft_reply_for_review"], "Thanks.")

    def test_failures_still_create_a_handoff_using_the_digest(self):
        degraded = TriageResult(ticket_id="T-1", urgency=Urgency.HIGH, action=Action.ESCALATE_TO_HUMAN, route_to=None,
                                analysis=None, status=ResultStatus.DEGRADED, status_reason="API down", digest="Customer c-1 ...")
        (handoff,) = plan_actions(degraded, "c-1")
        self.assertEqual(handoff.payload["team"], "triage_queue")
        self.assertEqual(handoff.payload["summary"], "Customer c-1 ...")
        self.assertEqual(handoff.payload["status"], "degraded")

    def test_feature_requests_are_not_logged_when_the_model_did_not_analyse(self):
        degraded = TriageResult(ticket_id="T-1", urgency=Urgency.HIGH, action=Action.ESCALATE_TO_HUMAN, route_to=None,
                                analysis=None, status=ResultStatus.DEGRADED, status_reason="x")
        self.assertNotIn(ActionKind.LOG_FEATURE_REQUEST, kinds(plan_actions(degraded)))

    def test_keys_are_stable_and_content_sensitive(self):
        a, b = plan_actions(result()), plan_actions(result())
        self.assertEqual([x.idempotency_key for x in a], [x.idempotency_key for x in b])
        other = plan_actions(result(analysis=make_analysis(feature_requests=["Something else"])))
        self.assertNotEqual(a[1].idempotency_key, other[1].idempotency_key)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "outbox"

    def lines(self, name):
        path = self.dir / name
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_actions_are_written_to_one_file_per_kind(self):
        summary = OutboxExecutor(self.dir).execute(plan_actions(result(), "cust-1"))
        self.assertEqual((summary.written, summary.duplicates_skipped), (2, 0))
        self.assertEqual(self.lines("replies.jsonl")[0]["payload"]["text"], "Here is how to enable dark mode.")
        self.assertEqual(self.lines("feature_requests.jsonl")[0]["payload"]["request"], "Scheduled dark mode")
        self.assertEqual(self.lines("handoffs.jsonl"), [])

    def test_running_twice_never_duplicates_even_with_a_new_executor(self):
        actions = plan_actions(result(), "cust-1")
        OutboxExecutor(self.dir).execute(actions)
        again = OutboxExecutor(self.dir).execute(actions)  # a fresh process re-reads the keys from disk
        self.assertEqual((again.written, again.duplicates_skipped), (0, 2))
        self.assertEqual(len(self.lines("replies.jsonl")), 1)

    def test_thai_text_is_stored_readably(self):
        thai = result(analysis=make_analysis(draft_reply="สวัสดีครับ", feature_requests=[]))
        OutboxExecutor(self.dir).execute(plan_actions(thai))
        self.assertIn("สวัสดีครับ", (self.dir / "replies.jsonl").read_text(encoding="utf-8"))

    def test_a_damaged_line_does_not_stop_the_executor(self):
        self.dir.mkdir()
        (self.dir / "replies.jsonl").write_text("{not json\n", encoding="utf-8")
        summary = OutboxExecutor(self.dir).execute(plan_actions(result()))
        self.assertEqual(summary.written, 2)

    def test_unwritable_outbox_raises_oserror(self):
        blocker = Path(self.tmp.name) / "file.txt"
        blocker.write_text("x", encoding="utf-8")
        with self.assertRaises(OSError):
            OutboxExecutor(blocker / "outbox").execute(plan_actions(result()))


if __name__ == "__main__":
    unittest.main()
