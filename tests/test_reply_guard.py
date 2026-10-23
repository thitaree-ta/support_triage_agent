import unittest

from tests.helpers import TICKETS, make_analysis
from triage.policy import PolicyEngine
from triage.reply_guard import ReplyGuard
from triage.schemas import Action, ToolTrace

KB_TEXT = "Password resets: reset emails arrive within 5 minutes."
KB_TRACE = ToolTrace(name="search_knowledge_base", arguments={"query": "x"}, ok=True,
                     output={"results": [{"id": "KB-301", "content": KB_TEXT}]})


def rules(reply, kb_text=KB_TEXT):
    return {i.rule for i in ReplyGuard().check(reply, kb_text)}


class ReplyGuardTests(unittest.TestCase):
    def test_claiming_an_action_was_done_is_flagged(self):
        for reply in ("We’ve noted your request.", "We have logged this for the product team.",
                      "Your refund has been processed.", "I’m sending this to our Billing team.",
                      "บันทึกคำขอของคุณแล้วครับ", "เรากำลังส่งเรื่องให้ทีมที่เกี่ยวข้อง", "ส่งต่อให้ทีมที่เกี่ยวข้องแล้ว"):
            with self.subTest(reply):
                self.assertIn("claims_action", rules(reply))

    def test_promises_are_flagged(self):
        for reply in ("We will refund you in full.", "You will receive a full refund.", "Your refund has been approved.",
                      "We guarantee a fix.", "เราจะคืนเงินให้ภายในวันนี้"):
            with self.subTest(reply):
                self.assertIn("promises_outcome", rules(reply))
        for reply in ("We will reply within 2 hours.", "Expect a fix by tomorrow.", "ภายใน 3 วันทำการ"):
            with self.subTest(reply):
                self.assertIn("promises_timeline", rules(reply))

    def test_a_time_quoted_from_a_retrieved_kb_article_is_allowed(self):
        self.assertEqual(rules("Reset emails arrive within 5 minutes."), set())
        self.assertIn("promises_timeline", rules("Reset emails arrive within 5 minutes.", kb_text=""))

    def test_internal_details_and_outage_confirmation_are_flagged(self):
        for reply in ("Our internal monitoring shows a 41.8% 5xx error rate.", "The on-call engineer is looking.",
                      "We have confirmed an outage.", "พบรายงานความผิดพลาดของระบบในระดับที่ต้องตรวจสอบ"):
            with self.subTest(reply):
                self.assertIn("internal_disclosure", rules(reply))

    def test_asking_for_secrets_and_echoing_cards_are_flagged(self):
        self.assertIn("asks_for_secrets", rules("Please send me your card number and CVV."))
        self.assertIn("asks_for_secrets", rules("กรุณาส่งรหัสผ่านของคุณ"))
        self.assertIn("card_number_in_reply", rules("You paid with 4242 4242 4242 4242."))

    def test_normal_helpful_replies_are_clean(self):
        clean = [
            "Dark mode is available on all plans. Update the app via Help > Check for Updates.",
            "Support chat cannot issue refunds; a billing specialist will review this.",
            "If your app has been updated to 5.3, System Default follows your Mac.",
            "A custom schedule isn’t supported yet. As a workaround, use System Default.",
            "Thanks for the suggestion. A member of our team will review this.",
            "", None,
        ]
        for reply in clean:
            with self.subTest(reply):
                self.assertEqual(rules(reply), set())

    def test_duplicate_matches_are_reported_once(self):
        issues = ReplyGuard().check("We’ve noted it. We’ve noted it.")
        self.assertEqual(len([i for i in issues if i.rule == "claims_action"]), 1)


class GuardInPolicyTests(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine.default()
        self.ticket = TICKETS["T-003"]

    def apply(self, **overrides):
        return self.engine.apply(self.ticket, make_analysis(**overrides), [KB_TRACE])

    def test_unsafe_auto_reply_is_held_for_a_person(self):
        out = self.apply(draft_reply="Dark mode is on all plans. We’ve noted your request.")
        self.assertEqual(out.action, Action.ROUTE_TO_SPECIALIST)
        self.assertIn("unsafe_auto_reply", {a.rule for a in out.adjustments})
        self.assertEqual([i.rule for i in out.reply_issues], ["claims_action"])

    def test_clean_auto_reply_is_untouched(self):
        out = self.apply()
        self.assertEqual((out.action, out.reply_issues), (Action.AUTO_RESPOND, []))

    def test_issues_in_a_holding_message_are_recorded_but_do_not_change_the_action(self):
        out = self.apply(recommended_action=Action.ESCALATE_TO_HUMAN, draft_reply="We will refund you within 3 days.")
        self.assertEqual(out.action, Action.ESCALATE_TO_HUMAN)
        self.assertTrue({i.rule for i in out.reply_issues} >= {"promises_outcome", "promises_timeline"})


if __name__ == "__main__":
    unittest.main()
