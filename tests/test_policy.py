import unittest

from tests.helpers import SLA, TICKETS, make_analysis
from triage.policy import PolicyEngine
from triage.schemas import Action, IssueType, Message, Sentiment, Team, Ticket, ToolTrace, Urgency

KB_OK = ToolTrace(
    name="search_knowledge_base", arguments={"query": "x"}, ok=True,
    output={"results": [{"id": "KB-301"}, {"id": "KB-303"}]},
)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine.default(low_confidence_threshold=0.6)
        self.benign = TICKETS["T-003"]  # Pro customer

    def apply(self, ticket=None, traces=(KB_OK,), **overrides):
        return self.engine.apply(ticket or self.benign, make_analysis(**overrides), list(traces))

    def rules(self, outcome):
        return {a.rule for a in outcome.adjustments}

    def test_benign_grounded_ticket_is_left_untouched(self):
        out = self.apply()
        self.assertEqual((out.urgency, out.action, out.route_to), (Urgency.LOW, Action.AUTO_RESPOND, None))
        self.assertEqual(out.adjustments, [])

    def test_money_raises_urgency_and_forces_human_even_if_model_says_auto(self):
        out = self.apply(
            TICKETS["T-001"], involves_money=True, primary_issue_type=IssueType.BILLING,
            urgency=Urgency.MEDIUM,
        )
        self.assertEqual(out.urgency, Urgency.HIGH)
        self.assertEqual(out.action, Action.ESCALATE_TO_HUMAN)
        self.assertEqual(out.route_to, Team.BILLING)  # inferred destination
        self.assertTrue({"money_issue_floor", "money_needs_human"} <= self.rules(out))

    def test_multi_user_outage_is_critical_and_escalated(self):
        out = self.apply(
            TICKETS["T-002"], primary_issue_type=IssueType.OUTAGE, multiple_users_affected=True,
            urgency=Urgency.HIGH, recommended_action=Action.ROUTE_TO_SPECIALIST,
        )
        self.assertEqual(out.urgency, Urgency.CRITICAL)
        self.assertEqual(out.action, Action.ESCALATE_TO_HUMAN)
        self.assertEqual(out.route_to, Team.INCIDENT_RESPONSE)

    def test_enterprise_high_urgency_escalates(self):
        out = self.apply(TICKETS["T-002"], urgency=Urgency.HIGH, recommended_action=Action.ROUTE_TO_SPECIALIST,
                         route_to=Team.TECHNICAL_SUPPORT)
        self.assertEqual(out.action, Action.ESCALATE_TO_HUMAN)
        self.assertIn("enterprise_high_needs_human", self.rules(out))

    def test_enterprise_low_urgency_is_not_forced_up(self):
        out = self.apply(TICKETS["T-002"], urgency=Urgency.LOW)
        self.assertEqual(out.action, Action.AUTO_RESPOND)

    def test_low_confidence_and_invalid_confidence_escalate(self):
        for confidence in (0.4, 1.7, -0.1, float("nan")):
            with self.subTest(confidence=confidence):
                self.assertEqual(self.apply(confidence=confidence).action, Action.ESCALATE_TO_HUMAN)

    def test_failed_tool_escalates(self):
        failed = ToolTrace(name="get_billing_charges", arguments={}, ok=False, error="boom")
        out = self.apply(traces=[KB_OK, failed])
        self.assertEqual(out.action, Action.ESCALATE_TO_HUMAN)
        self.assertIn("tool_failure", self.rules(out))

    def test_angry_customer_never_gets_bot_only_reply(self):
        out = self.apply(customer_sentiment=Sentiment.ANGRY)
        self.assertEqual(out.action, Action.ROUTE_TO_SPECIALIST)
        self.assertEqual(out.route_to, Team.TECHNICAL_SUPPORT)

    def test_auto_respond_must_be_grounded_in_retrieved_articles(self):
        cases = {
            "uncited": dict(kb_article_ids=[]),
            "hallucinated id": dict(kb_article_ids=["KB-999"]),
            "no draft": dict(draft_reply="  "),
        }
        for label, overrides in cases.items():
            with self.subTest(label):
                self.assertEqual(self.apply(**overrides).action, Action.ROUTE_TO_SPECIALIST)
        # citing an article that was not retrieved because KB was never searched
        self.assertEqual(self.apply(traces=[]).action, Action.ROUTE_TO_SPECIALIST)

    def test_policy_never_lowers_a_cautious_model_decision(self):
        out = self.apply(urgency=Urgency.CRITICAL, recommended_action=Action.ESCALATE_TO_HUMAN, route_to=Team.TECHNICAL_SUPPORT)
        self.assertEqual((out.urgency, out.action), (Urgency.CRITICAL, Action.ESCALATE_TO_HUMAN))
        low = self.apply(recommended_action=Action.ESCALATE_TO_HUMAN, route_to=Team.TECHNICAL_SUPPORT)
        self.assertEqual(low.action, Action.ESCALATE_TO_HUMAN)

    def test_auto_respond_never_carries_a_route(self):
        self.assertIsNone(self.apply(route_to=Team.PRODUCT).route_to)

    # ---- deadline and SLA -------------------------------------------------
    def test_imminent_customer_deadline_raises_urgency_to_high(self):
        out = self.apply(urgency=Urgency.MEDIUM, customer_deadline_minutes=120)
        self.assertEqual(out.urgency, Urgency.HIGH)
        self.assertIn("imminent_deadline_floor", self.rules(out))

    def test_distant_or_missing_deadline_does_not_change_urgency(self):
        self.assertEqual(self.apply(customer_deadline_minutes=2000).urgency, Urgency.LOW)
        self.assertEqual(self.apply(customer_deadline_minutes=None).urgency, Urgency.LOW)

    def aged_pro_ticket(self, minutes_ago: int) -> Ticket:
        return Ticket(id="X", customer=self.benign.customer, messages=[Message(minutes_ago=minutes_ago, text="hi")])

    def apply_with_sla(self, ticket, **overrides):
        return self.engine.apply(ticket, make_analysis(**overrides), [KB_OK], SLA)

    def test_breached_sla_on_high_urgency_escalates(self):
        # Pro/high target is 240 min; customer has waited 300.
        out = self.apply_with_sla(self.aged_pro_ticket(300), urgency=Urgency.HIGH,
                                  recommended_action=Action.ROUTE_TO_SPECIALIST, route_to=Team.TECHNICAL_SUPPORT)
        self.assertTrue(out.sla.breached)
        self.assertEqual(out.sla.remaining_minutes, -60)
        self.assertEqual(out.action, Action.ESCALATE_TO_HUMAN)
        self.assertIn("sla_breach", self.rules(out))

    def test_breached_sla_on_medium_urgency_routes_to_a_person(self):
        # Pro/medium target is 1440 min; waited 3000.
        out = self.apply_with_sla(self.aged_pro_ticket(3000), urgency=Urgency.MEDIUM)
        self.assertEqual(out.action, Action.ROUTE_TO_SPECIALIST)

    def test_sla_within_target_changes_nothing(self):
        out = self.apply_with_sla(self.aged_pro_ticket(30), urgency=Urgency.MEDIUM)
        self.assertFalse(out.sla.breached)
        self.assertEqual(out.action, Action.AUTO_RESPOND)

    def test_sla_is_evaluated_with_the_final_urgency(self):
        # Model says LOW (target 2880, not breached) but money floor makes it HIGH (target 240).
        out = self.apply_with_sla(self.aged_pro_ticket(300), involves_money=True)
        self.assertEqual(out.sla.urgency, Urgency.HIGH)
        self.assertTrue(out.sla.breached)


class RiskCategoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine.default()
        self.ticket = TICKETS["T-003"]

    def apply(self, traces=(KB_OK,), engine=None, **overrides):
        return (engine or self.engine).apply(self.ticket, make_analysis(**overrides), list(traces))

    def test_security_issue_is_critical_and_goes_to_the_security_team(self):
        out = self.apply(primary_issue_type=IssueType.SECURITY)
        self.assertEqual((out.urgency, out.action, out.route_to), (Urgency.CRITICAL, Action.ESCALATE_TO_HUMAN, Team.SECURITY))

    def test_legal_matter_even_as_a_secondary_topic_is_escalated(self):
        out = self.apply(secondary_issue_types=[IssueType.LEGAL_PRIVACY])
        self.assertEqual((out.urgency, out.action, out.route_to), (Urgency.HIGH, Action.ESCALATE_TO_HUMAN, Team.LEGAL_COMPLIANCE))

    def test_spam_is_never_auto_answered(self):
        out = self.apply(primary_issue_type=IssueType.SPAM)
        self.assertEqual(out.action, Action.ROUTE_TO_SPECIALIST)
        self.assertEqual(out.route_to, Team.TECHNICAL_SUPPORT)

    def status_trace(self, degraded: bool) -> ToolTrace:
        return ToolTrace(name="check_service_status", arguments={"region": "x"}, ok=True,
                         output={"internal_degraded": degraded})

    def test_degraded_monitoring_raises_access_tickets_even_if_model_missed_it(self):
        out = self.apply(traces=[KB_OK, self.status_trace(True)], primary_issue_type=IssueType.ACCOUNT_ACCESS,
                         multiple_users_affected=False)
        self.assertEqual(out.urgency, Urgency.HIGH)
        self.assertIn("outage_evidence_floor", {a.rule for a in out.adjustments})

    def test_healthy_monitoring_or_unrelated_topic_does_not_raise_urgency(self):
        healthy = self.apply(traces=[KB_OK, self.status_trace(False)], primary_issue_type=IssueType.ACCOUNT_ACCESS)
        self.assertEqual(healthy.urgency, Urgency.LOW)
        unrelated = self.apply(traces=[KB_OK, self.status_trace(True)], primary_issue_type=IssueType.HOW_TO)
        self.assertEqual(unrelated.urgency, Urgency.LOW)

    def test_kill_switch_turns_auto_respond_into_routing(self):
        off = PolicyEngine.default(auto_respond_enabled=False)
        out = self.apply(engine=off)
        self.assertEqual(out.action, Action.ROUTE_TO_SPECIALIST)
        self.assertIn("auto_respond_disabled", {a.rule for a in out.adjustments})
        self.assertEqual(self.apply().action, Action.AUTO_RESPOND)  # default engine unaffected


if __name__ == "__main__":
    unittest.main()
