import json
import unittest

from tests.helpers import SETTINGS, TICKETS, ScriptedLLM, calls, final, make_analysis, tool_call
from triage.factory import build_agent
from triage.schemas import Action, IssueType, ResultStatus, Sentiment, Team, Urgency


def agent_with(script, max_steps=6):
    from dataclasses import replace

    llm = ScriptedLLM(script)
    return build_agent(replace(SETTINGS, max_agent_steps=max_steps), llm=llm), llm


class SampleTicketTests(unittest.TestCase):
    def test_ticket_1_billing_is_escalated_to_billing(self):
        script = [
            calls(
                tool_call("get_billing_charges", "a", lookback_days=7),
                tool_call("search_knowledge_base", "b", query="duplicate pending charge refund"),
                tool_call("get_customer_history", "c", max_tickets=3),
            ),
            final(make_analysis(
                urgency=Urgency.HIGH, product="Billing / Pro upgrade", primary_issue_type=IssueType.BILLING,
                customer_sentiment=Sentiment.ANGRY, involves_money=True, kb_article_ids=["KB-102", "KB-103"],
                recommended_action=Action.ESCALATE_TO_HUMAN, route_to=Team.BILLING, confidence=0.9,
                handoff_summary="3 x $29.99 charged, plan still Free, sync error. Customer threatens dispute.",
            )),
        ]
        agent, llm = agent_with(script)
        result = agent.triage(TICKETS["T-001"])

        self.assertEqual((result.urgency, result.action, result.route_to),
                         (Urgency.HIGH, Action.ESCALATE_TO_HUMAN, Team.BILLING))
        self.assertEqual([t.name for t in result.tool_calls],
                         ["get_billing_charges", "search_knowledge_base", "get_customer_history"])
        self.assertTrue(all(t.ok for t in result.tool_calls))
        # tool results were fed back to the model, keyed by the model's call ids
        results, allow_tools = llm.session.received[1]
        self.assertEqual([r.call_id for r in results], ["a", "b", "c"])
        self.assertTrue(json.loads(results[0].output)["duplicate_charge_suspected"])
        self.assertTrue(allow_tools)
        self.assertEqual(result.adjustments, [])

    def test_ticket_2_under_triage_is_corrected_by_policy(self):
        """Model wrongly auto-responds to an Enterprise outage; guardrails must override it."""
        script = [
            calls(tool_call("check_service_status", "a", region="Thailand")),
            final(make_analysis(
                urgency=Urgency.HIGH, primary_issue_type=IssueType.OUTAGE, multiple_users_affected=True,
                customer_language="Thai", recommended_action=Action.AUTO_RESPOND, kb_article_ids=["KB-201"],
            )),
        ]
        agent, _ = agent_with(script)
        result = agent.triage(TICKETS["T-002"])

        self.assertEqual(result.urgency, Urgency.CRITICAL)
        self.assertEqual(result.action, Action.ESCALATE_TO_HUMAN)
        self.assertEqual(result.route_to, Team.INCIDENT_RESPONSE)
        self.assertFalse(result.auto_send)
        self.assertTrue({"multi_user_outage_floor", "critical_needs_human"}
                        <= {a.rule for a in result.adjustments})
        # Enterprise-premium critical target is 30 min; customer has waited 120 min.
        self.assertTrue(result.sla.breached)
        self.assertEqual((result.sla.target_minutes, result.sla.waiting_minutes, result.sla.remaining_minutes),
                         (30, 120, -90))
        self.assertTrue(json.loads(json.dumps(result.tool_calls[0].output))["status_page_may_be_stale"])

    def test_ticket_3_unresolved_dark_mode_is_reviewed_and_feature_request_logged(self):
        script = [
            calls(tool_call("search_knowledge_base", "a", query="dark mode system default schedule")),
            final(make_analysis(
                feature_requests=["Scheduled dark mode"], secondary_issue_types=[IssueType.FEATURE_REQUEST, IssueType.BUG],
                customer_sentiment=Sentiment.POSITIVE, kb_article_ids=["KB-301", "KB-303", "KB-302"],
            )),
        ]
        agent, _ = agent_with(script)
        result = agent.triage(TICKETS["T-003"])
        self.assertEqual(result.action, Action.ROUTE_TO_SPECIALIST)  # citations do not establish resolution
        self.assertFalse(result.auto_send)
        self.assertEqual(result.route_to, Team.TECHNICAL_SUPPORT)
        self.assertIn("unresolved_functional_issue", {a.rule for a in result.adjustments})
        self.assertEqual(result.urgency, Urgency.MEDIUM)
        self.assertEqual(result.analysis.feature_requests, ["Scheduled dark mode"])


class DeadlineAndSlaTests(unittest.TestCase):
    def test_ticket_1_deadline_floors_urgency_even_if_model_says_medium(self):
        script = [final(make_analysis(
            urgency=Urgency.MEDIUM, customer_deadline_minutes=120, kb_article_ids=[], draft_reply="ack",
            recommended_action=Action.ROUTE_TO_SPECIALIST, route_to=Team.BILLING,
        ))]
        agent, _ = agent_with(script)
        result = agent.triage(TICKETS["T-001"])
        self.assertEqual(result.urgency, Urgency.HIGH)
        self.assertIn("imminent_deadline_floor", {a.rule for a in result.adjustments})
        self.assertFalse(result.sla.breached)  # Free/high target 1440 min, waited 180

    def test_waiting_time_ignores_messages_already_answered(self):
        # T-003 had a human reply 2850 min ago; the oldest unanswered message is 1440 min old.
        self.assertEqual(TICKETS["T-003"].waiting_minutes(), 1440)
        self.assertEqual(TICKETS["T-002"].waiting_minutes(), 120)

    def test_model_sees_wait_time_and_sla_targets(self):
        agent, llm = agent_with([final(None)])
        agent.triage(TICKETS["T-002"])
        self.assertIn("waiting_without_human_reply_minutes: 120", llm.user_message)
        self.assertIn("enterprise-premium", llm.user_message)
        self.assertIn("critical 30m", llm.user_message)


class FailureModeTests(unittest.TestCase):
    def assert_safe_fallback(self, result, expected_reason):
        self.assertEqual((result.urgency, result.action), (Urgency.HIGH, Action.ESCALATE_TO_HUMAN))
        self.assertIsNone(result.analysis)
        self.assertEqual(result.status, ResultStatus.DEGRADED)
        self.assertIn(expected_reason, result.status_reason)
        self.assertFalse(result.auto_send)
        self.assertIn("Customer cust-", result.digest)  # a human gets context, not just an error

    def test_llm_exception_degrades_to_human_escalation(self):
        agent, _ = agent_with([RuntimeError("API down")])
        self.assert_safe_fallback(agent.triage(TICKETS["T-003"]), "RuntimeError: analysis unavailable")

    def test_empty_model_turn_degrades(self):
        agent, _ = agent_with([final(None)])
        self.assert_safe_fallback(agent.triage(TICKETS["T-003"]), "neither tool calls nor a valid analysis")

    def test_step_limit_forces_final_answer_then_falls_back(self):
        looping = calls(tool_call("search_knowledge_base", query="dark mode"))
        agent, llm = agent_with([looping, looping, looping], max_steps=3)
        result = agent.triage(TICKETS["T-003"])
        self.assert_safe_fallback(result, "final step")
        self.assertFalse(llm.session.received[-1][1])  # last call had tools disabled
        self.assertEqual(len(result.tool_calls), 2)  # trace of work done is preserved

    def test_failed_tool_forces_human_even_if_model_wants_auto_respond(self):
        script = [
            calls(tool_call("not_a_real_tool", "a"), tool_call("search_knowledge_base", "b", query="dark mode appearance")),
            final(make_analysis()),
        ]
        agent, _ = agent_with(script)
        result = agent.triage(TICKETS["T-003"])
        self.assertEqual(result.action, Action.ESCALATE_TO_HUMAN)
        self.assertIn("tool_failure", {a.rule for a in result.adjustments})

    def test_model_sees_ticket_text_escaped_and_tools_advertised(self):
        agent, llm = agent_with([final(None)])
        agent.triage(TICKETS["T-003"])
        self.assertIn("<ticket", llm.user_message)
        self.assertEqual({t.name for t in llm.tools},
                         {"search_knowledge_base", "get_customer_history", "get_billing_charges", "check_service_status"})


if __name__ == "__main__":
    unittest.main()
