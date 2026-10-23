import unittest

from tests.helpers import SLA, TICKETS
from triage.schemas import Message, Plan, Ticket, Urgency


class SlaTests(unittest.TestCase):
    def test_evaluate_computes_remaining_and_breach(self):
        status = SLA.evaluate(TICKETS["T-002"], Urgency.CRITICAL)
        self.assertEqual((status.tier, status.target_minutes, status.waiting_minutes), ("enterprise-premium", 30, 120))
        self.assertEqual(status.remaining_minutes, -90)
        self.assertTrue(status.breached)

    def test_exactly_on_target_is_not_a_breach(self):
        ticket = Ticket(id="X", customer=TICKETS["T-002"].customer, messages=[Message(minutes_ago=30, text="hi")])
        self.assertFalse(SLA.evaluate(ticket, Urgency.CRITICAL).breached)

    def test_no_messages_means_no_wait(self):
        ticket = Ticket(id="X", customer=TICKETS["T-003"].customer, messages=[])
        self.assertEqual(SLA.evaluate(ticket, Urgency.LOW).waiting_minutes, 0)

    def test_agent_reply_resets_the_clock(self):
        customer = TICKETS["T-003"].customer
        ticket = Ticket(id="X", customer=customer, last_agent_reply_minutes_ago=100,
                        messages=[Message(minutes_ago=500, text="old"), Message(minutes_ago=40, text="new")])
        self.assertEqual(ticket.waiting_minutes(), 40)

    def test_describe_lists_targets_from_critical_down(self):
        text = SLA.describe(Plan.ENTERPRISE)
        self.assertTrue(text.startswith("enterprise-premium"))
        self.assertLess(text.index("critical 30m"), text.index("low 1440m"))


if __name__ == "__main__":
    unittest.main()
