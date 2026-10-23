import unittest

from tests.helpers import TICKETS
from triage.preflight import CARD_PLACEHOLDER, Preflight, luhn_valid, mask_card_numbers
from triage.schemas import Message, Ticket


def ticket_with(*texts: str, reply=None) -> Ticket:
    base = TICKETS["T-003"]
    messages = [Message(minutes_ago=100 - i, text=t) for i, t in enumerate(texts)]
    return Ticket(id="P", customer=base.customer, messages=messages, last_agent_reply_minutes_ago=reply)


class CardMaskingTests(unittest.TestCase):
    def test_valid_card_numbers_are_masked_in_common_formats(self):
        for raw in ("4242 4242 4242 4242", "4242-4242-4242-4242", "4242424242424242"):
            with self.subTest(raw):
                text, masked = mask_card_numbers(f"my card {raw} was charged")
                self.assertTrue(masked)
                self.assertEqual(text, f"my card {CARD_PLACEHOLDER} was charged")

    def test_numbers_that_are_not_cards_are_left_alone(self):
        for text in ("order 1234 5678 9012 3456", "call 0812345678", "invoice 20260412", "ticket 12345"):
            with self.subTest(text):
                self.assertEqual(mask_card_numbers(text), (text, False))

    def test_luhn(self):
        self.assertTrue(luhn_valid("4242424242424242"))
        self.assertFalse(luhn_valid("4242424242424241"))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.pre = Preflight(max_message_chars=100, max_messages=3)

    def test_all_blank_ticket_bypasses_the_model(self):
        out = self.pre.run(ticket_with("   ", "\n\t"))
        self.assertIn("no readable customer message", out.bypass_reason)

    def test_blank_messages_are_dropped_and_noted(self):
        out = self.pre.run(ticket_with("hello", " "))
        self.assertEqual([m.text for m in out.ticket.messages], ["hello"])
        self.assertTrue(any("blank" in n for n in out.notes))
        self.assertIsNone(out.bypass_reason)

    def test_long_message_is_truncated_with_a_marker(self):
        out = self.pre.run(ticket_with("x" * 500))
        text = out.ticket.messages[0].text
        self.assertTrue(text.endswith("[TRUNCATED]"))
        self.assertLessEqual(len(text), 100 + len(" [TRUNCATED]"))
        self.assertTrue(any("truncated" in n for n in out.notes))

    def test_only_the_newest_messages_are_kept(self):
        out = self.pre.run(ticket_with("m1", "m2", "m3", "m4", "m5"))
        self.assertEqual([m.text for m in out.ticket.messages], ["m3", "m4", "m5"])
        self.assertTrue(any("latest 3 of 5" in n for n in out.notes))

    def test_cards_masked_and_original_ticket_untouched(self):
        original = ticket_with("card 4242 4242 4242 4242 charged twice")
        out = self.pre.run(original)
        self.assertIn(CARD_PLACEHOLDER, out.ticket.messages[0].text)
        self.assertIn("4242 4242 4242 4242", original.messages[0].text)
        self.assertTrue(any("card number masked" in n for n in out.notes))

    def test_agent_reply_marker_is_preserved(self):
        self.assertEqual(self.pre.run(ticket_with("hi", reply=50)).ticket.last_agent_reply_minutes_ago, 50)


if __name__ == "__main__":
    unittest.main()
