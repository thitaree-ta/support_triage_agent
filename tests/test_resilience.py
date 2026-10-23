import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import SETTINGS, TICKETS, OutcomeLLM, ScriptedLLM, SLA, calls, clone_ticket, final, make_analysis, tool_call
from triage.agent import TriageAgent
from triage.batch import BatchRunner, TicketPipeline
from triage.breaker import CircuitBreaker
from triage.datastore import load_tickets_lenient, parse_tickets
from triage.deadletter import DeadLetterWriter, load_dead_letter
from triage.factory import build_agent, build_registry
from triage.policy import PolicyEngine
from triage.preflight import CARD_PLACEHOLDER, Preflight
from triage.schemas import Action, Message, ResultStatus, Ticket

API_DOWN = RuntimeError("API down")


def make_runner(llm, threshold=3, dead_letter=None):
    agent = build_agent(SETTINGS, llm=llm)
    breaker = CircuitBreaker(threshold)
    pipeline = TicketPipeline(agent, Preflight(), breaker, SLA)
    return BatchRunner(pipeline, breaker, dead_letter), breaker


class AgentFailureTests(unittest.TestCase):
    def test_policy_crash_degrades_instead_of_raising(self):
        class Exploder:
            name = "exploder"

            def check(self, ctx):
                raise ZeroDivisionError("bug in a rule")

        agent = TriageAgent(ScriptedLLM([final(make_analysis())]), build_registry(SETTINGS),
                            PolicyEngine([], [Exploder()]), "system", sla_policy=SLA)
        result = agent.triage(TICKETS["T-003"])
        self.assertEqual(result.status, ResultStatus.DEGRADED)
        self.assertIn("ZeroDivisionError", result.status_reason)
        self.assertNotIn("bug in a rule", result.status_reason)
        self.assertEqual(result.action, Action.ESCALATE_TO_HUMAN)

    def test_ticket_time_budget_is_enforced_between_steps(self):
        clock = iter([0.0, 1.0, 2.0, 3.0, 500.0]).__next__  # deadline set at 0+90, step 1 at 1s, step 2 at 500s
        llm = ScriptedLLM([calls(tool_call("search_knowledge_base", query="dark mode")), final(make_analysis())])
        agent = build_agent(SETTINGS, llm=llm)
        agent._timeout, agent._clock = 90.0, clock
        result = agent.triage(TICKETS["T-003"])
        self.assertEqual(result.status, ResultStatus.DEGRADED)
        self.assertIn("time budget", result.status_reason)
        self.assertEqual(len(result.tool_calls), 1)  # work done so far is preserved
        self.assertIn("search_knowledge_base", result.digest)

    def test_degraded_result_still_carries_sla_and_digest(self):
        result = build_agent(SETTINGS, llm=ScriptedLLM([API_DOWN])).triage(TICKETS["T-002"])
        self.assertEqual(result.status, ResultStatus.DEGRADED)
        self.assertTrue(result.sla.breached)  # a failing model must not hide an SLA breach
        self.assertIn("status.company.com", result.digest)  # the latest message is quoted for the human


class PipelineTests(unittest.TestCase):
    def test_blank_ticket_never_reaches_the_model(self):
        llm = OutcomeLLM([API_DOWN])
        runner, _ = make_runner(llm)
        blank = Ticket(id="B", customer=TICKETS["T-003"].customer, messages=[Message(minutes_ago=3, text="  ")])
        report = runner.run([blank])
        result = report.results[0]
        self.assertEqual((result.status, result.action), (ResultStatus.BYPASSED, Action.ROUTE_TO_SPECIALIST))
        self.assertEqual(llm.sessions, 0)
        self.assertEqual(report.failures, 0)  # a content problem, not a system failure

    def test_circuit_opens_after_consecutive_failures_and_stops_calling_the_model(self):
        llm = OutcomeLLM([API_DOWN])
        runner, breaker = make_runner(llm, threshold=3)
        tickets = [clone_ticket(f"X{i}") for i in range(6)]
        report = runner.run(tickets)
        statuses = [r.status for r in report.results]
        self.assertEqual(statuses, [ResultStatus.DEGRADED] * 3 + [ResultStatus.BYPASSED] * 3)
        self.assertEqual(llm.sessions, 3)  # no wasted calls once the circuit is open
        self.assertTrue(report.breaker_opened and breaker.is_open)
        self.assertIn("circuit open", report.results[-1].status_reason)
        self.assertTrue(all(r.action is Action.ESCALATE_TO_HUMAN for r in report.results))
        self.assertEqual(report.failure_ratio, 1.0)

    def test_a_success_resets_the_failure_streak(self):
        good = final(make_analysis())
        llm = OutcomeLLM([API_DOWN, API_DOWN, good, API_DOWN, API_DOWN])
        runner, breaker = make_runner(llm, threshold=3)
        report = runner.run([clone_ticket(f"Y{i}") for i in range(5)])
        self.assertFalse(breaker.is_open)
        self.assertEqual(llm.sessions, 5)
        self.assertEqual([r.status for r in report.results].count(ResultStatus.OK), 1)

    def test_breaker_can_be_disabled(self):
        runner, breaker = make_runner(OutcomeLLM([API_DOWN]), threshold=None)
        runner.run([clone_ticket(f"Z{i}") for i in range(10)])
        self.assertFalse(breaker.is_open)

    def test_one_bad_ticket_does_not_stop_the_batch(self):
        good = final(make_analysis())
        runner, _ = make_runner(OutcomeLLM([good, API_DOWN, good]), threshold=5)
        report = runner.run([clone_ticket(f"W{i}") for i in range(3)])
        self.assertEqual([r.status for r in report.results],
                         [ResultStatus.OK, ResultStatus.DEGRADED, ResultStatus.OK])

    def test_pipeline_crash_is_contained(self):
        class BrokenPreflight:
            def run(self, ticket):
                raise KeyError("boom")

        agent = build_agent(SETTINGS, llm=OutcomeLLM([final(make_analysis())]))
        out = TicketPipeline(agent, BrokenPreflight(), CircuitBreaker(3), SLA).process(TICKETS["T-003"])
        self.assertEqual(out.result.status, ResultStatus.DEGRADED)
        self.assertTrue(out.retryable)


class DeadLetterAndInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "failed.jsonl"
        self.addCleanup(self.tmp.cleanup)

    def test_failed_tickets_are_saved_and_can_be_re_run(self):
        runner, _ = make_runner(OutcomeLLM([API_DOWN]), threshold=None, dead_letter=DeadLetterWriter(self.path))
        report = runner.run([clone_ticket("R1"), clone_ticket("R2")])
        self.assertEqual(report.dead_lettered, 2)
        lines = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([entry["ticket_id"] for entry in lines], ["R1", "R2"])
        self.assertEqual(lines[0]["status"], "degraded")
        tickets, rejected = parse_tickets(load_dead_letter(self.path))
        self.assertEqual([t.id for t in tickets], ["R1", "R2"])
        self.assertEqual(rejected, [])
        # the retry succeeds once the model is back
        retry_runner, _ = make_runner(OutcomeLLM([final(make_analysis())]))
        self.assertTrue(all(r.status is ResultStatus.OK for r in retry_runner.run(tickets).results))

    def test_dead_letter_never_stores_raw_card_numbers(self):
        base = TICKETS["T-003"]
        paying = Ticket(id="C1", customer=base.customer,
                        messages=[Message(minutes_ago=5, text="card 4242 4242 4242 4242 charged twice")])
        runner, _ = make_runner(OutcomeLLM([API_DOWN]), dead_letter=DeadLetterWriter(self.path))
        runner.run([paying])
        content = self.path.read_text(encoding="utf-8")
        self.assertNotIn("4242 4242", content)
        self.assertIn(CARD_PLACEHOLDER, content)

    def test_successes_and_blank_tickets_are_not_dead_lettered(self):
        runner, _ = make_runner(OutcomeLLM([final(make_analysis())]), dead_letter=DeadLetterWriter(self.path))
        blank = Ticket(id="B", customer=TICKETS["T-003"].customer, messages=[Message(minutes_ago=1, text="")])
        report = runner.run([clone_ticket("OK1"), blank])
        self.assertEqual(report.dead_lettered, 0)
        self.assertFalse(self.path.exists())

    def test_one_malformed_ticket_is_rejected_without_blocking_the_rest(self):
        good = TICKETS["T-003"].model_dump(mode="json")
        tickets, rejected = parse_tickets([good, {"id": "BAD", "messages": "not a list"}, "garbage"])
        self.assertEqual(len(tickets), 1)
        self.assertEqual([r.index for r in rejected], [1, 2])
        runner, _ = make_runner(OutcomeLLM([final(make_analysis())]), dead_letter=DeadLetterWriter(self.path))
        report = runner.run(tickets, rejected)
        by_id = {r.ticket_id: r for r in report.results}
        self.assertEqual(by_id["BAD"].status, ResultStatus.REJECTED)
        self.assertEqual(by_id["input-2"].status, ResultStatus.REJECTED)
        self.assertEqual(by_id["T-003"].status, ResultStatus.OK)
        self.assertEqual(report.dead_lettered, 2)
        self.assertAlmostEqual(report.failure_ratio, 2 / 3)

    def test_lenient_loader_reads_the_sample_file_and_rejects_non_arrays(self):
        tickets, rejected = load_tickets_lenient(SETTINGS.data_dir / "tickets.json")
        self.assertEqual((len(tickets), len(rejected)), (3, 0))
        not_a_list = Path(self.tmp.name) / "obj.json"
        not_a_list.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_tickets_lenient(not_a_list)


if __name__ == "__main__":
    unittest.main()
