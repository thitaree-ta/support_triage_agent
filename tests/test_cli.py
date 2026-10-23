import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import OutcomeLLM, SETTINGS, final, make_analysis
from triage import __main__ as cli
from triage.factory import build_agent, build_batch_runner


def run_cli(argv, llm, env=None):
    """Run the CLI with a fake LLM; returns (exit_code, stdout, stderr)."""
    env = {"OPENAI_API_KEY": "test-key", **(env or {})}
    out, err = io.StringIO(), io.StringIO()
    runner_factory = lambda settings: build_batch_runner(settings, build_agent(settings, llm=llm))
    with mock.patch.dict(os.environ, env), mock.patch.object(cli, "build_batch_runner", runner_factory), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dead_letter = str(Path(self.tmp.name) / "failed.jsonl")
        self.env = {"TRIAGE_DEAD_LETTER": self.dead_letter}

    def test_happy_path_prints_json_without_tool_outputs(self):
        llm = OutcomeLLM([final(make_analysis())])
        code, out, err = run_cli(["--json"], llm, self.env)
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual([d["ticket_id"] for d in data], ["T-001", "T-002", "T-003"])
        self.assertTrue(all(d["status"] == "ok" for d in data))
        self.assertIn("Summary: ok=3", err)

    def test_missing_api_key_is_a_usage_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), mock.patch.object(cli.Settings, "from_env", return_value=SETTINGS):
                self.assertEqual(cli.main([]), 2)
        self.assertIn("OPENAI_API_KEY", err.getvalue())

    def test_total_model_outage_alerts_and_saves_tickets_for_retry(self):
        llm = OutcomeLLM([RuntimeError("API down")])
        code, out, err = run_cli([], llm, self.env)
        self.assertEqual(code, 3)
        self.assertIn("ALERT", err)
        self.assertIn("circuit breaker opened", err)
        self.assertIn("--retry-failed", err)
        self.assertEqual(len(Path(self.dead_letter).read_text(encoding="utf-8").splitlines()), 3)
        self.assertIn("DEGRADED", out)
        # every ticket still gets a human-bound result with context
        self.assertEqual(out.count("Digest for the person picking this up"), 3)

        retry_code, retry_out, _ = run_cli(["--retry-failed", self.dead_letter, "--json"],
                                           OutcomeLLM([final(make_analysis())]), self.env)
        self.assertEqual(retry_code, 0)
        self.assertEqual(len(json.loads(retry_out)), 3)

    def test_unreadable_or_invalid_ticket_file_is_a_usage_error(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        for path in (str(bad), str(Path(self.tmp.name) / "missing.json")):
            with self.subTest(path):
                code, _, err = run_cli(["--tickets", path], OutcomeLLM([final(make_analysis())]), self.env)
                self.assertEqual(code, 2)
                self.assertIn("Cannot load tickets", err)

    def test_malformed_ticket_is_reported_but_others_still_run(self):
        good = json.loads((SETTINGS.data_dir / "tickets.json").read_text(encoding="utf-8"))[2]
        path = Path(self.tmp.name) / "mixed.json"
        path.write_text(json.dumps([good, {"id": "BROKEN"}]), encoding="utf-8")
        code, out, _ = run_cli(["--tickets", str(path), "--json"], OutcomeLLM([final(make_analysis())]), self.env)
        by_id = {d["ticket_id"]: d["status"] for d in json.loads(out)}
        self.assertEqual(by_id, {"BROKEN": "rejected", "T-003": "ok"})
        self.assertEqual(code, 3)  # 1 of 2 tickets lost to a system problem exceeds the 20% alert ratio

    def test_output_flag_writes_utf8_file_including_thai(self):
        out_path = Path(self.tmp.name) / "result.json"
        llm = OutcomeLLM([final(make_analysis(customer_language="Thai", draft_reply="สวัสดีครับ"))])
        code, out, err = run_cli(["--json", "--output", str(out_path)], llm, self.env)
        self.assertEqual(code, 0)
        self.assertEqual(out, "")  # nothing on stdout
        self.assertIn("Results written", err)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 3)
        self.assertIn("สวัสดีครับ", out_path.read_text(encoding="utf-8"))  # not escaped to \\u0e2a...

    def test_execute_writes_the_outbox_and_is_safe_to_repeat(self):
        outbox = Path(self.tmp.name) / "outbox"
        env = {**self.env, "TRIAGE_OUTBOX": str(outbox)}
        llm = OutcomeLLM([final(make_analysis())])  # clean auto-response for all 3 sample tickets
        # kb search is skipped by the fake, so the replies would be ungrounded: expect handoffs, not replies
        code, _, err = run_cli(["--json", "--execute"], llm, env)
        self.assertEqual(code, 0)
        self.assertIn("Actions executed: 3 written", err)
        self.assertEqual(len((outbox / "handoffs.jsonl").read_text(encoding="utf-8").splitlines()), 3)
        code, _, err = run_cli(["--json", "--execute"], OutcomeLLM([final(make_analysis())]), env)
        self.assertIn("0 written", err)
        self.assertIn("3 duplicate(s) skipped", err)

    def test_nothing_is_written_without_the_execute_flag(self):
        outbox = Path(self.tmp.name) / "outbox"
        run_cli(["--json"], OutcomeLLM([final(make_analysis())]), {**self.env, "TRIAGE_OUTBOX": str(outbox)})
        self.assertFalse(outbox.exists())

    def test_unwritable_outbox_is_reported_without_losing_results(self):
        blocker = Path(self.tmp.name) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        env = {**self.env, "TRIAGE_OUTBOX": str(blocker / "outbox")}
        code, out, err = run_cli(["--json", "--execute"], OutcomeLLM([final(make_analysis())]), env)
        self.assertEqual(code, 3)
        self.assertIn("could not write the outbox", err)
        self.assertEqual(len(json.loads(out)), 3)  # the triage results were still printed

    def test_unknown_ticket_id_is_a_usage_error(self):
        code, _, err = run_cli(["--ticket-id", "NOPE"], OutcomeLLM([final(make_analysis())]), self.env)
        self.assertEqual(code, 2)
        self.assertIn("No tickets matched", err)


if __name__ == "__main__":
    unittest.main()


def test_output_failure_preserves_results_and_does_not_execute(tmp_path):
    from unittest.mock import patch
    with patch.object(cli, "_execute") as execute:
        code, out, err = run_cli(
            ["--json", "--output", str(tmp_path), "--execute"],
            OutcomeLLM([final(make_analysis())]),
            {"TRIAGE_DEAD_LETTER": str(tmp_path / "failed.jsonl")},
        )
    assert code == 3
    assert len(json.loads(out)) == 3
    assert "No actions executed" in err
    execute.assert_not_called()
