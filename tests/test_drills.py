import json
import logging
import tempfile
import unittest
from pathlib import Path

from drills.__main__ import main as drills_main
from drills.scenarios import SCENARIOS
from tests.helpers import SETTINGS
from triage.datastore import load_json, load_tickets_lenient

INPUTS = Path(__file__).resolve().parent.parent / "examples" / "failure_inputs"


class DrillTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_every_drill_holds(self):
        self.assertGreaterEqual(len(SCENARIOS), 21)
        for scenario in SCENARIOS:
            with self.subTest(scenario.name):
                _, checks = scenario.run()
                failed = [c.label for c in checks if not c.ok]
                self.assertEqual(failed, [])

    def test_scenario_names_are_unique(self):
        names = [s.name for s in SCENARIOS]
        self.assertEqual(len(names), len(set(names)))

    def test_drills_cli_exit_codes(self):
        self.assertEqual(drills_main(["--scenario", "blank_ticket"]), 0)
        self.assertEqual(drills_main(["--scenario", "does_not_exist"]), 2)


class FailureInputFileTests(unittest.TestCase):
    def test_all_files_are_parseable_json(self):
        files = sorted(INPUTS.glob("*.json"))
        self.assertGreaterEqual(len(files), 5)
        for f in files:
            with self.subTest(f.name):
                self.assertIsInstance(load_json(f), list)

    def test_mixed_file_has_one_good_ticket_and_two_rejects(self):
        tickets, rejected = load_tickets_lenient(INPUTS / "mixed_valid_and_broken.json")
        self.assertEqual([t.id for t in tickets], ["F-GOOD"])
        self.assertEqual(len(rejected), 2)

    def test_files_with_a_utf8_bom_load(self):
        # PowerShell 5 writes a BOM when saving with -Encoding utf8
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.json"
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps([{"id": "x"}]).encode("utf-8"))
            self.assertEqual(load_json(path), [{"id": "x"}])

    def test_sample_data_still_loads(self):
        self.assertEqual(len(load_tickets_lenient(SETTINGS.data_dir / "tickets.json")[0]), 3)


if __name__ == "__main__":
    unittest.main()
