"""Master admission requires both dependency completion and valid final evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_master_baseline", ROOT / "scripts/run_master_baseline.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class TestMasterFinalAdmission(unittest.TestCase):
    def result(self, *, trace_passed=True, trace_events=4, source_passed=True, audit_status="SUCCEEDED") -> dict:
        statuses = {"plan": "SUCCEEDED", "measure": "SUCCEEDED", "audit": audit_status, "admit": "PENDING"}
        with patch.object(controller, "verify_execution_trace", return_value={"passed": trace_passed, "events": trace_events}), \
             patch.object(controller, "verify_source_manifest", return_value={"passed": source_passed}):
            return controller.finalize_result({}, controller.default_stage_graph(), statuses, Path("unused_trace"))

    def test_valid_final_evidence_admits_only_baseline_measurement(self) -> None:
        result = self.result()
        self.assertTrue(result["baseline_measured_and_audited"])
        self.assertEqual(result["statuses"]["admit"], "SUCCEEDED")
        self.assertFalse(result["predictive_admitted"])
        self.assertIsNone(result["official_rank"])

    def test_invalid_trace_cannot_coexist_with_success(self) -> None:
        result = self.result(trace_passed=False)
        self.assertFalse(result["baseline_measured_and_audited"])
        self.assertEqual(result["statuses"]["admit"], "BLOCKED")

    def test_empty_or_incomplete_trace_cannot_be_admitted(self) -> None:
        for events in (0, 2, 3, 5):
            with self.subTest(events=events):
                result = self.result(trace_events=events)
                self.assertFalse(result["baseline_measured_and_audited"])
                self.assertFalse(result["trace_complete"])

    def test_changed_source_and_unresolved_audit_cannot_be_admitted(self) -> None:
        for options in ({"source_passed": False}, {"audit_status": "UNRESOLVED"}, {"audit_status": "FAILED"}):
            with self.subTest(options=options):
                result = self.result(**options)
                self.assertFalse(result["baseline_measured_and_audited"])
                self.assertEqual(result["statuses"]["admit"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
