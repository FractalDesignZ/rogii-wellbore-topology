"""End-to-end falsifiers for measurement admission and retained evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestPadIterationCLI(unittest.TestCase):
    def test_no_donors_rejects_and_audit_catches_artifact_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "attempt"
            command = [sys.executable, str(ROOT / "scripts/measure_manifold_gr_synth.py"), "--n-train", "0", "--out", str(out)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["falsifier_book_ok"])
            self.assertFalse(summary["falsifier_pad_transfer_ok"])
            self.assertFalse(summary["ok"])
            self.assertFalse(summary["predictive_admitted"])
            self.assertTrue(json.loads((out / "audit.json").read_text())["payload"]["audit_passed"])
            previous = "0" * 64
            stages = []
            for line in (out / "steps_log.jsonl").read_text().splitlines():
                entry = json.loads(line)
                digest = entry.pop("entry_hash")
                self.assertEqual(entry["prev_hash"], previous)
                raw = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
                self.assertEqual(digest, hashlib.sha256((previous + raw).encode()).hexdigest())
                previous = digest
                stages.append(entry["stage"])
            self.assertEqual(stages, ["plan", "mutate", "verify", "admit"])
            contract = (out / "contract.json").read_bytes()
            repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual((out / "contract.json").read_bytes(), contract)
            summary["ok"] = True
            (out / "summary.json").write_text(json.dumps(summary))
            audit = subprocess.run([sys.executable, str(ROOT / "scripts/audit_pad_measurement.py"), "--run", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(audit.returncode, 0)
            self.assertIn("artifact digest mismatch", audit.stderr)

    def test_declared_book_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "attempt"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/measure_manifold_gr_synth.py"), "--donor-policy", "full_tvt", "--book-mae-limit", "0.001", "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["falsifier_book_ok"])
            self.assertTrue(summary["falsifier_pad_transfer_ok"])
            self.assertFalse(summary["ok"])
            self.assertEqual(summary["config"]["book_mae_limit"], 0.001)


if __name__ == "__main__":
    unittest.main()
