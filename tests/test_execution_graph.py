"""Source and process evidence falsifiers; no private harness dependency."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.execution_graph import (  # noqa: E402
    build_source_manifest, default_stage_graph, dependency_ready, run_bounded,
    validate_stage_graph, verify_execution_trace, verify_source_manifest,
)


class TestExecutionGraph(unittest.TestCase):
    def test_static_relative_import_closure_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "src/rwt"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "a.py").write_text("from . import b\ndef f(): return b.g()\n")
            (package / "b.py").write_text("def g(): return 2\n")
            (root / "entry.py").write_text("from rwt.a import f\nf()\n")
            report = build_source_manifest(root, ["entry.py"])
            self.assertEqual({row["path"] for row in report["files"]}, {"entry.py", "src/rwt/__init__.py", "src/rwt/a.py", "src/rwt/b.py"})
            self.assertEqual(report, build_source_manifest(root, ["entry.py"]))
            self.assertTrue(report["syntax_only"])
            self.assertFalse(report["dynamic_calls_resolved"])
            self.assertTrue(verify_source_manifest(root, report)["passed"])
            (package / "b.py").write_text("def g(): return 3\n")
            self.assertFalse(verify_source_manifest(root, report)["passed"])

    def test_fixed_stage_dependencies_and_unresolved(self) -> None:
        graph = default_stage_graph()
        self.assertEqual(validate_stage_graph(list(reversed(graph))), ["plan", "measure", "audit", "admit"])
        self.assertFalse(dependency_ready("measure", graph, {"plan": "UNRESOLVED"}))
        self.assertTrue(dependency_ready("measure", graph, {"plan": "SUCCEEDED"}))
        for bad in (
            [{"id": "plan", "depends_on": ["admit"]}, *graph[1:]],
            [graph[0], {"id": "measure", "depends_on": ["missing"]}, *graph[2:]],
        ):
            with self.assertRaises(ValueError):
                validate_stage_graph(bad)
        with self.assertRaises(ValueError):
            dependency_ready("measure", graph, {"plan": "caller_passed"})

    def test_real_exit_trace_and_dependency_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, trace = Path(tmp), Path(tmp) / "trace.jsonl"
            blocked = run_bounded([sys.executable, "-c", "raise RuntimeError('must not run')"], cwd=root, trace_path=trace, node_id="measure", timeout_s=2, dependencies=("plan",), statuses={"plan": "UNRESOLVED"})
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertFalse((root / "measure.stdout.log").exists())
            receipt = run_bounded([sys.executable, "-c", "print('worker evidence')"], cwd=root, trace_path=trace, node_id="plan", timeout_s=5)
            self.assertEqual(receipt["status"], "SUCCEEDED")
            self.assertEqual(receipt["returncode"], 0)
            self.assertTrue(verify_execution_trace(trace)["passed"])
            rows = trace.read_text().splitlines()
            changed = json.loads(rows[-1])
            changed["returncode"] = 7
            rows[-1] = json.dumps(changed)
            trace.write_text("\n".join(rows) + "\n")
            self.assertFalse(verify_execution_trace(trace)["passed"])

    def test_launch_failure_is_recorded_without_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = run_bounded([str(root / "missing_executable")], cwd=root, trace_path=root / "trace.jsonl", node_id="measure", timeout_s=2)
            self.assertEqual(receipt["status"], "FAILED")
            self.assertIsNone(receipt["returncode"])
            self.assertTrue(verify_execution_trace(root / "trace.jsonl")["passed"])

    def test_timeout_kills_spawned_child_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = "import time; from pathlib import Path; time.sleep(1.5); Path('child_survived').write_text('bad')"
            parent = "import subprocess,sys,time; from pathlib import Path; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); Path('child_started').write_text('yes'); time.sleep(10)"
            receipt = run_bounded([sys.executable, "-c", parent], cwd=root, trace_path=root / "trace.jsonl", node_id="measure", timeout_s=0.5)
            self.assertEqual(receipt["status"], "TIMEOUT")
            self.assertTrue(receipt["tree_kill_attempted"])
            self.assertTrue((root / "child_started").exists())
            time.sleep(1.7)
            self.assertFalse((root / "child_survived").exists())
            self.assertTrue(verify_execution_trace(root / "trace.jsonl")["passed"])

    def test_success_waits_for_child_after_leader_exits_and_seals_final_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = "import time; from pathlib import Path; time.sleep(.25); print('child done', flush=True); Path('child_done').write_text('yes')"
            parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-c'," + repr(child) + "])"
            receipt = run_bounded([sys.executable, "-c", parent], cwd=root, trace_path=root / "trace.jsonl", node_id="measure", timeout_s=5)
            self.assertEqual(receipt["status"], "SUCCEEDED")
            self.assertTrue((root / "child_done").exists())
            log = root / "measure.stdout.log"
            self.assertIn(b"child done", log.read_bytes())
            self.assertEqual(hashlib.sha256(log.read_bytes()).hexdigest(), receipt["stdout_sha256"])

    def test_timeout_remains_active_after_leader_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = "import time; from pathlib import Path; time.sleep(1); print('late output', flush=True); Path('child_survived').write_text('bad')"
            parent = "import subprocess,sys; from pathlib import Path; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); Path('leader_exited').write_text('yes')"
            receipt = run_bounded([sys.executable, "-c", parent], cwd=root, trace_path=root / "trace.jsonl", node_id="measure", timeout_s=.5)
            self.assertTrue((root / "leader_exited").exists())
            self.assertEqual(receipt["status"], "TIMEOUT")
            self.assertEqual(receipt["returncode"], 0)
            self.assertTrue(receipt["tree_kill_attempted"])
            time.sleep(.8)
            self.assertFalse((root / "child_survived").exists())
            self.assertEqual(hashlib.sha256((root / "measure.stdout.log").read_bytes()).hexdigest(), receipt["stdout_sha256"])
            self.assertTrue(verify_execution_trace(root / "trace.jsonl")["passed"])

    def test_tree_cleanup_leaves_unrelated_process_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            independent = "import time; from pathlib import Path; time.sleep(1); Path('unrelated_done').write_text('yes')"
            unrelated = subprocess.Popen([sys.executable, "-c", independent], cwd=root,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                child = "import time; time.sleep(10)"
                parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-c'," + repr(child) + "])"
                receipt = run_bounded([sys.executable, "-c", parent], cwd=root, trace_path=root / "trace.jsonl", node_id="measure", timeout_s=.5)
                self.assertEqual(receipt["status"], "TIMEOUT")
                self.assertEqual(unrelated.wait(timeout=5), 0)
                self.assertTrue((root / "unrelated_done").exists())
            finally:
                if unrelated.poll() is None:
                    unrelated.kill()
                    unrelated.wait(timeout=5)

    @unittest.skipIf(os.name == "nt", "POSIX process-table fallback")
    def test_posix_fallback_tracks_child_after_leader_exit(self) -> None:
        with patch("rwt.execution_graph.sys.platform", "darwin"):
            self.test_success_waits_for_child_after_leader_exits_and_seals_final_log()


if __name__ == "__main__":
    unittest.main()
