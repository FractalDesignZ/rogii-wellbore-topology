"""Public, finite wellbore execution evidence using portable standard-library APIs.

AST records describe syntax, not runtime call resolution or a security sandbox.
This module is an independent RWT implementation, not a native MaxOp adapter.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


_STAGES = ("plan", "measure", "audit", "admit")
_STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "TIMEOUT", "BLOCKED", "UNRESOLVED"}
_ZERO = "0" * 64


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes repository: {path}") from exc
    return resolved


def build_source_manifest(repo_root: Path, entry_paths: list[str | Path]) -> dict[str, Any]:
    """Hash entry scripts and their recursively resolvable static ``rwt`` imports.

    Relative imports and ``from package import module`` syntax are retained.
    Imports outside rwt are listed but not traversed. Dynamic imports and calls
    cannot be resolved by this syntax walk and are never claimed as a closure.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir() or not entry_paths:
        raise ValueError("a repository directory and explicit entry paths are required")
    entries = []
    for item in entry_paths:
        path = _contained(root, Path(item) if Path(item).is_absolute() else root / item)
        if path.suffix != ".py" or not path.is_file():
            raise ValueError(f"entry must be an existing Python source: {item}")
        entries.append(path.relative_to(root).as_posix())
    entries = sorted(set(entries))

    def module_path(name: str) -> Path | None:
        parts = name.split(".")
        if not parts or parts[0] != "rwt":
            return None
        base = root.joinpath("src", *parts)
        for candidate in (base / "__init__.py", base.with_suffix(".py")):
            if candidate.is_file():
                return _contained(root, candidate)
        return None

    pending = [root / entry for entry in entries]
    files: dict[str, dict[str, Any]] = {}
    edges = []
    while pending:
        path = _contained(root, pending.pop())
        relative = path.relative_to(root).as_posix()
        if relative in files:
            continue
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=relative)
        module_parts = path.relative_to(root).parts
        package = None
        if module_parts[:2] == ("src", "rwt"):
            parts = list(module_parts[1:])
            if parts[-1] == "__init__.py":
                parts.pop()
            else:
                parts[-1] = Path(parts[-1]).stem
                parts.pop()
            package = ".".join(parts)
        imports, calls = [], []
        file_id = _digest({"kind": "source", "path": relative})
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    bases = names
                    imported = names
                    syntax = "import"
                else:
                    name = node.module or ""
                    if node.level:
                        try:
                            name = importlib.util.resolve_name("." * node.level + name, package or "")
                        except (ImportError, ValueError):
                            name = "." * node.level + name
                    names = [name]
                    imported = [alias.name for alias in node.names]
                    bases = [name] + [name + "." + alias for alias in imported if alias != "*"]
                    syntax = "from"
                resolved = set()
                for name in bases:
                    candidate = module_path(name)
                    if candidate is not None:
                        resolved.add(candidate.relative_to(root).as_posix())
                        pending.append(candidate)
                        components = name.split(".")
                        for length in range(1, len(components)):
                            initializer = module_path(".".join(components[:length]))
                            if initializer is not None and initializer.name == "__init__.py":
                                pending.append(initializer)
                imports.append({"line": node.lineno, "syntax": syntax, "modules": names, "names": imported, "resolved_paths": sorted(resolved)})
                for target in sorted(resolved):
                    edge = {"source": file_id, "target": _digest({"kind": "source", "path": target}), "kind": "static_import", "line": node.lineno}
                    edges.append({"id": _digest(edge), **edge})
            elif isinstance(node, ast.Call):
                call = {"file": relative, "line": node.lineno, "column": node.col_offset, "expression": ast.unparse(node.func), "kind": "syntax_call"}
                calls.append({"id": _digest(call), **call})
        files[relative] = {
            "id": file_id, "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "imports": sorted(imports, key=lambda item: (item["line"], item["modules"])),
            "calls": sorted(calls, key=lambda item: (item["line"], item["column"])),
        }
    manifest = {
        "schema": "rwt.static-source-graph.v1", "entries": entries,
        "files": [files[key] for key in sorted(files)],
        "edges": sorted(edges, key=lambda item: item["id"]),
        "syntax_only": True, "dynamic_calls_resolved": False,
        "external_packages_hashed": False, "security_sandbox": False,
    }
    return {**manifest, "manifest_sha256": _digest(manifest)}


def verify_source_manifest(repo_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Rebuild static evidence so source, imports, and new resolutions are checked."""
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    findings = []
    if manifest.get("manifest_sha256") != _digest(body):
        findings.append("manifest_digest_mismatch")
    try:
        live = build_source_manifest(repo_root, manifest.get("entries", []))
        if live != manifest:
            findings.append("live_source_or_static_graph_mismatch")
    except (OSError, ValueError, SyntaxError) as exc:
        findings.append(f"source_verification_error:{type(exc).__name__}")
    return {"passed": not findings, "findings": findings, "syntax_only": True}


def default_stage_graph() -> list[dict[str, Any]]:
    return [{"id": stage, "depends_on": [] if index == 0 else [_STAGES[index - 1]]} for index, stage in enumerate(_STAGES)]


def validate_stage_graph(stages: list[dict[str, Any]]) -> list[str]:
    """Validate the finite plan→measure→audit→admit DAG and return its order."""
    if not isinstance(stages, list) or any(not isinstance(row, dict) for row in stages):
        raise ValueError("stages must be a list of stage records")
    ids = [row.get("id") for row in stages]
    if any(not isinstance(stage, str) for stage in ids) or len(set(ids)) != len(ids):
        raise ValueError("stage IDs must be unique strings")
    deps = {}
    for row in stages:
        values = row.get("depends_on")
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("stage dependencies must be explicit string lists")
        if len(set(values)) != len(values) or any(value not in ids for value in values):
            raise ValueError("missing or duplicate dependency")
        deps[row["id"]] = set(values)
    order = []
    while len(order) < len(ids):
        ready = sorted(stage for stage in ids if stage not in order and deps[stage] <= set(order))
        if not ready:
            raise ValueError("stage dependency cycle")
        order.extend(ready)
    expected = {row["id"]: set(row["depends_on"]) for row in default_stage_graph()}
    if deps != expected:
        raise ValueError("stage graph must be plan -> measure -> audit -> admit")
    return order


def dependency_ready(stage_id: str, stages: list[dict[str, Any]], statuses: dict[str, str]) -> bool:
    validate_stage_graph(stages)
    known = {row["id"] for row in stages}
    if stage_id not in known or any(key not in known or value not in _STATUSES for key, value in statuses.items()):
        raise ValueError("unknown stage or status")
    row = next(row for row in stages if row["id"] == stage_id)
    return statuses.get(stage_id, "PENDING") == "PENDING" and all(statuses.get(dep) == "SUCCEEDED" for dep in row["depends_on"])


def verify_execution_trace(trace_path: Path) -> dict[str, Any]:
    path = Path(trace_path)
    if not path.exists():
        return {"passed": True, "events": 0, "last_sha256": _ZERO}
    raw = path.read_bytes()
    previous = _ZERO
    count = 0
    try:
        if raw and not raw.endswith(b"\n"):
            raise ValueError("incomplete trace line")
        for line in raw.splitlines():
            row = json.loads(line)
            body = {key: value for key, value in row.items() if key != "sha256"}
            if row.get("sequence") != count + 1 or row.get("previous_sha256") != previous or row.get("sha256") != _digest(body):
                raise ValueError("trace chain mismatch")
            previous = row["sha256"]
            count += 1
    except (ValueError, TypeError, AttributeError, KeyError):
        return {"passed": False, "events": count, "last_sha256": previous}
    return {"passed": True, "events": count, "last_sha256": previous}


def _record(trace_path: Path, event: dict[str, Any]) -> None:
    state = verify_execution_trace(trace_path)
    if not state["passed"]:
        raise ValueError("refusing to append to altered execution trace")
    body = {"sequence": state["events"] + 1, "previous_sha256": state["last_sha256"], "utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": time.monotonic(), **event}
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("ab") as handle:
        handle.write(_canonical({**body, "sha256": _digest(body)}) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


class _PosixProcessGroup:
    popen_options = {"start_new_session": True}

    def attach(self, process: subprocess.Popen) -> None:
        self.pid = process.pid

    def active(self) -> bool:
        # Do not poll/wait for the leader here. Keeping it unreaped reserves its
        # PID and group ID until cleanup, even after its last child has exited.
        if sys.platform.startswith("linux") and Path("/proc/self/stat").is_file():
            for path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    fields = path.read_text().rpartition(")")[2].split()
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if len(fields) >= 3 and int(fields[2]) == self.pid and fields[0] not in {"Z", "X"}:
                    return True
            return False
        rows = subprocess.run(["ps", "-A", "-o", "pgid=", "-o", "stat="],
                              check=True, capture_output=True, text=True, timeout=5).stdout
        return any(int(group) == self.pid and state[0] not in {"Z", "X"}
                   for group, state in (row.split() for row in rows.splitlines() if row.strip()))

    def terminate(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def close(self) -> None:
        pass


class _WindowsProcessJob:
    """Own a job before the suspended leader is allowed to create children."""

    popen_options = {"creationflags": 0x00000004 | 0x00000200}  # SUSPENDED | NEW_PROCESS_GROUP

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
            "Thread32First": ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            "Thread32Next": ([wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
            "OpenThread": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("process_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io_counters", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process: subprocess.Popen) -> None:
        ctypes = self.ctypes
        from ctypes import wintypes

        class ThreadEntry(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("thread_id", wintypes.DWORD),
                        ("process_id", wintypes.DWORD), ("priority", wintypes.LONG),
                        ("delta_priority", wintypes.LONG), ("flags", wintypes.DWORD)]

        opened = snapshot = thread = None
        try:
            opened = self.api.OpenProcess(0x0101, False, process.pid)  # SET_QUOTA | TERMINATE
            if not opened or not self.api.AssignProcessToJobObject(self.handle, opened):
                raise ctypes.WinError(ctypes.get_last_error())
            snapshot = self.api.CreateToolhelp32Snapshot(0x00000004, 0)  # SNAPTHREAD
            if snapshot == ctypes.c_void_p(-1).value:
                snapshot = None
                raise ctypes.WinError(ctypes.get_last_error())
            entry = ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = self.api.Thread32First(snapshot, ctypes.byref(entry))
            while found and entry.process_id != process.pid:
                found = self.api.Thread32Next(snapshot, ctypes.byref(entry))
            if not found:
                raise OSError("suspended process has no primary thread")
            thread = self.api.OpenThread(0x0002, False, entry.thread_id)  # SUSPEND_RESUME
            if not thread or self.api.ResumeThread(thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            # Assignment can fail, so the job alone may not own the leader yet.
            process.kill()
            process.wait(timeout=10)
            raise
        finally:
            for handle in (thread, snapshot, opened):
                if handle:
                    self.api.CloseHandle(handle)

    def active(self) -> bool:
        ctypes = self.ctypes

        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_int64 * 4), ("page_faults", ctypes.c_uint32),
                        ("total", ctypes.c_uint32), ("active", ctypes.c_uint32), ("terminated", ctypes.c_uint32)]

        info = Accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(info.active)

    def terminate(self) -> None:
        if not self.api.TerminateJobObject(self.handle, 1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _kill_tree(process: subprocess.Popen, tree) -> dict[str, Any]:
    tree.terminate()
    deadline = time.monotonic() + 10
    while tree.active():
        if time.monotonic() >= deadline:
            raise RuntimeError("process tree did not stop; logs cannot be sealed")
        time.sleep(0.01)
    process.wait(timeout=10)
    return {"tree_kill_attempted": True, "pid": process.pid}


def run_bounded(
    command: list[str], *, cwd: Path, trace_path: Path, node_id: str,
    timeout_s: float, dependencies: tuple[str, ...] = (),
    statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one explicit command sequentially, without shell, preserving exit evidence.

    Dependencies must have SUCCEEDED. An unresolved/blocked node does not launch.
    Logs are exclusively created per node in the trace directory. The time limit
    covers the Windows job or POSIX process group, including children whose
    leader exits first. POSIX children deliberately leaving that group are
    outside this non-sandbox boundary. Logs are sealed after contained processes
    stop. Trace mutation by the same OS user is outside this evidence boundary;
    this function is not a concurrent scheduler or a CPU/RSS sandbox.
    """
    if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be an explicit nonempty string list")
    if Path(command[0]).name.lower() in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe", "sh", "bash"}:
        raise ValueError("shell commands are outside this finite launcher")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
        raise ValueError("node_id must be a simple stage identifier")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    statuses = dict(statuses or {})
    if any(value not in _STATUSES for value in statuses.values()):
        raise ValueError("unknown dependency status")
    if not isinstance(dependencies, tuple) or any(not isinstance(dep, str) for dep in dependencies):
        raise ValueError("dependencies must be an explicit tuple of stage IDs")
    trace_path, cwd = Path(trace_path), Path(cwd).resolve()
    if not cwd.is_dir():
        raise ValueError("cwd must be an existing directory")
    if not verify_execution_trace(trace_path)["passed"]:
        raise ValueError("execution trace is altered")
    if statuses.get(node_id, "PENDING") != "PENDING" or any(statuses.get(dep) != "SUCCEEDED" for dep in dependencies):
        receipt = {"node_id": node_id, "status": "BLOCKED", "returncode": None, "dependencies": list(dependencies), "reason": "dependencies_or_stage_unresolved"}
        _record(trace_path, {"event": "stage_blocked", **receipt})
        return receipt
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = trace_path.parent / f"{node_id}.stdout.log", trace_path.parent / f"{node_id}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        tree = None
        try:
            tree = _WindowsProcessJob() if os.name == "nt" else _PosixProcessGroup()
            process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, shell=False, **tree.popen_options)
            tree.attach(process)
        except BaseException as exc:
            if tree is not None:
                tree.close()
            if not isinstance(exc, OSError):
                raise
            receipt = {"node_id": node_id, "status": "FAILED", "returncode": None, "wall_seconds": time.monotonic() - started, "error": type(exc).__name__, "command": command}
            _record(trace_path, {"event": "process_failed_to_start", **receipt})
            return receipt
        try:
            _record(trace_path, {"event": "process_started", "node_id": node_id, "pid": process.pid, "command": command, "cwd": str(cwd), "dependencies": list(dependencies), "timeout_s": float(timeout_s)})
            killed = {}
            deadline = started + float(timeout_s)
            try:
                while tree.active():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout_s)
                    time.sleep(min(0.01, remaining))
                returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
                status = "SUCCEEDED" if returncode == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                killed = _kill_tree(process, tree)
                returncode, status = process.returncode, "TIMEOUT"
        except BaseException:
            _kill_tree(process, tree)
            raise
        finally:
            tree.close()
    receipt = {
        "node_id": node_id, "pid": process.pid, "status": status, "returncode": returncode,
        "wall_seconds": time.monotonic() - started, "dependencies": list(dependencies),
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        "stdout_sha256": _file_hash(stdout_path), "stderr_sha256": _file_hash(stderr_path),
        "wall_bound_only": True, "security_sandbox": False, **killed,
    }
    _record(trace_path, {"event": "process_finished", **receipt})
    return receipt
