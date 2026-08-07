"""Live progress log for train/assemble — JSON + line log for dashboards."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class ProgressTracker:
    """Thread-safe progress writer. Dashboard polls ``progress.json``."""

    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "progress.json"
        self.log_path = self.out_dir / "progress.log"
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._phase_t0 = self._t0
        self._heartbeat_stop = threading.Event()
        self.state: dict[str, Any] = {
            "phase": "init",
            "done": 0,
            "total": 0,
            "ok": 0,
            "failed": 0,
            "rate_per_min": 0.0,
            "eta_min": None,
            "workers": int(os.environ.get("RWT_WORKERS", "0") or 0),
            "last_item": None,
            "message": "",
            "events": [],
            "metrics": {},
            "t0": self._t0,
            "_phase_t0": self._phase_t0,
            "updated_at": self._t0,
            "elapsed_sec": 0.0,
        }
        self._flush()
        self._hb = threading.Thread(target=self._heartbeat_loop, name="rwt-progress-hb", daemon=True)
        self._hb.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(5.0):
            with self._lock:
                # keep dashboard clocks moving even when workers are on long wells
                self.state["message"] = self.state.get("message") or "heartbeat"
                self._flush()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()

    def _flush(self) -> None:
        now = time.time()
        self.state["updated_at"] = now
        self.state["elapsed_sec"] = now - self._t0
        done = int(self.state.get("done") or 0)
        phase_t0 = float(getattr(self, "_phase_t0", self._t0) or self._t0)
        # rate/ETA are per-phase so set_phase resets don't double-count history
        elapsed_min = max((now - phase_t0) / 60.0, 1e-6)
        self.state["rate_per_min"] = done / elapsed_min
        total = int(self.state.get("total") or 0)
        if total > 0 and self.state["rate_per_min"] > 0 and done < total:
            self.state["eta_min"] = (total - done) / self.state["rate_per_min"]
        else:
            self.state["eta_min"] = 0.0 if done >= total and total > 0 else None
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def set_phase(self, phase: str, *, total: int = 0, message: str = "") -> None:
        with self._lock:
            self.state["phase"] = phase
            if total:
                self.state["total"] = int(total)
                # per-phase counters (assemble vs spatial_attach must not double-count)
                self.state["done"] = 0
                self.state["ok"] = 0
                self.state["failed"] = 0
                self._phase_t0 = time.time()
                self.state["_phase_t0"] = self._phase_t0
            if message:
                self.state["message"] = message
            self._event(f"phase={phase} total={self.state['total']} {message}".strip())
            self._flush()

    def tick(
        self,
        *,
        item: str | None = None,
        ok: bool = True,
        message: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.state["done"] = int(self.state.get("done") or 0) + 1
            if ok:
                self.state["ok"] = int(self.state.get("ok") or 0) + 1
            else:
                self.state["failed"] = int(self.state.get("failed") or 0) + 1
            if item is not None:
                self.state["last_item"] = item
            if message:
                self.state["message"] = message
            if metrics:
                self.state.setdefault("metrics", {}).update(metrics)
            line = (
                f"[{self.state['phase']}] {self.state['done']}/{self.state['total']} "
                f"item={item} ok={ok} {message}"
            ).strip()
            self._event(line)
            self._flush()

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            self.state.setdefault("metrics", {}).update(metrics)
            self._flush()

    def finish(self, message: str = "done", **metrics: Any) -> None:
        self.stop_heartbeat()
        with self._lock:
            self.state["phase"] = "done"
            self.state["message"] = message
            if metrics:
                self.state.setdefault("metrics", {}).update(metrics)
            self._event(f"DONE {message} {metrics}")
            self._flush()

    def fail(self, message: str) -> None:
        self.stop_heartbeat()
        with self._lock:
            self.state["phase"] = "failed"
            self.state["message"] = message
            self._event(f"FAILED {message}")
            self._flush()

    def _event(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"{ts} {line}"
        ev = self.state.setdefault("events", [])
        ev.append(entry)
        if len(ev) > 200:
            del ev[:-200]
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")


_GLOBAL: ProgressTracker | None = None


def get_progress() -> ProgressTracker | None:
    return _GLOBAL


def init_progress(out_dir: Path | str) -> ProgressTracker:
    global _GLOBAL
    _GLOBAL = ProgressTracker(out_dir)
    return _GLOBAL
