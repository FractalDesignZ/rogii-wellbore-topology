#!/usr/bin/env python3
"""
Live train dashboard — serves HTML + charts from progress.json.

  python scripts/dashboard_server.py --out artifacts/full_train_parallel --port 8765

Open http://127.0.0.1:8765/  (auto-refresh every 3s)
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load_progress(out: Path) -> dict:
    p = out / "progress.json"
    if not p.is_file():
        return {"phase": "waiting", "done": 0, "total": 0, "message": "no progress.json yet", "events": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"phase": "error", "message": str(e), "events": []}


def _render_charts(out: Path, st: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # 1) progress bar chart
    fig, ax = plt.subplots(figsize=(8, 2.2))
    done = int(st.get("done") or 0)
    total = max(int(st.get("total") or 0), 1)
    ax.barh([0], [done], color="#3b82f6", height=0.5)
    ax.barh([0], [max(total - done, 0)], left=[done], color="#1e293b", height=0.5)
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"{done} / {total}  ·  phase={st.get('phase')}  ·  rate={st.get('rate_per_min', 0):.2f}/min")
    ax.set_title("RWT train progress")
    fig.tight_layout()
    fig.savefig(out / "chart_progress.png", dpi=120)
    plt.close(fig)

    # 2) rate / eta text panel as image
    fig, ax = plt.subplots(figsize=(8, 2.0))
    ax.axis("off")
    eta = st.get("eta_min")
    eta_s = f"{eta:.1f} min" if isinstance(eta, (int, float)) and eta is not None else "—"
    metrics = st.get("metrics") or {}
    lines = [
        f"workers={st.get('workers')}  elapsed={st.get('elapsed_sec', 0):.0f}s  ETA={eta_s}",
        f"ok={st.get('ok')}  failed={st.get('failed')}  last={st.get('last_item')}",
        f"message: {st.get('message', '')[:90]}",
    ]
    if metrics.get("oof_rmse_selected") is not None:
        lines.append(
            f"OOF selected={metrics.get('oof_rmse_selected')}  residual={metrics.get('use_residual')}  "
            f"{metrics.get('final_prior_type')}"
        )
    ax.text(0.02, 0.5, "\n".join(lines), va="center", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(out / "chart_status.png", dpi=120)
    plt.close(fig)

    # 3) baselines when present
    br = metrics.get("baseline_rmse") or {}
    if isinstance(br, dict) and br:
        names = list(br.keys())
        vals = [float(br[k]) for k in names]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        y = np.arange(len(names))
        ax.barh(y, vals, color="#22c55e")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("OOF RMSE (ft)")
        ax.set_title("Pure prior floors vs residual (from finished metrics)")
        if metrics.get("oof_rmse_selected") is not None:
            ax.axvline(float(metrics["oof_rmse_selected"]), color="#f59e0b", ls="--", label="selected")
            ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "chart_baselines.png", dpi=120)
        plt.close(fig)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="3"/>
  <title>RWT Train Dashboard</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#0b1220; color:#e2e8f0; margin:0; padding:1.2rem; }}
    h1 {{ margin:0 0 .5rem; font-size:1.35rem; }}
    .sub {{ color:#94a3b8; margin-bottom:1rem; font-size:.9rem; }}
    .grid {{ display:grid; gap:1rem; grid-template-columns: 1fr; max-width:980px; }}
    .card {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:1rem; }}
    img {{ max-width:100%; height:auto; border-radius:8px; background:#0f172a; }}
    pre {{ background:#0f172a; padding:.75rem; border-radius:8px; overflow:auto; max-height:280px; font-size:12px; }}
    .pill {{ display:inline-block; padding:.15rem .55rem; border-radius:999px; background:#1d4ed8; font-size:.8rem; }}
    a {{ color:#93c5fd; }}
  </style>
</head>
<body>
  <h1>RWT multi-core train dashboard</h1>
  <div class="sub">
    <span class="pill">{phase}</span>
    &nbsp; auto-refresh 3s · out=<code>{out}</code> ·
    <a href="/progress.json">progress.json</a> ·
    <a href="/progress.log">progress.log</a>
  </div>
  <div class="grid">
    <div class="card"><img src="/chart_progress.png?t={ts}" alt="progress"/></div>
    <div class="card"><img src="/chart_status.png?t={ts}" alt="status"/></div>
    <div class="card"><img src="/chart_baselines.png?t={ts}" alt="baselines" onerror="this.style.display='none'"/></div>
    <div class="card">
      <strong>event log (tail)</strong>
      <pre>{log}</pre>
    </div>
    <div class="card">
      <strong>hardware note</strong>
      <pre>CPU: multi-process well assemble (RWT_WORKERS) + LightGBM num_threads
GPU/NPU: not used by public open-core rwt (NumPy/OpenBLAS + LGBM CPU).
Snapdragon NPU would need a separate ONNX/QNN path — not claimed OPEN.</pre>
    </div>
  </div>
</body>
</html>
"""


def make_handler(out: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter
            pass

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            st = _load_progress(out)
            try:
                _render_charts(out, st)
            except Exception:
                pass
            if path in ("/", "/index.html"):
                events = st.get("events") or []
                log = "\n".join(events[-40:])
                body = HTML.format(
                    phase=st.get("phase", "?"),
                    out=str(out),
                    ts=int(time.time()),
                    log=log.replace("<", "&lt;"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            file_map = {
                "/progress.json": out / "progress.json",
                "/progress.log": out / "progress.log",
                "/chart_progress.png": out / "chart_progress.png",
                "/chart_status.png": out / "chart_status.png",
                "/chart_baselines.png": out / "chart_baselines.png",
                "/cv_report.json": out / "cv_report.json",
            }
            fp = file_map.get(path)
            if fp is None or not fp.is_file():
                self.send_error(404)
                return
            data = fp.read_bytes()
            ctype = "application/json" if fp.suffix == ".json" else (
                "image/png" if fp.suffix == ".png" else "text/plain; charset=utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "full_train_parallel"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    handler = make_handler(out)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard http://{args.host}:{args.port}/  (watching {out})", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
