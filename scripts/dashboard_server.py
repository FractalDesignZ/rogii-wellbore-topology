#!/usr/bin/env python3
"""
Live train dashboard — progress bar only auto-refreshes (JS poll).

Other views open in a pop-out drawer on demand.

  python scripts/dashboard_server.py --out artifacts/full_train_parallel --port 8765
  → http://127.0.0.1:8765/
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load_progress(out: Path) -> dict:
    p = out / "progress.json"
    if not p.is_file():
        return {
            "phase": "waiting",
            "done": 0,
            "total": 0,
            "message": "no progress.json yet",
            "events": [],
            "metrics": {},
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"phase": "error", "message": str(e), "events": [], "metrics": {}}


def _render_status_chart(out: Path, st: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chart_status.png"
    fig, ax = plt.subplots(figsize=(8, 2.0))
    ax.axis("off")
    eta = st.get("eta_min")
    eta_s = f"{eta:.1f} min" if isinstance(eta, (int, float)) and eta is not None else "—"
    metrics = st.get("metrics") or {}
    lines = [
        f"workers={st.get('workers')}  elapsed={st.get('elapsed_sec', 0):.0f}s  ETA={eta_s}",
        f"ok={st.get('ok')}  failed={st.get('failed')}  last={st.get('last_item')}",
        f"message: {str(st.get('message', ''))[:100]}",
    ]
    if metrics.get("oof_rmse_selected") is not None:
        lines.append(
            f"OOF selected={metrics.get('oof_rmse_selected')}  residual={metrics.get('use_residual')}  "
            f"{metrics.get('final_prior_type')}"
        )
    ax.text(0.02, 0.5, "\n".join(lines), va="center", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _render_baselines_chart(out: Path, st: dict) -> Path | None:
    metrics = st.get("metrics") or {}
    br = metrics.get("baseline_rmse") or {}
    if not isinstance(br, dict) or not br:
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / "chart_baselines.png"
    names = list(br.keys())
    vals = [float(br[k]) for k in names]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#22c55e")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("OOF RMSE (ft)")
    ax.set_title("Pure prior floors (finished metrics)")
    if metrics.get("oof_rmse_selected") is not None:
        ax.axvline(float(metrics["oof_rmse_selected"]), color="#f59e0b", ls="--", label="selected")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RWT Train</title>
  <style>
    :root {
      --bg: #0b1220;
      --card: #111827;
      --border: #1f2937;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --bar: #3b82f6;
      --bar-track: #1e293b;
      --ok: #22c55e;
      --warn: #f59e0b;
      --fail: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .shell {
      width: min(640px, 100%);
    }
    h1 {
      margin: 0 0 .35rem;
      font-size: 1.15rem;
      font-weight: 600;
      letter-spacing: .02em;
    }
    .meta {
      color: var(--muted);
      font-size: .85rem;
      margin-bottom: 1.25rem;
      display: flex;
      flex-wrap: wrap;
      gap: .5rem .75rem;
      align-items: center;
    }
    .pill {
      display: inline-block;
      padding: .12rem .55rem;
      border-radius: 999px;
      background: #1d4ed8;
      font-size: .75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .pill.done { background: #15803d; }
    .pill.failed { background: #b91c1c; }
    .pill.waiting { background: #334155; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.15rem 1.2rem 1.25rem;
    }
    .row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
      margin-bottom: .65rem;
      font-size: .9rem;
    }
    .counts { font-variant-numeric: tabular-nums; font-weight: 600; }
    .track {
      height: 14px;
      background: var(--bar-track);
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid #0f172a;
    }
    .fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #2563eb, #60a5fa);
      border-radius: 999px;
      transition: width .35s ease;
    }
    .fill.pulse { animation: pulse 1.6s ease-in-out infinite; }
    @keyframes pulse {
      0%, 100% { filter: brightness(1); }
      50% { filter: brightness(1.15); }
    }
    .stats {
      margin-top: .85rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: .45rem .75rem;
      font-size: .8rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .stats b { color: var(--text); font-weight: 600; }
    .toolbar {
      margin-top: 1rem;
      display: flex;
      flex-wrap: wrap;
      gap: .45rem;
    }
    button {
      appearance: none;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      border-radius: 999px;
      padding: .4rem .8rem;
      font-size: .8rem;
      cursor: pointer;
    }
    button:hover { border-color: #3b82f6; background: #172554; }
    button:focus-visible { outline: 2px solid #3b82f6; outline-offset: 2px; }
    .foot {
      margin-top: .85rem;
      font-size: .75rem;
      color: #64748b;
    }
    a { color: #93c5fd; }

    /* pop-out drawer */
    .backdrop {
      position: fixed;
      inset: 0;
      background: rgba(2, 6, 23, .65);
      opacity: 0;
      pointer-events: none;
      transition: opacity .2s ease;
      z-index: 40;
    }
    .backdrop.open { opacity: 1; pointer-events: auto; }
    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      height: 100%;
      width: min(440px, 100%);
      background: #0f172a;
      border-left: 1px solid var(--border);
      box-shadow: -12px 0 40px rgba(0,0,0,.35);
      transform: translateX(100%);
      transition: transform .22s ease;
      z-index: 50;
      display: flex;
      flex-direction: column;
    }
    .drawer.open { transform: translateX(0); }
    .drawer header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1rem 1.1rem;
      border-bottom: 1px solid var(--border);
    }
    .drawer header h2 {
      margin: 0;
      font-size: 1rem;
    }
    .drawer .body {
      padding: 1rem 1.1rem;
      overflow: auto;
      flex: 1;
    }
    .drawer pre {
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: .75rem;
      font-size: 11px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: none;
    }
    .drawer img {
      width: 100%;
      height: auto;
      border-radius: 10px;
      background: #020617;
      border: 1px solid var(--border);
    }
    .drawer .muted { color: var(--muted); font-size: .85rem; margin-bottom: .75rem; }
    .xbtn {
      border: none;
      background: transparent;
      color: var(--muted);
      font-size: 1.25rem;
      line-height: 1;
      padding: .25rem .4rem;
      cursor: pointer;
    }
    .xbtn:hover { color: var(--text); }
  </style>
</head>
<body>
  <div class="shell">
    <h1>RWT train</h1>
    <div class="meta">
      <span id="phase" class="pill waiting">waiting</span>
      <span id="last-item">—</span>
      <span id="poll-hint">bar refresh 2s</span>
    </div>

    <div class="card">
      <div class="row">
        <span id="phase-label">phase</span>
        <span class="counts"><span id="done">0</span> / <span id="total">0</span>
          (<span id="pct" style="color:var(--muted);font-weight:500"> (0%)</span>
        </span>
      </div>
      <div class="track" aria-label="progress">
        <div id="fill" class="fill pulse"></div>
      </div>
      <div class="stats">
        <div>rate <b id="rate">—</b></div>
        <div>ETA <b id="eta">—</b></div>
        <div>elapsed <b id="elapsed">—</b></div>
        <div>workers <b id="workers">—</b></div>
        <div style="grid-column:1/-1">message <b id="message">—</b></div>
      </div>

      <div class="toolbar">
        <button type="button" data-panel="status">Status</button>
        <button type="button" data-panel="log">Event log</button>
        <button type="button" data-panel="baselines">Baselines</button>
        <button type="button" data-panel="json">Raw JSON</button>
        <button type="button" data-panel="hw">Hardware</button>
      </div>
      <div class="foot">
        out: <code id="out-path">—</code>
        · <a href="/progress.json" target="_blank">progress.json</a>
        · <a href="/progress.log" target="_blank">progress.log</a>
      </div>
    </div>
  </div>

  <div id="backdrop" class="backdrop" hidden></div>
  <aside id="drawer" class="drawer" aria-hidden="true">
    <header>
      <h2 id="drawer-title">Details</h2>
      <button type="button" class="xbtn" id="close-drawer" aria-label="Close">×</button>
    </header>
    <div class="body" id="drawer-body"></div>
  </aside>

  <script>
    const OUT = __OUT__;
    document.getElementById('out-path').textContent = OUT;

    const el = (id) => document.getElementById(id);
    const drawer = el('drawer');
    const backdrop = el('backdrop');
    const drawerTitle = el('drawer-title');
    const drawerBody = el('drawer-body');
    let lastDone = -1;
    let lastPhase = '';

    function fmtEta(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return '—';
      if (v <= 0) return '0 min';
      if (v < 1) return (v * 60).toFixed(0) + 's';
      return v.toFixed(1) + ' min';
    }

    function setPhasePill(phase) {
      const p = el('phase');
      p.textContent = phase || 'waiting';
      p.className = 'pill';
      if (phase === 'done') p.classList.add('done');
      else if (phase === 'failed') p.classList.add('failed');
      else if (!phase || phase === 'waiting') p.classList.add('waiting');
    }

    function applyProgress(st) {
      const done = Number(st.done || 0);
      const total = Math.max(Number(st.total || 0), 0);
      const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
      el('done').textContent = String(done);
      el('total').textContent = String(total || '—');
      el('pct').textContent = total ? ` (${pct.toFixed(1)}%)` : '';
      el('fill').style.width = pct.toFixed(2) + '%';
      el('fill').classList.toggle('pulse', st.phase !== 'done' && st.phase !== 'failed');
      el('rate').textContent = st.rate_per_min != null ? Number(st.rate_per_min).toFixed(1) + '/min' : '—';
      el('eta').textContent = fmtEta(st.eta_min);
      el('elapsed').textContent = st.elapsed_sec != null ? Number(st.elapsed_sec).toFixed(0) + 's' : '—';
      el('workers').textContent = st.workers != null ? String(st.workers) : '—';
      el('message').textContent = st.message || '—';
      el('last-item').textContent = st.last_item ? ('last: ' + st.last_item) : '—';
      el('phase-label').textContent = st.phase || 'phase';
      setPhasePill(st.phase);
      lastDone = done;
      lastPhase = st.phase || '';
      window.__progress = st;
    }

    async function poll() {
      try {
        const r = await fetch('/progress.json?t=' + Date.now(), { cache: 'no-store' });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const st = await r.json();
        applyProgress(st);
        el('poll-hint').textContent = 'bar refresh 2s · ok';
      } catch (e) {
        el('poll-hint').textContent = 'poll error';
      }
    }

    function openDrawer(title, html) {
      drawerTitle.textContent = title;
      drawerBody.innerHTML = html;
      drawer.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      backdrop.hidden = false;
      requestAnimationFrame(() => backdrop.classList.add('open'));
    }

    function closeDrawer() {
      drawer.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
      backdrop.classList.remove('open');
      setTimeout(() => { backdrop.hidden = true; }, 200);
    }

    el('close-drawer').addEventListener('click', closeDrawer);
    backdrop.addEventListener('click', closeDrawer);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeDrawer();
    });

    async function showPanel(name) {
      const st = window.__progress || {};
      if (name === 'status') {
        const url = '/chart_status.png?t=' + Date.now();
        openDrawer('Status',
          '<p class="muted">Snapshot regenerated when you open this panel.</p>' +
          '<img src="' + url + '" alt="status chart"/>');
        return;
      }
      if (name === 'baselines') {
        const url = '/chart_baselines.png?t=' + Date.now();
        openDrawer('Baselines',
          '<p class="muted">Available after train metrics land in progress.json.</p>' +
          '<img src="' + url + '" alt="baselines" onerror="this.replaceWith(Object.assign(document.createElement(\'p\'),{className:\'muted\',textContent:\'No baseline metrics yet.\'}))"/>');
        return;
      }
      if (name === 'log') {
        let text = '(empty)';
        try {
          const r = await fetch('/progress.log?t=' + Date.now(), { cache: 'no-store' });
          if (r.ok) {
            const full = await r.text();
            const lines = full.trim().split(/\r?\n/);
            text = lines.slice(-80).join('\n') || '(empty)';
          }
        } catch (_) {}
        openDrawer('Event log', '<pre></pre>');
        drawerBody.querySelector('pre').textContent = text;
        return;
      }
      if (name === 'json') {
        openDrawer('Raw progress.json', '<pre></pre>');
        drawerBody.querySelector('pre').textContent = JSON.stringify(st, null, 2);
        return;
      }
      if (name === 'hw') {
        openDrawer('Hardware',
          '<pre>CPU: multi-process well assemble (RWT_WORKERS) + LightGBM num_threads\n' +
          'GPU/NPU: not used by public open-core rwt (NumPy/OpenBLAS + LGBM CPU).\n' +
          'Snapdragon NPU would need a separate ONNX/QNN path — not claimed OPEN.</pre>');
      }
    }

    document.querySelectorAll('[data-panel]').forEach((btn) => {
      btn.addEventListener('click', () => showPanel(btn.getAttribute('data-panel')));
    });

    poll();
    setInterval(poll, 2000);
  </script>
</body>
</html>
"""


def make_handler(out: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            st = _load_progress(out)

            if path in ("/", "/index.html"):
                # static shell — no full-page refresh; JS polls progress.json
                page = HTML.replace("__OUT__", json.dumps(str(out)))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/progress.json":
                # live source of truth for the bar only
                body = json.dumps(st).encode("utf-8")
                self._send(200, body, "application/json")
                return

            if path == "/progress.log":
                fp = out / "progress.log"
                data = fp.read_bytes() if fp.is_file() else b""
                self._send(200, data, "text/plain; charset=utf-8")
                return

            if path == "/chart_status.png":
                try:
                    fp = _render_status_chart(out, st)
                    self._send(200, fp.read_bytes(), "image/png")
                except Exception as e:  # noqa: BLE001
                    self._send(500, str(e).encode(), "text/plain; charset=utf-8")
                return

            if path == "/chart_baselines.png":
                try:
                    fp = _render_baselines_chart(out, st)
                    if fp is None or not fp.is_file():
                        self.send_error(404, "no baselines yet")
                        return
                    self._send(200, fp.read_bytes(), "image/png")
                except Exception as e:  # noqa: BLE001
                    self._send(500, str(e).encode(), "text/plain; charset=utf-8")
                return

            if path == "/cv_report.json":
                fp = out / "cv_report.json"
                if not fp.is_file():
                    self.send_error(404)
                    return
                self._send(200, fp.read_bytes(), "application/json")
                return

            self.send_error(404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "full_train_parallel"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(out))
    print(
        f"Dashboard http://{args.host}:{args.port}/  (bar polls progress.json; charts on demand)",
        flush=True,
    )
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
