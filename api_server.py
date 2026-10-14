"""
api_server.py - REST API and themed dashboard for Sentinel.

Started by rullall.py, or on its own:  python src/api_server.py

Endpoints
  GET  /                 dashboard
  GET  /api/health       service and model status
  GET  /api/stats        headline numbers and alert mix
  GET  /api/alerts       ?severity=critical|high|medium|low &limit=40
  GET  /api/stream       recent scored connections (drives the traffic ribbon)
  GET  /api/metrics      model scores from the last training run
  GET  /api/sample       ?kind=normal|dos|brute_force|port_scan|data_exfiltration
  POST /api/predict      classify one connection (JSON body)
  GET  /api/images       list of chart files, served at /charts/<name>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_from_directory  # pyright: ignore[reportMissingImports]

import src.threat_core as tc


def _read_csv(name: str) -> pd.DataFrame:
    path = tc.OUTPUTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(name: str) -> dict:
    path = tc.OUTPUTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def create_app() -> Flask:
    rf = joblib.load(tc.MODELS / "random_forest.joblib")
    iso = joblib.load(tc.MODELS / "isolation_forest.joblib")
    meta = json.loads((tc.MODELS / "meta.json").read_text())
    thr = meta["anomaly_threshold"]

    app = Flask(__name__)

    @app.get("/")
    def dashboard():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", models=["random_forest", "isolation_forest"],
                       trained_at=meta["trained_at"], classes=meta["classes"])

    @app.get("/api/stats")
    def stats():
        summary = _read_json("detection_summary.json")
        alerts = _read_csv("alerts.csv")
        mix = {}
        top = []
        if not alerts.empty:
            mix = alerts["predicted"].value_counts().to_dict()
            top = (alerts["src_ip"].value_counts().head(5)
                   .rename_axis("ip").reset_index(name="count").to_dict("records"))
        return jsonify(summary=summary, mix=mix, top_sources=top)

    @app.get("/api/alerts")
    def alerts():
        df = _read_csv("alerts.csv")
        if df.empty:
            return jsonify([])
        sev = request.args.get("severity")
        if sev:
            df = df[df["severity"] == sev]
        limit = max(1, min(request.args.get("limit", 40, type=int), 500))
        df = df.sort_values("timestamp", ascending=False).head(limit)
        return Response(df.to_json(orient="records"), mimetype="application/json")

    @app.get("/api/stream")
    def stream():
        df = _read_csv("stream_scored.csv")
        if df.empty:
            return jsonify([])
        cols = ["timestamp", "predicted", "severity", "anomaly_score", "alert"]
        return Response(df[cols].tail(400).to_json(orient="records"), mimetype="application/json")

    @app.get("/api/metrics")
    def metrics():
        return jsonify(_read_json("metrics.json"))

    @app.get("/api/sample")
    def sample():
        kind = request.args.get("kind", "normal")
        if kind not in tc.CLASSES:
            return jsonify(error=f"kind must be one of: {', '.join(tc.CLASSES)}"), 400
        df = tc.simulate_traffic(300, seed=int(np.random.randint(1, 1_000_000)), stealth=0)
        row = df[df["label"] == kind].iloc[0]
        rec = {f: row[f] for f in tc.NUMERIC_RAW}
        rec = {k: (round(float(v), 3) if k in ("duration", "syn_error_rate") else round(float(v), 1))
               for k, v in rec.items()}
        rec.update(protocol=str(row["protocol"]), service=str(row["service"]))
        return jsonify(rec)

    @app.post("/api/predict")
    def predict():
        try:
            rec = tc.clean_record(request.get_json(silent=True) or {})
        except ValueError as e:
            return jsonify(error=str(e)), 400
        X = tc.engineer_features(pd.DataFrame([rec]))[tc.FEATURES]
        proba = rf.predict_proba(X)[0]
        classes = list(rf.classes_)
        top = classes[int(proba.argmax())]
        score = float(iso.decision_function(X)[0])
        flagged = score < thr
        verdict = "unknown_anomaly" if (top == "normal" and flagged) else top
        return jsonify(
            verdict=verdict, label=tc.LABELS[verdict], severity=tc.SEVERITY[verdict],
            confidence=round(float(proba.max()), 4),
            probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
            anomaly_score=round(score, 4), anomaly_flag=bool(flagged),
            reasons=tc.explain(rec), action=tc.PLAYBOOK[verdict])

    @app.get("/api/images")
    def images():
        return jsonify(sorted(p.name for p in tc.IMAGES.glob("*.png")))

    @app.get("/charts/<path:name>")
    def charts(name):
        if not (tc.IMAGES / name).is_file():
            abort(404)
        return send_from_directory(tc.IMAGES, name)

    return app


# --------------------------------------------------------------- dashboard
DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel - AI threat detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap" rel="stylesheet">
<style>
:root{--deep:#0b1a2c;--panel:#112640;--raise:#173250;--line:#27476b;--text:#e4eef8;--muted:#8da6c1;
--safe:#3dd3b0;--low:#5fa8ff;--medium:#f4b942;--high:#ff8a4c;--critical:#ff4d5e;--radius:8px;
--display:"Chakra Petch","Segoe UI",system-ui,sans-serif;--body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--deep);color:var(--text);font:15px/1.55 var(--body)}
:focus-visible{outline:2px solid var(--low);outline-offset:2px}
.wrap{max-width:1280px;margin:0 auto;padding:0 clamp(16px,4vw,40px)}
header.top{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:18px 0;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:12px;font:600 22px var(--display);letter-spacing:.02em}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:6px 14px;color:var(--muted);font-size:13px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--safe)}
.hero{padding:40px 0 4px}
.hero h1{font:600 clamp(28px,4.6vw,48px)/1.08 var(--display);margin:0 0 12px;max-width:20ch}
.hero p{color:var(--muted);margin:0;max-width:62ch}
.ribbon{position:relative;margin-top:24px;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 14px 12px}
#ribbon{display:block;width:100%;height:200px;cursor:crosshair}
#tip{position:absolute;pointer-events:none;opacity:0;background:var(--raise);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font-size:12px;white-space:nowrap;transform:translate(-50%,-120%)}
.legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;margin-top:10px;color:var(--muted);font-size:13px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.legend button{margin-left:auto}
button{font:inherit;cursor:pointer}
.ghost{background:transparent;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 14px;font-size:13px}
.ghost:hover{border-color:var(--low)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);margin:30px 0 10px}
.stat{padding:0 22px;border-left:1px solid var(--line)}
.stat:first-child{padding-left:0;border-left:0}
.stat b{display:block;font:600 34px/1.1 var(--display)}
.stat span{color:var(--muted);font-size:14px}
.mix{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--raise);margin-top:16px}
.mix i{display:block;height:100%}
.mixkey{display:flex;flex-wrap:wrap;gap:4px 18px;margin-top:8px;font-size:13px;color:var(--muted)}
.mixkey i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px;margin-top:30px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px}
.panel h2{font:600 19px/1.2 var(--display);margin:0 0 4px}
.sub{margin:0 0 14px;color:var(--muted);font-size:14px}
.chips,.presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.chip{background:transparent;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:13px}
.chip[aria-pressed=true]{background:var(--raise);color:var(--text);border-color:var(--low)}
.presets button{background:var(--raise);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:6px 12px;font-size:13px}
.presets button:hover{border-color:var(--safe)}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-weight:500;color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid rgba(39,71,107,.55);white-space:nowrap}
td.ip{font-family:var(--mono);font-size:13px}
tr.s-critical td:first-child{box-shadow:inset 3px 0 var(--critical)}
tr.s-high td:first-child{box-shadow:inset 3px 0 var(--high)}
tr.s-medium td:first-child{box-shadow:inset 3px 0 var(--medium)}
tr.s-low td:first-child{box-shadow:inset 3px 0 var(--low)}
.empty{color:var(--muted);text-align:center;padding:22px}
.badge{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;color:#0b1a2c;background:var(--c)}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px 12px}
label{display:block;font-size:13px;color:var(--muted)}
input,select{width:100%;margin-top:4px;padding:8px 10px;background:var(--deep);color:var(--text);border:1px solid var(--line);border-radius:6px;font:inherit}
.primary{width:100%;margin-top:14px;padding:11px;background:var(--safe);color:#06251f;border:0;border-radius:6px;font:600 15px var(--body)}
.primary:hover{filter:brightness(1.08)}
.verdict{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:18px;padding:4px 0 4px 12px;border-left:4px solid var(--c);font:600 21px var(--display)}
.verdict .badge{font-family:var(--body)}
.note{color:var(--muted);font-size:14px;margin:8px 0 12px}
.bar{display:grid;grid-template-columns:130px 1fr 52px;align-items:center;gap:8px;margin:5px 0;font-size:13px}
.bar div{height:8px;background:var(--raise);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%}
.bar span:last-child{text-align:right;color:var(--muted)}
ul.why{margin:10px 0 0;padding-left:18px;color:var(--text);font-size:14px}
.action{margin-top:12px;padding:10px 12px;background:var(--raise);border-radius:6px;font-size:14px}
.models{margin-top:20px}
td.best{color:var(--safe);font-weight:600}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:6px}
figure{margin:0}
figure img{display:block;width:100%;border:1px solid var(--line);border-radius:6px}
figcaption{margin-top:6px;color:var(--muted);font-size:13px}
footer{margin:36px 0 48px;color:var(--muted);font-size:13px}
@media(max-width:960px){.grid{grid-template-columns:1fr}}
@media(max-width:760px){.stats{grid-template-columns:1fr 1fr;gap:20px 0}.stat:nth-child(odd){padding-left:0;border-left:0}}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <svg width="30" height="34" viewBox="0 0 30 34" fill="none" aria-hidden="true"><path d="M15 2 3 7v9c0 8 5 13.5 12 16 7-2.500 12-8 12-16V7L15 2Z" stroke="#3dd3b0" stroke-width="2.200" stroke-linejoin="round"/><path d="M8 17h4l2-5 3 9 2-4h3" stroke="#5fa8ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Sentinel
    </div>
    <div class="pill"><span class="dot"></span><span id="status">Connecting to models</span></div>
  </header>

  <section class="hero">
    <h1>Live traffic, sorted into safe and suspicious</h1>
    <p>Each tick is one network connection. Short teal ticks are normal traffic. Taller ticks are threats the models flagged, colored by severity. Hover a tick to inspect it.</p>
    <div class="ribbon">
      <canvas id="ribbon" role="img" aria-label="Timeline of scored network connections"></canvas>
      <div id="tip"></div>
      <div class="legend">
        <span><i style="background:var(--safe)"></i>Normal</span>
        <span><i style="background:var(--low)"></i>Low</span>
        <span><i style="background:var(--medium)"></i>Medium</span>
        <span><i style="background:var(--high)"></i>High</span>
        <span><i style="background:var(--critical)"></i>Critical</span>
        <button class="ghost" id="replay" type="button">Replay</button>
      </div>
    </div>
  </section>

  <section class="stats" aria-label="Detection summary">
    <div class="stat"><b id="st-events">-</b><span>connections scanned</span></div>
    <div class="stat"><b id="st-alerts">-</b><span>alerts raised</span></div>
    <div class="stat"><b id="st-rate">-</b><span>of simulated attacks caught</span></div>
    <div class="stat"><b id="st-fa">-</b><span>of normal traffic falsely flagged</span></div>
  </section>
  <div class="mix" id="mix" aria-hidden="true"></div>
  <div class="mixkey" id="mixkey"></div>

  <div class="grid">
    <section class="panel">
      <h2>Alerts</h2>
      <p class="sub">Newest first. Filter by severity.</p>
      <div class="chips" id="chips"></div>
      <div class="scroll">
        <table id="alerts">
          <thead><tr><th>Time</th><th>Source</th><th>Target</th><th>Threat</th><th>Severity</th><th>Confidence</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Check a connection</h2>
      <p class="sub">Load an example or enter your own values, then ask the model.</p>
      <div class="presets" id="presets"></div>
      <form id="form" autocomplete="off">
        <div class="fields" id="fields"></div>
        <button class="primary" type="submit">Check connection</button>
      </form>
      <div id="result" aria-live="polite"></div>
    </section>
  </div>

  <section class="panel models">
    <h2>Model scorecard</h2>
    <p class="sub">Measured on connections the models never saw during training.</p>
    <div class="scroll"><table id="models"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 score</th></tr></thead><tbody></tbody></table></div>
  </section>

  <section class="panel models">
    <h2>Charts from the last run</h2>
    <p class="sub">Open any chart in a new tab to save it for your README.</p>
    <div class="gallery" id="gallery"></div>
  </section>

  <footer>All traffic on this page is simulated. Nothing here connects to a real network.</footer>
</div>

<script>
const $ = s => document.querySelector(s);
const HEX = {critical:'#ff4d5e',high:'#ff8a4c',medium:'#f4b942',low:'#5fa8ff',none:'#3dd3b0'};
const LABEL = {normal:'Normal',dos:'DoS flood',brute_force:'Brute force',port_scan:'Port scan',data_exfiltration:'Data exfiltration',unknown_anomaly:'Unusual behaviour'};
const SEV = {normal:'none',dos:'critical',data_exfiltration:'critical',brute_force:'high',port_scan:'medium',unknown_anomaly:'low'};
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const get = async u => { const r = await fetch(u); if(!r.ok) throw new Error(u + ' ' + r.status); return r.json(); };
const clock = t => new Date(t.replace(' ','T')).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- traffic ribbon ---------- */
let stream = [], progress = 1;
function drawRibbon(){
  const c = $('#ribbon'), dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  c.width = w*dpr; c.height = h*dpr;
  const g = c.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0,0,w,h);
  const n = stream.length; if(!n) return;
  const base = h - 16, step = w/n, upto = Math.floor(n*progress);
  g.fillStyle = '#27476b'; g.fillRect(0, base, w, 1);
  const lift = {critical:1, high:.78, medium:.58, low:.4};
  for(let i=0;i<upto;i++){
    const e = stream[i], sev = e.alert ? e.severity : 'none';
    const mag = e.alert ? lift[sev] : Math.min(.3, Math.max(.08, .3 - e.anomaly_score));
    const th = Math.max(2, mag*(base-6));
    g.globalAlpha = e.alert ? 1 : .55; g.fillStyle = HEX[sev];
    g.fillRect(i*step, base-th, Math.max(1, step*.75), th);
  }
  g.globalAlpha = 1;
}
function replay(){
  if(reduceMotion){ progress = 1; drawRibbon(); return; }
  const t0 = performance.now();
  (function tick(t){ progress = Math.min(1, (t-t0)/2200); drawRibbon(); if(progress<1) requestAnimationFrame(tick); })(t0);
}
$('#ribbon').addEventListener('mousemove', ev => {
  if(!stream.length) return;
  const r = ev.target.getBoundingClientRect(), p = ev.target.parentElement.getBoundingClientRect();
  const i = Math.min(stream.length-1, Math.floor((ev.clientX - r.left)/r.width*stream.length));
  const e = stream[i], tip = $('#tip');
  tip.textContent = clock(e.timestamp) + '  ' + LABEL[e.predicted] + (e.alert ? ' (' + e.severity + ')' : '');
  tip.style.left = (ev.clientX - p.left) + 'px'; tip.style.top = (ev.clientY - p.top) + 'px'; tip.style.opacity = 1;
});
$('#ribbon').addEventListener('mouseleave', () => $('#tip').style.opacity = 0);
$('#replay').addEventListener('click', replay);
addEventListener('resize', drawRibbon);

/* ---------- stats ---------- */
async function loadStats(){
  const {summary:s, mix} = await get('/api/stats');
  if(!s.events) return;
  $('#st-events').textContent = s.events.toLocaleString();
  $('#st-alerts').textContent = s.alerts.toLocaleString();
  $('#st-rate').textContent = s.detection_rate + '%';
  $('#st-fa').textContent = s.false_alarm_rate + '%';
  const total = Object.values(mix).reduce((a,b)=>a+b,0) || 1;
  const keys = Object.keys(mix).sort((a,b)=>mix[b]-mix[a]);
  $('#mix').innerHTML = keys.map(k => `<i style="width:${mix[k]/total*100}%;background:${HEX[SEV[k]]}"></i>`).join('');
  $('#mixkey').innerHTML = keys.map(k => `<span><i style="background:${HEX[SEV[k]]}"></i>${LABEL[k]} ${mix[k]}</span>`).join('');
}

/* ---------- alerts ---------- */
let sevFilter = '';
function buildChips(){
  const opts = [['','All'],['critical','Critical'],['high','High'],['medium','Medium'],['low','Low']];
  $('#chips').innerHTML = opts.map(([v,t]) => `<button class="chip" type="button" data-v="${v}" aria-pressed="${v===sevFilter}">${t}</button>`).join('');
  document.querySelectorAll('.chip').forEach(b => b.addEventListener('click', () => { sevFilter = b.dataset.v; buildChips(); loadAlerts(); }));
}
async function loadAlerts(){
  const rows = await get('/api/alerts?limit=40' + (sevFilter ? '&severity=' + sevFilter : ''));
  $('#alerts tbody').innerHTML = rows.length ? rows.map(r => `
    <tr class="s-${r.severity}">
      <td>${clock(r.timestamp)}</td><td class="ip">${esc(r.src_ip)}</td>
      <td class="ip">${esc(r.dst_ip)} ${esc(r.service)}</td><td>${LABEL[r.predicted]}</td>
      <td><span class="badge" style="--c:${HEX[r.severity]}">${r.severity}</span></td>
      <td>${r.confidence == null ? 'n/a' : (r.confidence*100).toFixed(0) + '%'}</td></tr>`).join('')
    : '<tr><td colspan="6" class="empty">No alerts at this level. Pick another severity.</td></tr>';
}

/* ---------- check a connection ---------- */
const FIELDS = [['duration','Duration (seconds)'],['src_bytes','Bytes sent'],['dst_bytes','Bytes received'],
  ['pkts_per_sec','Packets per second'],['conn_count','Connections from source'],['failed_logins','Failed logins'],
  ['syn_error_rate','Half-open rate (0 to 1)'],['unique_ports','Distinct ports touched']];
const PRESETS = [['normal','Normal browsing'],['dos','SYN flood'],['brute_force','Password guessing'],['port_scan','Port scan'],['data_exfiltration','Large upload']];
function buildForm(){
  $('#fields').innerHTML = FIELDS.map(([k,t]) => `<label>${t}<input name="${k}" type="number" step="any" min="0" required></label>`).join('') +
    `<label>Protocol<select name="protocol"><option>tcp</option><option>udp</option><option>icmp</option></select></label>
     <label>Service<select name="service"><option>http</option><option>https</option><option>ssh</option><option>dns</option><option>ftp</option><option>smtp</option></select></label>`;
  $('#presets').innerHTML = PRESETS.map(([k,t]) => `<button type="button" data-k="${k}">${t}</button>`).join('');
  document.querySelectorAll('#presets button').forEach(b => b.addEventListener('click', async () => {
    const rec = await get('/api/sample?kind=' + b.dataset.k);
    for(const [k,v] of Object.entries(rec)) $('#form').elements[k].value = v;
    $('#form').requestSubmit();
  }));
}
$('#form').addEventListener('submit', async ev => {
  ev.preventDefault();
  const body = Object.fromEntries(new FormData(ev.target).entries());
  const res = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const r = await res.json();
  if(!res.ok){ $('#result').innerHTML = `<p class="note" style="color:var(--high)">${esc(r.error)}</p>`; return; }
  const c = HEX[r.severity];
  const bars = Object.entries(r.probabilities).sort((a,b)=>b[1]-a[1]).map(([k,p]) =>
    `<div class="bar"><span>${LABEL[k]}</span><div><i style="width:${p*100}%;background:${HEX[SEV[k]]}"></i></div><span>${(p*100).toFixed(1)}%</span></div>`).join('');
  $('#result').innerHTML = `
    <div class="verdict" style="--c:${c}"><span>${esc(r.label)}</span><span class="badge" style="--c:${c}">${r.severity === 'none' ? 'safe' : r.severity}</span></div>
    <p class="note">Model confidence ${(r.confidence*100).toFixed(1)}%. Anomaly detector: ${r.anomaly_flag ? 'flagged as unusual' : 'within the normal range'} (score ${r.anomaly_score}).</p>
    ${bars}
    ${r.reasons.length ? '<ul class="why">' + r.reasons.map(x => `<li>${esc(x)}</li>`).join('') + '</ul>' : ''}
    <div class="action">${esc(r.action)}</div>`;
});

/* ---------- scorecard and charts ---------- */
async function loadModels(){
  const m = await get('/api/metrics');
  const names = {random_forest:'Random forest', logistic_regression:'Logistic regression', isolation_forest:'Isolation forest (attack vs normal)'};
  const best = {}; ['accuracy','precision','recall','f1'].forEach(k => best[k] = Math.max(...Object.values(m).map(x => x[k])));
  $('#models tbody').innerHTML = Object.entries(m).map(([k,v]) => `<tr><td>${names[k]}</td>` +
    ['accuracy','precision','recall','f1'].map(c => `<td class="${v[c]===best[c]?'best':''}">${(v[c]*100).toFixed(1)}%</td>`).join('') + '</tr>').join('');
}
async function loadCharts(){
  const cap = {'confusion_matrix.png':'Where the random forest gets confused','attack_distribution.png':'How much of each traffic type the dataset holds',
    'feature_importance.png':'The signals the forest relies on most','anomaly_scores.png':'How the isolation forest separates attacks from normal','alert_timeline.png':'Alerts over the length of the stream'};
  const files = await get('/api/images');
  $('#gallery').innerHTML = files.map(f => `<figure><a href="/charts/${f}" target="_blank" rel="noopener"><img src="/charts/${f}" alt="${esc(cap[f]||f)}" loading="lazy"></a><figcaption>${esc(cap[f]||f)}</figcaption></figure>`).join('');
}

/* ---------- start ---------- */
(async () => {
  buildChips(); buildForm();
  try{
    const h = await get('/api/health');
    $('#status').textContent = 'Models ready, trained ' + h.trained_at.replace('T',' ');
    stream = await get('/api/stream'); replay();
    await Promise.all([loadStats(), loadAlerts(), loadModels(), loadCharts()]);
  }catch(e){
    $('#status').textContent = 'Could not load results. Run rullall.py first.';
    $('.dot').style.background = 'var(--critical)';
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000, debug=False)
