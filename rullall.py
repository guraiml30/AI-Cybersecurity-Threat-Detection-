#!/usr/bin/env python3
"""
rullall.py ("run all") - one command that deploys the whole Sentinel project.

    python rullall.py                 # run every pipeline, then start the API
    python rullall.py --no-api        # pipelines only
    python rullall.py --api-only      # reuse trained models, just start the API
    python rullall.py --open          # also open the dashboard in your browser

Pipelines, in order
  1. setup      create folders
  2. data       simulate network traffic (or load your own CSV with --data)
  3. clean      drop duplicates / bad rows
  4. features   engineer features, split train and test
  5. train      Random Forest, Logistic Regression, Isolation Forest
  6. evaluate   accuracy, precision, recall, F1 -> outputs/metrics.json
  7. detect     score a fresh simulated stream and raise alerts
  8. visualize  themed charts -> outputs/images/
  9. report     outputs/run_summary.json
 10. api        themed dashboard + REST API (Flask)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import warnings
import webbrowser
from datetime import datetime

warnings.filterwarnings("ignore")

from src.data_loader import stage_data, stage_setup
from src.detect import stage_detect
from src.evaluate import stage_evaluate
from src.features import stage_features
from src.preprocess import stage_clean
from src.report import stage_report
from src.train import stage_train
from src.visualize import stage_visualize
import src.threat_core as tc

# Each pipeline lives in its own file under src/. Add or reorder stages here.
PIPELINES = [
    ("setup", stage_setup), ("data", stage_data), ("clean", stage_clean),
    ("features", stage_features), ("train", stage_train), ("evaluate", stage_evaluate),
    ("detect", stage_detect), ("visualize", stage_visualize), ("report", stage_report),
]


def start_api(args) -> int:
    needed = ["random_forest.joblib", "isolation_forest.joblib", "meta.json"]
    if any(not (tc.MODELS / f).exists() for f in needed):
        print("No trained models found in models/. Run `python rullall.py` once without --api-only.")
        return 1
    from src.api_server import create_app
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Sentinel is live at {url}\n  Press Ctrl+C to stop.\n")
    if args.open:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    try:
        create_app().run(host=args.host, port=args.port, debug=False)
    except OSError as e:
        print(f"Could not start the API: {e}. Try another port, e.g. --port 8001")
        return 1
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Run every Sentinel pipeline, then start the API.")
    p.add_argument("--samples", type=int, default=20000, help="training connections to simulate")
    p.add_argument("--stream-size", type=int, default=1500, help="connections in the live detection stream")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", help="your own CSV (columns: %s, label)" % ", ".join(tc.RAW_FIELDS))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-api", action="store_true", help="run pipelines only")
    p.add_argument("--api-only", action="store_true", help="skip pipelines, start the API")
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    return p.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    print("\n  Sentinel | AI-powered cybersecurity threat detection\n")
    if args.api_only:
        return start_api(args)

    ctx: dict = {"log": []}
    for i, (name, fn) in enumerate(PIPELINES, 1):
        print(f"[{i}/{len(PIPELINES)}] {name}")
        t0 = time.time()
        try:
            fn(ctx, args)
            status = "ok"
        except Exception:
            traceback.print_exc()
            status = "failed"
        secs = round(time.time() - t0, 1)
        ctx["log"].append({"stage": name, "seconds": secs, "status": status})
        print(f"        {status} in {secs}s")
        if status == "failed":
            print(f"\nStopped at '{name}'. Fix the error above and run again.")
            return 1

    total = sum(s["seconds"] for s in ctx["log"])
    print(f"\nAll {len(PIPELINES)} pipelines finished in {total:.1f}s. Charts are in outputs/images/.")
    return 0 if args.no_api else start_api(args)


if __name__ == "__main__":
    sys.exit(main())
