"""
Pipeline 9: write outputs/run_summary.json.
"""
from __future__ import annotations

import json
from datetime import datetime

import src.threat_core as tc


def stage_report(ctx, args):
    out = {"finished_at": datetime.now().isoformat(timespec="seconds"),
           "settings": {"samples": args.samples, "stream_size": args.stream_size, "seed": args.seed},
           "stages": ctx["log"], "metrics": ctx.get("metrics"), "detection": ctx.get("detection")}
    (tc.OUTPUTS / "run_summary.json").write_text(json.dumps(out, indent=2))
    print("        wrote outputs/run_summary.json")
