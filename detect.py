"""
Pipeline 7: score a fresh simulated stream and raise alerts.
"""
from __future__ import annotations

import json

import numpy as np

import src.threat_core as tc


def stage_detect(ctx, args):
    """Score a brand-new simulated stream, exactly as a SOC pipeline would score live logs."""
    stream = tc.simulate_traffic(args.stream_size, args.seed + 1)
    X = tc.engineer_features(stream)[tc.FEATURES]
    proba = ctx["rf"].predict_proba(X)
    pred = ctx["rf"].classes_[proba.argmax(1)]
    score = ctx["iso"].decision_function(X)
    unusual = (pred == "normal") & (score < ctx["thr"])  # RF sees nothing, but behaviour is odd
    verdict = np.where(unusual, "unknown_anomaly", pred)
    stream["predicted"] = verdict
    stream["severity"] = [tc.SEVERITY[v] for v in verdict]
    conf = proba.max(1).round(4)
    stream["confidence"] = np.where(unusual, np.nan, conf)
    stream["anomaly_score"] = score.round(4)
    stream["alert"] = verdict != "normal"
    stream = stream.rename(columns={"label": "true_label"})

    cols = ["timestamp", "src_ip", "dst_ip", "protocol", "service", "true_label",
            "predicted", "severity", "confidence", "anomaly_score", "alert"]
    stream[cols].to_csv(tc.OUTPUTS / "stream_scored.csv", index=False)
    alerts = stream[stream["alert"]][cols]
    alerts.to_csv(tc.OUTPUTS / "alerts.csv", index=False)

    attack = stream["true_label"] != "normal"
    caught = int((stream["alert"] & attack).sum())
    false_alarms = int((stream["alert"] & ~attack).sum())
    summary = {
        "events": int(len(stream)), "alerts": int(stream["alert"].sum()),
        "attacks": int(attack.sum()), "detected": caught, "missed": int(attack.sum()) - caught,
        "false_alarms": false_alarms,
        "detection_rate": round(100 * caught / max(int(attack.sum()), 1), 1),
        "false_alarm_rate": round(100 * false_alarms / max(int((~attack).sum()), 1), 2),
    }
    (tc.OUTPUTS / "detection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"        {summary['alerts']} alerts from {summary['events']:,} events | "
          f"caught {caught}/{summary['attacks']} attacks ({summary['detection_rate']}%) | "
          f"false alarms {summary['false_alarm_rate']}%")
    ctx.update(stream=stream, alerts=alerts, detection=summary)
