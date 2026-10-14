"""
Pipeline 6: accuracy, precision, recall and F1 on the held-out test set.
"""
from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

import src.threat_core as tc


def _scores(y_true, y_pred, average):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}


def stage_evaluate(ctx, args):
    X_te, y_te = ctx["X_test"], ctx["y_test"]
    rf_pred = ctx["rf"].predict(X_te)
    lr_pred = ctx["lr"].predict(X_te)
    iso_score = ctx["iso"].decision_function(X_te)
    is_attack = (y_te != "normal").values
    metrics = {
        "random_forest": _scores(y_te, rf_pred, "weighted"),
        "logistic_regression": _scores(y_te, lr_pred, "weighted"),
        "isolation_forest": {**_scores(is_attack, iso_score < ctx["thr"], "binary"),
                             "note": "Attack vs normal only. Needs no attack labels."},
    }
    (tc.OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_te, rf_pred, labels=tc.CLASSES, zero_division=0)
    (tc.OUTPUTS / "classification_report.txt").write_text(report)
    for name, m in metrics.items():
        print(f"        {name:<20} acc {m['accuracy']:.3f}  prec {m['precision']:.3f}  "
              f"rec {m['recall']:.3f}  f1 {m['f1']:.3f}")
    ctx.update(metrics=metrics, rf_pred=rf_pred, iso_score=iso_score)
