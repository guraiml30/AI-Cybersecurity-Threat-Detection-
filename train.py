"""
Pipeline 5: train Random Forest, Logistic Regression and Isolation Forest, then save them.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.threat_core as tc


def stage_train(ctx, args):
    X_tr, y_tr = ctx["X_train"], ctx["y_train"]
    rf = Pipeline([("pre", tc.make_preprocessor()),
                   ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=args.seed))])
    lr = Pipeline([("pre", tc.make_preprocessor()),
                   ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    iso = Pipeline([("pre", tc.make_preprocessor()),
                    ("model", IsolationForest(n_estimators=200, random_state=args.seed, n_jobs=-1))])
    rf.fit(X_tr, y_tr)
    print("        Random Forest trained")
    lr.fit(X_tr, y_tr)
    print("        Logistic Regression trained")
    normal_only = X_tr[(y_tr == "normal").values]
    iso.fit(normal_only)  # learns what "normal" looks like, no attack labels needed
    thr = float(np.percentile(iso.decision_function(normal_only), 1))
    print(f"        Isolation Forest trained on {len(normal_only):,} normal rows (alert threshold {thr:.3f})")
    joblib.dump(rf, tc.MODELS / "random_forest.joblib")
    joblib.dump(lr, tc.MODELS / "logistic_regression.joblib")
    joblib.dump(iso, tc.MODELS / "isolation_forest.joblib")
    meta = {"classes": list(rf.classes_), "features": tc.FEATURES, "anomaly_threshold": thr,
            "trained_at": datetime.now().isoformat(timespec="seconds")}
    (tc.MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    ctx.update(rf=rf, lr=lr, iso=iso, thr=thr)
