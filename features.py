"""
Pipeline 4: engineer features and split into train and test sets.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

import src.threat_core as tc


def stage_features(ctx, args):
    df = ctx["df"]
    feats = tc.engineer_features(df)
    X, y = feats[tc.FEATURES], feats["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)
    pd.concat([X, y], axis=1).to_csv(tc.DATA_PROCESSED / "features.csv", index=False)
    print(f"        {len(tc.FEATURES)} features | train {len(X_tr):,} | test {len(X_te):,}")
    ctx.update(X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te)
