"""
Pipeline 3: clean the raw data (duplicates, bad values, unknown labels).
"""
from __future__ import annotations

import pandas as pd

import src.threat_core as tc


def stage_clean(ctx, args):
    df = ctx["df"]
    before = len(df)
    df = df.drop_duplicates().copy()
    for c in tc.NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lower=0)
    df = df.dropna(subset=tc.RAW_FIELDS + ["label"])
    df = df[df["label"].isin(tc.CLASSES)].copy()
    df["syn_error_rate"] = df["syn_error_rate"].clip(upper=1)
    print(f"        kept {len(df):,} of {before:,} rows")
    df.to_csv(tc.DATA_PROCESSED / "clean.csv", index=False)
    ctx["df"] = df.reset_index(drop=True)
