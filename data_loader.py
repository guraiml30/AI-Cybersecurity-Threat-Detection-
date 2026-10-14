"""
Pipelines 1-2: create folders, then simulate traffic (or load your own CSV).
"""
from __future__ import annotations

import pandas as pd

import src.threat_core as tc


def stage_setup(ctx, args):
    for d in tc.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"        project root: {tc.ROOT}")


def stage_data(ctx, args):
    if args.data:
        df = pd.read_csv(args.data)
        missing = [c for c in tc.RAW_FIELDS + ["label"] if c not in df.columns]
        if missing:
            raise ValueError(f"{args.data} is missing columns: {missing}")
        print(f"        loaded {len(df):,} rows from {args.data}")
    else:
        df = tc.simulate_traffic(args.samples, args.seed)
        print(f"        simulated {len(df):,} connections (seed {args.seed})")
    df.to_csv(tc.DATA_RAW / "traffic_raw.csv", index=False)
    print("        class counts:", df["label"].value_counts().to_dict())
    ctx["df"] = df
