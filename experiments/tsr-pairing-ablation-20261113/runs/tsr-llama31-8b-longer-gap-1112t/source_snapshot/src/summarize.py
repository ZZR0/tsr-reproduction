#!/usr/bin/env python3
"""Collect AlpacaEval results: summarize.py EVAL_ROOT [--csv OUT.csv]
EVAL_ROOT/<model>/leaderboard.csv -> one row per model: LC, Win (+SE), mean length, n."""
import argparse, glob, os
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--csv")
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.root, "*", "leaderboard.csv"))):
    lb = pd.read_csv(f, index_col=0).iloc[0]
    rows.append({"model": os.path.basename(os.path.dirname(f)), "LC": round(lb.get("length_controlled_winrate", float("nan")), 2),
                 "Win": round(lb["win_rate"], 2), "Win_SE": round(lb["standard_error"], 2), "length": int(lb["avg_length"]), "n": int(lb["n_total"])})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
if a.csv: df.to_csv(a.csv, index=False)
