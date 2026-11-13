#!/usr/bin/env python3
"""Paired bootstrap over the 805 AlpacaEval instructions: difference A - B in Win and LC, with a 95% percentile CI.
  paired_bootstrap.py EVAL_DIR_A EVAL_DIR_B [--n_boot 1000]
EVAL_DIR_* contain annotations.json from the same judge. Point estimates are the official leaderboard values;
note that alpaca_eval's LC fit depends on row order (cross-validation folds), which moves LC by up to ~0.8."""
import argparse, json, logging, warnings
import numpy as np, pandas as pd
from alpaca_eval.metrics.glm_winrate import get_length_controlled_winrate
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--n_boot", type=int, default=1000)
args = ap.parse_args()

def load(d):
    return pd.DataFrame(json.load(open(f"{d}/annotations.json"))).sort_values("instruction").reset_index(drop=True)
def official(d):
    lb = pd.read_csv(f"{d}/leaderboard.csv", index_col=0).iloc[0]
    return lb["win_rate"], lb["length_controlled_winrate"]
def stats(df):
    m = get_length_controlled_winrate(df.copy(), save_weights_dir=None, is_add_glm_preference_inplace=False, is_warn_extreme_changes=False)
    return m["win_rate"], m["length_controlled_winrate"]

A, B = load(args.a), load(args.b)
assert len(A) == len(B) == 805 and (A["instruction"].values == B["instruction"].values).all()
(wa, la), (wb, lb) = official(args.a), official(args.b)
rng = np.random.default_rng(0); dw, dl = [], []
for _ in range(args.n_boot):
    idx = rng.integers(0, 805, 805)
    (w1, l1), (w2, l2) = stats(A.iloc[idx].reset_index(drop=True)), stats(B.iloc[idx].reset_index(drop=True))
    dw.append(w1 - w2); dl.append(l1 - l2)
ci = lambda x: [round(v, 2) for v in np.percentile(x, [2.5, 97.5])]
print(json.dumps({"A": args.a, "B": args.b, "dWin": round(wa - wb, 2), "dWin_CI95": ci(dw),
                  "dLC": round(la - lb, 2), "dLC_CI95": ci(dl), "n_boot": args.n_boot}))
