#!/usr/bin/env python3
"""How often the future model's best score ties / beats / loses to the current model's best score.
  tie_stats.py CURRENT_judgement.json FUTURE_judgement.json
Under the repo rule (upgrade if max_future >= max_current) every tie becomes an upgrade."""
import collections, json, sys
C, F = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
cnt, top = collections.Counter(), collections.Counter()
for c, f in zip(C, F):
    mi, mf = max(c["scores"]), max(f["scores"])
    cnt[">" if mf > mi else "=" if mf == mi else "<"] += 1
    if mf == mi: top[mi] += 1
n = len(C)
print(json.dumps({"n": n, "future_higher": cnt[">"], "tie": cnt["="], "future_lower": cnt["<"],
                  "tie_rate": round(cnt["="] / n, 4), "upgrade_rate_repo_rule": round((cnt[">"] + cnt["="]) / n, 4),
                  "upgrade_rate_strict_rule": round(cnt[">"] / n, 4), "tie_top_score": {str(k): v for k, v in sorted(top.items())}}, indent=1))
