#!/usr/bin/env python3
"""Self-judging health stats for a TSR *_judgement.json (official format: judgments[k][0], scores[k])."""
import json, sys, re, collections, os
if os.environ.get('TSR_DROP_INVALID_SCORES') == '1':
    from score_validation import validation_summary
    print(json.dumps(dict(file=sys.argv[1], **validation_summary(json.load(open(sys.argv[1])))), indent=1))
    raise SystemExit(0)
d = json.load(open(sys.argv[1])); SC = re.compile(r"Score:\s*(\d+(?:\.\d+)?)")
n = 0; fail = 0; oor = 0; dist = collections.Counter(); alltie = 0; nmax5 = 0; lens = []
for it in d:
    s = it["scores"]; alltie += int(max(s) == min(s)); nmax5 += int(max(s) == 5)
    for j, sc, resp in zip(it["judgments"], s, it["responses"]):
        n += 1; t = j[0]; m = SC.search(t); lens.append(len(resp))
        if not m: fail += 1
        elif float(m.group(1)) > 5: oor += 1
        dist[sc if isinstance(sc, (int, float)) else str(sc)] += 1
print(json.dumps({"file": sys.argv[1], "n_items": len(d), "n_judgments": n,
    "parse_fail_rate": round(fail / n, 4), "out_of_range_rate": round(oor / n, 4),
    "group_all_tie_rate": round(alltie / len(d), 4), "group_max_is_5_rate": round(nmax5 / len(d), 4),
    "score_dist": {str(k): v for k, v in sorted(dist.items(), key=lambda x: float(x[0]))},
    "score_mean": round(sum(float(k) * v for k, v in dist.items()) / n, 3), "resp_mean_chars": round(sum(lens) / n)}, indent=1))
