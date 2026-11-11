#!/usr/bin/env python3
"""Re-implementation of the OFFICIAL TSR pair rules (settings/tsr/reformat2DPO_{past,future}.py) with stats and an
optional paper-style filter (keep only s_chosen > s_rejected, Algorithm 1 lines 11/19).
  past:   build_pairs.py past   --chosen C_judgement.json --rejected R_judgement.json --out DIR [--filter]
  future: build_pairs.py future --chosen C --rejected R --future F --out DIR [--filter]
  future only: add --future-only (always Mf chosen; implicitly require chosen score > rejected score).
In repo mode (no --filter) the output must equal the official script's output (checked by --check OFFICIAL_TRAIN_JSON)."""
import json, argparse, os, re, collections
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["past", "future"])
ap.add_argument("--chosen", required=True); ap.add_argument("--rejected", required=True); ap.add_argument("--future")
ap.add_argument("--out", required=True); ap.add_argument("--filter", action="store_true"); ap.add_argument("--check")
ap.add_argument("--strict", action="store_true", help="paper Alg.1 line 17: upgrade to Mf only if max(s_f) > max(s_i)")
ap.add_argument("--future-only", action="store_true", help="chosen only from Mf's highest-scored responses; automatically drop score gaps <= 0")
ap.add_argument("--score", choices=["plain", "conf"], default="plain", help="conf = official score x p(score token) from *_conf.json")
ap.add_argument("--cho_len", choices=["first", "longest", "shortest"], default="first", help="chosen among top-score ties: repo takes the first; shortest = Meta-Rewarding length control; longest = longest top-score response (on a current/future tie, across both models)")
ap.add_argument("--rej_len", choices=["first", "longest"], default="first", help="rejected among bottom-score ties (union of Mi and M0 pools on a tie): longest = Meta-Rewarding length control")
ap.add_argument("--within", choices=["first", "conf"], default="first", help="among same-score candidates of the selected source: repo takes the first; conf = highest p(score token) from *_conf.json")
ap.add_argument("--rej_tie", choices=["mi", "past"], default="mi", help="past/current tie for rejected: repo gives it to Mi; past = give it to M0")
a = ap.parse_args()
if a.mode == "future" and not a.future:
    ap.error("future mode requires --future")
if a.future_only:
    if a.mode != "future":
        ap.error("--future-only requires future mode")
    if a.strict:
        ap.error("--future-only cannot be combined with --strict (no current/future comparison)")
    if a.score != "plain":
        ap.error("--future-only requires plain scores for strict chosen > rejected filtering")
    a.filter = True
SYS = {"content": "You are a helpful, respectful, and knowledgeable assistant. Your goal is to provide accurate, concise, and relevant answers to user queries. If you are unsure about something, politely let the user know and avoid making assumptions. Always prioritize clarity and helpfulness in your responses.", "role": "system"}
C, R = json.load(open(a.chosen)), json.load(open(a.rejected))
F = json.load(open(a.future)) if a.mode == "future" else None
if a.score == "conf":
    for X, path in [(C, a.chosen), (R, a.rejected)] + ([(F, a.future)] if F is not None else []):
        Xc = json.load(open(path.replace("_judgement.json", "_conf.json")))
        for x, xc in zip(X, Xc): x["scores"] = [v["s"] * v["p"] if v["p"] is not None else v["s"] for v in xc]
assert len(C) == len(R) and (F is None or len(F) == len(C))
if a.within == "conf":
    CC, RC = [json.load(open(p.replace("_judgement.json", "_conf.json"))) for p in (a.chosen, a.rejected)]
    FC = json.load(open(a.future.replace("_judgement.json", "_conf.json"))) if F is not None else None
    def pick(x, xc, s):  # same-score candidates; highest p(score token), ties -> lowest index (as the official first-index rule)
        idx = [k for k, v in enumerate(x["scores"]) if v == s]
        assert all(xc[k]["s"] == x["scores"][k] for k in idx)
        return x["responses"][max(idx, key=lambda k: ((xc[k]["p"] if xc[k]["p"] is not None else -1.0), -k))]
st = collections.Counter(); pairs = []; gaps = []
for i in range(len(C)):
    c, r = C[i], R[i]; assert c["prompt"] == r["prompt"]
    i_max = c["scores"].index(max(c["scores"])); s_max = c["scores"][i_max]
    i1 = c["scores"].index(min(c["scores"])); s1 = c["scores"][i1]
    i2 = r["scores"].index(min(r["scores"])); s2 = r["scores"][i2]
    if s1 < s2 or (s1 == s2 and a.rej_tie == "mi"): rej, s_rej, src = c["responses"][i1], s1, "Mi"
    else: rej, s_rej, src = r["responses"][i2], s2, "M0"
    st[f"rejected_from_{src}"] += 1
    cho, s_cho = c["responses"][i_max], s_max
    if F is not None:
        f = F[i]; assert f["prompt"] == c["prompt"]
        i3 = f["scores"].index(max(f["scores"])); s3 = f["scores"][i3]
        keep_mi = False if a.future_only else (not (s3 > s_max) if a.strict else s_max > s3)
        if keep_mi: st["chosen_from_Mi"] += 1
        else:
            cho, s_cho = f["responses"][i3], s3
            st["chosen_from_Mf"] += 1; st["chosen_from_Mf_by_tie"] += int(s3 == s_max)
    if a.within == "conf":
        new_rej = pick(c, CC[i], s1) if src == "Mi" else pick(r, RC[i], s2)
        from_f = F is not None and not keep_mi
        new_cho = pick(f, FC[i], s3) if from_f else pick(c, CC[i], s_max)
        st["within_rej_changed"] += int(new_rej != rej); st["within_cho_changed"] += int(new_cho != cho); rej, cho = new_rej, new_cho
    if a.rej_len == "longest":
        bot = lambda x, s: [x["responses"][k] for k, v in enumerate(x["scores"]) if v == s]
        rpool = bot(c, s1) if s1 < s2 else bot(r, s2) if s2 < s1 else bot(c, s1) + bot(r, s2)
        new_rej = max(rpool, key=len); st["rej_longest_changed"] += int(new_rej != rej); rej = new_rej
        st["rej_longest_from_M0"] += int(rej in r["responses"] and rej not in c["responses"])
    if a.cho_len != "first":
        top = lambda x, s: [x["responses"][k] for k, v in enumerate(x["scores"]) if v == s]
        if F is None: pool = top(c, s_max)
        elif a.future_only: pool = top(f, s3)
        elif s3 > s_max or (s3 == s_max and not a.strict): pool = top(f, s3) + (top(c, s_max) if s3 == s_max else [])
        else: pool = top(c, s_max)
        new_cho = (max if a.cho_len == "longest" else min)(pool, key=len); st["cho_longest_changed"] += int(new_cho != cho); cho = new_cho
        st["cho_longest_from_Mf"] += int(F is not None and cho in F[i]["responses"])
    if cho is None or rej is None: continue
    st["identical_pair"] += int(cho == rej); st["tie_pair"] += int(s_cho == s_rej)
    if a.filter and not s_cho > s_rej: st["filtered_out"] += 1; continue
    if F is not None:
        st["retained_chosen_from_Mi" if keep_mi else "retained_chosen_from_Mf"] += 1
    st["retained_tie_pairs"] += int(s_cho == s_rej)
    gaps.append(s_cho - s_rej)
    pairs.append({"prompt": c["prompt"],
                  "chosen": [SYS, {"content": c["prompt"], "role": "user"}, {"content": cho, "role": "assistant"}],
                  "rejected": [SYS, {"content": c["prompt"], "role": "user"}, {"content": rej, "role": "assistant"}]})
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/train.json", "w", encoding="utf-8") as fo:
    for p in pairs: fo.write(json.dumps(p, ensure_ascii=False) + "\n")
st["n_prompts"] = len(C); st["n_pairs"] = len(pairs)
st["chosen_policy"] = "future_only" if a.future_only else "current_future_best" if F is not None else "current_best"
st["strict_score_gap"] = a.filter
st["gap_min"] = min(gaps) if gaps else None
st["gap_mean"] = round(sum(gaps) / max(len(gaps), 1), 3)
st["gap_median"] = sorted(gaps)[len(gaps) // 2] if gaps else None
if a.check:
    off = [json.loads(l) for l in open(a.check)]
    st["matches_official"] = off == pairs
json.dump(dict(st), open(f"{a.out}/stats.json", "w"), indent=1)
print(json.dumps(dict(st), indent=1))
