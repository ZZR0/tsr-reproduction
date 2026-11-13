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
ap.add_argument("--keep-equal-scores", action="store_true", help="Keep all score ties; chosen source ties prefer future, rejected source ties prefer current; no length selection")
ap.add_argument("--strict", action="store_true", help="paper Alg.1 line 17: upgrade to Mf only if max(s_f) > max(s_i)")
ap.add_argument("--min-length-gap", action="store_true", help="Jointly choose the closest character lengths from the highest/lowest score pools, without model-source priority")
ap.add_argument("--source-priority-length-gap", action="store_true", help="Strict score gap, then future/current source ties, then joint minimum character length gap")
ap.add_argument("--different-text", action="store_true", help="Require distinct stripped text in fixed source/score pools; first valid combination unless length matching is requested")
ap.add_argument("--future-only", action="store_true", help="chosen only from Mf's highest-scored responses; automatically drop score gaps <= 0")
ap.add_argument("--score", choices=["plain", "conf"], default="plain", help="conf = official score x p(score token) from *_conf.json")
ap.add_argument("--cho_len", choices=["first", "longest", "shortest"], default="first", help="chosen among top-score ties: repo takes the first; shortest = Meta-Rewarding length control; longest = longest top-score response (on a current/future tie, across both models)")
ap.add_argument("--rej_len", choices=["first", "longest"], default="first", help="rejected among bottom-score ties (union of Mi and M0 pools on a tie): longest = Meta-Rewarding length control")
ap.add_argument("--within", choices=["first", "conf"], default="first", help="among same-score candidates of the selected source: repo takes the first; conf = highest p(score token) from *_conf.json")
ap.add_argument("--rej_tie", choices=["mi", "past"], default="mi", help="past/current tie for rejected: repo gives it to Mi; past = give it to M0")
a = ap.parse_args()
if a.different_text and not a.source_priority_length_gap:
    if a.min_length_gap or a.strict or a.future_only or a.score != "plain" or a.within != "first" or a.cho_len != "first" or a.rej_len != "first" or a.rej_tie != "mi":
        ap.error("Distinct-text source-priority mode requires plain scores, future/current tie priorities and no length ranking")
    a.filter = True

if a.keep_equal_scores:
    if (a.filter or a.strict or a.future_only or a.min_length_gap or a.source_priority_length_gap or a.score != "plain"
            or a.within != "first" or a.cho_len != "first" or a.rej_len != "first" or a.rej_tie != "mi"):
        ap.error("--keep-equal-scores requires future/current source priorities, plain scores and no filtering or length selection")
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
if a.source_priority_length_gap and a.min_length_gap:
    ap.error("Choose one length-matching policy")
if a.min_length_gap or a.source_priority_length_gap:
    if a.future_only or a.strict or a.score != "plain" or a.within != "first" or a.cho_len != "first" or a.rej_len != "first" or a.rej_tie != "mi":
        ap.error("--min-length-gap replaces source/length tie rules and requires plain scores")
    a.filter = True
if a.different_text or a.min_length_gap or a.source_priority_length_gap:
    from length_pairing import select_min_length_pair, select_first_distinct_pair
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
st = collections.Counter(); pairs = []; gaps = []; length_gaps = []; chosen_lengths = []; rejected_lengths = []
chosen_pool_hist = collections.Counter(); rejected_pool_hist = collections.Counter()
for i in range(len(C)):
    c, r = C[i], R[i]; assert c["prompt"] == r["prompt"]
    if a.min_length_gap or a.source_priority_length_gap or a.different_text:
        f = F[i] if F is not None else None
        if f is not None:
            assert f["prompt"] == c["prompt"]
        if a.different_text and not a.source_priority_length_gap:
            selected = select_first_distinct_pair(c, r, f)
        else:
            selected = select_min_length_pair(c, r, f, source_priority=a.source_priority_length_gap, require_distinct_text=a.different_text)
        if not selected["valid"]:
            st["tie_pair"] += int(selected["chosen_score"] == selected["rejected_score"])
            st["filtered_out"] += 1
            st["filtered_equal_score_pairs"] += int(selected["chosen_score"] == selected["rejected_score"])
            st["filtered_no_distinct_text_pairs"] += int(selected.get("reason") == "no_distinct_text_pair")
            continue
        chosen_item, rejected_item = selected["chosen"], selected["rejected"]
        cho, rej = chosen_item["text"], rejected_item["text"]
        st["rejected_from_" + rejected_item["source"]] += 1
        st["retained_chosen_from_" + chosen_item["source"]] += 1
        if F is not None:
            st["chosen_from_" + chosen_item["source"]] += 1
            st["chosen_from_Mf_by_tie"] += int(chosen_item["source"] == "Mf" and max(c["scores"]) == max(f["scores"]))
        st["identical_pair"] += int(cho == rej)
        st["retained_tie_pairs"] += 0
        st["identical_text_combinations_excluded"] += selected["identical_text_combinations_excluded"]
        st["valid_pair_combinations_total"] += selected["valid_pair_combinations"]
        st["high_pool_candidates_total"] += selected["n_high_candidates"]
        st["low_pool_candidates_total"] += selected["n_low_candidates"]
        chosen_pool_hist[selected["n_high_candidates"]] += 1
        rejected_pool_hist[selected["n_low_candidates"]] += 1
        length_gaps.append(selected["length_gap"])
        chosen_lengths.append(len(cho)); rejected_lengths.append(len(rej))
        gaps.append(selected["chosen_score"] - selected["rejected_score"])
        pairs.append({"prompt":c["prompt"],
            "chosen":[SYS, {"content":c["prompt"],"role":"user"}, {"content":cho,"role":"assistant"}],
            "rejected":[SYS, {"content":c["prompt"],"role":"user"}, {"content":rej,"role":"assistant"}]})
        continue
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
    if a.keep_equal_scores:
        # Count tied extrema after model-source priorities, without retaining response dumps.
        chosen_scores = f["scores"] if F is not None and not keep_mi else c["scores"]
        rejected_scores = c["scores"] if src == "Mi" else r["scores"]
        chosen_pool_hist[sum(v == s_cho for v in chosen_scores)] += 1
        rejected_pool_hist[sum(v == s_rej for v in rejected_scores)] += 1
    gaps.append(s_cho - s_rej)
    length_gaps.append(abs(len(cho)-len(rej))); chosen_lengths.append(len(cho)); rejected_lengths.append(len(rej))
    pairs.append({"prompt": c["prompt"],
                  "chosen": [SYS, {"content": c["prompt"], "role": "user"}, {"content": cho, "role": "assistant"}],
                  "rejected": [SYS, {"content": c["prompt"], "role": "user"}, {"content": rej, "role": "assistant"}]})
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/train.json", "w", encoding="utf-8") as fo:
    for p in pairs: fo.write(json.dumps(p, ensure_ascii=False) + "\n")
st["n_prompts"] = len(C); st["n_pairs"] = len(pairs)
st["chosen_policy"] = "future_only" if a.future_only else "current_future_best" if F is not None else "current_best"
st["strict_score_gap"] = a.filter
st["base_chosen_source"] = "current"
st["chosen_tie_break"] = ("future_only" if a.future_only else "current" if a.strict else "future") if F is not None else "not_applicable"
st["rejected_tie_break"] = "current" if a.rej_tie == "mi" else "past"
if a.min_length_gap:
    st["pair_selection"] = "min_length_gap"
    st["chosen_tie_break"] = st["rejected_tie_break"] = "joint_minimum_length_gap"
    st["model_source_priority"] = False
    st["length_tie_break"] = "sha256(prompt + chosen_text + rejected_text); no source names"
if a.source_priority_length_gap:
    st["pair_selection"] = "future_tie_length_distinct" if a.different_text else "future_tie_length"
    st["filter_equal_scores"] = True
    st["model_source_priority"] = True
    st["selection_order"] = ["strict_score_gap", "model_source_priority", "joint_minimum_character_length_gap"]
    st["length_selection"] = "joint_minimum_after_source_priority"
    st["require_distinct_text"] = a.different_text
    st["text_comparison"] = "strip leading/trailing whitespace" if a.different_text else "unrestricted"
    st["length_tie_break"] = "sha256(prompt + chosen_text + rejected_text); deterministic"
if a.different_text and not a.source_priority_length_gap:
    st["pair_selection"] = "future_tie_distinct"
    st["filter_equal_scores"] = True
    st["model_source_priority"] = True
    st["selection_order"] = ["strict_score_gap", "model_source_priority", "first_distinct_combination"]
    st["length_selection"] = "none"
    st["require_distinct_text"] = True
    st["text_comparison"] = "strip leading/trailing whitespace"
if a.keep_equal_scores:
    st["pair_selection"] = "future_tie_unfiltered"
    st["filter_equal_scores"] = False
    st["model_source_priority"] = True
    st["length_selection"] = "none"
st["length_unit"] = "unicode_characters"
st["chosen_length_mean"] = sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else None
st["rejected_length_mean"] = sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else None
st["absolute_length_gap_mean"] = sum(length_gaps) / len(length_gaps) if length_gaps else None
st["absolute_length_gap_median"] = sorted(length_gaps)[len(length_gaps)//2] if length_gaps else None
st["absolute_length_gap_max"] = max(length_gaps) if length_gaps else None
st["zero_length_gap_pairs"] = sum(x == 0 for x in length_gaps)
st["gap_min"] = min(gaps) if gaps else None
st["gap_mean"] = round(sum(gaps) / max(len(gaps), 1), 3)
st["gap_median"] = sorted(gaps)[len(gaps) // 2] if gaps else None
if a.keep_equal_scores or a.source_priority_length_gap or a.different_text:
    def pool_summary(hist):
        values = sorted(k for k, count in hist.items() for _ in range(count))
        n = len(values)
        return {"n": n, "mean": sum(values) / n if n else None,
                "median": (values[(n - 1) // 2] + values[n // 2]) / 2 if n else None,
                "histogram": {str(k): hist[k] for k in sorted(hist)},
                "multiple_candidates_rate": sum(count for k, count in hist.items() if k > 1) / n if n else None}
    st["within_source_tie_selection"] = "joint minimum character length gap; deterministic hash for equal gaps" if a.source_priority_length_gap else "first valid distinct-text combination in original candidate order" if a.different_text else "first index; no random selection"
    st["pool_count_unit"] = "candidate entries, without text deduplication; after source priorities"
    st["chosen_candidate_pool"] = pool_summary(chosen_pool_hist)
    st["rejected_candidate_pool"] = pool_summary(rejected_pool_hist)
if a.check:
    off = [json.loads(l) for l in open(a.check)]
    st["matches_official"] = off == pairs
json.dump(dict(st), open(f"{a.out}/stats.json", "w"), indent=1)
print(json.dumps(dict(st), indent=1))
