#!/usr/bin/env python3
"""Confidence of existing TSR judgments (used by the score x p(score token) and within-tier confidence variants).
No re-sampling: for each saved judgment, feed the OFFICIAL judge prompt + the judgment text up to the score digit
(first match of the official regex "Score:\s*(\d+(?:\.\d+)?)") and read the judge's next-token distribution (T=1).
  tsr_conf.py --judge MODEL --inp X_judgement.json --out X_conf.json [--limit N]
Output per item: list of 7 dicts {s: official score, p: p(first digit token), pd: {d: p(d) for d in 0..9}, dec: bool}."""
import json, re, os, argparse, importlib.util, time
from vllm import LLM, SamplingParams
ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True); ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("J", os.path.join(os.environ["TSR_DIR"], "tsr", "self_generate_vllm_all_iterJudge.py"))
J = importlib.util.module_from_spec(spec); spec.loader.exec_module(J)
llm = LLM(model=a.judge, tensor_parallel_size=1, enable_prefix_caching=True, gpu_memory_utilization=0.85, seed=0)
G = J.SelfRewardGenerator.__new__(J.SelfRewardGenerator); G.tokenizer = llm.get_tokenizer()
tok = G.tokenizer
DIG = {str(d): tok.encode(str(d), add_special_tokens=False) for d in range(10)}
assert all(len(v) == 1 for v in DIG.values()); DIG = {d: v[0] for d, v in DIG.items()}
PAT = re.compile(r"Score:\s*(\d+(?:\.\d+)?)")
data = json.load(open(a.inp)); data = data[:a.limit] if a.limit else data
prompts, where = [], []
for i, d in enumerate(data):
    for k, resp in enumerate(d["responses"]):
        j = d["judgments"][k][0]
        m = PAT.search(j)
        if not m: where.append((i, k, None)); continue
        base = G.format_chat(G.construct_chat_eval(G.build_eval_user_prompt(d["prompt"], resp), "eval"))
        prompts.append(base + j[:m.start(1)]); where.append((i, k, m.group(1)))
t0 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=1.0, logprobs=20))
print(f"{len(prompts)} prompts in {time.time()-t0:.0f}s")
import math
res = [[None] * len(d["responses"]) for d in data]; it = iter(outs); n_dec = n_miss = 0
for i, k, sstr in where:
    s = data[i]["scores"][k]
    if sstr is None: res[i][k] = {"s": s, "p": None, "pd": {}, "dec": False}; n_miss += 1; continue
    lp = next(it).outputs[0].logprobs[0]
    pd = {d: math.exp(lp[t].logprob) if t in lp else 0.0 for d, t in DIG.items()}
    dec = "." in sstr; n_dec += dec
    res[i][k] = {"s": s, "p": pd[sstr[0]], "pd": pd, "dec": dec}
json.dump(res, open(a.out, "w"))
print(f"saved {a.out}: items {len(data)} no_match {n_miss} decimal {n_dec}")
