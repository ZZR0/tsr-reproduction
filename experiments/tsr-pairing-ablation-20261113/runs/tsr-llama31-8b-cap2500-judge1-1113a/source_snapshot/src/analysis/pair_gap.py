#!/usr/bin/env python3
"""Paper Figure 1 metric: an external judge scores chosen and rejected of a DPO pair set with the official TSR judge
prompt; reports mean(chosen - rejected) on a random sample of pairs. Uses JUDGE_API_KEY / JUDGE_BASE_URL.
  pair_gap.py --pairs SR0=DIR TSR0=DIR SR1=DIR TSR1=DIR [--model gpt-4o-mini] [--n 300] [--budget 1.0]
DIR is a pairs directory (train.json). We used gpt-4o-mini, n = 300, seed 0."""
import argparse, concurrent.futures as cf, importlib.util, json, os, random, re, sys, threading
import openai
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", nargs="+", required=True)
ap.add_argument("--model", default="gpt-4o-mini"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--budget", type=float, default=1.0)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("ij", os.path.join(os.environ["TSR_DIR"], "tsr", "self_generate_vllm_all_iterJudge.py"))
ij = importlib.util.module_from_spec(spec); argv = sys.argv; sys.argv = [argv[0]]; spec.loader.exec_module(ij); sys.argv = argv
G = ij.SelfRewardGenerator
SYS_EVAL = G.construct_chat_eval(None, "x", "eval")[0]["content"]
cl = openai.OpenAI(base_url=os.environ.get("JUDGE_BASE_URL") or None, api_key=os.environ["JUDGE_API_KEY"])
SC = re.compile(r"Score:\s*(\d+(?:\.\d+)?)")
cost = [0.0]; lock = threading.Lock(); stop = threading.Event()
def score(q, r):
    if stop.is_set(): return None
    for _ in range(3):
        try:
            out = cl.chat.completions.create(model=a.model, temperature=0, max_tokens=300,
                messages=[{"role": "system", "content": SYS_EVAL}, {"role": "user", "content": G.build_eval_user_prompt(None, q, r)}])
            with lock:
                cost[0] += out.usage.prompt_tokens * 0.15e-6 + out.usage.completion_tokens * 0.6e-6
                if cost[0] > a.budget: stop.set()
            m = SC.search(out.choices[0].message.content or "")
            s = float(m.group(1)) if m else None
            return s if s is not None and 0 <= s <= 5 else None
        except Exception as ex:
            print("error", type(ex).__name__, str(ex)[:150], file=sys.stderr)
    return None
res = {}
for spec_ in a.pairs:
    name, d = spec_.split("=", 1)
    P = [json.loads(l) for l in open(f"{d}/train.json")]
    idx = random.Random(0).sample(range(len(P)), min(a.n, len(P)))
    def pair(i):
        p = P[i]; q, c, r = p["prompt"], p["chosen"][-1]["content"], p["rejected"][-1]["content"]
        sc = score(q, c); return sc, (sc if c == r else score(q, r))
    with cf.ThreadPoolExecutor(4) as ex: out = list(ex.map(pair, idx))
    ok = [(x, y) for x, y in out if x is not None and y is not None]
    res[name] = {"n_ok": len(ok), "gap_mean": round(sum(x - y for x, y in ok) / len(ok), 3) if ok else None}
    print(name, res[name], "cost so far $%.3f" % cost[0], flush=True)
    if stop.is_set(): print("budget reached, stopping"); break
print(json.dumps(res, indent=1))
