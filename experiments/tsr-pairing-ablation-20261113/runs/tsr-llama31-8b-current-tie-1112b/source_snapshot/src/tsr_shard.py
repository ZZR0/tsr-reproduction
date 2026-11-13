#!/usr/bin/env python3
"""Data-parallel wrapper around the OFFICIAL TSR generate/judge code (TSR/tsr/*.py, unmodified).
Runs one shard of the prompts on one GPU; prompts, templates, sampling params, score parsing are the official ones.
  gen:   tsr_shard.py gen   --model M --input qa1.jsonl            --out X.shard0.json --shard 0 --nshards 2
  judge: tsr_shard.py judge --model M --input X_response.json      --out Y.shard0.json --shard 0 --nshards 2
  merge: tsr_shard.py merge --out X.json X.shard0.json X.shard1.json
Optional --limit N (smoke tests). Upstream code is loaded from $TSR_DIR/tsr."""
import sys, os, json, argparse, importlib.util
TSR = os.path.join(os.environ["TSR_DIR"], "tsr")

def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{TSR}/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["gen", "judge", "merge"])
ap.add_argument("--model"); ap.add_argument("--input"); ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--vllm_seed", type=int, default=None, help="seed for vLLM sampling (default: vLLM default, as in all main runs; 1234 = resampling replicate)")
ap.add_argument("files", nargs="*")
a = ap.parse_intermixed_args()

if a.mode == "merge":
    out = []
    for f in a.files: out += json.load(open(f))
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"merged {len(out)} -> {a.out}"); sys.exit(0)

mod = load("self_generate_vllm_all_generate" if a.mode == "gen" else "self_generate_vllm_all_iterJudge")
mod.seed_everything(a.seed)
data = mod.read_jsonl_file(a.input) if a.mode == "gen" else mod.read_json_file(a.input)
if a.limit: data = data[:a.limit]
n = len(data); lo, hi = a.shard * n // a.nshards, (a.shard + 1) * n // a.nshards
data = data[lo:hi]
print(f"[{a.mode}] shard {a.shard}/{a.nshards}: items {lo}:{hi} of {n}")
if a.vllm_seed is not None:
    import functools; mod.LLM = functools.partial(mod.LLM, seed=a.vllm_seed)
sr = mod.SelfRewardGenerator(a.model, 1)
prompts = [d["prompt"] for d in data]
if a.mode == "gen":
    responses = sr.generate_all_responses(prompts, num_responses=7, type="chat")
    results = sr.process_all(prompts, responses)
else:
    responses = [d["responses"] for d in data]
    judgments = sr.generate_all_judgments(prompts, responses, num_judgments=1, type="eval")
    results = sr.process_all(prompts, responses, judgments)
sr.save_results(results, a.out)
print(f"saved {len(results)} -> {a.out}")
