#!/usr/bin/env python3
"""Generate AlpacaEval 2.0 outputs (805 instructions) for one model.

Same prompt as training: the IFT system prompt + the instruction in the upstream chat template, BOS exactly once.
Sampling as in the paper (Appendix C): temperature 1.0, top_p 1.0, max_tokens 1024; stop at <|end_of_text|>
(id 128001) or a leaked role tag.
  gen_alpaca.py --model DIR --out model_outputs.json [--seed 0] [--temperature 1.0] [--limit N]
Writes the standard alpaca_eval format (list of {instruction, output, generator}) plus <out>.diag.json.
"""
import argparse, collections, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--reference_outputs", help="Frozen AlpacaEval reference JSON; preserves archived question order")
args = ap.parse_args()

import yaml
from alpaca_eval import constants as C
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TSR = os.environ["TSR_DIR"]
TEMPLATE = yaml.safe_load(open(f"{TSR}/settings/sft/config/config.yaml"))["chat_template"]
IFT_SYS = json.loads(open(f"{TSR}/data/sft_data/ift_train.jsonl").readline())["messages"][0]["content"]

data = json.load(open(args.reference_outputs)) if args.reference_outputs else list(C.get_alpaca_eval_data())
if args.limit: data = data[:args.limit]
tok = AutoTokenizer.from_pretrained(args.model)
tok.chat_template = TEMPLATE
prompts = []
for d in data:
    msgs = [{"role": "system", "content": IFT_SYS}, {"role": "user", "content": d["instruction"]}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert p.endswith("<|assistant|>\n"), repr(p[-40:])
    prompts.append(p)
ids0 = tok(prompts[0])["input_ids"]
assert ids0.count(tok.bos_token_id) == 1 and ids0[0] == tok.bos_token_id, "BOS must appear exactly once"

llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=4096, seed=args.seed)
llm.get_tokenizer().chat_template = TEMPLATE
sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=1024, n=1, seed=args.seed,
                    stop_token_ids=[128001], stop=["<|user|>", "<|system|>"], ignore_eos=False)
outs = llm.generate(prompts, sp)

name = os.path.basename(os.path.dirname(os.path.abspath(args.out)))
texts, fin = [], collections.Counter()
for o in outs:
    c = o.outputs[0]
    texts.append(c.text.strip())
    fin["truncated" if c.finish_reason == "length" else "role_stop" if c.stop_reason in ("<|user|>", "<|system|>") else "eos"] += 1
n = len(texts)
diag = dict(model=args.model, n=n, seed=args.seed, temperature=args.temperature, mean_chars=sum(map(len, texts)) / n,
            empty_rate=sum(1 for t in texts if not t) / n, **{f"{k}_rate": v / n for k, v in fin.items()})
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump([{"instruction": d["instruction"], "output": t, "generator": name} for d, t in zip(data, texts)],
          open(args.out, "w"), indent=1, ensure_ascii=False)
json.dump(diag, open(args.out.replace(".json", ".diag.json"), "w"), indent=1)
print(json.dumps(diag, indent=1))
