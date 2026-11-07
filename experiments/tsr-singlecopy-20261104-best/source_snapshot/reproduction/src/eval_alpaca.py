#!/usr/bin/env python3
"""AlpacaEval 2.0 annotation with an OpenAI-compatible judge (default gpt-4o-2024-08-06).

Annotator = the official weighted_alpaca_eval_gpt4_turbo setup (alpaca_eval_clf template, logprob weighting)
with the judge model swapped; reference = the official gpt4_1106_preview outputs; LC = length_controlled_v1.
Reads JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL from the environment (config.env).
  eval_alpaca.py --model_outputs model_outputs.json --name NAME --out_dir DIR
Writes DIR/leaderboard.csv and DIR/annotations.json. With fewer than 805 outputs (smoke test) only the raw
win rate is computed, because the length-controlled GLM needs the full set.
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--model_outputs", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()

os.environ["OPENAI_API_KEY"] = os.environ["JUDGE_API_KEY"]
os.environ["OPENAI_API_BASE"] = os.environ.get("JUDGE_BASE_URL") or "https://api.openai.com/v1"
judge = os.environ.get("JUDGE_MODEL") or "gpt-4o-2024-08-06"

import pandas as pd
from alpaca_eval import constants as C, evaluate

outputs = json.load(open(args.model_outputs))
ref = list(C.get_alpaca_eval_data())[:len(outputs)]
assert [o["instruction"] for o in outputs] == [r["instruction"] for r in ref], "outputs must follow AlpacaEval order"

# Annotator config next to the results, so its annotation cache never mixes judges or models.
cfg_dir = os.path.join(args.out_dir, "annotator")
os.makedirs(cfg_dir, exist_ok=True)
template = os.path.join(C.EVALUATORS_CONFIG_DIR, "alpaca_eval_clf_gpt4_turbo", "alpaca_eval_clf.txt")
with open(os.path.join(cfg_dir, "configs.yaml"), "w") as f:
    f.write(f"""weighted_alpaca_eval_judge:
  prompt_template: "{template}"
  fn_completions: "openai_completions"
  completions_kwargs:
    model_name: "{judge}"
    max_tokens: 1
    temperature: 1
    logprobs: true
    top_logprobs: 5
    num_procs: 3
  fn_completion_parser: "logprob_parser"
  completion_parser_kwargs:
    numerator_token: "m"
    denominator_tokens: ["m", "M"]
    is_binarize: false
  completion_key: "completions_all"
  batch_size: 1
""")

full = len(outputs) == 805
evaluate(model_outputs=pd.DataFrame([{**o, "generator": args.name} for o in outputs]),
         reference_outputs=pd.DataFrame([{"instruction": r["instruction"], "output": r["output"], "generator": "gpt4_turbo"} for r in ref]),
         annotators_config=cfg_dir, name=args.name, output_path=args.out_dir, is_return_instead_of_print=True,
         fn_metric="get_length_controlled_winrate" if full else "get_winrate",
         sort_by="length_controlled_winrate" if full else "win_rate")
lb = pd.read_csv(os.path.join(args.out_dir, "leaderboard.csv"), index_col=0)
print(lb.loc[[args.name]].to_string())
