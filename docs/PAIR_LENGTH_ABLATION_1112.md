# TSR: chosen-longer versus no length selection, 2026-11-12

Two new independent full pipelines, each on one SN5 8-H200 machine, with the same seed 42.
Both start from the verified Llama-3.1-8B Base and freshly train their own three-epoch SFT M0.
The canceled three-seed series remains canceled. No SFT weights, DPO models, candidate answers,
judgments, or preference datasets from older runs are reused.

Common rules in all four DPO datasets (Mf0/M1/Mf1/M2):

1. Remove candidates whose raw self-judge score extraction is invalid, using `explicit-score-v1`.
2. Require chosen_score > rejected_score. Never keep equal-score preference pairs.
3. Resolve score extrema and source priority: highest-scoring chosen from current/future,
   equal maxima prefer future; lowest-scoring rejected from current/past, equal minima prefer current.
   D1 has no future model, so its chosen pool is current-best.
4. Require distinct chosen/rejected text after stripping leading/trailing whitespace.
5. Do not relax scores or sources to satisfy text/length constraints. Drop the prompt when no legal pair exists.

Treatment A (`future_tie_longer_distinct`): within the fixed source/score pools, enumerate joint
chosen/rejected combinations, retain only pairs with `len(chosen) > len(rejected)`, then minimize
the positive difference `len(chosen) - len(rejected)`. Length is Python Unicode character count of
the actual response text, matching prior length experiments. It is not token count. Equal minimum
gaps are resolved by SHA256(prompt + chosen text + rejected text), deterministically.

Control B (`future_tie_distinct`): within the fixed source/score pools, take the first distinct-text
pair in original candidate order. No length preference, constraint, or ranking.

SFT/DPO use training seed and data_seed 42. vLLM engine seeds are 42 + shard index. Same archived
SFT and qa1/qa2 data, global batch 32, SFT lr 2e-6/three epochs, DPO lr 5e-7/beta 0.1/one epoch.
ZeRO-2, communication overlap disabled, DPO FlashAttention2 and original AdamW. Save final stage
weights only, with no optimizer-state checkpoints. Training candidate generation and self-judging
both use temperature 1, top_p 1, max output 1024; K=7 candidates per prompt; repetition penalty 1,
presence/frequency penalties 0.

Generate only final M2 evaluation answers: same 805 frozen AlpacaEval questions, temperature 0,
seed 0, top_p 1, max output 1024/context 4096, same prompts/tokenizer/BOS/stops. Score using the
existing GPT-4o-2024-08-06 weighted judge and LC settings, 16 concurrent requests per CPU worker.
Report LC Win, Win, output length, valid-pair counts and filtering reasons. Pair-count differences
between A and B are an expected result of A's stricter selection, not compensated by changing epochs.

Keep metrics, configurations, canonical DPO pairs, final models, evaluation outputs/annotations
and API request caches. Delete consumed raw training answer/judgment caches after training stages.
Retain no GPU tensors, full token batches, intermediate model checkpoints, or optimizer states.
