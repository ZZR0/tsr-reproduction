# TSR independent three-seed run, 2026-11-12

User-approved rule order:

1. Remove response candidates whose raw local self-judge output has no valid score.
2. All four DPO datasets, including future-model datasets, require chosen score > rejected score.
3. Chosen comes from the highest-score current/future pool; equal maxima prefer future. Rejected comes from the lowest-score current/past pool; equal minima prefer current.
4. Within these fixed score/source pools, take the first pair whose stripped texts differ. Do not rank by length, relax sources, or relax score extrema. If no pair exists, discard the prompt.

Future-model D1 has no available future source: chosen is current-best, rejected is current/past worst. The future tie rule applies to D2. SFT uses ordinary supervised examples rather than preference pairs.

Score validation requires one local judgment per response, containing exactly one explicit `Score:` label. Scores must be finite decimals in [0,5]; a genuine explicit zero is valid. Missing, malformed, repeated, and out-of-range scores are invalid. Invalid candidates are removed before determining extrema, without deleting other valid candidates for that prompt. Numeric zeroes from the upstream fallback are never trusted: raw judgments are reparsed. Entire prompts are dropped when no valid chosen/rejected pair remains. Missing or misaligned raw judgment arrays fail the stage rather than silently passing.

Three independent seeds: **42,43,44**. Each run starts from verified Llama-3.1-8B Base, freshly trains its own 3-epoch SFT M0, and trains Mf0, M1, Mf1, M2. No old SFT, candidate generation, self-judgment, DPO dataset or DPO model is reused. SFT/DPO use both seed and data_seed. vLLM engine seed is the run seed plus the data-parallel shard index; Python/NumPy/Torch use the run seed. The same archived SFT/qa1/qa2 inputs are used across runs.

Each job uses one complete 8-H200 SN5 node. A required pod anti-affinity over the three-run group keeps them on different hosts. Global batch remains 32: SFT 8 x 2 x accumulation 2; DPO 8 x 4 x accumulation 1. ZeRO-2, communication overlap off, DPO FlashAttention2, original AdamW, DPO learning rate 5e-7/beta 0.1/one epoch. SFT learning rate 2e-6, three epochs, context 2048. SFT retains its original gradient checkpointing. Each stage saves its final model only, without optimizer state.

Training candidate generation and local self-judge both use temperature **1**. K=7, top_p=1, maximum output=1024 tokens, repetition penalty=1, presence/frequency penalties=0. Final M2 evaluation uses temperature **0**, seed 0, the same 805 frozen AlpacaEval questions, prompts, BOS, tokenizer, stop tokens, context 4096 and maximum output 1024. GPT-4o judge and length-control settings are unchanged. Score only each run's final M2, with 16 API requests in flight per CPU worker. Eval seed is held fixed to compare independent training seeds.

Keep metrics, configs, stage summaries, canonical DPO pairs, final models, evaluation answers/annotations and API request caches. Delete consumed raw training candidate/judgment caches after stages finish. Record counts/reasons for invalid scores, score ties, identical-text exclusion, and prompts with no valid candidates. Never retain GPU tensors, token batches, per-update model/optimizer checkpoints or duplicate loader data after consumption.

Each frozen run-plan, source hash manifest and job manifest is authoritative for that run. Only EVALUATION_COMPLETED.json confirms all 805 final-model questions have been scored. A completed training marker alone is not a finished evaluation.
