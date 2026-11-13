# Chosen character cap 2500: self-judge temperature 1 versus 0

Two fresh TSR pipelines, one SN5 eight-H200 node each. Both use verified Llama-3.1-8B Base,
training/data seed 42, fresh three-epoch SFT M0, then Mf0, M1, Mf1 and final M2.
Do not reuse any previous SFT, DPO weights, training answers, judgments or preference datasets.
The canceled Meta-Rewarding-length-control SR baseline remains canceled and is not part of these runs.

Only configured treatment difference:

- A: local self-judge temperature **1**.
- B: local self-judge temperature **0**.

Both training candidate generation temperatures are **1**. K=7, top_p=1, maximum output=1024 tokens,
repetition penalty=1, presence/frequency penalties=0. Both final M2 evaluation answer generations
use temperature **0**, seed 0, the same frozen 805 AlpacaEval questions, templates, BOS, tokenizer,
stops, context 4096 and max output 1024. The GPT-4o-2024-08-06 weighted scoring judge retains its
existing settings, including judge temperature 1; CPU scoring concurrency 16 per run.

All four D1/D2 datasets follow this order:

1. Reparse raw self-judgments using `explicit-score-v1`. Remove invalid candidate responses before
   computing scores/extrema; never treat extraction failures as score zero. Valid explicit zero remains valid.
2. Require strict chosen_score > rejected_score. Filter equal-score preference pairs in every stage,
   including training of Mf0 and Mf1.
3. Fix score extrema and model sources. D2 chosen is highest-scoring current/future, with future
   preferred on equal maxima. Rejected is lowest-scoring current/past, with current preferred on
   equal minima. D1 has no future candidate: chosen is current-best.
4. In the fixed score/source pools, take the first candidate combination in original order satisfying
   both distinct stripped text and **len(chosen_text) <= 2500**. Character length is Python Unicode
   character count of the actual response, not UTF-8 bytes or tokens. The bound is inclusive.
5. If no valid combination exists, discard the prompt. Do not fall back to a different source,
   lower-score chosen, higher-score rejected, or trimmed text. No length-gap matching, shortest-answer
   preference, or chosen-longer constraint. Rejected has no character cap.

The 2500-character bound applies only to chosen responses in preference datasets, not SFT human
responses, raw sampled candidates, rejected responses, or evaluation answers. Raw candidates remain
available until their source/score pools have been fixed. No response text is truncated to force compliance.

Common optimization: global batch32; SFT 8x2xaccumulation2, lr2e-6/3epochs; DPO 8x4xaccumulation1,
lr5e-7/beta0.1/1epoch. ZeRO-2 with communication overlap off, original AdamW and DPO FlashAttention2.
vLLM engine seeds are 42 plus shard index. Scheduling/fused calls can change RNG consumption, so these
independently trained runs do not claim bitwise equivalence.

Store actual per-stage chosen maximum length, counts of over-cap candidates/combinations, prompts
without an eligible chosen, score ties, identical text, and invalid score extraction. Validate all
canonical DPO pair texts before training and again at pipeline completion. Retain metrics/configs,
canonical preference datasets, final stage model weights, evaluation outputs and annotations. Save
no optimizer or per-update checkpoints; remove consumed raw training generation/judgment caches.
