# Configuration

Everything below is what produced the numbers in the README. `src/make_cfg.py` writes the training configs from the
upstream ones; only paths and the per-GPU batch split differ.

## SFT (M0)

- Base model: Meta-Llama-3.1-8B.
- Data: upstream `data/sft_data`, IFT 5,000 + EFT 1,871 = 6,871 examples, packed into 2,048-token blocks (1,929 blocks).
  Test split: `ultrachat_test.json` (1,000 examples).
- Hyperparameters (upstream `settings/sft/config/config.yaml`): 3 epochs (180 steps), lr 2e-6, cosine schedule,
  warmup ratio 0.1, max_seq_length 2048, gradient checkpointing, seed 42, upstream chat template
  (`<|system|>` / `<|user|>` / `<|assistant|>`, each turn ending with `<|end_of_text|>`).
- Parallelism: DeepSpeed ZeRO-3 (upstream `settings/tsr/config/deepspeed_zero3.yaml`), bf16 mixed precision with fp32
  master weights; 2 × A800, per-device batch 2 × gradient accumulation 8 = global batch 32.
- Upstream `scripts/sft.sh` (DDP, pure bf16) does not reproduce the paper: about 95% of the parameter updates are lost to
  bf16 rounding and LC is 3.34. The paper text specifies ZeRO-3, which is what we use.

## Generation (training data)

- Upstream `tsr/self_generate_vllm_all_generate.py`: vLLM, temperature 1.0, top_p 1.0, max_tokens 1024, K = 7 responses
  per prompt, upstream system prompt.
- Prompts: TSR iter0 uses `qa1.jsonl`, iter1 `qa2.jsonl` (5,000 each). SR iter1 uses `qa2.jsonl`.

## Judging

- Upstream `tsr/self_generate_vllm_all_iterJudge.py`: the additive 5-point prompt and the evaluator system prompt,
  sampling as above, one judgement per response (`num_judgments=1`), the first `Score: N` match is the score
  (unparsable judgements get 0).
- TSR iter0 is judged by M0. TSR iter1 is judged by M1, including M0's and Mf1's responses.
  SR iter1 and w/o Future iter1 are judged by their starting model Mf0.

## Pair construction

Upstream rules (`reformat2DPO_past.py`, `reformat2DPO_future.py`); our re-implementation matches them pair for pair.

- chosen = the current model's highest-scored response (first index on ties);
- rejected = the lowest-scored response of the current model and of M0, the current model's on a tie;
- future upgrade: chosen becomes the future model's best response if its score is `>=` the current best.

No further filtering: pairs whose chosen and rejected scores are equal are kept (77 in iter1 D2), as in upstream.

| Pair set | Trains | chosen from future | rejected from M0 | mean length chosen / rejected (chars) |
|---|---|---|---|---|
| iter0 D1 | Mf0 (= SR iter0 = w/o Future iter0) | | | 1405 / 1256 |
| iter0 D2 | M1 | 4,689 / 5,000 (3,751 by tie) | | 1520 / 1256 |
| iter1 D1 | Mf1 | | 2,562 / 5,000 | 1776 / 1372 |
| iter1 D2 | M2 | 4,989 / 5,000 (4,968 by tie) | 2,562 / 5,000 | 1993 / 1372 |
| w/o Future iter1 | W1 | | 2,825 / 5,000 | 1564 / 1280 |
| SR iter1 | SR1 | | | 1564 / 1565 |

## DPO

### Local second-round default: future-only chosen

As of 2026-11-11, the default `run.sh iter1` uses `future_only`. For each prompt, choose
Mf1's highest-scored answer unconditionally, and reject the lowest-scored answer across
M1 and M0 (ties prefer M1). Scores remain M1's plain self-judgements. Keep the pair only
when `score(chosen) > score(rejected)`; discard equal or reversed scores. D1/Mf1 remain
unchanged. D2 is rebuilt on each invocation in `data/iter1_future_only/D2`, and M2 is
trained from M1 into `models/M2_future_only`. The input is still isolated by `make_cfg.py`
to prevent duplicate loading. `iter1 main` explicitly restores the historical selection
rule described above. The completed result is archived under `experiments/tsr-future-only-20261112`: M2 LC19.3324 /soft Win29.6734, versus Mf1 LC23.4019 /soft Win26.6419.

- Upstream `alignment-handbook/scripts/run_dpo.py` with patch 0001 (prepend BOS).
- Hyperparameters (upstream `settings/sr/config/dpo/config_full_iter0.yaml`, identical to the TSR configs except paths):
  lr 5e-7, β 0.1, 1 epoch (156 steps), cosine schedule, warmup ratio 0.1, max_length 2048, max_prompt_length 512,
  adamw_torch, gradient checkpointing, seed 42.
- Parallelism: upstream ZeRO-3 config, fp32 load + bf16 mixed precision, 16-bit weights saved;
  2 × A800, per-device batch 1 × gradient accumulation 16 = global batch 32.
- Initialisation and reference model: the round's starting model. Mf_i and M_{i+1} start from M_i (Mf1 from M1);
  W1 starts from Mf0 (w/o Future iter0); SR1 starts from Mf0 (SR iter0).
- The final weights are evaluated; no checkpoint selection.
- trl 0.12.2 always truncates prompts longer than 512 tokens to their last 512; trl 0.9.6 truncates only when prompt +
  response exceed 2,048 tokens. This affects about 5–6% of the pairs; retraining M2 with the older stack gave length 2697.

| Model | mean train loss |
|---|---|
| Mf0 (= SR iter0) | 0.571 |
| M1 | 0.245 |
| M1, strict upgrade rule | 0.512 |
| Mf1 | 0.417 |
| M2 | 0.167 |
| SR1 | 0.685 |
| W1 | 0.408 |

## AlpacaEval 2.0

- Generation (`src/gen_alpaca.py`): vLLM, temperature 1.0, top_p 1.0, max_tokens 1024, the IFT training system prompt,
  stop at `<|end_of_text|>` (id 128001) or the strings `<|user|>`, `<|system|>`; seed 0 (seeds 1 and 2 for the
  other two SFT runs).
- Evaluation (`src/eval_alpaca.py`): alpaca_eval 0.6.6, 805 instructions, reference `gpt4_1106_preview`
  (`alpaca_eval_gpt4_baseline`), annotator = `alpaca_eval_clf_gpt4_turbo` template with logprob weighting
  (max_tokens 1, temperature 1, top_logprobs 5), judge `gpt-4o-2024-08-06` (via OpenRouter), LC = `length_controlled_v1`.
- Judge snapshot matters: on the same SFT outputs `gpt-4o-2024-11-20` gives about 0.8 LC less than `2024-08-06`.
- alpaca_eval's LC fit uses cross-validation folds that depend on row order; shuffling the same annotations moves LC by
  up to about 0.8. Leaderboard values use the original order; the bootstrap CIs include this variation.

## Software and hardware

torch 2.5.1 (cu124), transformers 4.46.3, trl 0.12.2, accelerate 1.1.1, deepspeed 0.15.4, vllm 0.6.4.post1,
flash-attn 2.7.2.post1, datasets 2.21.0, alpaca_eval 0.6.6, Python 3.11.17; 2 × NVIDIA A800-SXM4-80GB.
Old-stack check: trl 0.9.6, transformers 4.43.4 (4.39.3 cannot load Llama-3.1), deepspeed 0.14.4, accelerate 0.33.0.


## Completed SN5 run and launch helpers

The full future-only result is in [the complete archive](../experiments/tsr-future-only-20261112/README.md).
SFT completed on2H200 (micro2, accumulation8,180updates); native final M0 was verified and reused.
The following four DPO stages used8H200 (micro1, accumulation4), keeping global batch32.
Mf0/M1/Mf1 each trained156updates; M2 trained148updates on4,732positive-gap pairs.
The GPU Job completed and released its allocation after training and response generation.

`python scripts/prepare_sn5.py --run-name NAME --gpus 8 --node any-sn5` freezes a single-node
SN5/H200 job specification but does not submit it. An exact `--node` is also supported.
`--sft-source-run PATH` reuses only a fully completed native M0 after verifying code, data,
base manifests, state and weight hashes. This is a stage-boundary reuse, not optimizer resume.
Do not delete the source M0 while another run references it.

`sn5_driver.py` validates the visible GPU count and global batch, runs an isolated2-update
SFT/DPO smoke plus K7generation/self-judging, and then executes the original pipeline stages.
On Lustre, `rank_training.py` gives ranks separate dataset caches and writes TensorBoard
locally before copying final event files to shared storage. No optimizer checkpoints are introduced.
Candidate generation/self-judging use8independent shards; changing the original2-way sharding
can change sampled candidates even when sampling parameters and seeds are unchanged.

`python scripts/score_when_ready.py --run RUN --credentials PRIVATE_JSON` runs without GPUs.
It reads JUDGE_API_KEY, JUDGE_BASE_URL and JUDGE_MODEL from a private JSON file (mode600;
parent directory mode700), waits for the7planned complete805-question outputs, and uses
the frozen scoring code and reference. Use `gpt-4o-2024-08-06` and your own compatible
endpoint. Never commit credential values or private service endpoints. API429/5xx/connection
failures get at most3attempts per model, preserving completed annotation caches.
After retry exhaustion, fix the endpoint and rerun the same command to resume.
Completed models are validated and reused. The scorer reports in `scoring/status.json`,
with `EVALUATION_COMPLETED.json` as the final seven-evaluation completion marker.
The frozen GPU driver's historical awaiting-credentials text does not describe CPU scoring.

`python scripts/early_eval.py --run RUN --models M1 Mf1` can generate those completed
models' planned seed0evaluations early on one temporarily allocated GPU. It publishes
complete outputs atomically, preserves the original generator settings, and avoids overwriting
outputs of the final driver. Supply USER and LOGNAME in nonroot containers whose numeric UID
is absent from /etc/passwd; use a writable HOME/cache directory. The CPU scorer consumes
these outputs and the final driver skips them.

Retention keeps stage-final models, canonical training pairs, configs and compact metrics;
consumed raw candidates/judgements and duplicate loader snapshots are removed. Public
archives contain final evaluations and statistics, not weights or raw training caches.
