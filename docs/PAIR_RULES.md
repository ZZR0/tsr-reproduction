# Current TSR pair rules: future chosen ties, filtered preference pairs

Latest requested variant: `future_tie_filtered`. Historical and canceled runs keep their frozen settings.

- Mf0/Mf1 (D1): chosen = current model's highest-scored response. No future candidate exists in D1. Rejected = lowest-scored response across current/past; equal source scores prefer current.
- M1/M2 (D2): chosen = highest-scored response across current/future; equal source scores prefer future. Rejected = lowest-scored response across current/past; equal source scores prefer current.
- All four DPO stages require `chosen_score > rejected_score`. Equal chosen/rejected scores are filtered, including during future-model training. No length criterion is used. Within one selected source, equal-score candidates retain the first-index rule.
- QA usage stays unchanged: qa1 for round 0, qa2 for round 1.

Source tie-breaking and final pair filtering are separate: a current/future chosen tie selects future first; the resulting chosen is then compared against rejected.

| Current best | Future best | Rejected | D2 result |
|---:|---:|---:|---|
| 5 | 5 | 1 | Keep; chosen from future |
| 4 | 5 | 1 | Keep; chosen from future |
| 5 | 4 | 1 | Keep; chosen from current |
| 5 | 5 | 5 | Discard: chosen/rejected scores equal |

Launch a fresh frozen run with `scripts/prepare_sn5.py --variant future_tie_filtered --gpus 8 --node any-sn5 --train-only --skip-smoke --sft-source-run <verified-SFT-run>`.
The launcher sets `TSR_PAIR_SELECTION=future_tie_filtered` for both rounds. Only verified completed SFT M0 is reused; Mf0/M1/Mf1/M2 are trained anew.
Global batch remains 32: DPO uses 8 GPUs x micro-batch 1 x gradient accumulation 4, one epoch, learning rate 5e-7, beta 0.1.

Before each DPO stage, actual pair files are checked for positive score gaps, zero retained equal-score pairs, and the requested model-source priorities. Source counts, filtering counts and training metrics are retained; consumed candidate/judgement caches are removed. No standalone tests, AlpacaEval generation or paid API scoring are launched for this run.

The preceding `future_tie_unfiltered` experiment (`tsr-llama31-8b-future-tie-all-1112g`) was canceled at user request and must not be resumed or relabeled as this variant.
Historical `current_tie`, `future_only` and `min_length_gap` archives remain unchanged.
