# Joint closest-length preference pairs (2026-11-12)

The min_length_gap variant keeps qa1 for round0 and qa2 for round1. It changes selection in all four DPO datasets.
First find the extreme scores within the eligible TSR sources, then choose the pair minimizing
`abs(len(chosen_text) - len(rejected_text))` jointly. Length means Unicode characters, matching reported answer length.

| Dataset | Highest-score chosen pool | Lowest-score rejected pool |
|---|---|---|
| D1 -> Mf0/Mf1 | all current responses at the current maximum | all current/past responses at the joint minimum |
| D2 -> M1/M2 | all current/future responses at their joint maximum | all current/past responses at the joint minimum |

D2 reselects rejected within the same lowest-score pool jointly with its chosen; the selected negative text
can differ from D1, since the chosen candidate pool changes.

Lower-scored chosen or higher-scored rejected responses are never admitted merely for a closer length.
Chosen score must strictly exceed rejected score. Equal-score positive/negative pairs are discarded.
All tied-source candidates enter the pools; no current, past or future model is preferred by source.
Residual equal length gaps use a deterministic hash of prompt/answer texts, without model labels.
Identical text may have multiple source labels; attribution then has no effect on the actual training text.

This is a joint optimization: choosing shortest positive and longest negative independently is not equivalent.
For positive lengths100/250 and negative lengths120/240, the selected pair is250/240 (gap10).

Use a fresh run because every DPO stage changes. Reuse only verified completed SFT M0.
Example launch preparation (does not submit):

```bash
python scripts/prepare_sn5.py --run-name tsr-llama31-8b-min-length-gap-1112f \
  --gpus 8 --node any-sn5 --variant min_length_gap --train-only --skip-smoke \
  --sft-source-run /path/to/completed-sft-run
```

Training-only runs skip standalone smoke tests, evaluation answer generation and API scoring as requested.
Local response generation and local model judging remain necessary to produce training pairs.
Actual training still checks its devices/configuration and requires valid nonempty preference data.
Pair statistics retain score gaps, source counts, mean chosen/rejected lengths, and absolute length gap mean/median/max.
Only final stage weights, canonical pairs, metrics/configs and hashes persist; no optimizer snapshots or raw token dumps.
No tests or judge API calls were requested for this variant.
