"""Validate the four D2 conditions against the same saved scores, without changing D1."""
import json
from pathlib import Path


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(base, gt, ge_filter, gt_filter, current_path, past_path, future_path):
    current, past, future = [json.load(open(p)) for p in [current_path, past_path, future_path]]
    n = len(current)
    assert len(past) == len(future) == len(base) == len(gt) == n
    keep, changed, ties, better = [], 0, 0, 0
    for i, (c, p, f, ge_row, gt_row) in enumerate(zip(current, past, future, base, gt)):
        assert c['prompt'] == p['prompt'] == f['prompt'] == ge_row['prompt'] == gt_row['prompt']
        assert ge_row['rejected'] == gt_row['rejected']
        sc, sf = max(c['scores']), max(f['scores'])
        # Greater and less comparisons must agree; equality must switch to the current model.
        expected = c['responses'][c['scores'].index(sc)] if sf == sc else ge_row['chosen'][-1]['content']
        assert gt_row['chosen'][-1]['content'] == expected
        assert gt_row['chosen'][:-1] == ge_row['chosen'][:-1]
        ties += sf == sc
        better += sf > sc
        changed += ge_row != gt_row
        rejected_score = min(min(c['scores']), min(p['scores']))
        if max(sc, sf) > rejected_score:
            keep.append(i)
    assert ge_filter == [base[i] for i in keep]
    assert gt_filter == [gt[i] for i in keep]
    return dict(status='PASS', original_pairs=n, filtered_pairs=len(keep),
                removed_tie_pairs=n-len(keep), future_current_best_ties=ties,
                future_strictly_better=better, chosen_text_changed_by_strict=changed,
                rejected_unchanged=True, filter_keeps_identical_prompt_set=True,
                d1_and_future_model_reused=True)
