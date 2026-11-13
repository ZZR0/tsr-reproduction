"""Priority order must beat a tempting shorter pair across sources or scores."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('length_pairing', ROOT / 'src/length_pairing.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(prompt, texts, scores):
    return dict(prompt=prompt, responses=texts, scores=scores)


class SourcePriorityLength(unittest.TestCase):
    def test_source_and_score_priorities_override_closer_lengths(self):
        c = record('p', ['c'*8, 'r'*10, 's'*105], [5, 1, 1])
        p = record('p', ['p'*100, 'q'*120], [1, 2])
        f = record('p', ['f'*100, 'g'*200, 't'*105], [5, 5, 4])
        out = module.select_min_length_pair(c, p, f, source_priority=True)
        self.assertEqual((out['chosen']['source'], out['rejected']['source']), ('Mf', 'Mi'))
        self.assertEqual((out['chosen']['text'], out['rejected']['text']), ('f'*100, 's'*105))
        self.assertEqual(out['length_gap'], 5)
        self.assertEqual((out['n_high_candidates'], out['n_low_candidates']), (2, 2))
        # Historical no-source-priority experiment remains a distinct implementation.
        old = module.select_min_length_pair(c, p, f)
        self.assertEqual(old['length_gap'], 0)
        self.assertEqual(old['rejected']['source'], 'M0')

    def test_strictly_better_scores_override_model_preferences(self):
        c = record('p', ['c'*8, 'd'*110, 'r'*10], [5, 5, 1])
        p = record('p', ['p'*100, 'q'*120], [0, 2])
        f = record('p', ['f'*100], [4])
        out = module.select_min_length_pair(c, p, f, source_priority=True)
        self.assertEqual((out['chosen']['source'], out['rejected']['source']), ('Mi', 'M0'))
        self.assertEqual(out['chosen']['text'], 'd'*110)
        self.assertEqual((out['chosen_score'], out['rejected_score'], out['length_gap']), (5, 0, 10))

    def test_equal_preference_scores_removed_with_and_without_future(self):
        c = record('p', ['a', 'bbb'], [5, 5])
        for f in [None, record('p', ['xx'], [5])]:
            self.assertFalse(module.select_min_length_pair(c, c, f, source_priority=True)['valid'])

    def test_four_real_stage_calls_apply_rules_and_record_pool_counts(self):
        with tempfile.TemporaryDirectory(prefix='tsr-priority-length-') as temp:
            root = Path(temp)
            current = [record('p', ['c'*8, 'r'*10, 's'*105], [5,1,1]), record('tie',['one','two'],[5,5])]
            past = [record('p', ['p'*100], [1]), record('tie',['past'],[5])]
            future = [record('p', ['f'*100, 'g'*200, 't'*105], [5,5,4]), record('tie',['future'],[5])]
            for folder, files in {
                'iter0': {'0_chosen_judgement.json':current, '0_future_judgement.json':future},
                'iter1': {'1_chosen_judgement.json':current, '1_rejected_judgement.json':past, '1_future_judgement.json':future},
            }.items():
                d=root/'data'/folder;d.mkdir(parents=True)
                for name, rows in files.items():(d/name).write_text(json.dumps(rows))
            script=r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
TSR_PAIR_SELECTION=future_tie_length
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
stage_iter0
stage_iter1 future_tie_length
'''
            env=dict(os.environ,TEST_REPO=str(ROOT),TEST_ROOT=str(root),TEST_PY=sys.executable)
            result=subprocess.run(['bash','-c',script],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for phase, dtype in [('iter0','D1'),('iter0','D2'),('iter1','D1'),('iter1_future_tie_length','D2')]:
                d=root/'data'/phase/dtype;s=json.loads((d/'stats.json').read_text())
                self.assertEqual((s['n_pairs'],s['filtered_out'],s['retained_tie_pairs']),(1,1,0))
                self.assertEqual(s['pair_selection'],'future_tie_length')
                self.assertTrue(s['model_source_priority'])
                self.assertEqual(s['chosen_candidate_pool']['n'],1)
                self.assertEqual(s['rejected_candidate_pool']['n'],1)
                pair=json.loads((d/'train.json').read_text())
                if dtype=='D2':
                    self.assertEqual(pair['chosen'][-1]['content'],'f'*100)
                    self.assertEqual(pair['rejected'][-1]['content'],'s'*105)


if __name__ == '__main__':
    unittest.main()
