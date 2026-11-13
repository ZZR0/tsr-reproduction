"""Distinct text is a constraint within fixed score/source pools, never a fallback."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('length_pairing',ROOT/'src/length_pairing.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def row(p,texts,scores):return dict(prompt=p,responses=texts,scores=scores)

class DistinctPriorityLength(unittest.TestCase):
    def test_skip_identical_closest_pair_and_keep_next_valid_pair(self):
        c=row('p',['current','same'],[5,1]);p=row('p',['past'],[1]);f=row('p',['same','longer'],[5,5])
        out=module.select_min_length_pair(c,p,f,source_priority=True,require_distinct_text=True)
        self.assertTrue(out['valid']);self.assertEqual(out['chosen']['text'],'longer')
        self.assertEqual(out['rejected']['source'],'Mi');self.assertEqual(out['rejected']['text'],'same')
        self.assertEqual(out['identical_text_combinations_excluded'],1)

    def test_no_source_or_score_fallback_when_fixed_pool_has_only_same_text(self):
        c=row('p',['current','same'],[5,1]);p=row('p',['past'],[1]);f=row('p',['same','lower-score-alternative'],[5,4])
        out=module.select_min_length_pair(c,p,f,source_priority=True,require_distinct_text=True)
        self.assertFalse(out['valid']);self.assertEqual(out['reason'],'no_distinct_text_pair')

    def test_ignore_leading_and_trailing_whitespace_for_identity(self):
        c=row('p',['current',' same '],[5,1]);p=row('p',['past'],[1]);f=row('p',['same\n'],[5])
        self.assertFalse(module.select_min_length_pair(c,p,f,source_priority=True,require_distinct_text=True)['valid'])

    def test_actual_pipeline_applies_distinct_rule_at_all_four_stages(self):
        with tempfile.TemporaryDirectory(prefix='tsr-distinct-') as tmp:
            root=Path(tmp);prompts=['no-fallback','valid-alternative','all-duplicates','score-tie']
            current=[row(prompts[0],['current','same'],[5,1]),row(prompts[1],['current','same'],[5,1]),row(prompts[2],['dup','dup'],[5,1]),row(prompts[3],['c','r'],[5,5])]
            past=[row(p,['past'],[5 if p=='score-tie' else 1]) for p in prompts]
            future=[row(prompts[0],['same'],[5]),row(prompts[1],['same','futureLong'],[5,5]),row(prompts[2],['dup'],[5]),row(prompts[3],['f'],[5])]
            for phase,files in {'iter0':{'0_chosen_judgement.json':current,'0_future_judgement.json':future},'iter1':{'1_chosen_judgement.json':current,'1_rejected_judgement.json':past,'1_future_judgement.json':future}}.items():
                d=root/'data'/phase;d.mkdir(parents=True)
                for name,data in files.items():(d/name).write_text(json.dumps(data))
            script=r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
TSR_PAIR_SELECTION=future_tie_length_distinct
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
stage_iter0
stage_iter1 future_tie_length_distinct
'''
            proc=subprocess.run(['bash','-c',script],env=dict(os.environ,TEST_REPO=str(ROOT),TEST_ROOT=str(root),TEST_PY=sys.executable),capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            for phase,dtype in [('iter0','D1'),('iter0','D2'),('iter1','D1'),('iter1_future_tie_length_distinct','D2')]:
                d=root/'data'/phase/dtype;s=json.loads((d/'stats.json').read_text())
                self.assertEqual(s['pair_selection'],'future_tie_length_distinct');self.assertTrue(s['require_distinct_text'])
                self.assertEqual(s['retained_tie_pairs'],0);self.assertEqual(s.get('identical_pair',0),0)
                self.assertEqual(s['filtered_equal_score_pairs'],1)
                rows=[json.loads(x) for x in (d/'train.json').read_text().splitlines()]
                self.assertTrue(all(x['chosen'][-1]['content'].strip()!=x['rejected'][-1]['content'].strip() for x in rows))
                if dtype=='D2':
                    self.assertEqual(s['n_pairs'],1);self.assertEqual(s['filtered_no_distinct_text_pairs'],2)
                    self.assertEqual(rows[0]['chosen'][-1]['content'],'futureLong')
                else:
                    self.assertEqual(s['n_pairs'],2);self.assertEqual(s['filtered_no_distinct_text_pairs'],1)

if __name__=='__main__':unittest.main()
