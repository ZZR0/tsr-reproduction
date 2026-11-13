"""No length ranking is allowed after resolving scores and model source ties."""
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

class FirstDistinct(unittest.TestCase):
    def test_first_valid_combination_wins_even_when_another_has_closer_length(self):
        c=row('p',['current','same'],[5,1]);past=row('p',['past'],[1]);f=row('p',['same','a much longer answer','tiny'],[5,5,5])
        out=module.select_first_distinct_pair(c,past,f)
        self.assertTrue(out['valid']);self.assertEqual(out['chosen']['text'],'a much longer answer')
        self.assertEqual(out['rejected']['text'],'same');self.assertEqual(out['chosen']['source'],'Mf')
        matched=module.select_min_length_pair(c,past,f,source_priority=True,require_distinct_text=True)
        self.assertEqual(matched['chosen']['text'],'tiny')

    def test_no_source_or_score_fallback_and_no_equal_score_pairs(self):
        c=row('p',['current','same'],[5,1]);past=row('p',['past'],[1]);f=row('p',['same','lower'],[5,4])
        self.assertFalse(module.select_first_distinct_pair(c,past,f)['valid'])
        equal=row('p',['a','b'],[5,5])
        self.assertFalse(module.select_first_distinct_pair(equal,equal,equal)['valid'])

    def test_all_four_stages_filter_scores_and_text_without_length_matching(self):
        with tempfile.TemporaryDirectory(prefix='tsr-first-distinct-') as temp:
            root=Path(temp)
            current=[row('ok',['current','same'],[5,1]),row('tie',['a','b'],[5,5]),row('duplicate',['same','same'],[5,1])]
            past=[row('ok',['past'],[1]),row('tie',['past'],[5]),row('duplicate',['other'],[1])]
            future=[row('ok',['same','a much longer answer','tiny'],[5,5,5]),row('tie',['future'],[5]),row('duplicate',['same'],[5])]
            for phase,files in {'iter0':{'0_chosen_judgement.json':current,'0_future_judgement.json':future},'iter1':{'1_chosen_judgement.json':current,'1_rejected_judgement.json':past,'1_future_judgement.json':future}}.items():
                d=root/'data'/phase;d.mkdir(parents=True)
                for name,data in files.items():(d/name).write_text(json.dumps(data))
            script=r'''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
TSR_PAIR_SELECTION=future_tie_distinct
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
stage_iter0
stage_iter1 future_tie_distinct
'''
            p=subprocess.run(['bash','-c',script],env=dict(os.environ,TEST_REPO=str(ROOT),TEST_ROOT=str(root),TEST_PY=sys.executable),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            for phase,typ in [('iter0','D1'),('iter0','D2'),('iter1','D1'),('iter1_future_tie_distinct','D2')]:
                d=root/'data'/phase/typ;s=json.loads((d/'stats.json').read_text());pair=json.loads((d/'train.json').read_text())
                self.assertEqual(s['pair_selection'],'future_tie_distinct');self.assertEqual(s['length_selection'],'none')
                self.assertEqual(s['n_pairs'],1);self.assertEqual(s['filtered_equal_score_pairs'],1);self.assertEqual(s['filtered_no_distinct_text_pairs'],1)
                self.assertEqual(s.get('identical_pair',0),0);self.assertEqual(s['retained_tie_pairs'],0)
                self.assertEqual(s['chosen_candidate_pool']['n'],1);self.assertEqual(s['rejected_candidate_pool']['n'],1)
                if typ=='D2':self.assertEqual(pair['chosen'][-1]['content'],'a much longer answer')

if __name__=='__main__':unittest.main()
