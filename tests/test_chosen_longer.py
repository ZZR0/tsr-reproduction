import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
if ROOT.name=='tests': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from length_pairing import select_min_length_pair, select_first_distinct_pair


def row(p,texts,scores):
    return dict(prompt=p,responses=texts,scores=scores,
                judgments=[[f'Score: {s}'] for s in scores])


def choose(c,r,f=None):
    return select_min_length_pair(c,r,f,source_priority=True,require_distinct_text=True,require_chosen_longer=True)


class LongerPair(unittest.TestCase):
    def test_joint_minimum_positive_gap(self):
        c=row('p',['current high is much longer','abcdefghij','abcdefghijkl'],[5,1,1])
        r=row('p',['past'],[1])
        f=row('p',['abcdefghi','ABCDEFGHIJKLM'],[5,5])
        out=choose(c,r,f)
        self.assertTrue(out['valid'])
        self.assertEqual(out['chosen']['text'],'ABCDEFGHIJKLM')
        self.assertEqual(out['rejected']['text'],'abcdefghijkl')
        self.assertEqual(out['signed_length_gap'],1)
        self.assertEqual(out['chosen']['source'],'Mf')
        self.assertEqual(out['rejected']['source'],'Mi')
        self.assertEqual(select_first_distinct_pair(c,r,f)['chosen']['text'],'abcdefghi')

    def test_never_relax_scores_sources_or_strict_length(self):
        c=row('p',['very long current high answer','current low'],[5,1])
        r=row('p',['p'],[1]);f=row('p',['f','future lower score but very long'],[5,4])
        out=choose(c,r,f)
        self.assertFalse(out['valid']);self.assertEqual(out['reason'],'no_chosen_longer_pair')
        equal=row('p',['abc','xyz'],[5,5])
        self.assertFalse(choose(equal,equal,equal)['valid'])
        same=row('p',['same  ','same'],[5,1])
        self.assertFalse(choose(same,same)['valid'])
        c2=row('p',['abc','xyz'],[5,1])
        self.assertFalse(choose(c2,c2)['valid'])  # equal lengths are forbidden.

    def test_all_four_real_stages_and_invalid_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            c=[row('ok',['abcdefghijklm','abcdefghij','abcdefghijkl','bad'],[5,1,1,0]),
               row('length impossible',['hi','longer'],[5,1]),
               row('ties',['a','b'],[5,5]),row('invalid',['bad'],[0])]
            c[0]['judgments'][-1]=['missing score'];c[-1]['judgments']=[['missing score']]
            r=[row('ok',['past'],[1]),row('length impossible',['p'],[1]),
               row('ties',['past'],[5]),row('invalid',['bad'],[0])]
            r[-1]['judgments']=[['Score: NaN']]
            f=[row('ok',['abcdefghi','ABCDEFGHIJKLM','bad out of range'],[5,5,99]),
               row('length impossible',['f'],[5]),row('ties',['future'],[5]),row('invalid',['bad'],[0])]
            f[-1]['judgments']=[['Score: NaN']]
            for phase,files in {'iter0':{'0_chosen_judgement.json':c,'0_future_judgement.json':f},
                                'iter1':{'1_chosen_judgement.json':c,'1_rejected_judgement.json':r,'1_future_judgement.json':f}}.items():
                d=root/'data'/phase;d.mkdir(parents=True)
                for name,data in files.items():(d/name).write_text(json.dumps(data))
            script='''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
TSR_PAIR_SELECTION=future_tie_longer_distinct
TSR_DROP_INVALID_SCORES=1
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
stage_iter0
stage_iter1 future_tie_longer_distinct
'''
            p=subprocess.run(['bash','-c',script],env=dict(os.environ,TEST_REPO=str(ROOT),TEST_ROOT=str(root),TEST_PY=sys.executable),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            for phase,ds in [('iter0','D1'),('iter0','D2'),('iter1','D1'),('iter1_future_tie_longer_distinct','D2')]:
                d=root/'data'/phase/ds;s=json.loads((d/'stats.json').read_text())
                pairs=[json.loads(line) for line in (d/'train.json').read_text().splitlines()]
                self.assertEqual(s['n_pairs'],1)
                self.assertEqual(s['filtered_no_chosen_longer_pairs'],1)
                self.assertEqual(s['filtered_equal_score_pairs'],1)
                self.assertEqual(s['filtered_no_valid_score_candidates'],1)
                self.assertEqual(s['signed_length_gap_min'],1)
                self.assertEqual(s['chosen_longer_pairs'],1)
                self.assertEqual(s['chosen_shorter_pairs'],0)
                self.assertEqual(s['invalid_score_filter']['parser'],'explicit-score-v1')
                self.assertEqual(pairs[0]['rejected'][-1]['content'],'abcdefghijkl')
                self.assertEqual(pairs[0]['chosen'][-1]['content'],'ABCDEFGHIJKLM' if ds=='D2' else 'abcdefghijklm')

if __name__=='__main__': unittest.main()
