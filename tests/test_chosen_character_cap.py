import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
if ROOT.name=='tests':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from length_pairing import select_first_distinct_pair


def row(prompt,texts,scores):
    return dict(prompt=prompt,responses=texts,scores=scores,judgments=[[f'Score: {s}'] for s in scores])


class CharacterCap(unittest.TestCase):
    def test_inclusive_unicode_cap_and_next_candidate_in_same_fixed_pool(self):
        c=row('p',['a'*2501,'中'*2500,'short high','r'],[5,5,5,1])
        p=row('p',['past'],[1])
        out=select_first_distinct_pair(c,p,max_chosen_characters=2500)
        self.assertTrue(out['valid']);self.assertEqual(out['chosen']['text'],'中'*2500)
        self.assertGreater(len(out['chosen']['text'].encode()),2500)  # Count characters, not bytes.
        f=row('p',['z'*2501,'f'*2400,'tiny'],[5,5,5])
        out=select_first_distinct_pair(c,p,f,max_chosen_characters=2500)
        self.assertEqual(out['chosen']['text'],'f'*2400)  # No shortest-length ranking.
        self.assertEqual(out['chosen']['source'],'Mf');self.assertEqual(out['rejected']['source'],'Mi')

    def test_cap_never_relaxes_source_or_score(self):
        c=row('p',['short current','r'],[5,1]);p=row('p',['past'],[1])
        f=row('p',['f'*2501,'short lower future'],[5,4])
        out=select_first_distinct_pair(c,p,f,max_chosen_characters=2500)
        self.assertFalse(out['valid']);self.assertEqual(out['reason'],'no_chosen_within_character_limit')
        c=row('p',['c'*2501,'short lower','r'],[5,4,1])
        self.assertFalse(select_first_distinct_pair(c,p,max_chosen_characters=2500)['valid'])

    def test_all_four_real_training_pair_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            c=[row('ok',['x'*2501,'中'*2500,'r'],[5,5,1]),
               row('over',['x'*2501,'r'],[5,1]),
               row('tie',['a','b'],[5,5]),row('same',['same','same'],[5,1]),
               row('invalid',['bad'],[0])]
            p=[row('ok',['past'],[1]),row('over',['past'],[1]),row('tie',['past'],[5]),
               row('same',['past'],[1]),row('invalid',['bad'],[0])]
            f=[row('ok',['y'*2501,'f'*2400,'bad high'],[5,5,99]),
               row('over',['y'*2501,'short lower'],[5,4]),row('tie',['future'],[5]),
               row('same',['same'],[5]),row('invalid',['bad'],[0])]
            for rows in [c,p,f]:rows[-1]['judgments']=[['missing score']]
            for phase,files in {'iter0':{'0_chosen_judgement.json':c,'0_future_judgement.json':f},
                                'iter1':{'1_chosen_judgement.json':c,'1_rejected_judgement.json':p,'1_future_judgement.json':f}}.items():
                d=root/'data'/phase;d.mkdir(parents=True)
                for name,rows in files.items():(d/name).write_text(json.dumps(rows))
            shell='''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
TSR_PAIR_SELECTION=future_tie_charcap_distinct
TSR_DROP_INVALID_SCORES=1
TSR_MAX_CHOSEN_CHARACTERS=2500
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
stage_iter0
stage_iter1 future_tie_charcap_distinct
'''
            result=subprocess.run(['bash','-c',shell],env=dict(os.environ,TEST_REPO=str(ROOT),TEST_ROOT=str(root),TEST_PY=sys.executable),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for phase,ds in [('iter0','D1'),('iter0','D2'),('iter1','D1'),('iter1_future_tie_charcap_distinct','D2')]:
                d=root/'data'/phase/ds;stats=json.loads((d/'stats.json').read_text())
                pairs=[json.loads(line) for line in (d/'train.json').read_text().splitlines()]
                self.assertEqual(stats['n_pairs'],1)
                self.assertEqual(stats['filtered_no_chosen_within_character_limit'],1)
                self.assertEqual(stats['filtered_equal_score_pairs'],1)
                self.assertEqual(stats['filtered_no_distinct_text_pairs'],1)
                self.assertEqual(stats['filtered_no_valid_score_candidates'],1)
                self.assertEqual(stats['chosen_over_character_limit_pairs'],0)
                self.assertEqual(stats['max_chosen_characters'],2500)
                self.assertEqual(pairs[0]['chosen'][-1]['content'],'f'*2400 if ds=='D2' else '中'*2500)
                self.assertEqual(stats['length_selection'],'character_cap_only_no_ranking')

if __name__=='__main__':unittest.main()
