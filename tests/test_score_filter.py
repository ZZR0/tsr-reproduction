import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'tests':
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from score_validation import extract_score, annotate_record, remove_invalid_candidates
from length_pairing import select_first_distinct_pair


def row(prompt, texts, judgments, numeric=None):
    return dict(prompt=prompt, responses=texts, judgments=[[j] for j in judgments],
                scores=numeric if numeric is not None else [0]*len(texts))


class Scores(unittest.TestCase):
    def test_explicit_zero_valid_and_bad_scores_never_default_to_zero(self):
        self.assertEqual(extract_score(['Score: 0']), (0.0, None))
        self.assertEqual(extract_score(['Feedback\nScore: **5**']), (5.0, None))
        for text in ['no score', 'Score: NaN', 'Score: infinity', 'Score: 6',
                     'Score: -1', 'Score: 5/5', 'Score: 5\nScore: 1']:
            self.assertIsNone(extract_score([text])[0], text)

    def test_reparse_raw_judgments_and_preserve_candidate_alignment(self):
        r=row('p', ['bad', 'good', 'zero'], ['bad format', 'Score: 5', 'Score: 0'], [5,0,0])
        a=annotate_record(r)
        self.assertEqual(a['scores'], [None,5.0,0.0])
        c=remove_invalid_candidates(a)
        self.assertEqual(c['responses'], ['good','zero'])
        self.assertEqual(c['original_candidate_indices'], [1,2])

    def test_empty_source_is_not_a_score_or_source_fallback(self):
        empty=dict(prompt='p', responses=[], scores=[])
        f=dict(prompt='p', responses=['future'], scores=[5])
        past=dict(prompt='p', responses=['past'], scores=[1])
        self.assertFalse(select_first_distinct_pair(empty,past)['valid'])
        selected=select_first_distinct_pair(empty,past,f)
        self.assertEqual(selected['chosen']['source'], 'Mf')
        self.assertEqual(selected['rejected']['source'], 'M0')

    def test_all_four_real_pairing_stages(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            c=[row('ok',['current','invalid','low'],['Score: 5','missing','Score: 1'],[5,0,1]),
               row('tie',['a','b'],['Score: 5','Score: 5'],[5,5]),
               row('duplicate',['same','same'],['Score: 5','Score: 1'],[5,1]),
               row('invalid',['bad'],['Score: NaN'],[0])]
            past=[row('ok',['past'],['Score: 1'],[1]),row('tie',['past'],['Score: 5'],[5]),
                  row('duplicate',['other'],['Score: 1'],[1]),row('invalid',['bad'],['missing'],[0])]
            future=[row('ok',['bad_high','future'],['Score: 99','Score: 5'],[99,5]),
                    row('tie',['future'],['Score: 5'],[5]),row('duplicate',['same'],['Score: 5'],[5]),
                    row('invalid',['bad'],['Score: 7'],[7])]
            for phase, files in {'iter0':{'0_chosen_judgement.json':c,'0_future_judgement.json':future},
                                 'iter1':{'1_chosen_judgement.json':c,'1_rejected_judgement.json':past,'1_future_judgement.json':future}}.items():
                d=root/'data'/phase; d.mkdir(parents=True)
                for name, rows in files.items(): (d/name).write_text(json.dumps(rows))
            code='''
set -euo pipefail
. "$TEST_REPO/scripts/lib.sh"
. "$TEST_REPO/scripts/stages.sh"
DATA="$TEST_ROOT/data"; MODELS="$TEST_ROOT/models"; M0="$MODELS/M0"; TSR_DIR="$TEST_ROOT/upstream"
PY="$TEST_PY"; LOGS="$TEST_ROOT/logs"; mkdir -p "$LOGS"
TSR_PAIR_SELECTION=future_tie_distinct
TSR_DROP_INVALID_SCORES=1
need() { :; }; gen() { :; }; judge() { :; }; dpo() { :; }
stage_iter0
stage_iter1 future_tie_distinct
'''
            p=subprocess.run(['bash','-c',code], env=dict(os.environ,TEST_REPO=str(ROOT),TEST_ROOT=str(root),TEST_PY=sys.executable),capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            for phase, dataset in [('iter0','D1'),('iter0','D2'),('iter1','D1'),('iter1_future_tie_distinct','D2')]:
                d=root/'data'/phase/dataset
                stats=json.loads((d/'stats.json').read_text())
                pairs=[json.loads(line) for line in (d/'train.json').read_text().splitlines()]
                self.assertEqual(len(pairs),1)
                self.assertTrue(stats['drop_invalid_scores'])
                self.assertEqual(stats['filtered_equal_score_pairs'],1)
                self.assertEqual(stats['filtered_no_distinct_text_pairs'],1)
                self.assertEqual(stats['filtered_no_valid_score_candidates'],1)
                self.assertEqual(pairs[0]['rejected'][-1]['content'],'low')
                self.assertEqual(pairs[0]['chosen'][-1]['content'],'future' if dataset=='D2' else 'current')

    def test_all_stage_configs_receive_independent_seed(self):
        import yaml
        repo=ROOT
        if not (repo/'third_party/TSR/settings/sft/config/config.yaml').exists():
            self.skipTest('Run setup.sh to provide the pinned upstream config templates')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); data=root/'pairs'; data.mkdir(); (data/'train.json').write_text('{}\n')
            for mode in ['sft','dpo']:
                for seed in [42,43,44]:
                    cfg=root/f'{mode}{seed}.yaml'
                    subprocess.run([sys.executable,str(ROOT/'src/make_cfg.py'),mode,'--name','test','--init','/base','--data',str(data),'--out_dir',str(root/'model'),'--ngpu','8','--cfg',str(cfg)],
                        env=dict(os.environ,TSR_DIR=str(repo/'third_party/TSR'),TSR_TRAIN_SEED=str(seed)), check=True, capture_output=True)
                    c=yaml.safe_load(cfg.read_text())
                    self.assertEqual(c['seed'],seed); self.assertEqual(c['data_seed'],seed)
                    self.assertEqual(c['per_device_train_batch_size']*c['gradient_accumulation_steps']*8,32)

if __name__=='__main__': unittest.main()
