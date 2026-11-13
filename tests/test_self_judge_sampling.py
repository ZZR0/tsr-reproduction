"""Judge temperature overrides must reach every actual call without mutating generation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('inference_sampling',ROOT/'src/inference_sampling.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Sampling(unittest.TestCase):
    def test_new_parameters_and_generation_restored_on_exception(self):
        original=SimpleNamespace(temperature=1.,top_p=1.,max_tokens=1024,stop=['EOS'])
        obj=SimpleNamespace(sampling_params=original)
        with self.assertRaisesRegex(RuntimeError,'test exception'):
            with module.self_judge_sampling(obj,SimpleNamespace,0):
                self.assertIsNot(obj.sampling_params,original);self.assertEqual(obj.sampling_params.temperature,0.)
                self.assertEqual(original.temperature,1.);self.assertTrue(obj._tsr_self_judging)
                raise RuntimeError('test exception')
        self.assertIs(obj.sampling_params,original);self.assertFalse(obj._tsr_self_judging)

    def test_real_shard_execution_separates_fused_and_standalone_sampling(self):
        upstream=r'''
import json
from types import SimpleNamespace as NS
SamplingParams=NS
LLM=NS
def seed_everything(seed):pass
def read_jsonl_file(p):return [json.loads(x) for x in open(p)]
def read_json_file(p):return json.load(open(p))
class Engine:
 def generate(self,prompts,params):return [NS(prompt_token_ids=[1],outputs=[NS(token_ids=[2],text=str(params.temperature))]) for p in prompts]
class SelfRewardGenerator:
 def __init__(self,model,world):
  self.sampling_params=SamplingParams(temperature=1.,top_p=1.,max_tokens=1024,stop=['EOS']);self.llm=Engine()
 def generate_all_responses(self,prompts,num_responses,type):
  return [[o.outputs[0].text]*num_responses for o in self.llm.generate(prompts,self.sampling_params)]
 def generate_all_judgments(self,prompts,responses,num_judgments,type):
  return [[[o.outputs[0].text] for r in row] for o,row in zip(self.llm.generate(prompts,self.sampling_params),responses)]
 def process_all(self,prompts,responses,judgments=None):
  return [dict(prompt=p,responses=r,judgments=j) for p,r,j in zip(prompts,responses,judgments or [None]*len(prompts))]
'''
        with tempfile.TemporaryDirectory(prefix='tsr-judge-temperature-') as tmp:
            root=Path(tmp);d=root/'tsr';d.mkdir()
            for name in ['self_generate_vllm_all_iterJudge.py','self_generate_vllm_all_generate.py']:(d/name).write_text(upstream)
            qa=root/'qa.jsonl';qa.write_text(json.dumps({'prompt':'p'})+'\n')
            past=root/'past.json';past.write_text(json.dumps([{'prompt':'p','responses':['past']*7}]))
            env=dict(os.environ,TSR_DIR=str(root),TSR_SELF_JUDGE_TEMPERATURE='0',TSR_PERFORMANCE_PROFILE='reference')
            for mode in ['gen','judge','genjudge']:
                out=root/(mode+'.json');extra=root/'extra.json'
                cmd=[sys.executable,str(ROOT/'src/tsr_shard.py'),mode,'--model','fake','--input',str(past if mode=='judge' else qa),'--out',str(out)]
                if mode=='genjudge':cmd += ['--extra-judge-input',str(past),'--extra-judge-out',str(extra)]
                p=subprocess.run(cmd,env=env,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
                calls=json.loads(Path(str(out)+'.metrics.json').read_text())['calls']
                self.assertEqual([c['temperature'] for c in calls],{'gen':[1.],'judge':[0.],'genjudge':[1.,0.,0.]}[mode])
                if mode=='genjudge':
                    self.assertEqual(json.loads(out.read_text())[0]['responses'],['1.0']*7)
                    self.assertEqual(json.loads(extra.read_text())[0]['judgments'],[['0.0']]*7)

if __name__=='__main__':unittest.main()
