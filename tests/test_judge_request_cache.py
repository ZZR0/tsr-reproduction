import collections
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('judge_request_cache', REPO/'src/judge_request_cache.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def choice(text='m'):
    return [{'text':text, 'total_tokens':1, 'logprobs':{'content':[{'token':'m','logprob':-.2,
        'top_logprobs':[{'token':'m','logprob':-.2},{'token':'M','logprob':-1.7}]}]}}]


class JudgeCacheTests(unittest.TestCase):
    def test_success_is_reused_and_namespace_and_payload_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def original(args,**kwargs):calls.append(args);return choice()
            one=cache.cached_helper(original,tmp,'provider1')
            self.assertEqual(one(('p',1),model='gpt4o'),choice())
            self.assertEqual(one(('p',1),model='gpt4o'),choice())
            self.assertEqual(len(calls),1)
            one(('different',1),model='gpt4o')
            cache.cached_helper(original,tmp,'provider2')(('p',1),model='gpt4o')
            self.assertEqual(len(calls),3)
            for p in Path(tmp).glob('*.json'):
                saved=json.loads(p.read_text());self.assertEqual(saved['choices'],choice())

    def test_failures_and_empty_fallbacks_are_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            def failure(*args,**kwargs):raise RuntimeError('HTTP429')
            with self.assertRaises(RuntimeError):cache.cached_helper(failure,tmp,'n')(('p',1))
            self.assertEqual(list(Path(tmp).glob('*.json')),[])
            for invalid in [[],[{'text':'','logprobs':None}],[{'text':'m'}]]:
                wrapped=cache.cached_helper(lambda *a,**k:invalid,tmp,'n')
                self.assertEqual(wrapped(('p',1)),invalid)
                self.assertEqual(list(Path(tmp).glob('*.json')),[])

    def test_interrupted_atomic_write_leaves_prior_success_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapped=cache.cached_helper(lambda *a,**k:choice(),tmp,'n')
            wrapped(('good',1))
            (Path(tmp)/'.unfinished.tmp').write_text('{')
            self.assertEqual(wrapped(('good',1)),choice())
            self.assertEqual(len(list(Path(tmp).glob('*.json'))),1)

    def test_official_threaded_decoder_keeps_successes_when_another_request_fails(self):
        from alpaca_eval.decoders import openai as decoder
        prompts=[f'<|im_start|>user\nq{i}<|im_end|>' for i in range(3)]
        calls=collections.Counter()
        def fake(args,**kwargs):
            key=json.dumps(args,sort_keys=True);calls[key]+=1
            if 'q0' in key and calls[key]==1:raise RuntimeError('transient HTTP429')
            return choice()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,
             JUDGE_REQUEST_CACHE=tmp,JUDGE_API_KEY='test-only',JUDGE_BASE_URL='http://unused',JUDGE_MODEL='gpt-4o-2024-08-06'):
            with patch.object(decoder,'_openai_completion_helper',fake):
                with cache.official_completion_cache():
                    with self.assertRaises(RuntimeError):
                        decoder.openai_completions(prompts,model_name='gpt-4o-2024-08-06',num_procs=3,max_tokens=1)
                    self.assertEqual(len(list(Path(tmp).glob('*.json'))),2)
                    result=decoder.openai_completions(prompts,model_name='gpt-4o-2024-08-06',num_procs=3,max_tokens=1)
                    self.assertEqual(result['completions'],['m']*3)
                    self.assertEqual(len(list(Path(tmp).glob('*.json'))),3)
                    self.assertEqual(sorted(calls.values()),[1,1,2])
                self.assertIs(decoder._openai_completion_helper,fake)

    def test_decoder_is_restored_after_exception(self):
        from alpaca_eval.decoders import openai as decoder
        original=decoder._openai_completion_helper
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,JUDGE_REQUEST_CACHE=tmp,JUDGE_API_KEY='test-only'):
            with self.assertRaises(RuntimeError):
                with cache.official_completion_cache():raise RuntimeError('stop')
            self.assertIs(decoder._openai_completion_helper,original)


if __name__=='__main__':unittest.main()
