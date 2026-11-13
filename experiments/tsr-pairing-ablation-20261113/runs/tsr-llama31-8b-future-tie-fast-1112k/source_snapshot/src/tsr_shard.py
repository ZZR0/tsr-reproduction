#!/usr/bin/env python3
"""Data-parallel official TSR inference with optional throughput settings.

Reference mode preserves the upstream API. genjudge reuses one loaded engine
for current-model generation/self-judging and optionally judging past answers.
Only scheduling/cache settings change; prompts, K=7, scores and sampling stay
with the pinned upstream implementation. Fused calls advance the engine RNG;
no bitwise equivalence to separate engine launches is claimed.
"""
import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import time


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(p)


ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['gen', 'judge', 'genjudge', 'merge'])
ap.add_argument('--model')
ap.add_argument('--input')
ap.add_argument('--out', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--vllm_seed', type=int, default=None)
ap.add_argument('--extra-judge-input')
ap.add_argument('--extra-judge-out')
ap.add_argument('files', nargs='*')
a = ap.parse_intermixed_args()
if a.mode == 'merge':
    rows = []
    for filename in a.files:
        rows.extend(json.loads(Path(filename).read_text()))
    write_json(a.out, rows)
    print(f'merged {len(rows)} -> {a.out}', flush=True)
    raise SystemExit(0)
if bool(a.extra_judge_input) != bool(a.extra_judge_out) or (a.extra_judge_input and a.mode != 'genjudge'):
    ap.error('Extra judging requires genjudge and both extra paths')

name = 'self_generate_vllm_all_generate' if a.mode == 'gen' else 'self_generate_vllm_all_iterJudge'
path = Path(os.environ['TSR_DIR']) / 'tsr' / (name + '.py')
spec = importlib.util.spec_from_file_location(name, path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.seed_everything(a.seed)
data = mod.read_jsonl_file(a.input) if a.mode in ('gen', 'genjudge') else mod.read_json_file(a.input)
if a.limit:
    data = data[:a.limit]
n = len(data)
lo, hi = a.shard * n // a.nshards, (a.shard + 1) * n // a.nshards
data = data[lo:hi]
print(f'[{a.mode}] shard {a.shard}/{a.nshards}: items {lo}:{hi} of {n}', flush=True)
engine_options = {}
if a.vllm_seed is not None:
    engine_options['seed'] = a.vllm_seed
if os.environ.get('TSR_PERFORMANCE_PROFILE') == 'h200-fast':
    engine_options.update(
        max_num_batched_tokens=int(os.environ.get('TSR_VLLM_MAX_BATCHED_TOKENS', '16384')),
        max_num_seqs=int(os.environ.get('TSR_VLLM_MAX_SEQS', '512')),
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
    )
if engine_options:
    mod.LLM = functools.partial(mod.LLM, **engine_options)
started = time.monotonic()
sr = mod.SelfRewardGenerator(a.model, 1)
metrics = dict(mode=a.mode, shard=a.shard, prompts=len(data), engine_options=engine_options,
               model_load_s=time.monotonic()-started, calls=[])
original_generate = sr.llm.generate


def measured_generate(*args, **kwargs):
    begin = time.monotonic()
    outputs = original_generate(*args, **kwargs)
    metrics['calls'].append(dict(
        requests=len(outputs), elapsed_s=time.monotonic()-begin,
        input_tokens=sum(len(o.prompt_token_ids) for o in outputs),
        output_tokens=sum(len(c.token_ids) for o in outputs for c in o.outputs),
    ))
    print('INFERENCE_METRICS ' + json.dumps(metrics['calls'][-1]), flush=True)
    return outputs


sr.llm.generate = measured_generate
prompts = [row['prompt'] for row in data]
if a.mode in ('gen', 'genjudge'):
    responses = sr.generate_all_responses(prompts, num_responses=7, type='chat')
else:
    responses = [row['responses'] for row in data]
if a.mode == 'gen':
    results = sr.process_all(prompts, responses)
else:
    judgments = sr.generate_all_judgments(prompts, responses, num_judgments=1, type='eval')
    results = sr.process_all(prompts, responses, judgments)
write_json(a.out, results)
if a.extra_judge_input:
    extra = mod.read_json_file(a.extra_judge_input)
    if a.limit:
        extra = extra[:a.limit]
    assert len(extra) == n
    extra = extra[lo:hi]
    assert [row['prompt'] for row in extra] == prompts
    extra_responses = [row['responses'] for row in extra]
    judgments = sr.generate_all_judgments(prompts, extra_responses, num_judgments=1, type='eval')
    write_json(a.extra_judge_out, sr.process_all(prompts, extra_responses, judgments))
metrics['total_s'] = time.monotonic() - started
write_json(a.out + '.metrics.json', metrics)
print(f'saved {len(results)} -> {a.out}', flush=True)
