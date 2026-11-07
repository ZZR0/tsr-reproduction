#!/usr/bin/env python3
"""Single-copy TSR rerun with four parallel, fixed-M1 D2 experiments."""
import concurrent.futures as futures
from contextlib import contextmanager
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ablation_pairs import rows, validate

REPO = Path('/media/ardeawang/TSR/tsr-reproduction')
OPS = Path('/media/ardeawang/TSR/repro/tsr-singlecopy-20261104')
OLD = REPO / 'work'
WORK = REPO / 'work-tsr-singlecopy-20261104'
MODELS, DATA, EVAL = WORK / 'models', WORK / 'data', WORK / 'alpaca_eval'
PY = REPO / '.venv/bin/python'
RUNTIME = Path('/opt/tsr-reproduction-aa7200a')
CASES = [('M2', [], [0, 1]), ('M2_gt_keep', ['--strict'], [2, 3]),
         ('M2_ge_filter', ['--filter'], [4, 5]), ('M2_gt_filter', ['--strict', '--filter'], [6, 7])]
COMMIT = 'aa7200a3538a1a6e89518016dc275226363fc52c'
GLOBAL_FD = None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def status(stage, state='running', **extra):
    item = dict(time=datetime.datetime.now().astimezone().isoformat(), stage=stage, status=state, **extra)
    write_json(OPS / 'status.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


@contextmanager
def gpu_lease(groups):
    """Acquire one complete GPU group; generation can use any free single GPU."""
    (OPS / 'gpu-locks').mkdir(exist_ok=True)
    handles = []
    while not handles:
        for group in groups:
            acquired = []
            try:
                for gpu in sorted(group):
                    h = (OPS / 'gpu-locks' / str(gpu)).open('a')
                    acquired.append(h)
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handles = acquired
                selected = list(group)
                break
            except BlockingIOError:
                for h in acquired:
                    h.close()
        if not handles:
            time.sleep(2)
    try:
        yield selected, [h.fileno() for h in handles]
    finally:
        for h in handles:
            h.close()


def call(operation, args, label, gpus=None, port=29531, gpu_fds=()):
    command = ['bash', str(OPS / 'run-stage.sh'), operation,
               ','.join(map(str, gpus or [0, 1])), str(port), *map(str, args)]
    with (OPS / (label + '.log')).open('a') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       pass_fds=(GLOBAL_FD, *gpu_fds), env=os.environ.copy())


def gpu_call(operation, args, label, gpus=(0, 1), port=29531):
    with gpu_lease([gpus]) as (selected, fds):
        call(operation, args, label, selected, port, fds)


def train(name, raw_data, gpus=(0, 1), port=29531):
    clean = WORK / 'trainsets' / name
    clean.mkdir(parents=True, exist_ok=True)
    source, target = raw_data / 'train.json', clean / 'train.json'
    if target.exists():
        assert digest(target) == digest(source), 'Existing training input changed'
    else:
        shutil.copyfile(source, target)
    env = dict(os.environ, HF_HOME='/media/ardeawang/TSR/.cache/huggingface',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    loading = json.loads(subprocess.check_output(
        [str(PY), str(OPS / 'inspect_dpo_loading.py'), str(clean), '--repetitions', '1'],
        env=env, text=True))
    write_json(OPS / (name + '.data_loading.json'), loading)
    init = MODELS / ('M0' if name in ['Mf0', 'M1'] else 'M1')
    write_json(OPS / (name + '.training.json'), dict(status='running', gpus=list(gpus), init=str(init), **loading))
    gpu_call('dpo', [name, init, clean], 'train_' + name, gpus, port)
    state = json.load((MODELS / name / 'trainer_state.json').open())
    result = json.load((MODELS / name / 'train_results.json').open())
    assert result['train_samples'] == loading['loaded_rows']
    assert state['global_step'] == loading['expected_steps']
    write_json(OPS / (name + '.training.json'), dict(status='completed', gpus=list(gpus), init=str(init),
               global_step=state['global_step'], epoch=state['epoch'], **loading))


def evaluate(name):
    try:
        write_json(OPS / (name + '.evaluation.json'), dict(status='waiting_for_gpu'))
        with gpu_lease([[g] for g in [2, 3, 0, 1, 4, 5, 6, 7]]) as (gpus, fds):
            call('eval_generate', [name], 'eval_generate_' + name, gpus, gpu_fds=fds)
        answers = json.load((EVAL / name / 'model_outputs.json').open())
        reference = json.load((EVAL / 'M0/model_outputs.json').open())
        assert len(answers) == 805
        assert [x['instruction'] for x in answers] == [x['instruction'] for x in reference]
        write_json(OPS / (name + '.evaluation.json'), dict(status='judging', questions=805))
        call('eval_score', [name], 'eval_score_' + name)
        result = next(csv.DictReader((EVAL / name / 'leaderboard.csv').open()))
        assert int(float(result['n_total'])) == 805
        assert all(math.isfinite(float(result[k])) for k in ['length_controlled_winrate', 'win_rate', 'avg_length'])
        write_json(OPS / (name + '.evaluation.json'), dict(status='completed', result=result))
        return result
    except Exception as exc:
        write_json(OPS / (name + '.evaluation.json'), dict(status='failed', error_type=type(exc).__name__))
        raise


def verify_source():
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    result = subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet'])
    if result.returncode == 0:
        return
    assert result.returncode == 1
    # Allow only the exact repository fix requested and deployed during this run.
    approved = json.load((OPS / 'repository-source-fix.json').open())
    assert approved['base_commit'] == COMMIT
    changed = set(subprocess.check_output(['git', '-C', str(REPO), 'diff', '--name-only', '-z'], text=True).rstrip('\0').split('\0'))
    assert changed == set(approved['tracked_files'])
    assert all(digest(REPO / name) == checksum for name, checksum in approved['tracked_files'].items())


def setup():
    verify_source()
    for p in [MODELS, DATA, EVAL, WORK / 'logs', WORK / 'trainsets']:
        p.mkdir(parents=True, exist_ok=True)
    assert (OLD / 'models/M0/.done').exists()
    assert json.load((OLD / 'models/M0/train_results.json').open())['train_samples'] == 6871
    if not (MODELS / 'M0').exists():
        (MODELS / 'M0').symlink_to(OLD / 'models/M0', target_is_directory=True)
    for name in ['M0', 'M0_seed1', 'M0_seed2']:
        assert (OLD / 'alpaca_eval' / name / 'leaderboard.csv').exists()
        if not (EVAL / name).exists():
            (EVAL / name).symlink_to(OLD / 'alpaca_eval' / name, target_is_directory=True)
    # Reserve four new checkpoints on shared storage and three on the root disk.
    shared = ['Mf0', 'M1', 'Mf1', 'M2']
    local = ['M2_gt_keep', 'M2_ge_filter', 'M2_gt_filter']
    for names, location in [(shared, MODELS), (local, RUNTIME / 'tsr-singlecopy-20261104/models')]:
        location.mkdir(parents=True, exist_ok=True)
        pending = sum(not (MODELS / n / '.done').exists() for n in names)
        assert shutil.disk_usage(location).free > (pending * 16 + 8) * 1024**3, 'Insufficient reserved storage'
        for name in names:
            target = location / name
            target.mkdir(exist_ok=True)
            if location != MODELS and not (MODELS / name).exists():
                (MODELS / name).symlink_to(target, target_is_directory=True)
    # These cached responses/scores depend only on the unchanged SFT model.
    reused = {}
    for rel in ['iter0/0_chosen_response.json', 'iter0/0_chosen_judgement.json', 'iter1/1_rejected_response.json']:
        source, target = OLD / 'data' / rel, DATA / rel
        records = json.load(source.open())
        assert len(records) == 5000 and all(len(r['responses']) == 7 for r in records)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        reused[rel] = dict(source=str(source), sha256=digest(source), rows=len(records))
    write_json(OPS / 'reused-sft-inputs.json', reused)
    write_json(OPS / 'plan.json', dict(commit=COMMIT, work=str(WORK), sft_reused=str(OLD / 'models/M0'),
        sft_mask_system_user=False, dpo_epochs=1, data_repetitions=1, global_batch_size=32, train_seed=42,
        retrain_common=['Mf0', 'M1', 'Mf1'], sr1=False, wof=False,
        variants=[dict(model=n, pair_flags=f, gpus=g) for n, f, g in CASES],
        eval=dict(benchmark='AlpacaEval', seed=0, temperature=1, top_p=1, max_tokens=1024,
                  judge='gpt-4o-2024-08-06')))


def main():
    global GLOBAL_FD
    OPS.mkdir(parents=True, exist_ok=True)
    lock = (OPS / 'launcher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    gpu_lock = Path('/media/ardeawang/TSR/repro/gpu-work.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    GLOBAL_FD = gpu_lock.fileno()
    os.environ.update(TZ='Asia/Shanghai', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    setup()
    if '--prepare-only' in sys.argv:
        status('prepared', 'ready')
        return
    qa = REPO / 'third_party/TSR/data/SelfPlayBeginData'
    w0, w1 = DATA / 'iter0', DATA / 'iter1'
    c0 = w0 / '0_chosen_judgement.json'
    c1, r1, f1 = [w1 / ('1_' + role + '_judgement.json') for role in ['chosen', 'rejected', 'future']]
    eval_jobs = []
    with futures.ThreadPoolExecutor(max_workers=1) as evaluator:
        status('train_Mf0')
        call('pairs', ['past', w0 / 'D1', c0, c0], 'pairs_iter0_D1')
        train('Mf0', w0 / 'D1'); eval_jobs.append(evaluator.submit(evaluate, 'Mf0'))
        status('iter0_future_rollout_and_judging')
        gpu_call('gen', [MODELS / 'Mf0', qa / 'qa1.jsonl', w0 / '0_future_response.json'], 'gen_iter0_future')
        gpu_call('judge', [MODELS / 'M0', w0 / '0_future_response.json', w0 / '0_future_judgement.json'], 'judge_iter0_future')
        call('pairs', ['future', w0 / 'D2', c0, c0, w0 / '0_future_judgement.json'], 'pairs_iter0_D2')
        status('train_M1')
        train('M1', w0 / 'D2'); eval_jobs.append(evaluator.submit(evaluate, 'M1'))
        status('iter1_current_past_rollout_and_judging')
        gpu_call('gen', [MODELS / 'M1', qa / 'qa2.jsonl', w1 / '1_chosen_response.json'], 'gen_iter1_current')
        gpu_call('judge', [MODELS / 'M1', w1 / '1_chosen_response.json', c1], 'judge_iter1_current')
        gpu_call('judge', [MODELS / 'M1', w1 / '1_rejected_response.json', r1], 'judge_iter1_past')
        call('pairs', ['past', w1 / 'D1', c1, r1], 'pairs_iter1_D1')
        status('train_Mf1')
        train('Mf1', w1 / 'D1'); eval_jobs.append(evaluator.submit(evaluate, 'Mf1'))
        status('iter1_future_rollout_and_judging')
        gpu_call('gen', [MODELS / 'Mf1', qa / 'qa2.jsonl', w1 / '1_future_response.json'], 'gen_iter1_future')
        gpu_call('judge', [MODELS / 'M1', w1 / '1_future_response.json', f1], 'judge_iter1_future')
        inputs = [c1, r1, f1, w1 / 'D1/train.json']
        inputs += sorted(p for p in (MODELS / 'M1').iterdir() if p.is_file() and p.suffix in {'.json', '.bin', '.safetensors'})
        write_json(OPS / 'shared-input-sha256.json', {str(p): digest(p) for p in inputs})
        datasets = []
        for name, flags, gpus in CASES:
            data = DATA / 'four-settings' / name / 'D2'
            call('pairs', ['future', data, c1, r1, f1, '--', *flags], 'pairs_' + name)
            datasets.append(data)
        verification = validate(*[rows(d / 'train.json') for d in datasets], c1, r1, f1)
        write_json(OPS / 'pair-verification.json', verification)
        status('four_M2_parallel', groups=[dict(model=n, gpus=g) for n, f, g in CASES])
        failures = []
        with futures.ThreadPoolExecutor(max_workers=4) as trainers:
            jobs = {trainers.submit(train, n, d, g, 29540+i): n
                    for i, ((n, f, g), d) in enumerate(zip(CASES, datasets))}
            for job in futures.as_completed(jobs):
                name = jobs[job]
                try:
                    job.result()
                    eval_jobs.append(evaluator.submit(evaluate, name))
                except Exception as exc:
                    failures.append(dict(model=name, error_type=type(exc).__name__))
        if failures:
            write_json(OPS / 'training-errors.json', failures)
            raise RuntimeError('One or more M2 training jobs failed; inspect per-model logs')
        (OPS / 'training.done').touch()
        status('alpaca_evaluation')
        for job in eval_jobs:
            job.result()
    summary = []
    for name, flags, gpus in CASES:
        row = next(csv.DictReader((EVAL / name / 'leaderboard.csv').open()))
        summary.append(dict(model=name, pair_flags=flags, LC=float(row['length_controlled_winrate']),
                            Win=float(row['win_rate']), length=float(row['avg_length']), n=int(float(row['n_total']))))
    write_json(OPS / 'comparison.json', summary)
    with (OPS / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (OPS / 'evaluation.done').touch()
    status('all', 'completed')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('pipeline', 'failed', error_type=type(exc).__name__)
        raise SystemExit(1)
