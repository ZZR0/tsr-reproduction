#!/usr/bin/env python3
"""Publish compact, offline-verifiable TSR experiment records without weights or credentials."""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

EXCLUDED_RUNS = {'tsr-llama31-8b-future-only-1111e8'}  # Already committed in the prior archive.
PRIVATE_FIELDS = {'base_url','credential_file','JUDGE_API_KEY','JUDGE_BASE_URL','HF_TOKEN','api_key','access_token','authorization'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items() if k not in PRIVATE_FIELDS}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def compressed_bytes(path):
    data=path.read_bytes()
    return gzip.decompress(data) if path.suffix=='.gz' else data


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    paths=set()
    for item in manifest['files']:
        p=root/item['path'];assert root.resolve() in p.resolve().parents and not p.is_symlink()
        assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],item['path']
        paths.add(item['path'])
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert paths==actual,('Manifest coverage mismatch',paths^actual)
    for item in json.loads((root/'provenance/source-files.json').read_text()):
        p=root/item['path']
        assert hashlib.sha256(compressed_bytes(p)).hexdigest()==item['export_uncompressed_sha256'],str(p)
        if not item['sanitized']:
            assert item['source_sha256']==item['export_uncompressed_sha256']
    ref=json.loads((root/'inputs/alpaca_reference.json').read_text())
    assert len(ref)==805
    results=json.loads((root/'metrics/results.json').read_text())['evaluations']
    for row in results:
        d=root/'runs'/row['run']/'evaluation'/row['model']
        outputs=json.loads(compressed_bytes(d/'model_outputs.json.gz'))
        annotations=json.loads(compressed_bytes(d/'annotations.json.gz'))
        marker=json.loads((d/'GPT4O_SCORE_COMPLETE.json').read_text())
        assert len(outputs)==len(annotations)==row['questions']==805
        assert [x['instruction'] for x in outputs]==[x['instruction'] for x in ref]
        assert hashlib.sha256(compressed_bytes(d/'model_outputs.json.gz')).hexdigest()==marker['input_sha256']
        assert digest(root/'inputs/alpaca_reference.json')==marker['reference_sha256']
        assert all(isinstance(a['preference'],(int,float)) and math.isfinite(a['preference']) and 1<=a['preference']<=2 for a in annotations)
        for key in ['lc_win','win','mean_characters']:
            assert math.isfinite(row[key])
    print(f"Verified {len(paths)} files, {len(results)} complete evaluations, {len(results)*805} scored answers")


def export(repo,out):
    if out.exists():raise RuntimeError('Refusing to overwrite existing archive')
    out.mkdir(parents=True)
    provenance=[];evaluations=[];inventory=[];pair_rows=[]
    def copy(source,target,*,compress=False,sanitize=False):
        source=Path(source);p=out/target;p.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if sanitize:data=(json.dumps(clean(json.loads(data)),ensure_ascii=False,indent=2)+'\n').encode()
        p.write_bytes(gzip.compress(data,mtime=0) if compress else data)
        provenance.append(dict(path=target,source_sha256=digest(source),export_uncompressed_sha256=hashlib.sha256(data).hexdigest(),sanitized=sanitize))
    reference_source=repo/'experiments/tsr-singlecopy-20261104-best/alpaca_eval/reference_outputs.json'
    copy(reference_source,'inputs/alpaca_reference.json')
    reference=json.loads(reference_source.read_text())
    for run in sorted((repo/'work/runs').iterdir()):
        if not (run/'run-plan.json').exists():continue
        plan=json.loads((run/'run-plan.json').read_text())
        completed=json.loads((run/'EVALUATION_COMPLETED.json').read_text()) if (run/'EVALUATION_COMPLETED.json').exists() else None
        canceled=(run/'CANCELED.json').exists();failed=(run/'FAILED.json').exists()
        stages=[m.name for m in (run/'trial/models').glob('*') if (m/'.done').exists()]
        status='completed' if completed else 'canceled' if canceled else 'failed' if failed else 'incomplete_no_final_evaluation'
        inv=dict(run=run.name,status=status,created_utc=plan.get('created_utc'),saved_stage_models=stages,
                 completed_evaluation_models=list(completed['models']) if completed else [],
                 already_archived=(run.name in EXCLUDED_RUNS))
        inventory.append(inv)
        if run.name in EXCLUDED_RUNS:continue
        prefix=f'runs/{run.name}'
        save(out/prefix/'lifecycle.json',inv)
        for f in ['run-plan.json','CANCELED.json','model-source.json','EXECUTION_REQUEST.json',
                  'TRAINING_COMPLETED.json','TRAINING_AND_GENERATION_COMPLETED.json','EVALUATION_COMPLETED.json',
                  'dataset-manifest.json','base-model-manifest.json','source-sha256.json',
                  'gpu-validated.json','submission.json','job.json','score-job.json','m2-score-job.json']:
            if (run/f).is_file():copy(run/f,prefix+'/provenance/'+f,sanitize=True)
        if failed:
            failure=json.loads((run/'FAILED.json').read_text())
            save(out/prefix/'provenance/failure-summary.json',dict(type=failure.get('type'),source_sha256=digest(run/'FAILED.json'),details='See retained local run logs; raw exception text is not exported'))
        if not completed:continue
        for model,metric in completed['models'].items():
            d=run/'trial/alpaca_eval'/model
            outputs=json.loads((d/'model_outputs.json').read_text())
            annotations=json.loads((d/'annotations.json').read_text())
            marker=json.loads((d/'GPT4O_SCORE_COMPLETE.json').read_text())
            diag=json.loads((d/'model_outputs.diag.json').read_text())
            assert len(outputs)==len(annotations)==805
            assert [x['instruction'] for x in outputs]==[x['instruction'] for x in reference]
            assert digest(d/'model_outputs.json')==marker['input_sha256']
            assert digest(reference_source)==marker['reference_sha256']
            evaluations.append(dict(run=run.name,model=model,questions=805,
                pair_selection=plan.get('pair_selection',plan.get('source_training_pair_selection','evaluation_only')),
                self_judge_temperature=plan.get('generation',{}).get('self_judge_temperature',plan.get('source_training_self_judge_temperature',1.0)),
                generation_temperature=diag['temperature'],generation_seed=diag['seed'],
                max_chosen_characters=plan.get('chosen_character_cap',{}).get('max_characters'),
                lc_win=metric['length_controlled_winrate'],win=metric['win_rate'],
                mean_characters=diag['mean_chars'],truncation_percent=100*diag.get('truncated_rate',0),
                completed_utc=completed['completed_utc'],judge=marker['judge'],
                source_run=plan.get('source_run',run.name)))
            for f in ['model_outputs.json','annotations.json']:
                copy(d/f,f'{prefix}/evaluation/{model}/{f}.gz',compress=True)
            for f in ['model_outputs.diag.json','GPT4O_SCORE_COMPLETE.json']:
                copy(d/f,f'{prefix}/evaluation/{model}/{f}',sanitize=True)
            copy(d/'leaderboard.csv',f'{prefix}/evaluation/{model}/leaderboard.csv')
        for file in sorted((run/'trial/data').glob('*/D*/stats.json')):
            dataset=str(file.parent.relative_to(run/'trial/data'))
            stats=json.loads(file.read_text());data=file.parent/'train.json'
            evidence=dict(stats=stats,train_json_sha256=digest(data),train_json_bytes=data.stat().st_size)
            save(out/prefix/'pair_statistics'/dataset/'stats.json',evidence)
            pair_rows.append(dict(run=run.name,dataset=dataset,n_pairs=stats['n_pairs'],
                score_ties_filtered=stats.get('filtered_equal_score_pairs',0),
                identical_text_filtered=stats.get('filtered_no_distinct_text_pairs',0),
                no_chosen_longer_pair=stats.get('filtered_no_chosen_longer_pairs',0),
                no_chosen_within_cap=stats.get('filtered_no_chosen_within_character_limit',0),
                chosen_max_characters=stats.get('chosen_length_max'),chosen_mean_characters=stats.get('chosen_length_mean'),
                mean_absolute_gap=stats.get('absolute_length_gap_mean'),
                invalid_score_candidates_removed=stats.get('invalid_score_filter',{}).get('invalid_candidates_removed',0)))
        for m in (run/'trial/models').glob('*'):
            for f in ['trainer_state.json','train_results.json','all_results.json']:
                if (m/f).is_file():copy(m/f,f'{prefix}/training/{m.name}/{f}',sanitize=True)
        for f in (run/'trial/configs').glob('*.yaml'):
            copy(f,f'{prefix}/configs/{f.name}')
        for m in (run/'trial/models').glob('*'):
            source=m.resolve().parent.parent/'configs'/(m.resolve().name+'.yaml')
            target=f'{prefix}/configs/{m.name}.yaml'
            if source.is_file() and not (out/target).exists():copy(source,target)
        for f in (run/'trial/metrics').rglob('*.json'):
            if f.stat().st_size<3_000_000:
                copy(f,f'{prefix}/training_metrics/{f.relative_to(run/"trial/metrics")}',sanitize=True)
        if (run/'bootstrap/pip-freeze.txt').is_file():copy(run/'bootstrap/pip-freeze.txt',prefix+'/provenance/pip-freeze.txt')
        # Preserve the actual frozen experiment implementation, not merely the current checkout.
        bundle=run/'bundle'
        for folder in ['src','scripts','configs','patches']:
            for f in (bundle/folder).rglob('*'):
                if f.is_file() and f.suffix in ['.py','.sh','.yaml','.yml','.json','.patch']:
                    copy(f,prefix+'/source_snapshot/'+str(f.relative_to(bundle)))
    save(out/'metrics/run-inventory.json',inventory)
    save(out/'metrics/results.json',dict(evaluations=evaluations,notes=[
        'Win is the AlpacaEval win_rate from probability-based judge preferences; LC Win is length controlled.',
        'Evaluation-only rows reuse source weights; inspect model-source.json for provenance.',
        'M0_seed1 and M0_seed2 are generation seeds of one SFT model, not independent training seeds.',
        'Fresh SFT runs use seed 42 unless otherwise recorded; independently trained models are not bitwise identical.',
        'Canceled/failed/incomplete runs are not presented as completed results.',
        'For older bundles without an explicit self_judge_temperature field, 1.0 is inferred from the pinned upstream sampling implementation.',
        'Prior future-only results remain in ../tsr-future-only-20261112.']))
    for name,rows in [('results',evaluations),('pair-filtering',pair_rows)]:
        with (out/f'metrics/{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Latest four fresh-SFT experiments are the primary controlled comparisons.
    latest=[r for r in evaluations if any(tag in r['run'] for tag in ['longer-gap-1112t','distinct-only-1112t','cap2500-judge'])]
    text='''# TSR 配对规则和温度实验记录（2026-11-13）

本目录保存本轮实验的完整评测结果、实际配置、训练指标、配对过滤统计，以及当时运行的代码快照。
模型为 Llama-3.1-8B；最新四组均从 Base 独立训练 SFT，训练种子 42、全局 batch 32。
所有表格的 Win 均指结果文件的 `win_rate`，不是按题目硬判胜负统计的离散胜率。

## 最新四组结果

每组均完成 805 题 AlpacaEval。self-generate=1，评测回答生成=0，GPT-4o 评分裁判设置保持一致。

| 实验 | self-judge 温度 | LC Win (%) | Win (%) | 平均字符数 | 输出截断比例 (%) |
|---|---:|---:|---:|---:|---:|
'''
    labels={'longer-gap-1112t':'chosen 更长且正长度差最小','distinct-only-1112t':'仅要求文本不同，不限长度',
            'cap2500-judge1':'chosen ≤2500 字符，self-judge=1','cap2500-judge0':'chosen ≤2500 字符，self-judge=0'}
    for row in latest:
        label=next(v for k,v in labels.items() if k in row['run'])
        text+=f"| {label} | {row['self_judge_temperature']:g} | {row['lc_win']:.2f} | {row['win']:.2f} | {row['mean_characters']:.0f} | {row['truncation_percent']:.2f} |\n"
    text+='''
字符约束在分数与模型来源优先级确定后执行；无合法 pair 则丢弃该题，不降低分数或切换来源。
字符上限不通过截短文本实现，也不应用于评测回答。两轮四个 DPO 阶段都执行相应规则。
最新 cap2500 两组同时过滤同分、相同文本和异常分数。实验差异和过滤口径见各 run-plan 和配对统计。

这轮 self-judge=0、chosen≤2500 的最终 LC Win 为 29.28%，Win 为 27.19%。
这些是单训练种子实验；长度规则会改变保留数据量，不能只凭一次比较断言普遍优劣。
训练候选和评测回答仍共享 1024 输出 token 上限，输出截断比例已单列。

## 文件结构

- `metrics/results.csv`、`results.json`：所有新归档完整评测及温度、长度信息。
- `metrics/pair-filtering.csv`：各训练阶段的配对数量和过滤原因。
- `metrics/run-inventory.json`：完成、取消、失败和未完成运行的索引。
- `runs/<run>/evaluation/<model>/`：完整 805 题回答和评分注释（gzip）、成绩和生成参数。
- `runs/<run>/training/`、`training_metrics/`：原生训练状态、标量指标及推理耗时。
- `runs/<run>/configs/`、`provenance/`、`source_snapshot/`：实际配置、版本/数据校验和执行代码。
- `manifest.json`、`provenance/source-files.json`：归档完整性及原文件校验记录。

只归档分析所需记录。不包含模型权重、优化器状态、训练完整 batch、视觉张量、API 凭据或请求缓存。
训练 preference 数据只保存哈希、规模和统计；评测回答与注释完整保留。
复用历史权重的评测变体见 `model-source.json`；评测种子变化不可当成多训练种子结果。
服务地址和秘密相关配置字段已从 JSON 记录中移除，此类变换在 source-files.json 标注为 sanitized。

## 离线验证

```bash
python scripts/archive_pairing_experiments.py --verify experiments/tsr-pairing-ablation-20261113
```

取消的三种子运行没有完成最终评测，不作为多种子结果。用户取消的 Meta-Rewarding length-control SR baseline
没有提交训练，也没有实现代码纳入本次提交。
'''
    (out/'README.md').write_text(text)
    save(out/'provenance/source-files.json',provenance)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),format='tsr-pairing-experiments-v1',
                  evaluations=len(evaluations),run_inventory_entries=len(inventory),files=[])
    for p in sorted(out.rglob('*')):
        if p.is_file():manifest['files'].append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p)))
    save(out/'manifest.json',manifest)
    verify(out)
    print(f'Archive size: {sum(f["bytes"] for f in manifest["files"])/1024/1024:.2f} MiB')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',type=Path)
    ap.add_argument('--runs-root',type=Path,help='Repository root containing work/runs (defaults to this checkout)')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.verify:return verify(a.verify.resolve())
    repo=a.runs_root.resolve() if a.runs_root else Path(__file__).resolve().parents[1]
    out=a.output.resolve() if a.output else repo/'experiments/tsr-pairing-ablation-20261113'
    export(repo,out)

if __name__=='__main__':main()
