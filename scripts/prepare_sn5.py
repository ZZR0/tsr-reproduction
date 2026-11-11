#!/usr/bin/env python3
"""Freeze a full TSR run with global batch 32. Preparation never submits a Job."""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import json
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "work/runs/tsr-llama31-8b-future-only-1111a"
NODE = "hyperpod-i-03fcebc230d22fd76"
IMAGE = "pytorch/pytorch@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
UPSTREAM = "020266aa9ef3cc35bf81384c9b59f98cff62aa1c"


def resource_plan(gpus, node):
    if gpus not in (2, 4, 8, 16):
        raise ValueError("Supported GPU counts: 2, 4, 8, 16")
    if gpus > 8:
        raise ValueError("This launcher supports only single-node runs (at most 8 GPUs)")
    expressions = [
        {"key":"group", "operator":"In", "values":["shennong-5", "shennong-5-dev"]},
        {"key":"node.kubernetes.io/instance-type", "operator":"In", "values":["ml.p5e.48xlarge"]},
    ]
    if node != "any-sn5":
        expressions.append({"key":"kubernetes.io/hostname", "operator":"In", "values":[node]})
    return {
        "gpus":gpus, "global_batch":32, "sft_micro_batch":2,
        "sft_gradient_accumulation":32 // (2 * gpus),
        "dpo_micro_batch":1, "dpo_gradient_accumulation":32 // gpus,
        "node_expressions":expressions,
        "requests":{"cpu":str(8*gpus), "memory":f"{96*gpus}Gi", "nvidia.com/gpu":str(gpus)},
        "limits":{"cpu":str(16*gpus), "memory":f"{192*gpus}Gi", "nvidia.com/gpu":str(gpus)},
    }


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="tsr-llama31-8b-future-only-1111a")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--node", default=NODE, help="Exact node, or any-sn5 (still constrained to SN5 H200)")
    parser.add_argument("--sft-source-run", type=Path, help="Reuse a completed M0 at a stage boundary; never resume partial weights")
    options = parser.parse_args()
    resources = resource_plan(options.gpus, options.node)
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,51}", options.run_name):
        raise SystemExit("Invalid run name")
    RUN = REPO / "work/runs" / options.run_name
    if RUN.exists():
        raise SystemExit("Run already exists; refusing to overwrite a frozen run")
    up = REPO / "third_party/TSR"
    if subprocess.check_output(["git", "-C", str(up), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        raise SystemExit("Upstream revision differs")
    base = REPO / "work/base/Llama-3.1-8B"
    manifest = json.loads((base / "verified-manifest.json").read_text())
    assert manifest["all_files_verified_against_official"]
    bundle = RUN / "bundle"
    bundle.mkdir(parents=True)
    for folder in ["src", "scripts", "configs", "patches"]:
        shutil.copytree(REPO / folder, bundle / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["run.sh", "setup.sh", "requirements.txt", "README.md"]:
        shutil.copy2(REPO / name, bundle / name)
    shutil.copytree(up, bundle / "third_party/TSR", ignore=shutil.ignore_patterns(".git", "data", "__pycache__", "*.pyc", "*.egg-info"))
    archive = REPO / "experiments/tsr-singlecopy-20261104-best"
    archived = json.loads((archive / "manifest.json").read_text())
    dataset = []
    for e in archived["files"]:
        if not e["path"].startswith("inputs/"):
            continue
        rel = e["path"][len("inputs/"):]
        src = up / "data" / rel
        assert sha(src) == e["sha256"], rel
        dst = bundle / "third_party/TSR/data" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        try:
            records = json.loads(src.read_text())
            count = len(records)
        except json.JSONDecodeError:
            count = sum(1 for line in src.open() if line.strip())
        dataset.append({"path":rel, "sha256":e["sha256"], "bytes":e["bytes"], "rows":count})
    (bundle / ".venv").symlink_to(REPO / ".venv", target_is_directory=True)
    config = f'''GPUS={','.join(map(str, range(options.gpus)))}
WORK_DIR={RUN}/trial
BASE_MODEL={base}
JUDGE_MODEL=gpt-4o-2024-08-06
JUDGE_BASE_URL=${{JUDGE_BASE_URL:-https://api.openai.com/v1}}
JUDGE_API_KEY=${{JUDGE_API_KEY:-}}
HF_TOKEN=${{HF_TOKEN:-}}
'''
    (bundle / "config.env").write_text(config)
    (RUN / "inputs").mkdir()
    shutil.copy2(archive / "alpaca_eval/reference_outputs.json", RUN / "inputs/alpaca_reference.json")
    smoke = RUN / "trial/smoke/data/smoke-pairs"
    smoke.mkdir(parents=True)
    with gzip.open(archive / "data/iter1/D1/train.json.gz", "rt") as f:
        rows = [next(f) for _ in range(64)]
    (smoke / "train.json").write_text("".join(rows))
    dump(RUN / "base-model-manifest.json", manifest)
    dump(RUN / "dataset-manifest.json", dataset)
    dump(RUN / "source-sha256.json", {str(p.relative_to(bundle)):sha(p) for p in bundle.rglob("*") if p.is_file() and not p.is_symlink() and ".venv" not in p.parts})
    dump(RUN / "run-plan.json", {
        "run_dir":str(RUN), "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpus":options.gpus, "node":options.node, "image":IMAGE, "upstream_commit":UPSTREAM,
        "base_model":"meta-llama/Llama-3.1-8B", "base_directory":str(base),
        "initialization":"fresh Base -> SFT M0 -> Mf0 -> M1 -> Mf1 -> M2_future_only",
        **{key:resources[key] for key in ["global_batch", "sft_micro_batch", "sft_gradient_accumulation", "dpo_micro_batch", "dpo_gradient_accumulation"]},
        "sft_epochs":3, "sft_lr":2e-6, "dpo_epochs":1, "dpo_lr":5e-7,
        "beta":0.1, "warmup_ratio":0.1, "train_seed":42,
        "second_round_chosen":"Mf1 only, first highest-scored answer",
        "second_round_rejected":"minimum among M1/M0; ties prefer M1",
        "second_round_filter":"chosen_score > rejected_score",
        "generation":{"K":7,"temperature":1.0,"top_p":1.0,"max_tokens":1024,"judge_count":1,
                      "data_parallel_shards":options.gpus,
                      "note":"Changing inference sharding can change sampled answers; no bitwise equivalence claimed"},
        "eval":{"questions":805,"judge":"gpt-4o-2024-08-06","reference":"gpt4_1106_preview",
                "model_outputs":["M0","M0_seed1","M0_seed2","Mf0","M1","Mf1","M2_future_only"],
                "scoring":"pending user-supplied API; GPU job exits after generation"},
        "retention":"metrics/configs/final stage models/canonical DPO datasets; generated candidate and judgement caches removed after consumption; no optimizer checkpoints or token dumps",
        "gpu_validation":f"2 SFT updates + 2 DPO updates + {7*options.gpus} generations/self-judgements in separate smoke workdir",
    })
    if options.sft_source_run:
        from sn5_driver import inherit_completed_sft
        inherit_completed_sft(RUN, options.sft_source_run)
    job = {
      "apiVersion":"batch/v1", "kind":"Job",
      "metadata":{"name":"zaninzeng-" + options.run_name, "namespace":"sn5-system-intern",
                  "labels":{"owner":"zaninzeng","app":"tsr-reproduction","toolkits/creator":"zaninzeng"}},
      "spec":{"backoffLimit":0,"activeDeadlineSeconds":172800,"ttlSecondsAfterFinished":604800,
        "template":{"metadata":{"labels":{"owner":"zaninzeng","app":"tsr-reproduction","run":options.run_name}},
          "spec":{"restartPolicy":"Never","automountServiceAccountToken":False,
            "affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":resources["node_expressions"]}]}}},
            "tolerations":[{"key":key,"operator":"Exists","effect":"NoSchedule"} for key in ["nvidia.com/gpu","role","glm5sft"]],
            "containers":[{"name":"train","image":IMAGE,"command":["bash",str(bundle/"scripts/sn5_bootstrap.sh"),str(RUN)],
               "resources":{"requests":resources["requests"], "limits":resources["limits"]},
               "volumeMounts":[{"name":"fsx","mountPath":"/mnt/sn-007"},{"name":"shm","mountPath":"/dev/shm"}]}],
            "volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"youtu-sn2-007"}},
                       {"name":"shm","emptyDir":{"medium":"Memory","sizeLimit":"64Gi"}}]}}}}
    dump(RUN / "job.json", job)
    (RUN / "README.md").write_text(f"# TSR future-only, {options.gpus} H200\n\nNode constraint: `{options.node}`.\n\n"
        "TSR chain with only iter1 D2 changed. Scientific configuration follows the archived single-copy experiment. "
        f"Global batch remains 32. SFT: {options.gpus} x 2 x {resources['sft_gradient_accumulation']}; "
        f"DPO: {options.gpus} x 1 x {resources['dpo_gradient_accumulation']}. "
        "When sft-provenance.json exists, M0 is reused from a fully finished prior SFT, without retraining or mid-stage resume. "
        f"Original candidate generation and self-judging execute as {options.gpus} independent shards; "
        "sharding can change samples but K, temperature, top_p and output limits stay the same. "
        "Each final stage model is saved once; no optimizer-state checkpoints. Consumed raw generation caches are removed.\n\n"
        "Job success covers GPU training and generation; the full scientific evaluation remains pending until original GPT-4o credentials arrive. "
        "Do not call pending scoring a completed replication.\n")
    print(RUN)


if __name__ == "__main__":
    main()
