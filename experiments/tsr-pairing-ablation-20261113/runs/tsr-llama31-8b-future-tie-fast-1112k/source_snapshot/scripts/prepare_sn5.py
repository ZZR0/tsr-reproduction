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
RUN = REPO / "work/runs/tsr-llama31-8b-current-tie-1112b"
NODE = "hyperpod-i-03fcebc230d22fd76"
IMAGE = "pytorch/pytorch@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
UPSTREAM = "020266aa9ef3cc35bf81384c9b59f98cff62aa1c"


def resource_plan(gpus, node, dpo_micro_batch=1):
    if gpus not in (2, 4, 8, 16):
        raise ValueError("Supported GPU counts: 2, 4, 8, 16")
    if gpus > 8:
        raise ValueError("This launcher supports only single-node runs (at most 8 GPUs)")
    expressions = [
        {"key":"group", "operator":"In", "values":["shennong-5", "shennong-5-dev"]},
        {"key":"node.kubernetes.io/instance-type", "operator":"In", "values":["ml.p5e.48xlarge", "ml.p5en.48xlarge"]},
    ]
    if node != "any-sn5":
        expressions.append({"key":"kubernetes.io/hostname", "operator":"In", "values":[node]})
    if dpo_micro_batch not in (1, 2, 4) or 32 % (gpus * dpo_micro_batch):
        raise ValueError("DPO micro-batch must divide global batch 32")
    return {
        "gpus":gpus, "global_batch":32, "sft_micro_batch":2,
        "sft_gradient_accumulation":32 // (2 * gpus),
        "dpo_micro_batch":dpo_micro_batch, "dpo_gradient_accumulation":32 // (gpus * dpo_micro_batch),
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
    parser.add_argument("--run-name", default="tsr-llama31-8b-current-tie-1112b")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--variant", choices=["current_tie", "min_length_gap", "future_tie_unfiltered", "future_tie_filtered"], default="current_tie")
    parser.add_argument("--train-only", action="store_true", help="Train four DPO stages only; no evaluation generation or API scoring")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip standalone GPU smoke tests when explicitly requested")
    parser.add_argument("--node", default=NODE, help="Exact node, or any-sn5 (still constrained to SN5 H200)")
    parser.add_argument("--sft-source-run", type=Path, help="Reuse a completed M0 at a stage boundary; never resume partial weights")
    parser.add_argument("--performance-profile", choices=["reference", "h200-fast"], default="reference")
    parser.add_argument("--zero-stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--dpo-micro-batch", type=int, choices=[1, 2, 4])
    parser.add_argument("--reuse-iter0-source-run", type=Path, help="Reuse verified M0 candidate/judge data only; no DPO weights")
    options = parser.parse_args()
    fast = options.performance_profile == "h200-fast"
    zero = options.zero_stage if options.zero_stage is not None else (2 if fast else 3)
    micro = options.dpo_micro_batch if options.dpo_micro_batch is not None else (4 if fast else 1)
    resources = resource_plan(options.gpus, options.node, micro)
    if fast:
        resources["requests"].update(cpu=str(4 * options.gpus), memory=f"{64 * options.gpus}Gi")
    keep_ties = options.variant == "future_tie_unfiltered"
    prefer_future_ties = options.variant in ("future_tie_unfiltered", "future_tie_filtered")
    score_filter = "none; retain equal-score pairs" if keep_ties else "chosen_score > rejected_score"
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
TSR_PAIR_SELECTION={options.variant}
BASE_MODEL={base}
JUDGE_MODEL=gpt-4o-2024-08-06
JUDGE_BASE_URL=${{JUDGE_BASE_URL:-https://api.openai.com/v1}}
JUDGE_API_KEY=${{JUDGE_API_KEY:-}}
HF_TOKEN=${{HF_TOKEN:-}}
'''
    if fast:
        ds = {"train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto",
              "gradient_accumulation_steps": "auto", "gradient_clipping": "auto",
              "bf16": {"enabled": "auto"},
              "zero_optimization": {"stage": zero, "overlap_comm": False,
                                    "contiguous_gradients": True, "reduce_bucket_size": 50000000,
                                    "allgather_bucket_size": 50000000},
              "steps_per_print": 1000000, "wall_clock_breakdown": False}
        dump(bundle / "configs/deepspeed-fast.json", ds)
        import yaml
        accelerate_cfg = yaml.safe_load((up / "settings/tsr/config/deepspeed_zero3.yaml").read_text())
        accelerate_cfg["deepspeed_config"] = {"deepspeed_config_file": str(bundle / "configs/deepspeed-fast.json"),
                                               "zero3_init_flag": zero == 3}
        accelerate_cfg.pop("mixed_precision", None)  # BF16 belongs to the external DeepSpeed config
        accelerate_cfg["num_processes"] = options.gpus
        (bundle / "configs/accelerate-fast.yaml").write_text(yaml.safe_dump(accelerate_cfg))
        config += f"""TSR_PERFORMANCE_PROFILE=h200-fast
TSR_ZERO_STAGE={zero}
TSR_DPO_MICRO_BATCH={micro}
TSR_GRADIENT_CHECKPOINTING=0
TSR_ACCELERATE_CONFIG={bundle}/configs/accelerate-fast.yaml
TSR_VLLM_MAX_BATCHED_TOKENS=16384
TSR_VLLM_MAX_SEQS=512
TSR_FUSE_INFERENCE=1
"""
    (bundle / "config.env").write_text(config)
    (RUN / "inputs").mkdir()
    shutil.copy2(archive / "alpaca_eval/reference_outputs.json", RUN / "inputs/alpaca_reference.json")
    if not options.skip_smoke:
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
        "initialization":f"fresh Base -> SFT M0 -> Mf0 -> M1 -> Mf1 -> M2_{options.variant}",
        "iter1_variant":options.variant, "pair_selection":options.variant,
        "training_only":options.train_only, "skip_smoke":options.skip_smoke,
        "performance_profile":options.performance_profile,
        "performance":{"zero_stage":zero, "dpo_micro_batch":micro,
                       "dpo_gradient_accumulation":resources["dpo_gradient_accumulation"],
                       "gradient_checkpointing":not fast, "communication_overlap":False if fast else None, "attention":("flash_attention_2" if fast else "upstream default"),
                       "optimizer":"adamw_torch",
                       "drop_incomplete_final_global_batch":fast,
                       "memory_fallback":"halve micro-batch, preserve global batch 32; then enable checkpointing; restart stage from init" if fast else None,
                       "vllm_max_num_batched_tokens":16384 if fast else 512,
                       "vllm_max_num_seqs":512 if fast else 256,
                       "prefix_caching":fast, "fused_inference":fast,
                       "target_gpu_elapsed_seconds":3600 if fast else None,
                       "target_is_estimate_not_forced_early_stop":True},
        "first_round_chosen":("highest score among M0/Mf0; joint min length gap with lowest-score current/past pool" if options.variant == "min_length_gap" else "highest score among M0/Mf0; ties prefer Mf0" if prefer_future_ties else "highest score among M0/Mf0; ties prefer M0"),
        "first_round_filter":score_filter,
        "filter_equal_scores":not keep_ties,
        "pair_rule_scope":["iter0/D1 (Mf0)", "iter0/D2 (M1)", "iter1/D1 (Mf1)", f"iter1_{options.variant}/D2 (M2)"],
        **{key:resources[key] for key in ["global_batch", "sft_micro_batch", "sft_gradient_accumulation", "dpo_micro_batch", "dpo_gradient_accumulation"]},
        "sft_epochs":3, "sft_lr":2e-6, "dpo_epochs":1, "dpo_lr":5e-7,
        "beta":0.1, "warmup_ratio":0.1, "train_seed":42,
        "future_model_pairs":{"models":["Mf0", "Mf1"], "filter":score_filter,
                              "chosen":"current best only; never replaced by past",
                              "rejected":("minimum among current/past; joint min length gap; no source priority" if options.variant == "min_length_gap" else "minimum among current/past; ties prefer current")},
        "second_round_chosen":("highest score pool across M1/Mf1; joint min length gap; no source priority" if options.variant == "min_length_gap" else "highest score among M1/Mf1; ties prefer Mf1" if prefer_future_ties else "highest score among M1/Mf1; ties prefer M1; future only if strictly higher"),
        "second_round_rejected":("lowest score pool across M1/M0; joint min length gap; no source priority" if options.variant == "min_length_gap" else "minimum among M1/M0; ties prefer M1"),
        "second_round_filter":score_filter,
        "generation":{"K":7,"temperature":1.0,"top_p":1.0,"max_tokens":1024,"judge_count":1,
                      "data_parallel_shards":options.gpus,
                      "note":"Changing inference sharding can change sampled answers; no bitwise equivalence claimed"},
        "eval":{"enabled":not options.train_only,"questions":805,"judge":"gpt-4o-2024-08-06","reference":"gpt4_1106_preview",
                "model_outputs":["M0","M0_seed1","M0_seed2","Mf0","M1","Mf1",f"M2_{options.variant}"],
                "scoring":("disabled by user; no API scoring job" if options.train_only else "separate CPU worker using configured GPT-4o service; GPU job exits after generation")},
        "retention":"metrics/configs/final stage models/canonical DPO datasets; generated candidate and judgement caches removed after consumption; no optimizer checkpoints or token dumps",
        "gpu_validation":("standalone smoke skipped at user request; actual training retains device/config/data checks" if options.skip_smoke else f"2 SFT updates + 2 DPO updates + {7*options.gpus} generations/self-judgements in separate smoke workdir"),
    })
    if options.sft_source_run:
        from sn5_driver import inherit_completed_sft
        inherit_completed_sft(RUN, options.sft_source_run)
    if options.reuse_iter0_source_run:
        from reuse_iter0 import inherit_iter0_data
        inherit_iter0_data(RUN, options.reuse_iter0_source_run)
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
    (RUN / "README.md").write_text(f"# TSR {options.variant}, {options.gpus} H200\n\nNode constraint: `{options.node}`.\n\n"
        f"TSR pair-selection variant: {options.variant}. All D1/D2 filtering: {score_filter}. "
        + ("Chosen ties prefer future; rejected ties prefer current; no length selection. " if prefer_future_ties else "") +
        "For min_length_gap, source ties are resolved jointly by minimum absolute character length difference, with no model-source priority. Other scientific settings follow the archived single-copy experiment. "
        f"Global batch remains 32. SFT: {options.gpus} x 2 x {resources['sft_gradient_accumulation']}; "
        f"DPO: {options.gpus} x {micro} x {resources['dpo_gradient_accumulation']}. "
        "When sft-provenance.json exists, M0 is reused from a fully finished prior SFT, without retraining or mid-stage resume. "
        f"Original candidate generation and self-judging execute as {options.gpus} independent shards; "
        "sharding, scheduling and fused calls can change sampled answers; K, temperature, top_p and output limits stay the same. "
        f"Performance profile: {options.performance_profile}; ZeRO-{zero}; DPO micro-batch {micro}. "
        "Each final stage model is saved once; no optimizer-state checkpoints. Consumed raw generation caches are removed.\n\n"
        f"Training-only: {options.train_only}; skip standalone smoke: {options.skip_smoke}. "
        "When training-only is true, no evaluation answers are generated and no API scoring job is launched. "
        "Otherwise job success covers GPU training and evaluation generation, with separate CPU scoring; "
        "only EVALUATION_COMPLETED.json confirms that all seven evaluations finished. "
        "Do not call pending scoring a completed replication.\n")
    print(RUN)


if __name__ == "__main__":
    main()
