"""One explicitly owned, synchronous TSR run; stages are original reproduction helpers."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_summary(work, name, *, write_metrics=True):
    model = work / "models" / name
    if not (model / ".done").is_file():
        raise RuntimeError(f"Model completion marker missing: {name}")
    state = json.loads((model / "trainer_state.json").read_text())
    if state["global_step"] <= 0:
        raise RuntimeError(f"No optimizer update: {name}")
    for row in state.get("log_history", []):
        for key, val in row.items():
            if isinstance(val, float) and not math.isfinite(val):
                raise RuntimeError(f"Non-finite {name}/{key}")
    indices = list(model.glob("*.index.json"))
    weights = []
    for idx in indices:
        obj = json.loads(idx.read_text())
        if "weight_map" in obj:
            weights.extend(model / s for s in set(obj["weight_map"].values()))
    if not weights:
        weights = list(model.glob("model.safetensors")) + list(model.glob("pytorch_model.bin"))
    if not weights or any(not f.is_file() or f.stat().st_size == 0 for f in weights):
        raise RuntimeError(f"Incomplete model weights: {name}")
    result = dict(model=name, global_step=state["global_step"], epoch=state.get("epoch"),
                  log_history=state.get("log_history", []), weight_bytes=sum(p.stat().st_size for p in weights))
    for filename in ["train_results.json", "all_results.json"]:
        path = model / filename
        if path.exists():
            result[filename] = json.loads(path.read_text())
    if write_metrics:
        atomic(work / "metrics" / f"{name}.json", result)
    return result


def inherit_completed_sft(run, source):
    """Reuse only a final, native saved M0; do not treat weights as optimizer recovery."""
    source = source.resolve()
    run = run.resolve()
    if source == run or source.parent != run.parent:
        raise RuntimeError("SFT source must be a different run in the same owned runs directory")
    old = json.loads((source / "run-plan.json").read_text())
    new = json.loads((run / "run-plan.json").read_text())
    for key in ["upstream_commit", "base_model", "base_directory", "global_batch", "sft_micro_batch",
                "sft_epochs", "sft_lr", "warmup_ratio", "train_seed"]:
        if old[key] != new[key]:
            raise RuntimeError(f"SFT source configuration differs: {key}")
    for manifest in ["dataset-manifest.json", "base-model-manifest.json"]:
        if json.loads((source / manifest).read_text()) != json.loads((run / manifest).read_text()):
            raise RuntimeError(f"SFT source differs: {manifest}")
    # Preserve the actual training code and input template, not just matching plan labels.
    for rel in ["third_party/TSR/alignment-handbook/scripts/run_sft.py",
                "third_party/TSR/settings/sft/config/config.yaml"]:
        if sha256(source / "bundle" / rel) != sha256(run / "bundle" / rel):
            raise RuntimeError(f"SFT source code/template differs: {rel}")
    summary = model_summary(source / "trial", "M0", write_metrics=False)
    import yaml
    cfg = yaml.safe_load((source / "trial/configs/M0.yaml").read_text())
    if cfg["max_steps"] != -1 or cfg["num_train_epochs"] != 3 or summary.get("epoch", 0) < 2.9:
        raise RuntimeError("SFT source is a smoke test or unfinished training")
    model = source / "trial/models/M0"
    dst = run / "trial/models/M0"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(model, target_is_directory=True)
    provenance = dict(source_run=str(source), source_model=str(model), source_gpus=old["gpus"],
                      optimizer_resume=False, training_repeated=False, global_batch=32,
                      native_global_step=summary["global_step"],
                      source_config_sha256=sha256(source / "trial/configs/M0.yaml"),
                      files={p.name:dict(bytes=p.stat().st_size, sha256=sha256(p))
                             for p in model.iterdir() if p.is_file()})
    atomic(run / "sft-provenance.json", provenance)
    atomic(run / "sft-complete.json", dict(models=[summary], inherited_from=str(source), completed_unix=time.time()))
    new["initialization"] = "completed prior SFT M0 -> Mf0 -> M1 -> Mf1 -> " + "M2_" + new.get("iter1_variant", "future_only")
    new["sft_source_run"] = str(source)
    new["sft_source_gpus"] = old["gpus"]
    atomic(run / "run-plan.json", new)
    model_summary(run / "trial", "M0")


def discard_consumed(work, phase, models):
    """Only remove named generated caches owned by this run, after native training returned."""
    summaries = []
    phase_dir = work / "data" / phase
    for path in sorted(phase_dir.glob("*.json")):
        if not (path.name.endswith(("_response.json", "_judgement.json", ".official.json"))):
            continue
        record = dict(path=str(path.relative_to(work)), bytes=path.stat().st_size, sha256=sha256(path))
        if path.name.endswith("_judgement.json"):
            from collections import Counter
            rows = json.loads(path.read_text())
            record["prompts"] = len(rows)
            record["scores"] = dict(Counter(str(s) for row in rows for s in row["scores"]))
        summaries.append(record)
    if summaries:
        atomic(work / "metrics" / f"{phase}-consumed-caches.json", summaries)
    # Make the small manifest durable before unlinking the consumed generation/judgement cache.
    for row in summaries:
        (work / row["path"]).unlink()
    for name in models:
        # The canonical D1/D2 datasets remain; this directory is a duplicate loader snapshot.
        snapshot = work / "configs" / f"{name}.data"
        if snapshot.is_dir():
            if any(p.name != "train.json" or not p.is_file() for p in snapshot.iterdir()):
                raise RuntimeError(f"Unexpected snapshot content: {snapshot}")
            shutil.rmtree(snapshot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    bundle = run / "bundle"
    work = run / "trial"
    owner = json.loads((run / "run-plan.json").read_text())
    ngpu = owner["gpus"]
    variant = owner.get("iter1_variant", "future_only")
    if variant not in ("current_tie", "future_only"):
        raise RuntimeError("Unsupported frozen iter1 variant")
    final_model = f"M2_{variant}"
    if owner["run_dir"] != str(run) or ngpu not in (2, 4, 8):
        raise RuntimeError("Run ownership/resources differ")
    if owner["sft_micro_batch"] * owner["sft_gradient_accumulation"] * ngpu != 32:
        raise RuntimeError("SFT effective batch changed")
    if owner["dpo_micro_batch"] * owner["dpo_gradient_accumulation"] * ngpu != 32:
        raise RuntimeError("DPO effective batch changed")
    env = dict(os.environ, TSR_BUNDLE=str(bundle), PYTHONDONTWRITEBYTECODE="1")
    env["TSR_DIR"] = str(bundle / "third_party/TSR")
    started = time.time()
    def status(phase, **details):
        atomic(run / "status.json", dict(phase=phase, updated_unix=time.time(), elapsed_s=time.time()-started, **details))
    def shell(script, *, smoke=False):
        local = dict(env, SMOKE="1" if smoke else "0")
        subprocess.run(["bash", "-euc", '. "$TSR_BUNDLE/scripts/lib.sh"; . "$TSR_BUNDLE/scripts/stages.sh"; load_config; ' + script],
                       env=local, check=True, cwd=bundle)
    try:
        import torch
        if torch.cuda.device_count() != ngpu:
            raise RuntimeError(f"Expected exactly {ngpu} visible GPUs, got {torch.cuda.device_count()}")
        atomic(run / "gpu-validated.json", {"count":ngpu, "devices":[torch.cuda.get_device_name(i) for i in range(ngpu)],
                                           "torch":torch.__version__, "cuda":torch.version.cuda})
        if not (run / "GPU_SMOKE_PASSED.json").exists():
            status("gpu_smoke")
            shell(f'LIMIT={ngpu}; sft; dpo plumbing "$M0" "$WORK_DIR/data/smoke-pairs"; '
                  'gen "$MODELS/plumbing" "$(QA 1)" "$DATA/smoke_response.json"; '
                  'judge "$MODELS/plumbing" "$DATA/smoke_response.json" "$DATA/smoke_judgement.json"', smoke=True)
            smoke = work / "smoke"
            checked = [model_summary(smoke, n) for n in ["M0", "plumbing"]]
            assert all(x["global_step"] == 2 for x in checked)
            rows = json.loads((smoke / "data/smoke_judgement.json").read_text())
            assert len(rows) == ngpu and all(len(x["responses"]) == 7 and len(x["scores"]) == 7 for x in rows)
            atomic(run / "GPU_SMOKE_PASSED.json", dict(models=checked, prompts=ngpu, candidates=7*ngpu, completed_unix=time.time()))
            # The formal work directory and official Base are untouched by smoke optimization.
            shutil.rmtree(smoke / "models")
            for p in (smoke / "data").glob("smoke_*.json"):
                p.unlink()
        for label, command, models, phase in [
            ("sft", "stage_sft", ["M0"], None),
            ("iter0", "stage_iter0", ["Mf0", "M1"], "iter0"),
            (f"iter1_{variant}", f"stage_iter1 {variant}", ["Mf1", final_model], "iter1"),
        ]:
            marker = run / f"{label}-complete.json"
            if marker.exists():
                for name in models:
                    model_summary(work, name)
                if phase:
                    discard_consumed(work, phase, models)
                continue
            status(label)
            shell(command)
            results = [model_summary(work, name) for name in models]
            atomic(marker, dict(models=results, completed_unix=time.time()))
            if phase:
                discard_consumed(work, phase, models)
        stats = json.loads((work / f"data/iter1_{variant}/D2/stats.json").read_text())
        assert stats["strict_score_gap"] and stats["gap_min"] > 0 and stats["retained_tie_pairs"] == 0
        if variant == "future_only":
            assert stats["chosen_policy"] == "future_only" and stats.get("retained_chosen_from_Mi", 0) == 0
        else:
            assert stats["chosen_policy"] == "current_future_best" and stats["chosen_tie_break"] == "current"
            assert stats["rejected_tie_break"] == "current"
        for d1 in ["iter0/D1", "iter1/D1"]:
            d1_stats = json.loads((work / "data" / d1 / "stats.json").read_text())
            assert d1_stats["strict_score_gap"] and d1_stats["retained_tie_pairs"] == 0 and d1_stats["gap_min"] > 0
        status("alpaca_generation")
        reference = run / "inputs/alpaca_reference.json"
        reference_rows = json.loads(reference.read_text())
        # Each evaluation retains its single-GPU batch and seed. Parallelize independent evaluations.
        evaluations = [("M0", "M0", 0), ("Mf0", "Mf0", 0), ("M0_seed1", "M0", 1),
                       ("M1", "M1", 0), ("M0_seed2", "M0", 2), ("Mf1", "Mf1", 0),
                       (final_model, final_model, 0)]
        def generate(item, gpu):
            name, model, seed = item
            out = work / "alpaca_eval" / name / "model_outputs.json"
            if not out.exists():
                local = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
                log = work / "logs" / f"alpaca_gen_{name}.log"
                with log.open("w") as f:
                    subprocess.run([sys.executable, str(bundle / "src/gen_alpaca.py"), "--model", str(work / "models" / model),
                                    "--out", str(out), "--seed", str(seed), "--reference_outputs", str(reference)],
                                   env=local, stdout=f, stderr=subprocess.STDOUT, check=True)
            rows = json.loads(out.read_text())
            if len(rows) != 805 or [x["instruction"] for x in rows] != [x["instruction"] for x in reference_rows]:
                raise RuntimeError(f"Evaluation incomplete or reordered: {name}")
        for offset in range(0, len(evaluations), ngpu):
            with concurrent.futures.ThreadPoolExecutor(max_workers=ngpu) as pool:
                futures = [pool.submit(generate, item, gpu) for gpu, item in enumerate(evaluations[offset:offset+ngpu])]
                for f in futures:
                    f.result()
        atomic(run / "TRAINING_AND_GENERATION_COMPLETED.json", dict(completed_unix=time.time(),
               evaluations=[x[0] for x in evaluations], questions_each=805, m2_pairs=stats["n_pairs"],
               scoring="handled by separate CPU worker; see SCORING_CONFIGURED.json and EVALUATION_COMPLETED.json"))
        status("gpu_work_completed", gpu_work_complete=True, release_gpu=True)
    except BaseException as e:
        status("failed", error_type=type(e).__name__, error=str(e))
        atomic(run / "FAILED.json", dict(type=type(e).__name__, error=str(e), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
