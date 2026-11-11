"""Generate complete planned evaluation outputs early on one temporary GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def check(path, reference):
    rows = json.loads(path.read_text())
    if len(rows) != 805 or [r["instruction"] for r in rows] != [r["instruction"] for r in reference]:
        raise RuntimeError("Incomplete or reordered early evaluation output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["M0", "Mf0"],
                    choices=["M0", "Mf0", "M1", "Mf1", "M2_future_only"])
    args = ap.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "run-plan.json").read_text())
    if plan["run_dir"] != str(run):
        raise RuntimeError("Run ownership differs")
    if len(args.models) != len(set(args.models)) or any(n not in plan["eval"]["model_outputs"] for n in args.models):
        raise RuntimeError("Unexpected or duplicate evaluation models")
    state_name = "early-evaluation" if args.models == ["M0", "Mf0"] else "early-evaluation-" + "-".join(args.models).lower()
    status = run / state_name / "status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    lock = (status.parent / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = run / "inputs/alpaca_reference.json"
    reference_rows = json.loads(reference.read_text())
    env = dict(os.environ, TSR_DIR=str(run / "bundle/third_party/TSR"),
               CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
    completed = []
    delegated = []
    try:
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Early generation requires exactly one GPU")
        for name in args.models:
            model = run / "trial/models" / name
            if not (model / ".done").is_file():
                raise RuntimeError(f"Model has not completed training: {name}")
            directory = run / "trial/alpaca_eval" / name
            directory.mkdir(parents=True, exist_ok=True)
            out = directory / "model_outputs.json"
            if out.exists():
                check(out, reference_rows)
                completed.append(name)
                continue
            atomic(status, dict(phase="generating", model=name, completed=completed, questions=805, seed=0))
            temporary = directory / "model_outputs.early.json"
            with (status.parent / f"generate_{name}.log").open("w") as log:
                subprocess.run([sys.executable, str(run / "bundle/src/gen_alpaca.py"),
                                "--model", str(model), "--out", str(temporary), "--seed", "0",
                                "--reference_outputs", str(reference)], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=1200)
            check(temporary, reference_rows)
            diag = directory / "model_outputs.early.diag.json"
            gpu_status = json.loads((run / "status.json").read_text())
            if out.exists():
                # Never replace an output that the final driver has already published.
                check(out, reference_rows)
                temporary.unlink()
                diag.unlink()
            elif gpu_status["phase"] == "alpaca_generation":
                # The final driver does not take our lock. If its generation has started,
                # let it publish the canonical file instead of racing two writers.
                temporary.unlink()
                diag.unlink()
                delegated.append(name)
                continue
            else:
                diag.replace(directory / "model_outputs.diag.json")
                temporary.replace(out)
            completed.append(name)
            print(json.dumps(dict(model=name, questions=805, seed=0, published=True)), flush=True)
        atomic(status, dict(phase="completed", models=completed, delegated_to_final_driver=delegated, questions_each=805, seed=0,
                            completed_unix=time.time(), release_gpu=True))
    except Exception as e:
        atomic(status, dict(phase="failed", completed=completed, error_type=type(e).__name__))
        raise


if __name__ == "__main__":
    main()
