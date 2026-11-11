"""Score only this run's seven planned AlpacaEval outputs, without allocating a GPU."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_ready(path, reference):
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False  # The GPU writer may not have finished writing this JSON yet.
    if len(rows) != 805:
        return False
    if [r.get("instruction") for r in rows] != [r["instruction"] for r in reference]:
        raise RuntimeError("Evaluation question order differs from the frozen reference")
    if not all(isinstance(r.get("output"), str) for r in rows):
        raise RuntimeError("Evaluation output is not text")
    return True


def validate_result(directory, name):
    with (directory / "leaderboard.csv").open() as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if r.get("") == name), None)
    if row is None or int(float(row["n_total"])) != 805:
        raise RuntimeError(f"Incomplete leaderboard: {name}")
    metrics = {k:float(row[k]) for k in ["win_rate", "length_controlled_winrate", "avg_length"]}
    if not all(math.isfinite(x) for x in metrics.values()):
        raise RuntimeError(f"Non-finite metrics: {name}")
    annotations = json.loads((directory / "annotations.json").read_text())
    if len(annotations) != 805:
        raise RuntimeError(f"Incomplete annotations: {name}")
    if not all(isinstance(a.get("preference"), (int, float)) and math.isfinite(a["preference"])
               and 1 <= a["preference"] <= 2 for a in annotations):
        raise RuntimeError(f"Invalid weighted preference: {name}")
    return dict(metrics, n_total=805)


def call(script, arguments, env, log, timeout=7200):
    """Bound scoring time and redact credentials from third-party library errors."""
    proc = subprocess.Popen([sys.executable, str(script), *map(str, arguments)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
        log.write_text(output.replace(env["JUDGE_API_KEY"], "[REDACTED]"))
        raise RuntimeError("Scoring subprocess timed out")
    log.write_text(output.replace(env["JUDGE_API_KEY"], "[REDACTED]"))
    if proc.returncode:
        raise RuntimeError(f"Scoring subprocess failed ({proc.returncode}); inspect {log.name}")


def retryable_log(text):
    # Some compatible proxies report an expired *upstream* key as HTTP429 even
    # when the next request succeeds. Retry finitely; never retry HTTP401/403.
    return any(s in text for s in [
        "Error code: 429", "Error code: 500", "Error code: 502", "Error code: 503", "Error code: 504",
        "openai.APIConnectionError", "openai.APITimeoutError",
    ])


def score_with_retries(script, arguments, env, state_dir, name, completed):
    stamp = time.time_ns()
    for attempt in range(1, 4):
        log = state_dir / f"{name}-{stamp}-attempt{attempt}.log"
        atomic(state_dir / "status.json", dict(phase="scoring", model=name, attempt=attempt,
                                               completed=list(completed)))
        try:
            call(script, arguments, env, log)
            return
        except RuntimeError:
            if attempt == 3 or not log.exists() or not retryable_log(log.read_text()):
                raise
            atomic(state_dir / "status.json", dict(phase="retry_wait", model=name, attempt=attempt,
                   completed=list(completed), reason="transient_api_error", keep_annotation_cache=True))
            time.sleep(30 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--credentials", type=Path, required=True)
    ap.add_argument("--timeout-hours", type=float, default=24)
    ap.add_argument("--poll-seconds", type=float, default=30)
    args = ap.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "run-plan.json").read_text())
    if plan["run_dir"] != str(run):
        raise RuntimeError("Run ownership differs")
    state_dir = run / "scoring"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    credentials = json.loads(args.credentials.read_text())
    if credentials["JUDGE_MODEL"] != plan["eval"]["judge"]:
        raise RuntimeError("Judge model differs from the frozen experiment")
    if args.credentials.stat().st_mode & 0o077:
        raise RuntimeError("Credentials must be readable only by their owner")
    env = dict(os.environ, **credentials, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OPENAI_LOG="warn", HF_HUB_DISABLE_XET="1")
    reference = run / "inputs/alpaca_reference.json"
    reference_rows = json.loads(reference.read_text())
    if len(reference_rows) != 805:
        raise RuntimeError("Frozen reference must contain 805 questions")
    names = plan["eval"]["model_outputs"]
    if len(names) != 7 or any(Path(n).name != n for n in names):
        raise RuntimeError("Unexpected evaluation model list")
    configured = dict(judge=credentials["JUDGE_MODEL"], base_url=credentials["JUDGE_BASE_URL"],
                      credential_file=str(args.credentials), gpu_required=False, names=names,
                      frozen_eval_sha256=digest(run / "bundle/src/eval_alpaca.py"))
    atomic(run / "SCORING_CONFIGURED.json", configured)
    completed = {}
    start = time.monotonic()
    try:
        while len(completed) < len(names):
            for name in names:
                if name in completed:
                    continue
                directory = run / "trial/alpaca_eval" / name
                outputs = directory / "model_outputs.json"
                if not input_ready(outputs, reference_rows):
                    continue
                marker = directory / "GPT4O_SCORE_COMPLETE.json"
                input_hash = digest(outputs)
                if marker.exists():
                    prior = json.loads(marker.read_text())
                    if prior["input_sha256"] != input_hash or prior["judge"] != credentials["JUDGE_MODEL"]:
                        raise RuntimeError(f"Existing scoring provenance differs: {name}")
                atomic(state_dir / "status.json", dict(phase="scoring", model=name, completed=list(completed)))
                if not marker.exists():
                    score_with_retries(run / "bundle/src/eval_alpaca.py",
                         ["--model_outputs", outputs, "--name", name, "--out_dir", directory,
                          "--reference_outputs", reference], env, state_dir, name, completed)
                metrics = validate_result(directory, name)
                result = dict(model=name, judge=credentials["JUDGE_MODEL"], base_url=credentials["JUDGE_BASE_URL"],
                              input_sha256=input_hash, reference_sha256=digest(reference), **metrics)
                atomic(marker, result)
                completed[name] = result
                atomic(state_dir / "results.json", completed)
                print(json.dumps(dict(model=name, **metrics)), flush=True)
            if len(completed) == len(names):
                break
            if (run / "FAILED.json").exists():
                raise RuntimeError("GPU training/generation failed before all evaluation outputs became available")
            if time.monotonic() - start > args.timeout_hours * 3600:
                raise RuntimeError("Timed out waiting for evaluation outputs")
            atomic(state_dir / "status.json", dict(phase="waiting_for_outputs", completed=list(completed),
                   waiting=[n for n in names if n not in completed], updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            time.sleep(args.poll_seconds)
        with (state_dir / "summary.csv").open("w") as f:
            fields = ["model", "win_rate", "length_controlled_winrate", "avg_length", "n_total"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(completed[n] for n in names)
        atomic(run / "EVALUATION_COMPLETED.json", dict(models=completed, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        atomic(state_dir / "status.json", dict(phase="completed", models=list(completed), gpu_required=False))
    except Exception as exc:
        # Never persist raw exception strings that might contain a vendor's credential echo.
        atomic(state_dir / "status.json", dict(phase="failed", error_type=type(exc).__name__, completed=list(completed)))
        raise SystemExit("Scoring failed; see the redacted per-model scoring logs") from None


if __name__ == "__main__":
    main()
