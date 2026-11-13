#!/usr/bin/env python3
"""Export final TSR evaluation artifacts and compact training evidence; never export credentials/caches."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil

STAGES = ["M0", "Mf0", "M1", "Mf1", "M2_future_only"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit("Archive destination already exists; refusing to overwrite")
    plan = json.loads((run / "run-plan.json").read_text())
    if not (run / "EVALUATION_COMPLETED.json").exists():
        raise SystemExit("All seven evaluations must finish before exporting a complete archive")
    reference = json.loads((run / "inputs/alpaca_reference.json").read_text())
    rows, completed = [], {}
    for name in plan["eval"]["model_outputs"]:
        root = run / "trial/alpaca_eval" / name
        outputs = json.loads((root / "model_outputs.json").read_text())
        annotations = json.loads((root / "annotations.json").read_text())
        marker = json.loads((root / "GPT4O_SCORE_COMPLETE.json").read_text())
        if len(outputs) != 805 or len(annotations) != 805:
            raise SystemExit(f"Incomplete evaluation: {name}")
        if [r["instruction"] for r in outputs] != [r["instruction"] for r in reference]:
            raise SystemExit(f"Question order differs: {name}")
        if marker["input_sha256"] != sha(root / "model_outputs.json"):
            raise SystemExit(f"Scored output hash differs: {name}")
        if not all(isinstance(a.get("preference"), (int, float)) and math.isfinite(a["preference"])
                   and 1 <= a["preference"] <= 2 for a in annotations):
            raise SystemExit(f"Invalid preferences: {name}")
        with (root / "leaderboard.csv").open() as f:
            row = next(r for r in csv.DictReader(f) if r[""] == name)
        diag = json.loads((root / "model_outputs.diag.json").read_text())
        result = dict(model=name, seed=int(diag["seed"]), n=805,
                      lc=float(row["length_controlled_winrate"]), soft_win=float(row["win_rate"]),
                      discrete_win=float(row["discrete_win_rate"]), average_chars=float(row["avg_length"]),
                      truncation_rate=diag.get("truncated_rate", 0),
                      judge=marker["judge"], reference_sha256=marker["reference_sha256"])
        rows.append(result)
        completed[name] = {k:v for k,v in marker.items() if k != "base_url"}
    out.mkdir(parents=True)
    source_files = []
    def copy(source, relative, compress=False):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with source.open("rb") as src, target.open("wb") as dst:
                with gzip.GzipFile(filename="", fileobj=dst, mode="wb", mtime=0) as z:
                    shutil.copyfileobj(src, z)
        else:
            shutil.copy2(source, target)
        source_files.append(dict(path=relative, uncompressed_sha256=sha(source), gzip=compress))
    for name in plan["eval"]["model_outputs"]:
        root = run / "trial/alpaca_eval" / name
        for filename in ["model_outputs.json", "model_outputs.diag.json", "leaderboard.csv"]:
            copy(root / filename, f"evaluation/{name}/{filename}")
        copy(root / "annotations.json", f"evaluation/{name}/annotations.json.gz", compress=True)
        # The annotator config contains model/template/decoder settings, never API credentials.
        copy(root / "annotator/configs.yaml", f"evaluation/{name}/annotator.yaml")
    sft_source = Path(plan.get("sft_source_run", str(run)))
    for name in STAGES:
        model = run / "trial/models" / name
        cfg_root = sft_source if name == "M0" else run
        copy(cfg_root / f"trial/configs/{name}.yaml", f"configs/{name}.yaml")
        for filename in ["trainer_state.json", "train_results.json"]:
            copy(model / filename, f"training/{name}/{filename}")
    pairs = {}
    for path in sorted((run / "trial/data").glob("*/D*/stats.json")):
        key = str(path.parent.relative_to(run / "trial/data"))
        data = path.parent / "train.json"
        pairs[key] = dict(statistics=json.loads(path.read_text()), train_json_sha256=sha(data), bytes=data.stat().st_size)
    write_json(out / "metrics/pair_statistics.json", pairs)
    for filename in ["run-plan.json", "dataset-manifest.json", "base-model-manifest.json",
                     "source-sha256.json", "sft-provenance.json", "gpu-validated.json", "GPU_SMOKE_PASSED.json"]:
        copy(run / filename, "provenance/" + filename)
    copy(run / "bootstrap/pip-freeze.txt", "provenance/pip-freeze.txt")
    write_json(out / "provenance/scoring_completed.json", completed)
    write_json(out / "provenance/reference.json", dict(
        archive="../tsr-singlecopy-20261104-best/alpaca_eval/reference_outputs.json",
        sha256=sha(run / "inputs/alpaca_reference.json"), questions=805))
    baseline = [r for r in rows if r["model"] in ["M0", "M0_seed1", "M0_seed2"]]
    mean = {k:sum(r[k] for r in baseline)/len(baseline) for k in ["lc", "soft_win", "discrete_win", "average_chars"]}
    write_json(out / "metrics/results.json", dict(evaluations=rows, m0_three_generation_seed_mean=mean,
               notes=["Three generation seeds of one SFT model, not three independent training seeds.",
                      "LC and soft Win use the same frozen reference and scoring code for all models.",
                      "Paper Win soft/hard implementation has not been verified; numeric comparisons are descriptive.",
                      "M2 future-only changes only second-round D2, not iter0 or Mf1 D1."]))
    with (out / "metrics/results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "provenance/source_files.json", source_files)
    print(f"Exported {len(rows)} complete 805-question evaluations to {out}")


if __name__ == "__main__":
    main()
