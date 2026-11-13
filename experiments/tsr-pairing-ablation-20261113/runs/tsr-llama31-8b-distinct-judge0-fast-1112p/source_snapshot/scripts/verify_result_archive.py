#!/usr/bin/env python3
"""Verify all archived hashes, source-file copies, and all seven 805-question evaluations offline."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    args = ap.parse_args()
    root = args.archive.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest["files"]:
        p = root / row["path"]
        if root not in p.resolve().parents or p.is_symlink():
            raise SystemExit("Archive paths must be real files inside the archive")
        data = p.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"File hash/size mismatch: {row['path']}")
    for row in json.loads((root / "provenance/source_files.json").read_text()):
        p = root / row["path"]
        data = gzip.decompress(p.read_bytes()) if row["gzip"] else p.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["uncompressed_sha256"]:
            raise SystemExit(f"Original source hash differs: {row['path']}")
    ref = json.loads((root / "provenance/reference.json").read_text())
    source_reference = root / ref["archive"]
    if hashlib.sha256(source_reference.read_bytes()).hexdigest() != ref["sha256"]:
        raise SystemExit("Frozen reference hash differs")
    reference = json.loads(source_reference.read_text())
    results = json.loads((root / "metrics/results.json").read_text())["evaluations"]
    if len(results) != 7:
        raise SystemExit("Expected seven evaluations")
    for row in results:
        d = root / "evaluation" / row["model"]
        outputs = json.loads((d / "model_outputs.json").read_text())
        annotations = json.loads(gzip.decompress((d / "annotations.json.gz").read_bytes()))
        if len(outputs) != 805 or len(annotations) != 805:
            raise SystemExit("Incomplete evaluation")
        if [x["instruction"] for x in outputs] != [x["instruction"] for x in reference]:
            raise SystemExit("Evaluation order differs")
        if not all(math.isfinite(x["preference"]) and 1 <= x["preference"] <= 2 for x in annotations):
            raise SystemExit("Invalid weighted preference")
    print(f"Verified {len(manifest['files'])} files and seven complete 805-question evaluations")


if __name__ == "__main__":
    main()
