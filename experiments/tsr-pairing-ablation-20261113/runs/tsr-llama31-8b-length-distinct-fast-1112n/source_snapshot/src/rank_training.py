"""Assign a private datasets cache before importing the unchanged native training entry."""
import os
import hashlib
from pathlib import Path
import runpy
import shutil
import sys


def tensorboard_directory(config):
    root = os.environ.get("TSR_TENSORBOARD_LOCAL_ROOT")
    if not root:
        return None
    config = Path(config).resolve()
    suffix = hashlib.sha256(str(config).encode()).hexdigest()[:12]
    return Path(root) / f"{config.stem}-{suffix}"


def sync_tensorboard(config, destination):
    local = tensorboard_directory(config)
    if local is not None and local.is_dir():
        shutil.copytree(local, Path(destination) / local.name, dirs_exist_ok=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--sync-tensorboard":
        sync_tensorboard(sys.argv[2], sys.argv[3])
        raise SystemExit(0)
    if len(sys.argv) < 2:
        raise SystemExit("Expected native training script and its arguments")
    root = os.environ.get("TSR_RANK_CACHE_ROOT")
    if root:
        rank = int(os.environ["LOCAL_RANK"])
        if rank < 0:
            raise SystemExit("Negative LOCAL_RANK")
        cache = Path(root).resolve() / f"rank-{rank}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache)
    entry = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    if len(sys.argv) >= 2:
        logdir = tensorboard_directory(sys.argv[1])
        if logdir is not None:
            logdir.mkdir(parents=True, exist_ok=True)
            sys.argv.append("--logging_dir=" + str(logdir))
    if os.environ.get("TSR_PERFORMANCE_PROFILE") == "h200-fast" and entry.name == "run_dpo.py":
        import math
        import numbers
        from trl import DPOTrainer
        original_log = DPOTrainer.log
        def finite_log(self, logs, *args, **kwargs):
            for key, value in logs.items():
                if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
                    raise FloatingPointError(f"Non-finite training metric {key}={value}; stopping before invalid model export")
            return original_log(self, logs, *args, **kwargs)
        DPOTrainer.log = finite_log
    runpy.run_path(str(entry), run_name="__main__")
