#!/usr/bin/env python3
"""Write an SFT or DPO config from the upstream TSR configs.

Only paths and the per-GPU batch split change; the global batch stays 32 on any number of GPUs.
  sft: template settings/sft/config/config.yaml (upstream: 8 GPUs x 4 x accumulation 1)
       -> per-device 2, accumulation 32 / (2 * ngpu). Sequences are packed to 2048 tokens, so this is equivalent.
  dpo: template settings/sr/config/dpo/config_full_iter0.yaml (upstream: 8 GPUs x 2 x accumulation 2).
       The TSR DPO configs are identical to it except paths (and iter0_tmp, which uses per-device 1, i.e. a global
       batch of 16 on 8 GPUs; we use 32 for every DPO run). -> per-device 1, accumulation 32 / ngpu.
       Only DIR/train.json is copied into <config stem>.data/ for loading; nested validation files are excluded.

  make_cfg.py {sft,dpo} --name N --init MODEL --data DIR --out_dir DIR --ngpu G --cfg OUT.yaml [--max_steps S]
"""
import argparse, os, shutil, tempfile, yaml
from pathlib import Path


def dpo_training_data(data, cfg):
    """Snapshot the canonical pairs without recursively importing verification copies."""
    source = Path(data) / "train.json"
    if not source.is_file():
        raise SystemExit(f"DPO pairs file not found: {source}")
    target_dir = Path(cfg).resolve().with_suffix(".data")
    if target_dir.is_symlink():
        raise SystemExit(f"DPO input directory must not be a symlink: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in target_dir.iterdir() if p.name != "train.json" or not p.is_file()]
    if unexpected:
        raise SystemExit(f"Unexpected files in isolated DPO input directory {target_dir}: {unexpected}")
    # Stage outside the dataset directory, then replace atomically. A failed copy leaves the old input intact.
    fd, temporary = tempfile.mkstemp(prefix=f".{target_dir.name}-", suffix=".tmp", dir=target_dir.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target_dir / "train.json")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(target_dir)

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["sft", "dpo"])
ap.add_argument("--name", required=True); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--out_dir", required=True); ap.add_argument("--ngpu", type=int, required=True)
ap.add_argument("--cfg", required=True); ap.add_argument("--max_steps", type=int, default=-1)
a = ap.parse_args()
tsr = os.environ["TSR_DIR"]

if a.mode == "sft":
    c = yaml.safe_load(open(f"{tsr}/settings/sft/config/config.yaml"))
    assert c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] * 8 == 32
    micro = 2
    c.update({"per_device_eval_batch_size": 2, "save_strategy": "no"})
else:
    c = yaml.safe_load(open(f"{tsr}/settings/sr/config/dpo/config_full_iter0.yaml"))
    assert c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] * 8 == 32
    micro = int(os.environ.get("TSR_DPO_MICRO_BATCH", "1"))
    c.update({"report_to": ["tensorboard"]})
    if os.environ.get("TSR_PERFORMANCE_PROFILE") == "h200-fast":
        c.update({"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2",
                  "gradient_checkpointing": os.environ.get("TSR_GRADIENT_CHECKPOINTING", "0") == "1",
                  "optim": "adamw_torch", "save_safetensors": True,
                  "logging_nan_inf_filter": False, "logging_steps": 1,
                  "dataloader_drop_last": True})
if 32 % (micro * a.ngpu):
    raise SystemExit(f"global batch 32 is not divisible by {micro} x {a.ngpu} GPUs; use 1, 2, 4, 8, 16 or 32 GPUs")
data = dpo_training_data(a.data, a.cfg) if a.mode == "dpo" else a.data
c.update({"model_name_or_path": a.init, "dataset_mixer": {data: 1.0}, "output_dir": a.out_dir,
          "hub_model_id": a.name, "per_device_train_batch_size": micro,
          "gradient_accumulation_steps": 32 // (micro * a.ngpu), "max_steps": a.max_steps})
os.makedirs(os.path.dirname(a.cfg), exist_ok=True)
yaml.safe_dump(c, open(a.cfg, "w"), sort_keys=False)
print(a.cfg)
