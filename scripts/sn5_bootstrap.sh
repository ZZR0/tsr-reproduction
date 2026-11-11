#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
repo=/mnt/sn-007/zaninzeng/tsr-reproduction
export PATH="$repo/.venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
export HF_HOME="$repo/.runtime/hf-cache" HF_DATASETS_CACHE="$run_dir/trial/dataset-cache"
export TSR_RANK_CACHE_ROOT="$run_dir/runtime/datasets"
export TSR_TENSORBOARD_LOCAL_ROOT="/tmp/tsr-tensorboard-${HOSTNAME}"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export TORCH_EXTENSIONS_DIR="$run_dir/runtime/torch-extensions" TRITON_CACHE_DIR="$run_dir/runtime/triton"
export VLLM_CACHE_ROOT="$run_dir/runtime/vllm" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$run_dir/bundle/third_party/TSR/alignment-handbook/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda
mkdir -p "$run_dir/bootstrap" "$run_dir/runtime" "$run_dir/trial/logs"
exec > >(tee -a "$run_dir/bootstrap/job.log") 2>&1
"$repo/.venv/bin/python" --version
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$run_dir/bootstrap/gpu-start.csv"
"$repo/.venv/bin/python" -m pip freeze > "$run_dir/bootstrap/pip-freeze.txt"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader >> "$run_dir/bootstrap/gpu.csv" || true
    sleep 30
  done
) &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
"$repo/.venv/bin/python" -u "$run_dir/bundle/scripts/sn5_driver.py" "$run_dir"
