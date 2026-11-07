#!/usr/bin/env bash
set -euo pipefail
REPO=/media/ardeawang/TSR/tsr-reproduction
OPS=/media/ardeawang/TSR/repro/tsr-singlecopy-20261104
RUNTIME=/opt/tsr-reproduction-aa7200a
operation=$1; selected_gpus=$2; selected_port=$3; shift 3
export TZ=Asia/Shanghai PYTHONUTF8=1 PYTHONUNBUFFERED=1
export IS_ALPACA_EVAL_2=True OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda
export TMPDIR="$RUNTIME/tmp" TORCH_EXTENSIONS_DIR="$RUNTIME/torch_extensions"
export TRITON_CACHE_DIR="$RUNTIME/triton" VLLM_CACHE_ROOT="$RUNTIME/vllm"
cd "$REPO"
. scripts/lib.sh
load_config
# Runtime paths and GPU placement only; retain the repository's scientific helpers.
export WORK_DIR="$REPO/work-tsr-singlecopy-20261104" GPUS="$selected_gpus"
IFS=, read -ra GPU_LIST <<< "$GPUS"; NGPU=${#GPU_LIST[@]}
MODELS=$WORK_DIR/models; DATA=$WORK_DIR/data; LOGS=$WORK_DIR/logs; EVAL=$WORK_DIR/alpaca_eval
M0=$MODELS/M0
mkdir -p "$MODELS" "$DATA" "$LOGS" "$EVAL"
_train() {
  local cfg=$1 script=$2 logf=$3
  ( cd "$TSR_DIR/alignment-handbook" && CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch \
      --config_file "$TSR_DIR/settings/tsr/config/deepspeed_zero3.yaml" --num_processes "$NGPU" \
      --main_process_port "$selected_port" "$script" "$cfg" ) > "$logf" 2>&1
}
case "$operation" in
  gen|judge|pairs|dpo) "$operation" "$@" ;;
  eval_generate)
    name=$1; need "$MODELS/$name/.done"
    if [ ! -f "$EVAL/$name/model_outputs.json" ]; then
      CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" src/gen_alpaca.py --model "$MODELS/$name" \
        --out "$EVAL/$name/model_outputs.json" --seed 0 > "$LOGS/alpaca_gen_$name.log" 2>&1
    fi ;;
  eval_score)
    need "$EVAL/$1/model_outputs.json"
    alpaca "$1" "$MODELS/$1" ;;
  *) exit 2 ;;
esac
