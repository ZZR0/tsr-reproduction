#!/bin/bash
# set your conda_path and hf_token here
CONDA_PATH=""
HF_TOKEN=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WORLD_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')

source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/handbook"
cd "${PROJECT_ROOT}/alignment-handbook"

export HF_HOME="${PROJECT_ROOT}/cache"
huggingface-cli login --token "${HF_TOKEN}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${PROJECT_ROOT}/settings/sft/config/multi_gpu.yaml" \
    --num_processes "${WORLD_SIZE}" \
    scripts/run_sft.py "${PROJECT_ROOT}/settings/sft/config/config.yaml" \
    --load_in_4bit=false