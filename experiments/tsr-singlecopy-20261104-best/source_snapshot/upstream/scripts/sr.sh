#!/bin/bash

CONDA_PATH=""
HF_TOKEN=""

### train from sft
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"


BASE_MODEL_PATH="${PROJECT_ROOT}/models/Llama-3.1-8B-mix-sft-epoch3"
export CUDA_VISIBLE_DEVICES=1,2,3,4
WORLD_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
TEST_MODE=False
# Loop over each iteration
ITERATIONS=(0 1 2 3)
for iter in "${ITERATIONS[@]}"
do
    echo "==============================="
    echo "Starting iteration: iter${iter}"
    echo "==============================="

    # Activate the first Conda environment
    source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/myenv"
    echo "Activated Conda environment: myenv"

    # Navigate to the tsr directory
    cd "${PROJECT_ROOT}/tsr"
    echo "Changed directory to tsr"

    # Run generate_vllm.py with the current iteration
    mkdir -p "${PROJECT_ROOT}/settings/sr/data_dpo/iter"
    if [ "${iter}" -eq 0 ]; then
        echo "Running self_generate_vllm_all.py for iter${iter}"
        python self_generate_vllm_all.py --iter "${iter}" --model "${BASE_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa1.jsonl" --output_dir "${PROJECT_ROOT}/settings/sr/data_dpo/iter" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}"
    else
        PREV_ITER=$((iter - 1))
        QA_ITER=$(( (iter % 5) + 1 ))
        echo "Running self_generate_vllm_all.py for iter${iter}"
        python self_generate_vllm_all.py --iter "${iter}" --model "${PROJECT_ROOT}/models/Llama-3.1-8B-mix-sft-epoch3-self_rewarding-iter${PREV_ITER}" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa${QA_ITER}.jsonl" --output_dir "${PROJECT_ROOT}/settings/sr/data_dpo/iter" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}"
    fi

    # Run reformat2DPO.py to format the output
    mkdir -p "${PROJECT_ROOT}/settings/sr/data_dpo/iter${iter}"
    echo "Running reformat2DPO.py for iter${iter}"
    python "${PROJECT_ROOT}/settings/sr/reformat2DPO.py" \
        --input_path "${PROJECT_ROOT}/settings/sr/data_dpo/iter/${iter}.json" \
        --output_path "${PROJECT_ROOT}/settings/sr/data_dpo/iter${iter}/train.json"

    # Activate the second Conda environment
    source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/handbook"
    echo "Activated Conda environment: handbook"

    # Navigate to the alignment-handbook directory
    cd "${PROJECT_ROOT}/alignment-handbook"
    echo "Changed directory to alignment-handbook"

    # Set necessary environment variables
    export HF_HOME="${PROJECT_ROOT}/cache"
    # export HF_ENDPOINT=https://hf-mirror.com
    echo "Set environment variables HF_HOME and HF_ENDPOINT"



    echo "Logging into Hugging Face CLI"
    huggingface-cli login --token "${HF_TOKEN}"

    # Run the accelerate launch command with the current iteration's config
    echo "Running accelerate launch for iter${iter}"
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${PROJECT_ROOT}/settings/sr/config/deepspeed_zero3.yaml" \
        --num_processes "${WORLD_SIZE}" \
        scripts/run_dpo.py "${PROJECT_ROOT}/settings/sr/config/dpo/config_full_iter${iter}.yaml"

    echo "==============================="
    echo "Completed iteration: iter${iter}"
    echo "==============================="
    echo ""
done

echo "All iterations completed successfully."
