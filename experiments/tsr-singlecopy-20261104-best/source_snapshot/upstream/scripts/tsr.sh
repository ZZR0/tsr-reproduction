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
ITERATIONS=(0 1)
for iter in "${ITERATIONS[@]}"
do
    if [ "${iter}" -eq 0 ]; then
        EVAL_MODEL_PATH="${BASE_MODEL_PATH}"
    else
        PREV_ITER=$((iter - 1))
        EVAL_MODEL_PATH="${PROJECT_ROOT}/models/Llama-3.1-8B-mix-sft-epoch3-temporal-iterjudge-iter${PREV_ITER}"
    fi

    echo "==============================="
    echo "Starting iteration: iter${iter}"
    echo "==============================="

    # Activate the first Conda environment
    source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/myenv"
    echo "Activated Conda environment: myenv"

    # Navigate to the tsr directory
    cd "${PROJECT_ROOT}/tsr"
    echo "Changed directory to tsr"

    mkdir -p "${PROJECT_ROOT}/tsr/iter/iterJudge"

    # Run generate_vllm.py with the current iteration
    if [ "${iter}" -eq 0 ]; then
        echo "Running self_generate_vllm_all_generate.py (chosen) for iter${iter}"
        python self_generate_vllm_all_generate.py --iter "${iter}" --model "${BASE_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa1.jsonl" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type chosen
        python self_generate_vllm_all_iterJudge.py --iter "${iter}" --model "${EVAL_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/tsr/iter/iterJudge/0_chosen_response.json" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type chosen

        echo "Running self_generate_vllm_all_generate.py (rejected) for iter${iter}"
        python self_generate_vllm_all_generate.py --iter "${iter}" --model "${BASE_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa1.jsonl" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type rejected
        python self_generate_vllm_all_iterJudge.py --iter "${iter}" --model "${EVAL_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/tsr/iter/iterJudge/0_rejected_response.json" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type rejected

    else
        PREV_ITER=$((iter - 1))
        QA_ITER=$((iter + 1))
        echo "Running self_generate_vllm_all_generate.py (chosen) for iter${iter}"
        python self_generate_vllm_all_generate.py --iter "${iter}" --model "${PROJECT_ROOT}/models/Llama-3.1-8B-mix-sft-epoch3-temporal-iterjudge-iter${PREV_ITER}" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa${QA_ITER}.jsonl" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type chosen
        python self_generate_vllm_all_iterJudge.py --iter "${iter}" --model "${EVAL_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_chosen_response.json" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type chosen

        echo "Running self_generate_vllm_all_generate.py (rejected) for iter${iter}"
        python self_generate_vllm_all_generate.py --iter "${iter}" --model "${BASE_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa${QA_ITER}.jsonl" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type rejected
        python self_generate_vllm_all_iterJudge.py --iter "${iter}" --model "${EVAL_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_rejected_response.json" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type rejected

    fi

    # Run reformat2DPO_past.py to format the output
    mkdir -p "${PROJECT_ROOT}/settings/tsr/data_dpo/iter${iter}_tmp"
    echo "Running reformat2DPO_past.py for iter${iter}"
    python "${PROJECT_ROOT}/settings/tsr/reformat2DPO_past.py" \
        --input_path_chosen "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_chosen_judgement.json" \
        --input_path_rejected "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_rejected_judgement.json" \
        --output_path "${PROJECT_ROOT}/settings/tsr/data_dpo/iter${iter}_tmp/train.json"

    # Activate the second Conda environment
    source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/handbook"
    echo "Activated Conda environment: handbook"
    # Navigate to the alignment-handbook directory
    cd "${PROJECT_ROOT}/alignment-handbook"
    echo "Changed directory to alignment-handbook"
    # Set necessary environment variables
    export HF_HOME="${PROJECT_ROOT}/cache"
    echo "Set environment variables HF_HOME and HF_ENDPOINT"
    echo "Logging into Hugging Face CLI"
    huggingface-cli login --token "${HF_TOKEN}"
    # Run the accelerate launch command with the current iteration's config
    echo "Running accelerate launch for iter${iter}_tmp"
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${PROJECT_ROOT}/settings/tsr/config/deepspeed_zero3.yaml" \
        --num_processes "${WORLD_SIZE}" \
        scripts/run_dpo.py "${PROJECT_ROOT}/settings/tsr/config/dpo/config_full_iter${iter}_tmp.yaml"
    echo "Completed DPO_tmp: iter${iter}_tmp"

    # Activate the first Conda environment
    source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/myenv"
    echo "Activated Conda environment: myenv"

    # Navigate to the tsr directory
    cd "${PROJECT_ROOT}/tsr"
    echo "Changed directory to tsr"

    if [ "${iter}" -eq 0 ]; then
        echo "Running self_generate_vllm_all_generate.py (future) for iter${iter}"
        python self_generate_vllm_all_generate.py --iter "${iter}" --model "${PROJECT_ROOT}/models/Llama-3.1-8B-mix-sft-epoch3-temporal-iterjudge-iter${iter}_tmp" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa1.jsonl" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type future
        python self_generate_vllm_all_iterJudge.py --iter "${iter}" --model "${EVAL_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/tsr/iter/iterJudge/0_future_response.json" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type future

    else
        PREV_ITER=$((iter - 1))
        QA_ITER=$((iter + 1))
        echo "Running self_generate_vllm_all_generate.py (future) for iter${iter}"
        python self_generate_vllm_all_generate.py --iter "${iter}" --model "${PROJECT_ROOT}/models/Llama-3.1-8B-mix-sft-epoch3-temporal-iterjudge-iter${iter}_tmp" --input_dir "${PROJECT_ROOT}/data/SelfPlayBeginData/qa${QA_ITER}.jsonl" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type future
        python self_generate_vllm_all_iterJudge.py --iter "${iter}" --model "${EVAL_MODEL_PATH}" --input_dir "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_future_response.json" --output_dir "${PROJECT_ROOT}/tsr/iter/iterJudge" --world_size "${WORLD_SIZE}" --test "${TEST_MODE}" --type future
    fi

    # Run reformat2DPO_future.py to format the output
    mkdir -p "${PROJECT_ROOT}/settings/tsr/data_dpo/iter${iter}"
    echo "Running reformat2DPO_future.py for iter${iter}"
    python "${PROJECT_ROOT}/settings/tsr/reformat2DPO_future.py" \
        --input_path_chosen "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_chosen_judgement.json" \
        --input_path_future "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_future_judgement.json" \
        --input_path_rejected "${PROJECT_ROOT}/tsr/iter/iterJudge/${iter}_rejected_judgement.json" \
        --output_path "${PROJECT_ROOT}/settings/tsr/data_dpo/iter${iter}/train.json"

    # Activate the second Conda environment
    source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/handbook"
    echo "Activated Conda environment: handbook"

    # Navigate to the alignment-handbook directory
    cd "${PROJECT_ROOT}/alignment-handbook"
    echo "Changed directory to alignment-handbook"

    # Set necessary environment variables
    export HF_HOME="${PROJECT_ROOT}/cache"
    echo "Set environment variables HF_HOME and HF_ENDPOINT"

    echo "Logging into Hugging Face CLI"
    huggingface-cli login --token "${HF_TOKEN}"

    # Run the accelerate launch command with the current iteration's config
    echo "Running accelerate launch for iter${iter}"
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${PROJECT_ROOT}/settings/tsr/config/deepspeed_zero3.yaml" \
        --num_processes "${WORLD_SIZE}" \
        scripts/run_dpo.py "${PROJECT_ROOT}/settings/tsr/config/dpo/config_full_iter${iter}.yaml"
    echo "Completed DPO: iter${iter}"

    echo "==============================="
    echo "Completed iteration: iter${iter}"
    echo "==============================="
    echo ""
done

echo "All iterations completed successfully."
