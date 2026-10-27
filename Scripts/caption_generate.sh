#!/bin/bash

# ============================================
# Caption Generation Script
# Calls caption_generator.py to generate
# image captions using LVLM models.
# ============================================

# Record script start time
start_time=$(date +%s)

# -------------------- Configuration --------------------

# Path to the caption generator script (modify as needed)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/Eval/caption_generator.py"

# Image folder path (REQUIRED - modify this)
FOLDER_PATH="/path/to/your/image/folder"

# Output CSV file path
OUTPUT_FILE="${SCRIPT_DIR}/Eval/output/captions.csv"

# Model name: use "gpt-xxx" for OpenAI, or model name for vLLM (e.g., "Qwen3-VL-30B-A3B")
MODEL_NAME="Qwen3-VL-30B-A3B"

# Number of concurrent threads
MAX_WORKERS=4

# Maximum retry attempts
MAX_RETRIES=3

# -------------------- Environment Variables --------------------
# Uncomment and set the appropriate API keys based on your model choice:

# For OpenAI models (gpt-xxx):
# export OPENAI_API_KEY="your-openai-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"

# For vLLM models:
# export VLLM_API_KEY="your-vllm-api-key"
# export VLLM_BASE_URL="http://localhost:8080/v1"

# -------------------- Execution --------------------

echo "=========================================="
echo "  Caption Generation"
echo "=========================================="
echo "[Config] script_path   = ${SCRIPT_PATH}"
echo "[Config] folder_path   = ${FOLDER_PATH}"
echo "[Config] output_file   = ${OUTPUT_FILE}"
echo "[Config] model_name    = ${MODEL_NAME}"
echo "[Config] max_workers   = ${MAX_WORKERS}"
echo "[Config] max_retries   = ${MAX_RETRIES}"
echo "=========================================="

retries=0

while [ $retries -lt $MAX_RETRIES ]; do
    python "${SCRIPT_PATH}" \
        --folder_path "${FOLDER_PATH}" \
        --output_file "${OUTPUT_FILE}" \
        --model_name "${MODEL_NAME}" \
        --max_workers ${MAX_WORKERS}

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Caption generation completed successfully!"
        break
    fi

    retries=$((retries + 1))
    echo "Script failed with exit code ${exit_code}, retrying... (${retries}/${MAX_RETRIES})"
    sleep 5
done

if [ $retries -ge $MAX_RETRIES ]; then
    echo "Reached maximum retry attempts (${MAX_RETRIES}). Script execution failed."
fi

# -------------------- Summary --------------------

end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo "=========================================="
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "=========================================="