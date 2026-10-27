#!/bin/bash

# ============================================
# Graph-based Caption Evaluation Script
# Calls eval.py to evaluate image captions
# using graph-based metrics.
# ============================================

# Record script start time
start_time=$(date +%s)

# -------------------- Configuration --------------------

# Path to the evaluation script (modify as needed)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/Eval/eval.py"

# Maximum retry attempts
MAX_RETRIES=3

# -------------------- Environment Variables --------------------
# Set the appropriate API keys based on your --api_backend choice:

# For Azure OpenAI backend (--api_backend azure):
# export AZURE_API_KEY="your-azure-api-key"
# export AZURE_ENDPOINT="https://your-endpoint.openai.azure.com"
# export AZURE_API_VERSION="2024-12-01-preview"

# For OpenAI backend (--api_backend openai):
# export OPENAI_API_KEY="your-openai-api-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"

# -------------------- Execution --------------------

echo "=========================================="
echo "  Graph-based Caption Evaluation"
echo "=========================================="
echo "[Config] script_path   = ${SCRIPT_PATH}"
echo "[Config] max_retries   = ${MAX_RETRIES}"
echo "=========================================="

retries=0

while [ $retries -lt $MAX_RETRIES ]; do
    python "${SCRIPT_PATH}" \
        --mode "combine" \
        --cap_file_gt "${SCRIPT_DIR}/GC-Bench/GC-Benchmark_1009_186_with_gt_entities_reorganized.csv" \
        --cap_file_model "/path/to/model_captions.csv" \
        --embedding_model_name "Qwen/Qwen3-Embedding-0.6B" \
        --rerank_model_name "Qwen/Qwen3-Reranker-4B" \
        --api_model_name "gpt-5" \
        --api_backend "openai" \
        --device "cuda:0" \
        --generate_max_workers 16 \
        --save_path "/path/to/output_results.csv"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Evaluation completed successfully!"
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