#!/bin/bash
set -e

echo "=========================================="
echo "  GC-Cap Environment Installation Script"
echo "=========================================="

# -----------------------------------------------
# Step 1: Core Python dependencies
# -----------------------------------------------
echo ""
echo "[1/2] Installing core Python dependencies..."
pip install openai \
            pillow \
            pandas \
            networkx

# -----------------------------------------------
# Step 2: vLLM (optional, for self-hosted models)
# -----------------------------------------------
echo ""
echo "[2/2] Installing vLLM..."
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation

# -----------------------------------------------
# Done
# -----------------------------------------------
echo ""
echo "=========================================="
echo "  Installation complete!"
echo ""
echo "  Next steps:"
echo "  1. Set your API keys:"
echo "     export OPENAI_API_KEY='your-key'"
echo "     export OPENAI_BASE_URL='https://api.openai.com/v1'"
echo "     # (Azure users)"
echo "     export AZURE_API_KEY='your-key'"
echo "     export AZURE_ENDPOINT='https://your-endpoint.openai.azure.com'"
echo "     export AZURE_API_VERSION='2024-12-01-preview'"
echo ""
echo "  2. The following models will be auto-downloaded on first run:"
echo "     - Qwen/Qwen3-Embedding-0.6B  (entity matching)"
echo "     - Qwen/Qwen3-Reranker-4B     (description similarity)"
echo ""
echo "  3. Run evaluation:"
echo "     bash Scripts/caption_generate.sh"
echo "     bash Scripts/eval.sh"
echo "=========================================="
