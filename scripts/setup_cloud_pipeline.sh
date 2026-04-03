#!/bin/bash
# Setup a 2x GPU Vast.ai instance for PipelineRL training
#
# Installs TRL main + vLLM (with --no-deps to avoid transformers downgrade)
# This allows vLLM server on GPU 0 and TRL trainer on GPU 1 to coexist.
#
# Usage: bash scripts/setup_cloud_pipeline.sh

set -e

echo "=== Setting up PipelineRL environment ==="

# Core dependencies
pip install -q transformers accelerate ddgs trafilatura rapidfuzz \
    requests python-dotenv peft datasets pyyaml jmespath

# TRL from git main (Qwen3 chat template support)
pip install -q git+https://github.com/huggingface/trl.git

# vLLM (--no-deps to avoid downgrading transformers)
# TRL's server mode communicates with vLLM over HTTP, so version mismatch is OK
pip install -q vllm --no-deps

# Install vLLM's actual runtime deps (minus transformers)
pip install -q openai aiohttp uvicorn fastapi

echo ""
echo "=== Verifying installations ==="
python -c "
import transformers, trl
print(f'transformers={transformers.__version__}')
print(f'trl={trl.__version__}')
try:
    import vllm
    print(f'vllm={vllm.__version__}')
except Exception as e:
    print(f'vllm import issue: {e}')
from trl import GRPOTrainer
print('GRPOTrainer: OK')
"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "=== Ready ==="
echo "To start training: bash scripts/train_pipeline.sh"
