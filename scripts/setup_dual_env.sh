#!/bin/bash
# Setup two separate conda environments for PipelineRL:
#   vllm_env (GPU 0): vLLM + transformers 4.x → serves generation
#   train_env (GPU 1): TRL main + transformers 5.x → training + tool calling
#
# They communicate over HTTP (localhost:8000).
# Usage: bash scripts/setup_dual_env.sh

set -e

echo "============================================"
echo "Setting up dual-env PipelineRL"
echo "============================================"

# Install conda if not present
if ! command -v conda &> /dev/null; then
    echo "[0/2] Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    export PATH="/opt/miniconda/bin:$PATH"
    echo 'export PATH="/opt/miniconda/bin:$PATH"' >> ~/.bashrc
    conda init bash 2>/dev/null
fi
export PATH="/opt/miniconda/bin:$PATH"

# ===== ENV 1: vLLM server =====
echo ""
echo "[1/2] Creating vllm_env (GPU 0 — generation server)..."
conda create -n vllm_env python=3.11 -y -q 2>&1 | tail -1
conda run -n vllm_env pip install -q vllm 2>&1 | tail -3
echo "  Verifying..."
conda run -n vllm_env python -c "
import vllm, transformers
print(f'  vllm={vllm.__version__}, transformers={transformers.__version__}')
"

# ===== ENV 2: TRL trainer =====
echo ""
echo "[2/2] Creating train_env (GPU 1 — training + tool calling)..."
conda create -n train_env python=3.11 -y -q 2>&1 | tail -1
conda run -n train_env pip install -q \
    torch torchvision torchaudio \
    "transformers>=5.3" accelerate peft datasets \
    git+https://github.com/huggingface/trl.git \
    ddgs trafilatura rapidfuzz requests python-dotenv pyyaml jmespath \
    2>&1 | tail -3
echo "  Verifying..."
conda run -n train_env python -c "
import transformers, trl
print(f'  transformers={transformers.__version__}, trl={trl.__version__}')
from trl import GRPOTrainer
print('  GRPOTrainer: OK')
"

echo ""
echo "============================================"
echo "Both environments ready!"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
echo "To start training: bash scripts/train_dual_env.sh"
echo "============================================"
