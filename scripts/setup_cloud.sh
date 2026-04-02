#!/usr/bin/env bash
# Bootstrap script for Vast.ai cloud instance
# Run after SSH-ing into the instance: bash scripts/setup_cloud.sh

set -euo pipefail

echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y tmux htop

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements-cloud.txt

echo "=== Verifying GPU access ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

echo "=== Setup complete ==="
