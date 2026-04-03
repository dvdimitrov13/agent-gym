#!/bin/bash
# PipelineRL-style training: vLLM server on GPU 0, trainer on GPU 1
#
# This decouples generation from training:
# - GPU 0: vLLM serves the model for fast batched generation
# - GPU 1: Trainer handles RL loop, tool calling, gradient updates
# - TRL syncs weights from trainer → vLLM after each step
#
# Usage: bash scripts/train_pipeline.sh [config_path]

set -e

CONFIG=${1:-"src/training/configs/cloud_14b_pipeline.yaml"}
MODEL="Qwen/Qwen3-14B"
PORT=8000
LOG_DIR="/root"

echo "============================================"
echo "PipelineRL Training Setup"
echo "  Model: $MODEL"
echo "  Config: $CONFIG"
echo "  vLLM server: GPU 0, port $PORT"
echo "  Trainer: GPU 1"
echo "============================================"

# Step 1: Start vLLM server on GPU 0
echo "[1/2] Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model "$MODEL" \
    --port $PORT \
    --gpu_memory_utilization 0.90 \
    --max_model_len 4096 \
    --dtype bfloat16 \
    --log_level info \
    > "$LOG_DIR/vllm_server.log" 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"
echo "  Log: $LOG_DIR/vllm_server.log"

# Wait for server to be ready (check both /health and /v1/models endpoints)
echo "  Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1 || \
       curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "  vLLM server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM server died. Check $LOG_DIR/vllm_server.log"
        tail -20 "$LOG_DIR/vllm_server.log"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1 && \
   ! curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "  ERROR: vLLM server not ready after 600s"
    tail -20 "$LOG_DIR/vllm_server.log"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Step 2: Start trainer on GPU 1
echo "[2/2] Starting trainer on GPU 1..."
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH=/root/agent-gym \
    python -m src.training.train --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/train.log"

TRAIN_EXIT=$?

# Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Training finished with exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
