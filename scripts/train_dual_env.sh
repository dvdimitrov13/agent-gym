#!/bin/bash
# PipelineRL: inference server on GPU 0, TRL trainer on GPU 1
# No vLLM — uses plain transformers model.generate()
#
# Usage: bash scripts/train_dual_env.sh [config_path]

set -e

CONFIG=${1:-"src/training/configs/cloud_14b_pipeline.yaml"}
MODEL="Qwen/Qwen3-14B"
PORT=8000
LOG_DIR="/root"

echo "============================================"
echo "PipelineRL (no vLLM)"
echo "  GPU 0: inference server (transformers)"
echo "  GPU 1: TRL trainer"
echo "  Model: $MODEL"
echo "  Config: $CONFIG"
echo "============================================"

# Step 1: Start inference server on GPU 0
echo ""
echo "[1/2] Starting inference server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=/root/agent-gym \
    python -m src.training.inference_server \
    --model "$MODEL" \
    --port $PORT \
    > "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "  Server PID: $SERVER_PID"
echo "  Log: $LOG_DIR/server.log"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/health 2>/dev/null | grep -q "ok"; then
        echo "  Server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  ERROR: Server died"
        tail -20 "$LOG_DIR/server.log"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:$PORT/health 2>/dev/null | grep -q "ok"; then
    echo "  ERROR: Server not ready after 600s"
    tail -20 "$LOG_DIR/server.log"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Step 2: Start trainer on GPU 1
echo ""
echo "[2/2] Starting trainer on GPU 1..."
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH=/root/agent-gym \
    python -m src.training.train --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/train.log"

TRAIN_EXIT=$?

# Cleanup
echo ""
echo "Stopping inference server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Training finished with exit code: $TRAIN_EXIT"
exit $TRAIN_EXIT
