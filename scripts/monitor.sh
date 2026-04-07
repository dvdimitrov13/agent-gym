#!/bin/bash
# Monitor GPU memory and training process
# Logs to /root/monitor.log every 60s, alerts on OOM or crash
while true; do
    if ! pgrep -f "src.training.train" > /dev/null 2>&1; then
        echo "$(date): ALERT - training process died!" >> /root/monitor.log
        tail -5 /root/pipeline.log >> /root/monitor.log 2>/dev/null
        break
    fi
    MEM0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
    MEM1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
    STEPS=$(grep -c "step_time" /root/pipeline.log 2>/dev/null)
    echo "$(date): steps=$STEPS gpu0=${MEM0}MiB gpu1=${MEM1}MiB" >> /root/monitor.log
    if [ "$MEM0" -gt 47000 ] 2>/dev/null; then
        echo "$(date): WARN GPU0 HIGH ${MEM0}MiB" >> /root/monitor.log
    fi
    sleep 60
done
