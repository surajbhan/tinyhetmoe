#!/usr/bin/env bash
# batch_3.sh — launch medical + legal on Atlas GPUs 0+1.
#
# Runs after batch 2 (general + thinker) finishes. Same pattern as
# batch_2.sh: raw nohup, pidfile management, refuse if already running.
set -euo pipefail

REPO=/data/sematic-embedding/tinyhetmoe
RUNS=$REPO/runs
cd "$REPO"

echo "[batch3] freeing general+thinker corpora ..."
rm -f data_v7/v7_train_general.bin data_v7/v7_train_thinker.bin
rm -f data_v7/v7_val_general.bin   data_v7/v7_val_thinker.bin
df -h /data | tail -1

for d in medical legal; do
  cp training/configs/tiny_hetmoe_v7_${d}.json training/configs/tiny_hetmoe_v7_${d}.host.json
done

for d in medical legal; do
  pidf=$RUNS/tiny_hetmoe_v7_${d}.pid
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null \
     && ps -p "$(cat "$pidf")" -o cmd= | grep -q train_tiny_hetmoe; then
    echo "[batch3] $d already running pid $(cat "$pidf"); aborting"
    exit 1
  fi
  rm -f "$pidf"
done

LOG_MEDICAL=$RUNS/tiny_hetmoe_v7_medical.log
LOG_LEGAL=$RUNS/tiny_hetmoe_v7_legal.log

echo "[batch3] launching medical (GPU 0)"
nohup env CUDA_VISIBLE_DEVICES=0 python3 -u training/train_tiny_hetmoe.py \
  --config training/configs/tiny_hetmoe_v7_medical.host.json \
  >"$LOG_MEDICAL" 2>&1 < /dev/null &
echo $! > $RUNS/tiny_hetmoe_v7_medical.pid

echo "[batch3] launching legal (GPU 1)"
nohup env CUDA_VISIBLE_DEVICES=1 python3 -u training/train_tiny_hetmoe.py \
  --config training/configs/tiny_hetmoe_v7_legal.host.json \
  >"$LOG_LEGAL" 2>&1 < /dev/null &
echo $! > $RUNS/tiny_hetmoe_v7_legal.pid

sleep 5
for d in medical legal; do
  pid=$(cat $RUNS/tiny_hetmoe_v7_${d}.pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "[batch3] $d alive pid=$pid"
  else
    echo "[batch3] $d DIED IMMEDIATELY — see $RUNS/tiny_hetmoe_v7_${d}.log"
    tail -20 $RUNS/tiny_hetmoe_v7_${d}.log
  fi
done
