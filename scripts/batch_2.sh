#!/usr/bin/env bash
# batch_2.sh — launch general + thinker on Atlas GPUs 0+1.
#
# Uses raw nohup (no `bash -lc` abstraction) since that's what we
# verified works for batch 1 resume. Writes pidfiles for the watcher
# / status command. Frees the code_*.bin corpora first since those
# domains are done and the bins are 3 GB on /data.
set -euo pipefail

REPO=/data/sematic-embedding/tinyhetmoe
RUNS=$REPO/runs
cd "$REPO"

echo "[batch2] freeing code_* corpora ..."
rm -f data_v7/v7_train_code_py.bin data_v7/v7_train_code_js.bin
rm -f data_v7/v7_val_code_py.bin   data_v7/v7_val_code_js.bin
df -h /data | tail -1

# Pre-build host configs that point at data_v7/ paths (the configs are
# already correct since we kept v7_general / v7_thinker .json templates
# untouched). The trainer reads from train_file/val_file/meaning_emb_path
# in the config so we just copy the canonical config to .host.json.
for d in general thinker; do
  cp training/configs/tiny_hetmoe_v7_${d}.json training/configs/tiny_hetmoe_v7_${d}.host.json
done

# Refuse to relaunch if either is already running.
for d in general thinker; do
  pidf=$RUNS/tiny_hetmoe_v7_${d}.pid
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null \
     && ps -p "$(cat "$pidf")" -o cmd= | grep -q train_tiny_hetmoe; then
    echo "[batch2] $d already running pid $(cat "$pidf"); aborting"
    exit 1
  fi
  rm -f "$pidf"
done

LOG_GENERAL=$RUNS/tiny_hetmoe_v7_general.log
LOG_THINKER=$RUNS/tiny_hetmoe_v7_thinker.log

echo "[batch2] launching general (GPU 0)"
nohup env CUDA_VISIBLE_DEVICES=0 python3 -u training/train_tiny_hetmoe.py \
  --config training/configs/tiny_hetmoe_v7_general.host.json \
  >"$LOG_GENERAL" 2>&1 < /dev/null &
echo $! > $RUNS/tiny_hetmoe_v7_general.pid

echo "[batch2] launching thinker (GPU 1)"
nohup env CUDA_VISIBLE_DEVICES=1 python3 -u training/train_tiny_hetmoe.py \
  --config training/configs/tiny_hetmoe_v7_thinker.host.json \
  >"$LOG_THINKER" 2>&1 < /dev/null &
echo $! > $RUNS/tiny_hetmoe_v7_thinker.pid

sleep 5
for d in general thinker; do
  pid=$(cat $RUNS/tiny_hetmoe_v7_${d}.pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "[batch2] $d alive pid=$pid"
  else
    echo "[batch2] $d DIED IMMEDIATELY — see $RUNS/tiny_hetmoe_v7_${d}.log"
    tail -20 $RUNS/tiny_hetmoe_v7_${d}.log
  fi
done
