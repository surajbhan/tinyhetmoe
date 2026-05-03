#!/usr/bin/env bash
# v7_watcher.sh — overnight orchestrator for v7 batches 2 + 3.
#
# Polls every 60s. When BOTH trainings in the current batch have
# "[tiny] done. final:" in their log, advances to the next batch.
# Exits cleanly after batch 3 finishes (or after a hard 16-hour
# wall budget — safety net so a hung trainer doesn't loop forever).
#
# Run via:
#   nohup bash scripts/v7_watcher.sh >runs/v7_watcher.log 2>&1 &
#   echo $! > runs/v7_watcher.pid
set -euo pipefail

REPO=/data/sematic-embedding/tinyhetmoe
RUNS=$REPO/runs
cd "$REPO"

POLL_SEC=60
WALL_BUDGET_SEC=$((16 * 3600))   # 16 hours
START_T=$(date +%s)

batch_done() {
  for d in "$@"; do
    grep -q "\[tiny\] done. final:" "$RUNS/tiny_hetmoe_v7_${d}.log" 2>/dev/null || return 1
  done
  return 0
}

batch_failed() {
  # A batch member is "failed" if its pid is not alive AND the log lacks
  # the done marker. Means the trainer crashed without finishing.
  local d="$1"
  local pidf="$RUNS/tiny_hetmoe_v7_${d}.pid"
  local logf="$RUNS/tiny_hetmoe_v7_${d}.log"
  [ -f "$pidf" ] || return 1
  local pid; pid=$(cat "$pidf")
  if kill -0 "$pid" 2>/dev/null; then return 1; fi  # alive
  grep -q "\[tiny\] done. final:" "$logf" 2>/dev/null && return 1  # finished
  return 0  # dead and unfinished -> failed
}

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- phase 1: wait for batch 1 (code_py + code_js) -------------------
log "watcher started. waiting for batch 1 (code_py + code_js) to finish."
while true; do
  if batch_done code_py code_js; then
    log "batch 1 finished."
    break
  fi
  for d in code_py code_js; do
    if batch_failed "$d"; then
      log "batch 1 FAILED: $d trainer died without [tiny] done. exiting."
      exit 2
    fi
  done
  if (( $(date +%s) - START_T > WALL_BUDGET_SEC )); then
    log "wall budget exhausted in phase 1. exiting."
    exit 3
  fi
  sleep "$POLL_SEC"
done

# ---- phase 2: launch batch 2, wait for it ----------------------------
log "launching batch 2 (general + thinker)"
bash scripts/batch_2.sh 2>&1 | sed "s/^/[batch2] /"
log "batch 2 launched. polling for completion."
while true; do
  if batch_done general thinker; then
    log "batch 2 finished."
    break
  fi
  for d in general thinker; do
    if batch_failed "$d"; then
      log "batch 2 FAILED: $d trainer died without [tiny] done. exiting."
      exit 2
    fi
  done
  if (( $(date +%s) - START_T > WALL_BUDGET_SEC )); then
    log "wall budget exhausted in phase 2. exiting."
    exit 3
  fi
  sleep "$POLL_SEC"
done

# ---- phase 3: launch batch 3, wait for it ----------------------------
log "launching batch 3 (medical + legal)"
bash scripts/batch_3.sh 2>&1 | sed "s/^/[batch3] /"
log "batch 3 launched. polling for completion."
while true; do
  if batch_done medical legal; then
    log "batch 3 finished. ALL 6 v7 TRAININGS DONE."
    break
  fi
  for d in medical legal; do
    if batch_failed "$d"; then
      log "batch 3 FAILED: $d trainer died without [tiny] done. exiting."
      exit 2
    fi
  done
  if (( $(date +%s) - START_T > WALL_BUDGET_SEC )); then
    log "wall budget exhausted in phase 3. exiting."
    exit 3
  fi
  sleep "$POLL_SEC"
done

# ---- summary ----------------------------------------------------------
log "==== v7 done. summary ===="
for d in code_py code_js general thinker medical legal; do
  logf="$RUNS/tiny_hetmoe_v7_${d}.log"
  bf16_best=$(grep "best_val" "$logf" 2>/dev/null | tail -1 || true)
  qat_best=$(grep "best_qat.pt" "$logf" 2>/dev/null | tail -1 || true)
  done_line=$(grep "\[tiny\] done. final:" "$logf" 2>/dev/null | tail -1)
  log "  $d:"
  log "    bf16 best (last): ${bf16_best:-(none)}"
  log "    qat  best (last): ${qat_best:-(none)}"
  log "    done: ${done_line:-(none)}"
done
log "checkpoints under /home/suraj/v7_runs/. Probe with scripts/probe_v7.py."
