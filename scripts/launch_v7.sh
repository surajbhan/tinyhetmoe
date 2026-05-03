#!/usr/bin/env bash
# launch_v7.sh — orchestrate v7 trainings on Atlas (101).
#
# 132 + 109 sidelined: 132 has nohup-stderr buffering issues that mask
# trainer death, and 109 hard-locks under simultaneous 2-GPU load.
# Atlas alone is the smooth path — runs 2 domains in parallel (one per
# GPU) and picks up the next pair as each finishes.
#
# Pair schedule (3 batches × 2 GPUs):
#   batch 1: code_py (GPU 0), code_js (GPU 1)
#   batch 2: general (GPU 0), thinker (GPU 1)
#   batch 3: medical (GPU 0), legal   (GPU 1)
#
# Subcommands:
#   launch <d>    start training for one domain (foreground nohup)
#   launch-batch <n>   start batch 1, 2, or 3 (the two domains in that pair)
#   queue-next    when current pair is done, advance to the next batch
#   status        per-domain pid liveness + nvidia-smi
#   tail <d>      tail the training log for a domain
#   stop <d>      kill the trainer for a domain

set -euo pipefail

REPO_LOCAL="/data/sematic-embedding/tinyhetmoe"

# host  user      code_dir                              corpus_dir            gpu  domain
declare -A HOST_USER=(
  [atlas]="suraj"      [r132]="yoctotta"             [r109]="yoctotta"
)
declare -A HOST_IP=(
  [atlas]="127.0.0.1"  [r132]="192.168.0.132"        [r109]="192.168.0.109"
)
declare -A HOST_CODE=(
  [atlas]="/data/sematic-embedding/tinyhetmoe"
  [r132]="/home/yoctotta/tinyhetmoe"
  [r109]="/home/yoctotta/tinyhetmoe"
)
declare -A HOST_CORPUS=(
  [atlas]="data_v7"
  [r132]="/data/v7_corpus"
  [r109]="/home/yoctotta/v7_corpus"
)

# Atlas-only placement. All 6 domains run on Atlas, paired into 3
# batches of 2 (one per GPU). Larger corpora pair together so wall-time
# per batch is balanced.
declare -A DOMAIN_PLACEMENT=(
  [code_py]="atlas:0"
  [code_js]="atlas:1"
  [general]="atlas:0"
  [thinker]="atlas:1"
  [medical]="atlas:0"
  [legal]="atlas:1"
)

DOMAINS=(code_py code_js general thinker medical legal)
declare -a BATCH_1=(code_py code_js)
declare -a BATCH_2=(general thinker)
declare -a BATCH_3=(medical legal)

# ── helpers ────────────────────────────────────────────────────────────

ssh_host() {
  local host="$1"; shift
  local ip="${HOST_IP[$host]}"
  if [[ "$host" == "atlas" ]]; then
    bash -lc "$*"
  else
    ssh "$ip" "$*"
  fi
}

rsync_to() {
  local host="$1"
  local ip="${HOST_IP[$host]}"
  local code="${HOST_CODE[$host]}"
  if [[ "$host" == "atlas" ]]; then return 0; fi
  echo "[sync-code] -> $host ($ip):$code"
  ssh "$ip" "mkdir -p $code"
  rsync -av --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'runs/' --exclude 'logs/' \
    --exclude 'data/' --exclude 'data_v7/' \
    --exclude '.git/' --exclude 'wasm/target/' \
    --exclude 'node_modules/' --exclude 'docs/' \
    "$REPO_LOCAL/" "$ip:$code/"
}

# Patch a config in-place on the remote so its file paths point at the
# host's actual corpus dir. Writes a host-specific copy alongside the
# original (NOT a mutation of the canonical config).
prepare_remote_config() {
  local host="$1" domain="$2"
  local code="${HOST_CODE[$host]}"
  local corpus="${HOST_CORPUS[$host]}"
  local src="training/configs/tiny_hetmoe_v7_${domain}.json"
  local dst="training/configs/tiny_hetmoe_v7_${domain}.host.json"
  ssh_host "$host" "cd $code && python3 -c \"
import json
c = json.load(open('$src'))
c['train_file']      = '$corpus/v7_train_${domain}.bin'
c['val_file']        = '$corpus/v7_val_${domain}.bin'
c['meaning_emb_path']= '$corpus/meaning_axes_full_132.npy'
json.dump(c, open('$dst','w'), indent=2)
print('wrote $dst')
\""
}

# ── subcommands ────────────────────────────────────────────────────────

cmd_sync_code() {
  rsync_to r132
  rsync_to r109
}

cmd_smoke() {
  for host in atlas r132 r109; do
    echo
    echo "── smoke @ $host ──────────────────────────────────────────"
    local code="${HOST_CODE[$host]}"
    local corpus="${HOST_CORPUS[$host]}"
    ssh_host "$host" "cd $code && python3 scripts/verify_audit_fixes.py \
      --val-bin $corpus/v7_val_general.bin \
      --meaning $corpus/meaning_axes_full_132.npy \
      --vocab 151665 --batch 4 --seq-len 512 --qat" | tail -10
  done
}

cmd_launch_one() {
  local domain="$1"
  local placement="${DOMAIN_PLACEMENT[$domain]}"
  local host="${placement%%:*}"
  local gpu="${placement##*:}"
  local code="${HOST_CODE[$host]}"
  local logf="$code/runs/tiny_hetmoe_v7_${domain}.log"
  local pidf="$code/runs/tiny_hetmoe_v7_${domain}.pid"

  echo "[launch] $domain -> $host:gpu$gpu, log $logf"
  prepare_remote_config "$host" "$domain"
  # nohup + setsid + & detaches the python child fully from the SSH
  # session so the trainer survives ssh disconnect and tmux death.
  # If a previous run is still alive, refuse to relaunch.
  # Detach via `nohup … &` so SSH disconnect doesn't SIGHUP the trainer.
  # `$!` is the python PID (no `setsid` wrapper that would fork an
  # intermediate shell whose PID is what `$!` would actually capture).
  # Verify the captured PID is actually a `python3` process before
  # writing the pidfile, since stale pidfiles are silently misleading.
  ssh_host "$host" "cd $code && mkdir -p runs && \
    if [ -f $pidf ]; then \
      old=\$(cat $pidf); \
      if kill -0 \$old 2>/dev/null && ps -p \$old -o cmd= | grep -q train_tiny_hetmoe; then \
        echo 'already running pid '\$old; exit 0; \
      else \
        echo 'pidfile stale, replacing'; rm -f $pidf; \
      fi; \
    fi && \
    nohup env CUDA_VISIBLE_DEVICES=$gpu python3 -u training/train_tiny_hetmoe.py --config training/configs/tiny_hetmoe_v7_${domain}.host.json >$logf 2>&1 < /dev/null & \
    pid=\$!; echo \$pid > $pidf; \
    sleep 2; \
    if kill -0 \$pid 2>/dev/null; then echo 'launched pid '\$pid; \
    else echo 'TRAINER DIED IMMEDIATELY — see '$logf; tail -20 $logf; fi"
}

cmd_launch_batch() {
  local n="$1"
  case "$n" in
    1) for d in "${BATCH_1[@]}"; do cmd_launch_one "$d"; done ;;
    2) for d in "${BATCH_2[@]}"; do cmd_launch_one "$d"; done ;;
    3) for d in "${BATCH_3[@]}"; do cmd_launch_one "$d"; done ;;
    *) echo "unknown batch: $n (use 1, 2, or 3)"; exit 1 ;;
  esac
}

# `queue-next` looks at which batches still have unfinished trainings and
# advances. A domain is "finished" when its log contains the
# "[tiny] done. final:" line that the trainer writes on clean exit.
domain_finished() {
  local d="$1"
  local code="${HOST_CODE[atlas]}"
  local logf="$code/runs/tiny_hetmoe_v7_${d}.log"
  [ -f "$logf" ] && grep -q "\[tiny\] done. final:" "$logf"
}

batch_done() {
  local -n batch=$1
  for d in "${batch[@]}"; do
    if ! domain_finished "$d"; then return 1; fi
  done
  return 0
}

cmd_queue_next() {
  if ! batch_done BATCH_1; then
    echo "[queue] batch 1 (code_py + code_js) still running. Status:"
    cmd_status; return 0
  fi
  if ! batch_done BATCH_2; then
    echo "[queue] batch 1 done. Launching batch 2 (general + thinker)."
    cmd_launch_batch 2; return 0
  fi
  if ! batch_done BATCH_3; then
    echo "[queue] batch 2 done. Launching batch 3 (medical + legal)."
    cmd_launch_batch 3; return 0
  fi
  echo "[queue] all batches finished."
}

cmd_status() {
  local code="${HOST_CODE[atlas]}"
  echo "── atlas ──────────────────────────────────────────────────"
  for d in "${DOMAINS[@]}"; do
    local pidf="$code/runs/tiny_hetmoe_v7_${d}.pid"
    local logf="$code/runs/tiny_hetmoe_v7_${d}.log"
    if [ -f "$pidf" ]; then
      local pid; pid=$(cat "$pidf")
      if kill -0 "$pid" 2>/dev/null; then
        local last_step; last_step=$(grep -E "^step " "$logf" 2>/dev/null | tail -1 | awk '{print $2}')
        echo "  $d  pid=$pid alive  last_step=${last_step:-(none)}"
      else
        if grep -q "\[tiny\] done. final:" "$logf" 2>/dev/null; then
          echo "  $d  FINISHED"
        else
          echo "  $d  pid=$pid DEAD (no clean exit)"
        fi
      fi
    else
      echo "  $d  not launched"
    fi
  done
  echo
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
}

cmd_tail() {
  local domain="$1"
  local placement="${DOMAIN_PLACEMENT[$domain]}"
  local host="${placement%%:*}"
  local code="${HOST_CODE[$host]}"
  ssh_host "$host" "tail -f $code/runs/tiny_hetmoe_v7_${domain}.log"
}

cmd_stop() {
  local domain="$1"
  local placement="${DOMAIN_PLACEMENT[$domain]}"
  local host="${placement%%:*}"
  local code="${HOST_CODE[$host]}"
  local pidf="$code/runs/tiny_hetmoe_v7_${domain}.pid"
  ssh_host "$host" "if [ -f $pidf ]; then pid=\$(cat $pidf); kill \$pid 2>/dev/null && echo killed \$pid || echo no live process; rm -f $pidf; else echo no pidfile; fi"
}

# ── dispatch ───────────────────────────────────────────────────────────

case "${1:-}" in
  launch)        cmd_launch_one "$2" ;;
  launch-batch)  cmd_launch_batch "$2" ;;
  queue-next)    cmd_queue_next ;;
  status)        cmd_status ;;
  tail)          cmd_tail "$2" ;;
  stop)          cmd_stop "$2" ;;
  *)
    echo "usage: $0 {launch <domain>|launch-batch <1|2|3>|queue-next|status|tail <domain>|stop <domain>}"
    echo "domains: ${DOMAINS[*]}"
    echo "batches: 1=(code_py code_js)  2=(general thinker)  3=(medical legal)"
    exit 1
    ;;
esac
