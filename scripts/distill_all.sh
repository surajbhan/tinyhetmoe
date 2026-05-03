#!/usr/bin/env bash
# distill_all.sh — sequentially distill all 6 v7 experts.
#
# Per expert (bf16-distill from best.pt → QAT-finetune from distilled bf16):
#   1. bf16: 5000 steps, lr 5e-5, top-K CE, sft anchor 0.1
#   2. QAT-finetune: 1500 steps, lr 2e-5, same loss
#
# Teacher = Qwen2.5-0.5B (base) for general/medical/legal
#         = Qwen2.5-Math-1.5B for thinker
#         = Qwen2.5-Coder-0.5B for code_py/code_js
# All Qwen2.5-family share vocab so logits map 1:1.
#
# Outputs:
#   /home/suraj/v7_runs/tiny_hetmoe_v7_<domain>_distill_bf16/checkpoints/best_distill_qat.pt
#   /home/suraj/v7_runs/tiny_hetmoe_v7_<domain>_distill_qat/checkpoints/best_distill_qat.pt  ← deploy artifact

set -euo pipefail

REPO=/data/sematic-embedding/tinyhetmoe
RUNS=/home/suraj/v7_runs
cd "$REPO"

declare -A TEACHERS=(
  [general]="Qwen/Qwen2.5-0.5B"
  [thinker]="Qwen/Qwen2.5-Math-1.5B"
  [code_py]="Qwen/Qwen2.5-Coder-0.5B"
  [code_js]="Qwen/Qwen2.5-Coder-0.5B"
  [medical]="Qwen/Qwen2.5-0.5B"
  [legal]="Qwen/Qwen2.5-0.5B"
)

DOMAINS=(general thinker code_py code_js medical legal)

BF16_STEPS=5000
QAT_STEPS=1500

t_start=$(date +%s)
echo "[$(date +%H:%M:%S)] distill_all started — domains: ${DOMAINS[*]}"

for d in "${DOMAINS[@]}"; do
  TEACHER=${TEACHERS[$d]}
  src_bf16=$RUNS/tiny_hetmoe_v7_${d}/checkpoints/best.pt
  out_bf16=$RUNS/tiny_hetmoe_v7_${d}_distill_bf16
  out_qat=$RUNS/tiny_hetmoe_v7_${d}_distill_qat
  log_bf16=$RUNS/tiny_hetmoe_v7_${d}_distill_bf16.log
  log_qat=$RUNS/tiny_hetmoe_v7_${d}_distill_qat.log

  if [ ! -f "$src_bf16" ]; then
    echo "[$(date +%H:%M:%S)] $d: missing $src_bf16, SKIP"
    continue
  fi

  echo
  echo "[$(date +%H:%M:%S)] ── $d → teacher=$TEACHER ──"

  # Stage 1: bf16 distillation
  if [ -f "$out_bf16/checkpoints/best_distill_qat.pt" ]; then
    echo "  bf16 stage already done, skipping"
  else
    echo "  bf16 distill: $BF16_STEPS steps lr 5e-5"
    python3 -u training/distill_v7.py \
      --student-ckpt "$src_bf16" \
      --teacher "$TEACHER" \
      --domain "$d" \
      --out-dir "$out_bf16" \
      --max-steps $BF16_STEPS \
      --lr 5e-5 --batch-size 4 --max-seq-len 512 \
      --top-k 50 --sft-anchor 0.1 \
      --log-interval 100 --val-interval 500 --save-interval 5000 \
      >"$log_bf16" 2>&1
    echo "  bf16 done. tail:"
    tail -3 "$log_bf16"
  fi

  src_qat="$out_bf16/checkpoints/best_distill_qat.pt"

  # Stage 2: QAT-finetune
  if [ -f "$out_qat/checkpoints/best_distill_qat.pt" ]; then
    echo "  QAT stage already done, skipping"
  else
    echo "  QAT-finetune: $QAT_STEPS steps lr 2e-5"
    python3 -u training/distill_v7.py \
      --student-ckpt "$src_qat" \
      --teacher "$TEACHER" \
      --domain "$d" \
      --out-dir "$out_qat" \
      --max-steps $QAT_STEPS \
      --lr 2e-5 --batch-size 4 --max-seq-len 512 \
      --top-k 50 --sft-anchor 0.1 \
      --qat \
      --log-interval 100 --val-interval 500 --save-interval 1500 \
      >"$log_qat" 2>&1
    echo "  QAT done. tail:"
    tail -3 "$log_qat"
  fi

  # Free old intermediate ckpts to keep disk healthy
  rm -f "$out_bf16/checkpoints/ckpt_distill_"*.pt
  rm -f "$out_qat/checkpoints/ckpt_distill_"*.pt

  elapsed=$(( $(date +%s) - t_start ))
  echo "  elapsed: $((elapsed/60))m $((elapsed%60))s"
  df -h / | tail -1
done

echo
echo "[$(date +%H:%M:%S)] ALL 6 EXPERTS DISTILLED. total time: $(( ($(date +%s) - t_start) / 60 ))m"
echo "Deploy artifacts at: \$RUNS/tiny_hetmoe_v7_<domain>_distill_qat/checkpoints/best_distill_qat.pt"
