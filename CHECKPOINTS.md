# TinyHetMoE Checkpoint Registry

Manually-maintained list of meaningful checkpoints. Most snapshots live
outside the repo (`/tmp/` or in `runs/`) since they're too large to commit.

## Current — deployed model

`docs/tiny.bin` ← exported from `/tmp/v5b_best_qat.pt` (val **1.5977 / PPL 4.94**).
Live demo at https://surajbhan.github.io/tinyhetmoe/

## Snapshot registry

| Snapshot | Run | Step | Type | val | PPL | Notes |
|---|---|---|---|---|---|---|
| `/tmp/v5b_bf16_best.pt`   | v5b | 15000 | bf16 | 1.6056 | 4.98 | bf16 floor of v5b |
| `/tmp/v5b_best_qat.pt`    | v5b | 27500 | QAT  | **1.5977** | **4.94** | **Currently deployed** to docs/tiny.bin |
| `/tmp/v3_bf16_best_step24000.pt` | v3 | 24000 | bf16 | 1.5155 | 4.55 | Older bf16 floor (best across runs but on small vocab) |
| `runs/tiny_hetmoe_v4a/checkpoints/best_qat.pt`         | v4a | 38500 | QAT | 1.5734 | 4.82 | Previous deployment artifact (now superseded by v5b) |
| `runs/tiny_hetmoe_v4a/checkpoints/ckpt_final_40000.pt` | v4a | 40000 | QAT | — | — | End-of-run record |
| `runs/tiny_hetmoe_v5b/checkpoints/best.pt`             | v5b | 15000 | bf16 | 1.6056 | 4.98 | Same as /tmp/v5b_bf16_best.pt |
| `runs/tiny_hetmoe_v5b/checkpoints/best_qat.pt`         | v5b | 27500 | QAT  | 1.5977 | 4.94 | Same as /tmp/v5b_best_qat.pt |

## Comparison across all runs

| Run | Recipe summary | bf16 best | QAT best |
|---|---|---|---|
| v1 | QAT-from-zero, vote-backward, low min_lr | n/a | val 3.05 / PPL 21 |
| v2 | bf16-then-QAT, low min_lr | 1.54 / 4.67 | 2.55 / 12.7 |
| v3 | + min_lr=2e-4 | 1.5155 / 4.55 | 2.40 / 11.0 |
| v4a | + STE backward | 1.5155 / 4.55 | 1.5734 / 4.82 |
| v5 | + var-length training | 1.6107 / 5.01 | (didn't reach QAT phase) |
| **v5b** | + restart with --skip-optimizer + plateau-bug fix | **1.6056 / 4.98** | **1.5977 / 4.94** |

## Re-export workflow

```bash
python3 scripts/export_model.py --ckpt <path-to-best_qat.pt> --out docs/tiny.bin
git add docs/tiny.bin docs/tiny.meta.json
git commit -m "Update model: <run> step <N> (PPL <P>)"
git push
```

## Why two best.pt files

The trainer saves `best.pt` (lifetime lowest val — usually bf16 phase) AND
`best_qat.pt` (lowest val while QAT was on). Always deploy `best_qat.pt`
since that's the actual ternary deployment artifact.
