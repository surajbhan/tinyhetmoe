# TinyHetMoE Checkpoint Registry

Manually-maintained list of meaningful checkpoints. Most snapshots live
outside the repo (`/tmp/` or in `runs/`) since they're too large to commit.

## Current

| Snapshot | Run | Step | Type | val | PPL | Notes |
|---|---|---|---|---|---|---|
| `/tmp/v3_bf16_best_step24000.pt` | v3 | 24000 | bf16 | 1.5155 | 4.55 | bf16 floor — used to seed v4a |
| `/tmp/v4a_best_qat.pt`           | v4a | 38500 | QAT | 1.5734 | 4.82 | **Currently deployed** to docs/tiny.bin |
| `runs/tiny_hetmoe_v4a/checkpoints/best_qat.pt` | v4a | 38500 | QAT | 1.5734 | 4.82 | Same as above (in run dir) |
| `runs/tiny_hetmoe_v4a/checkpoints/ckpt_final_40000.pt` | v4a | 40000 | QAT | — | — | End-of-run record |
| `runs/tiny_hetmoe_v5/checkpoints/best.pt`     | v5 | (live) | bf16 | (live) | (live) | v5 in progress |
| `runs/tiny_hetmoe_v5/checkpoints/best_qat.pt` | v5 | (pending) | QAT | — | — | Will save once QAT phase fires |

## Deleted (to free disk)

- `runs/tiny_hetmoe/` (v1, val 3.05 / PPL 21, QAT-from-zero with vote-backward — superseded)
- `runs/tiny_hetmoe_v2/` (val 2.55 / PPL 12, QAT phase starved by low min_lr — superseded)
- `runs/tiny_hetmoe_v3/` (val 2.40 / PPL 11, vote-backward at 38M scale was wrong choice — superseded)
- v4a intermediate `ckpt_36-40K.pt` (kept only `best_qat.pt` and `ckpt_final_40000.pt`)

## Currently deployed

`docs/tiny.bin` ← exported from `/tmp/v4a_best_qat.pt` (PPL 4.82).
Live demo at https://surajbhan.github.io/tinyhetmoe/

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
