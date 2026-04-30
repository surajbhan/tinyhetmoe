# TinyHetMoE Checkpoint Registry

Manually-maintained list of meaningful checkpoints we don't want to lose.
Stored outside the repo (in `/tmp/` or `/data/.../runs/`) since they're
too large to commit.

| Snapshot path | Run | Step | Type | val | PPL | Notes |
|---|---|---|---|---|---|---|
| `/tmp/v2_bf16_best_step17000.pt` | v2 | 17000 | bf16 best | 1.5418 | 4.67 | First clean Eldan & Li-baseline bf16 floor |
| `/tmp/v3_bf16_best_step24000.pt` | v3 | 24000 | bf16 best | 1.5155 | 4.55 | Improved bf16 floor with high min_lr |
| `/tmp/v4a_best_qat.pt`           | v4a | 38500 | QAT best | 1.5734 | 4.82 | First QAT to nearly match bf16 (STE backward) |
| `runs/tiny_hetmoe_v4a/checkpoints/best_qat.pt` | v4a | (live) | QAT best | (live) | (live) | Currently in v4a's run dir |
| `runs/tiny_hetmoe_v4a/checkpoints/best.pt`     | v4a | (none yet) | — | — | — | Resume init had bf16=1.5155 from v3; never beaten in QAT mode |

## Currently deployed

`docs/tiny.bin` ← exported from `/tmp/v4a_best_qat.pt` (PPL 4.82).
GitHub Pages serves this at https://surajbhan.github.io/tinyhetmoe/

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
since `best.pt` may contain FP weights that lose information when
post-quantized at export time.
