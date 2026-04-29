# model/

Model definition for TinyHetMoE.

Will house:
- `tiny_hetmoe.py` — the model class (Highway + B + QAT + HetMoE,
  scaled to ~13M params: hidden 256, 4 layers, 4 heads, 4 experts)
- `qat_utils.py` — copy of the parent project's QAT utilities (ternary
  forward, vote-backward, QK-Norm). Vendored here so the folder is
  self-contained for GitHub.

Empty for now. Phase b step 4 populates this.
