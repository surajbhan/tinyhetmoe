# training/

Training script and configs for TinyHetMoE.

Will house:
- `train_tiny_hetmoe.py` — fork of the parent's `train_highway_b.py`,
  scaled down (single GPU, no DDP, single corpus, smaller embeddings)
- `configs/tiny_hetmoe.json` — model + training hyperparameters

Same recipe as production: B (trainable meaning), Highway expansion,
QAT-from-zero, QK-Norm, plateau-detect LR halving, variable-length
training (here: {128, 256, 512} since TinyStories is short).

Empty for now. Phase b step 4.
