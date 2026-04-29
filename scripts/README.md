# scripts/

Data prep and one-off utilities.

Will house:
- `download_tinystories.py` — pull the dataset from HuggingFace
- `tokenize_corpus.py` — apply the 5K tokenizer, write `train.bin` /
  `val.bin` (uint16, since vocab fits)
- `make_meaning_axes.py` — extract 32-axis contextual meaning embeddings
  by running TinyStories vocab through Qwen2.5-Coder layer-12 and
  projecting via QR-orthogonalized anchor directions
- `eval_generation.py` — sample stories from the trained model, eyeball
  for coherence
- `export_trace.py` — run the model on a prompt and dump the per-token
  per-layer state (attention, meaning, expert routing, hiddens) for the
  UI to consume

Empty for now. Phase b steps 1, 2, 3, 6, 7.
