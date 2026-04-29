# tokenizer/

Custom 5K-vocab tokenizer for TinyStories.

Three-tier vocab:
1. ~3,500 word-level tokens (most common TinyStories words, kept whole)
2. ~1,200 BPE merges over the residual
3. ~256 byte fallbacks for anything else
4. Special tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`

See [docs/design.md §3](../docs/design.md) for the rationale.

Will house:
- `train_tokenizer.py` — script to train the tokenizer
- `tokenizer.json` — trained tokenizer artifact (~200 KB)

Empty for now. Phase b step 1.
