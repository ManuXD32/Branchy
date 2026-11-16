# Trained models

Two models trained from scratch for the article, one pass over TinyStories (537M tokens, 131,000 steps), same
tokenizer (8,192 entries), context 1,024 tokens, 32-bit float weights:

| Folder | Model | Parameters | Held-out loss |
|---|---|---:|---:|
| `sequential-313M/inference.pt` | ordinary transformer, 19 blocks of width 1,152 | 313.4M | 1.143 |
| `branched8-309M/inference.pt` | trunk of 2 blocks, 8 parallel branches of 7 blocks (width 616), linear fusion, 2 final blocks; trained with 20% branch dropout and a one-token delay on remote branches | 309.4M | 1.255 |

They write short children's stories in simple English and nothing else; they exist to compare the two architectures.
Licence of the weights: CC BY 4.0 (`LICENSE-DOCS-AND-MODELS.md`): use and adapt them freely, give credit.
Training data: TinyStories (Eldan and Li, 2023); check its licence before you redistribute derived weights.

Each file is about 1.25 GB. That is above GitHub's limit for files inside a repository (100 MB), so publish them
as **release assets** (limit 2 GB per file) or on a model hub, and point `scripts/get_models.sh` at that place:

```
branchy-model-sequential-313M.zip
branchy-model-branched8-309M.zip
tokenizer.json
SHA256SUMS
```

After downloading, the repository expects:

```
models/sequential-313M/inference.pt
models/branched8-309M/inference.pt
models/tokenizer.json
```

Training them yourself: `src/proxy/train_300m.py` (about 20 hours each on a 20 GB GPU; data preparation with
`src/tools/prepare_tinystories_full.py`), see `REPRODUCE.md`.

The LFM3-1.7B and Qwen3.5-0.8B weights used in the pretrained-model experiments are not included; get them from
their publishers under their own licences (`src/pretrained/lfm3_export.py` cuts LFM3 into the stacks the workers load).
