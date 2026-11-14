# LISA-GCN Core

LISA-GCN is a standalone research model derived from the original PRDetect
**two-layer GCN**, not from RGCN V1/V3. It keeps spaCy dependency graphs,
legacy frozen RoBERTa node embeddings, binary labels (`human=1`, `AI=0`), and
source-only train/validation splits.

## Implemented Core

- Sentence-aligned windows of 96, 128, and 160 spaCy tokens. Each token occurs
  once per partition; any overlong sentence is deterministically split and
  recorded in graph metadata.
- Shared two-layer GCN and per-window logits; document logit is the mean of
  all window logits, followed by one sigmoid.
- Paired 128 vs. 96/160 views, with BCE-with-logits classification loss and a
  warm-started probability consistency term.
- Source-train length quantiles and log-space GroupDRO. Missing groups in an
  effective batch are excluded, never treated as zero loss.
- `--micro-batch-size` controls documents forwarded at once;
  `--effective-batch-size` controls the set of document losses combined into
  one GroupDRO update and optimizer step. `--window-chunk-size` independently
  limits how many graph windows enter the GCN in one call.
- Checkpoints record model, optimizer, source-fitted group boundaries, GroupDRO
  state, config, seed and preprocessing version.

Legacy caches do not store sentence boundaries. LISA therefore rebuilds its
graphs from raw JSONL and does not slice cached nodes by index.

## Run

```bash
python train_lisa_gcn.py \
  --train-json data/original_text/benchmarks/hc3/hc3_train.json \
  --val-json data/original_text/benchmarks/hc3/hc3_val.json \
  --seed 2025 --output model/lisa_gcn

python infer_lisa_gcn.py --checkpoint model/lisa_gcn/lisa_gcn_seed2025.pt \
  --text "Example text to classify."
```

The JSONL input must have `text` and `label`; labels follow the existing
PRDetect convention. This Core deliberately excludes RGCN/V3 integration,
syntax-aware swap, EMA/SWA, R-Drop, CMixStyle, restricted embeddings,
perturbation consistency, ensembles and hyperparameter search.

## Verified baseline difference

The original [`GCN2`](../model/GCN2.py) returns sigmoid probabilities and is
trained with `BCELoss`. LISA returns logits and uses `BCEWithLogitsLoss` so
mean-logit document aggregation is mathematically well-defined. The original
baseline code is untouched.
