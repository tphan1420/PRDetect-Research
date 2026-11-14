# PRISM-Lite F1

F1 is an isolated experiment: the legacy two-layer GCN receives a separate
PRISM graph cache containing sentence-local sequential fallback edges and has
two residual + LayerNorm paths.  The original GCN, its caches, and its
checkpoints are unchanged. Work is on branch `prism`.

## Audited baseline and F1 contract

The implementation was audited against [GCN2](../model/GCN2.py),
[graph construction](../building_graph.py), [training](../gcn.py), and
[evaluation](../test.py). No baseline source file was edited.

- Node `i` is spaCy token `i`. Frozen RoBERTa receives token strings through
  `convert_tokens_to_ids`, **not** BPE realignment. Features remain float32.
- One document is one graph. The baseline embeds groups of 512 tokens and
  concatenates their embeddings; these are **not** separate graph chunks.
  Consequently sentence-local fallback can cross an embedding-group boundary,
  but never a sentence/document/graph boundary.
- Dependency edges remain directed dependent → head, including existing
  root self-loops. Fallback adds both directions for consecutive sentence
  tokens, with no new self-loops. Directed duplicates are removed, dependency
  type `0` takes precedence over fallback type `1`.
- Two PyG `GCNConv` layers retain default normalization/self-loop behavior.
  Each layer computes `Dropout(ReLU(LayerNorm(skip(x) + GCN(x))))`.
  Dimensions are 768 → 256 → 64; dropout is 0.5. The original node linear
  head → graph mean → sigmoid ordering is preserved.
- Adam: lr `1e-4`, weight decay `1e-5`; BCE on probabilities; batch 32;
  40 epochs maximum; patience 5. There is no scheduler (as in the baseline).
  Validation accuracy **ties count as improvement**, save, and reset patience.
  Neither test labels nor target metrics select checkpoints or thresholds.

## Environment

Use the same spaCy parser, frozen RoBERTa assets and library versions for
paired baseline/F1 runs. A working modern PyTorch with `weights_only=True`,
PyTorch Geometric, scikit-learn, spaCy + `en_core_web_sm`, transformers and
tqdm is required. PRISM requires real PyG; it does not silently substitute
the approximate no-PyG fallback from the legacy model.

## Build / train / evaluate

Build source caches from the same raw HC3 splits used by the baseline:

```bash
python -m prism_lite.graph_builder --input-json data/original_text/benchmarks/hc3/hc3_train.json --output prism_graph_data/hc3_train_f1.pkl
python -m prism_lite.graph_builder --input-json data/original_text/benchmarks/hc3/hc3_val.json --output prism_graph_data/hc3_val_f1.pkl
python -m prism_lite.train --train-cache prism_graph_data/hc3_train_f1.pkl --val-cache prism_graph_data/hc3_val_f1.pkl --output model/prism_f1_hc3_seed2025.pt --seed 2025
python -m prism_lite.evaluate --checkpoint model/prism_f1_hc3_seed2025.pt --cache prism_graph_data/hc3_val_f1.pkl --output result/prism_f1_hc3_val_seed2025.json
python -m prism_lite.infer --checkpoint model/prism_f1_hc3_seed2025.pt --text "Example text to classify."
```

Each cache/output checkpoint path must be new; existing files are not silently
overwritten. To force CPU, pass `--cpu` to each applicable command. Caches
under `prism_graph_data/` and default PRISM model artifacts are Git-ignored.

For Tables 2/3/4 build a new cache for each desired raw test JSONL and pass it
to the same evaluator. Example (Table 3 clean, only **after** source training):

```bash
python -m prism_lite.graph_builder --input-json data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json --output prism_graph_data/gpt35_test_f1.pkl
python -m prism_lite.evaluate --checkpoint model/prism_f1_hc3_seed2025.pt --cache prism_graph_data/gpt35_test_f1.pkl --output result/prism_f1_table3_clean_seed2025.json
```

Table 2 uses HC3 clean + `data/perturbed_text/hc3/lexical_synonym/` at
5/10/20/30%; Table 3 uses GPT-3.5 clean +
`data/perturbed_text/gpt35_mixed/lexical_synonym/` at 5/10/20/30%; Table 4
uses HC3 clean + `data/perturbed_text/hc3/structural_noise/` Insert, Repeat,
Replace, Delete 25%. These inputs are reparsed by the same builder. The report
contains accuracy, AUC, human-F1, macro-F1, class recalls, confusion matrix,
original source-row IDs and per-sample probabilities at threshold 0.5.
The AUC is `null` for a single-class input. No benchmark is run automatically.

## Cache and checkpoint provenance

Cache v2 stores `graphs`, `y`, sentence node lists, original row IDs, skipped
empty-row IDs, counts/overlap/growth statistics, source SHA256 and preprocessing
fingerprint. The fingerprint includes builder + baseline source hashes, parser
metadata, tokenizer vocabulary hash, encoder configuration/revision and library
versions. Loading checks shapes, bounds, labels, sentence partitions and the
exact fallback edge set. It rejects legacy or v1 PRISM caches: rebuild these.

Checkpoint includes model + optimizer state, `scheduler_state=None`, epoch,
seed, config/hash, source train/validation hashes, preprocessing fingerprint,
parameter count and validation history. A separate `.train.json` logs every
epoch's train/validation loss/accuracy, source-val class recall, LR, elapsed
time and peak CUDA memory. Inference/evaluation strictly reject mismatching
preprocessing fingerprints. Caches use Python pickle and must be your own
trusted build artifacts; checkpoints load with `weights_only=True`.

Local encoder weights are not content-hashed by this first implementation;
use unchanged assets (including any local `roberta-base/` directory) for a
paired comparison. Matching metadata is not proof of identical local weights.

## Verification and remaining work

```bash
python -m unittest discover -s prism_lite/tests -v
```

Offline tests cover fallback direction/overlap/boundaries, legacy root
self-loops, empty-edge forward/backward, baseline GCN optimizer step, graph
batch isolation and row IDs, baseline feature/dependency parity over 512-token
embedding groups, two-epoch F1 train/save/load/evaluate probability parity,
and rejection of mismatching preprocessing or invalid fallback edges.

The integration fixture replaces spaCy/RoBERTa with controlled token/encoder
objects while running the real baseline and PRISM construction/training code.
It proves call-path parity, not numerical parity with pretrained RoBERTa.
The default F1 model has **427,009 trainable parameters**. Seven offline
tests passed on CPU (PyTorch 2.15.0+cpu / PyG 2.8.1), including raw-text
inference equivalence. The controlled two-epoch smoke is not a pilot result.
Local spaCy, transformers and the English parser model are not installed.
Real-asset feature parity, GPU runtime/VRAM and HC3/Table 2/3/4 accuracy remain
unverified. No claims of benchmark improvement are made.

Only F1 training/inference/evaluation is implemented. Dependency-only cache
construction exists for a future R0 control, but R0 training, E1/P1
attention/gates, automatic table-suite orchestration, pretrained B0→F1
checkpoint conversion, and resume-training CLI are **not** implemented.
