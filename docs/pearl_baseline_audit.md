# PEARL P0 — baseline audit

Audit commit: `1bc2460bab43e769e4112280e77a51833d48aef5`. Scope expanded by the subsequent user request to end-to-end PEARL-Lite, including weighted GCN, joint fine-tuning and benchmark smoke evaluation. No baseline retraining/overwriting.

## Raw text → graph

[building_graph.py](../building_graph.py) reads JSONL (despite `.json` extension), parses each entire document using `spacy.load("en_core_web_sm")`. Nodes follow spaCy `token.i`. Parsing is not chunked. Embeddings are chunked at 512 spaCy tokens: `RobertaTokenizer.convert_tokens_to_ids(token.text list)`, then `RobertaModel`, concatenate last hidden states. This is legacy vocabulary lookup, **not BPE alignment**. Preserve this behavior. The builder uses `torch.no_grad()`; `from_pretrained` sets the encoder to eval mode (confirmed at runtime, correcting the initial audit's concern about dropout). RoBERTa comes from local `roberta-base/` if present or Hugging Face; neither revision nor checksum is pinned in the baseline. PEARL records local hashes/resolved revision and preserves the same loading/lookup behavior.

Each token contributes exactly one tensor edge `(token.i, token.head.i)`, dependent → head, including ROOT self-loop. Default is directed, unsymmetrized; `--bidirectional` adds reverse edges only for non-root tokens. Relation labels are encoded through [dep_vocab.py](../dep_vocab.py) when requested; ordinary GCN ignores them. Cache contains `all_token_embeddings`, `all_edge_index`, `y`, optionally `all_edge_type`; no character offsets, UIDs, source row IDs, sentence IDs or text hashes. Failed/empty documents are filtered in current builder; therefore cache row position cannot safely identify a source row without verification.

## Architecture and normalization

[GCN2](../model/GCN2.py) has GCNConv 768→256→64 by default, ReLU and dropout 0.5 after both layers, linear 64→1 per node, mean pool per document, then sigmoid. Training uses BCELoss, prediction threshold ≥0.5. Human=1 if label contains `human`, AI=0 otherwise (numeric labels pass through).

PyG GCNConv defaults normalize=True, source_to_target, add remaining self-loops (ROOT loop retained, missing loops filled), degree accumulated on destinations, coefficient `deg[src]^-1/2 * deg[dst]^-1/2`. It does **not** make this directed graph symmetric. The optional local fallback in the same file differs: adds 1 to degree even for existing ROOT loop and uses a biased linear layer after aggregation. PyG and fallback checkpoint keys/operators differ. PEARL reuses the exact native PyG layers or preserves the fallback operator explicitly; cross-backend state loading fails rather than converting weights.

[Author copy](../TacGia/PRDetect/model/GCN2.py) uses the same two layers and mean pooling for one graph, requires PyG. Current root adds batched pooling and dependency fallback. Author builder uses same directed edges/legacy embeddings, contains environment-specific paths and unresolved globals in its script entry point. Current root is the executable project baseline, author copy is archival evidence.

## Training and evaluation

[gcn.py](../gcn.py) loads prebuilt `graph_data/{hc3_train,hc3_val,hc3_test}.pkl`; seed default 2024, torch/CUDA seed only; Adam lr=1e-4, weight_decay=1e-5; batch=32, max=40 epochs, patience=5; checkpoint selected by validation accuracy (ties overwrite); no scheduler. Default checkpoint `model/{dataset}_gcn_model_{seed}.pth`. Test reloads best checkpoint, reports accuracy, binary Human-positive F1 and AUROC, appends existing results. NumPy/Python RNG, parser/model revisions and split hashes are not saved in this legacy checkpoint.

HC3 source train and validation JSONL files both exist under [HC3](../data/original_text/benchmarks/hc3/). PEARL will use these existing partitions, check IDs/parent IDs and exact/whitespace-normalized text overlap before generating edits, record file hashes and selected IDs. Split-generation history/group IDs cannot be recovered from text/label-only rows: no claim of question-level isolation beyond available metadata. Official source/target test data will not be parsed or used for fitting. Existing checkpoint metrics are historical, not newly reproduced baseline scores.

## Sidecar and adjustments to V2

Generate parser-only metadata from raw text using the same model name, preserving every spaCy token (including whitespace), offsets, sentence membership, POS, dependency, head and tensor edge position. Arc `i` maps to original directed tensor column `i`; exclude self-loop/ROOT from retention loss only, retain them in metadata. Record reverse edge positions when bidirectional is explicitly requested. Exact metadata-to-existing-cache equality remains **unverified** until the actual cache/parser revision can be supplied. No `.pkl` graph caches are present locally at initial audit. No new RoBERTa embeddings or `.pkl` caches are required for P1/P2.

Environment initially: project venv Python 3.13; torch/sklearn/numpy present, spaCy/model/pytest absent. Legacy requirements pin spaCy 3.7.4 and Python 3.8 CUDA wheels, incompatible with this runtime. Installed spaCy 3.8.7 / en_core_web_sm 3.8.0 is a separately versioned **pilot**, not a claim of historical cache parity. Python environment tools initially lacked a client; on continuation configuration succeeded for the same project venv. No original code/checkpoints/results are edited. File hashes, dependency versions and split checks are saved with experiment artifacts.

Split audit: source files have 8,000 train / 1,000 validation rows. Exact-content overlap initially identified 15 distinct hashes; normalized-content/group quarantine removes 17 train rows in the PEARL derivative. Original files remain unchanged. Existing baseline weights may themselves have seen these overlapping texts during historical training; quarantine cannot undo that history. No claim of a leakage-free historical baseline is made.

Current tests verify epsilon=1 parity against the historical HC3 checkpoint on freshly parsed graphs. End-to-end smoke uses that same checkpoint and real frozen RoBERTa embeddings. The diagnostics CLI checks embeddings and edge_index against the actual unchanged builder under the installed parser. Historical cache identity remains unverified because those caches/parser revisions are unavailable. This implementation smoke is not scientific acceptance of PEARL quality; see the smoke report.
