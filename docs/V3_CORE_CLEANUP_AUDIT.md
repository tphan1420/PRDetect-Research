# V3 Core cleanup audit

Scope: retain Quang's V3 baseline and remove only code paths dedicated to the
discarded experimental variants.  This document records the classification made
before deletion; it is not a deletion list for research artefacts.

## Classification

| Classification | Files / area | Reason |
| --- | --- | --- |
| KEEP | `model/QuangV3.py`, `train_v3.py`, `evaluate_v3_quang.py` | V3 Core model, train, and Table 2/3/4 evaluation path. |
| KEEP | `building_graph.py`, `dep_vocab.py`, `dep_vocab.json`, `graph_data/` | Legacy RoBERTa graph-cache construction and dependency relation encoding. |
| KEEP | `model/V3_quang/` | Existing Quang V3 checkpoints; no checkpoint was overwritten or removed. |
| REFACTOR | `rgcn.py`, `test_rgcn.py` | Inline the legacy-cache validation formerly imported from the removed R-Drop helper; restore single-forward BCE behavior. |
| DELETE | V3 adapter modules, R-Drop helper, ablation runner, their focused tests, and V3-only ablation notebooks | They have no Core dependency and implement one of the removed experiments. |
| UNCERTAIN / RETAIN | `result/`, historical documents, `audit_table3.py`, data, and checkpoints outside V3 Core | They are research artefacts or retrospective analyses.  They may mention removed experiments, but are not imported by the Core runtime and are retained to avoid deleting evidence. |

## Removed source paths

- `model/V3ClassSwap.py`
- `model/V3CMixStyle.py`
- `model/V3Restricted.py`
- `model/V3StructuralMap.py`
- `model/v3_mixstyle.py`
- `rdrop.py`
- `run_v3_ablation.py`
- `audit_correctedstruct.py`
- Focused experimental tests and Kaggle notebooks 05, 07, 08, and 09.
- `docs/MeanLogit_Readout_Experiment.md` and `docs/Table3_V1_V3_Diagnosis.md`,
  because their runnable workflows referred to the removed variants.

## Core compatibility constraints

The retained V3 model continues to use basis-decomposed RGCN layers,
projection/LayerNorm/residual connections, depth/children/distance structural
features, original syntax-aware swap, and mean/max/attention `pool3` readout.
The Core trainer retains AdamW, warmup/cosine scheduling, EMA, `lossaug`
selection, early stopping, and checkpoint fields needed by the evaluator.

The cleanup intentionally does not remove data, cached graphs, result archives,
or the explicitly retained historical material under `docs/archive/`.  Those files
should be archived separately only after a research-owner review.

## Verification performed

- `python -m compileall` passed for the retained V3 entrypoints, legacy RGCN
  runners, and the replacement V3 Core test.
- `python -m unittest tests.test_v3_core -v` passed six offline checks.  They
  cover the Core-only CLI surface, Table 2/3/4 case availability, legacy
  checkpoint filename discovery, and removal of experimental runtime hooks.
- All retained Kaggle notebooks parse as JSON, and `train_v3` plus
  `evaluate_v3_quang` import successfully without Torch.

The current virtual environment does not contain PyTorch or NumPy, so the
forward/backward/optimizer, syntax-swap numerical, strict checkpoint-load, EMA
selection, and predicted-probability comparison checks could not be executed.
No local legacy graph cache was available for a same-checkpoint before/after
prediction comparison.  These checks remain required before claiming numerical
equivalence in a GPU research environment.
