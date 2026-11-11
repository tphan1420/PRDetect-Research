# V3-CorrectedStruct code audit

Date: 2026-11-11. Scope: code and the artifacts available in this checkout.
No training, benchmark, cache building, checkpoint deletion, or checkpoint
overwrite was performed.

## Findings

| Area | Status | Evidence |
| --- | --- | --- |
| Corrected swap dispatch | PASS | `CorrectedStructuralMappingNet` overrides `_augment`; it refuses any mode other than `directed`. |
| All valid dependency IDs form separate swap roles | PASS in code; runtime test pending | The adapter assigns `node_rel[src] = rel` for every non-self edge, with `R` as the root/unknown sentinel. It does not use `R // 2`. |
| Class-preserving swap was not added | PASS | The override accepts `labels` only for base-class compatibility and never reads it. |
| Swap limits and probabilities | PASS in code | It preserves V3's `rho ~ U(0, aug_max)`, node selection, `both` 50/50 mask choice, and donor shuffle. |
| Directed structural features | PASS in code; runtime test pending | `fwd_mask(..., "directed")` retains every relation; `parent_array` removes self loops and enforces one non-self parent. |
| Root/unknown separation | PASS in code | Root self loops are excluded. Nodes without a non-self dependency edge use sentinel `R`, which cannot equal a valid ID in `[0, R)`. |
| Parent cycle validation | FIXED | A directed cycle previously reached the `MAXD` loop in `struct_features` and produced artificial depth. `parent_array` now rejects it before feature construction. This only affects malformed directed caches. |
| Legacy `low` / `high` behavior | PASS in source diff; runtime test pending | The only changed branch in `fwd_mask` is opt-in `directed`; legacy masks retain their old predicates. R-GCN message passing is unchanged. |
| Graph builder default schema | PASS for the workflow code | `building_graph.py` appends only `dependent -> head` by default. Reverse edges are added only with `--bidirectional`; `run_v3_ablation.py` invokes the builder without that flag. |
| Real cache direction, relation IDs, and inverse edges | CHƯA XÁC MINH | This checkout has no retained `.pkl` caches from the CorrectedStruct run. |
| `R=51` against the actual caches | CHƯA XÁC MINH | `dep_vocab.json` contains 60 labels, while the experiment must prove that its training cache maximum is 50 and thus inferred `R` is 51. |
| Seed-2025/2026 checkpoint metadata | CHƯA XÁC MINH | No `v3_structmap_seed2025.pt` or `v3_structmap_seed2026.pt` exists in this checkout. |
| Trainer/evaluator wiring | PASS in code | Training maps `structmap` to `CorrectedStructuralMappingNet`; it requires `directed`. The evaluator checks method and stored mode, then constructs the same adapter with `directed` before strict state loading. |

## Exact code locations

- Directed relation masking, parent validation, depth/children/distance features: `model/QuangV3.py`, `fwd_mask`, `parent_array`, and `struct_features`.
- Corrected syntax-aware swap grouping: `model/V3StructuralMap.py`, `CorrectedStructuralMappingNet._augment`.
- Training model selection and directed-only guard: `train_v3.py`, `validate_args` and `build_model`.
- Checkpoint fields: `train_v3.py`, `train_one` checkpoint payload.
- Evaluation model selection and strict restoration: `evaluate_v3_quang.py`, `_load_model` and graph construction.
- Graph direction: `building_graph.py`, `build_graph`.

## Tests

`tests/test_v3_structmap.py` now covers the following runtime invariants:

1. IDs `12, 25, 34, 42, 50` all produce parent/child/depth/distance features in `directed`; the same graph differs under legacy `low`.
2. A node with relation 34 cannot swap with nodes of relation 25, even though both IDs are at least 25.
3. Multiple non-self parents and cycles are rejected in `directed`.
4. Legacy `low` and `high` masks and parent assignment retain their original formulas.

Run on Kaggle, where PyTorch and NumPy are installed:

```python
import subprocess, sys

for pattern in ("test_v3_quang.py", "test_v3_structmap.py"):
    run = subprocess.run([
        sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v",
    ])
    assert run.returncode == 0, pattern
```

The local static checks passed. The local selected environment has neither
PyTorch nor NumPy, so the seven numerical CorrectedStruct tests were collected
but skipped; this is not evidence that they passed at runtime.

## Read-only artifact audit on Kaggle

The new `audit_correctedstruct.py` reads checkpoint metadata and graph caches,
then writes a new JSON report only if that path does not already exist. It does
not build data or run inference. First locate retained caches:

```python
from pathlib import Path

WORK = Path('/kaggle/working/v3_structmap_table3_table4')  # adjust to the completed run
cache_paths = sorted(WORK.rglob('*.pkl'))
for path in cache_paths:
    print(path)
```

Then audit both seed checkpoints and the relevant HC3/Table-3/Table-4 caches:

```python
import subprocess, sys

report = WORK / 'correctedstruct_code_data_audit.json'
command = [
    sys.executable, 'audit_correctedstruct.py',
    '--checkpoint-dir', str(WORK / 'model'),
    '--seeds', '2025', '2026',
    '--graph-cache', *map(str, cache_paths),
    '--output', str(report),
]
run = subprocess.run(command, check=True, text=True)
print('Audit report:', report)
```

The report prints method, mode, `R`, readout, optimizer/training settings,
completion status, provenance, observed relation IDs, range violations,
duplicate parents, reverse-edge pairs, and low-vs-directed feature differences.

## Checkpoint decision

The cycle validation patch does not alter a valid dependency forest and does
not modify the R-GCN weights, readout, optimizer, swap probability, or legacy
mapping modes. Therefore an existing CorrectedStruct checkpoint remains valid
for comparison only after its retained graph audit reports no malformed parent
forest and in-range relation IDs. If the report finds a cycle, duplicate parent,
out-of-range ID, or inverse-edge schema, that checkpoint is not a clean test of
the intended directed mapping and must be retrained from a new work directory.

The observed accuracy decrease is not evidence of an implementation fault. The
code-level change intentionally exposes structural features and swap roles for
relations 25--50 that legacy `low` ignored. Whether that hurts cross-domain
generalization is a hypothesis for a subsequent controlled experiment, not a
reason to change the implementation.
