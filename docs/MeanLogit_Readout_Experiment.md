# MeanLogit Readout: Table 3 and Table 4

`meanlogit` is a named V3 variant. It keeps the V3 encoder, relation schema,
augmentation, optimizer, validation selection, and all graph inputs unchanged.
The only architectural change is the classifier readout: it applies one logit
head to each node and averages those logits per graph. The paired `baseline`
variant remains the V3 `pool3` readout.

Run the following cells after the clone/dependency cells. They train the two
variants with the same HC3 source cache, seeds, and low structural schema, then
evaluate both Table 3 and Table 4 at threshold 0.5.

```python
from pathlib import Path
import os
import sys

REPO = Path('/kaggle/working/PRDetect-Research')
os.chdir(REPO)
sys.path.insert(0, str(REPO))

WORK = Path('/kaggle/working/v3_meanlogit_table3_table4')
WORK.mkdir(parents=True, exist_ok=False)  # Pick a new name when rerunning.
```

```python
!python run_v3_ablation.py \
  --work-dir "$WORK" \
  --experiment-name v3_meanlogit_table3_table4 \
  --variants baseline meanlogit \
  --seeds 2025 2026 \
  --suite all \
  --struct-mode low \
  --batch-size 32 \
  --micro-batch-size 8 \
  --val-batch-size 64 \
  --eval-batch-size 4 \
  --max-graphs 4 \
  --epochs 10 \
  --patience 10 \
  --device cuda
```

```python
import pandas as pd

comparison = pd.read_csv(WORK / 'comparison.csv')
display(comparison[comparison['suite'].isin(['table3', 'table4'])])
print('Archive:', WORK / 'v3_meanlogit_table3_table4_ablation_results.zip')
print('Baseline checkpoints:', WORK / 'model')
```

`run_v3_ablation.py` constructs `baseline` with `pool3` and `meanlogit` with
the MeanLogit readout even though the shared CLI default is `pool3`. The saved
checkpoint names are `v3_baseline_seed*.pt` and `v3_meanlogit_seed*.pt`; each
records its effective readout and the evaluator rejects a MeanLogit checkpoint
whose metadata does not say `meanlogit`.

Select a candidate only if it improves the two-seed Table 3 result without a
material Table 4 structural-noise regression. Then rerun that selected variant
with `--suite full` to include Table 2.
