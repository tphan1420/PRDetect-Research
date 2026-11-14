# V3 Structure-Guided Restricted Embeddings

## Mục tiêu và phạm vi

Biến thể `restricted` kiểm tra giả thuyết rằng V3 học quá mạnh các thành phần
RoBERTa gắn với chủ đề/phong cách HC3, làm giảm khả năng chuyển sang GPT-3.5.
Nó giữ nguyên encoder đóng băng, graph, relation vocabulary, R-GCN, Swap,
EMA, optimizer và protocol chọn checkpoint; chỉ thay đổi cách hai nhóm đặc
trưng đầu vào đi vào R-GCN.

| Nhóm đặc trưng | Cơ chế |
|---|---|
| Semantic | RoBERTa node embedding sau `proj` + `LayerNorm` được nhân với gate theo từng channel trong `(0, 1)` |
| Structural | Structural projection từ Depth / Children / Distance được nhân với gate riêng theo từng channel trong `(0, 2)` |

Semantic gate khởi tạo ở `0.9`. Trong train, loss có thêm soft constraint:

```text
L = 0.5 * BCE(clean) + 0.5 * BCE(swap) + λ * max(mean(g_semantic) - budget, 0)^2
```

Mặc định `budget=0.8`, `λ=0.1`: model được khuyến khích bỏ bớt capacity
semantic nhưng tự chọn channel cần giữ cho bài toán AI/Human. Structural gate
khởi tạo đúng `1.0`, không bị budget penalty, nên có thể tăng/giảm đóng góp
structural thay vì bị triệt tiêu mặc định.

Đây là adaptation có thể ablation của ý tưởng **restricted embeddings**, không
phải bản sao paper. Paper gốc nghiên cứu xóa fixed linear subspace/head/coordinate
bằng các chiến lược như coordinate removal và concept erasure; còn V3 không
giữ attention-head activations và chỉ có node embedding đã trích xuất, nên dùng
gate differentiable theo channel là khả thi hơn. Paper cho thấy loại subspace
gây nhiễu có thể tăng robust cross-domain/cross-generator, nhưng không đảm bảo
cơ chế này sẽ tăng Table 3 của PRDetect. Nguồn: [Kuznetsov et al., 2024](https://aclanthology.org/2024.findings-emnlp.992/).

## Tổ chức code

| File | Vai trò |
|---|---|
| [model/V3Restricted.py](../model/V3Restricted.py) | Adapter có hai gate, semantic-budget loss và thống kê gate |
| [model/QuangV3.py](../model/QuangV3.py) | Thêm hook structural tùy chọn sau `sproj`; baseline không truyền hook nên phép tính cũ giữ nguyên |
| [train_v3.py](../train_v3.py) | Variant `restricted`, regularization loss, checkpoint metadata, log gate |
| [evaluate_v3_quang.py](../evaluate_v3_quang.py) | Strict-load checkpoint `v3_restricted_seed*.pt` với đúng hyperparameter |
| [run_v3_ablation.py](../run_v3_ablation.py) | Workflow memory-safe baseline vs restricted, tối đa 4 graph cache |
| [tests/test_v3_restricted.py](../tests/test_v3_restricted.py) | Test gate, budget loss, gradient và yêu cầu structural features |

Do có thêm hai vector gate học được, checkpoint V3 cũ **không** strict-load
vào `restricted`; phải train lại từ đầu. Đây là đúng cho ablation công bằng.

## Cell Kaggle — Table 3 HC3 → GPT-3.5

Bật GPU và Internet. Các cell giả định code đã được push lên branch đang clone.

### Cell 1 — clone

```python
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
if not REPO.exists():
    !git clone --branch feat/rgcn-rdrop https://github.com/tphan1420/PRDetect-Research.git "{REPO}"
    assert _exit_code == 0

%cd /kaggle/working/PRDetect-Research
assert (REPO / 'model/V3Restricted.py').is_file(), 'Clone chưa có variant restricted.'
```

### Cell 2 — môi trường

```python
%pip install -q -r requirements-v3-eval.txt
!python -m spacy download en_core_web_sm
assert _exit_code == 0

!python -c "import torch, spacy, sklearn; assert torch.cuda.is_available(), 'Enable GPU'; spacy.load('en_core_web_sm'); print('Environment OK:', torch.__version__, spacy.__version__)"
assert _exit_code == 0
```

Không cài lại Torch của Kaggle. Nếu notebook yêu cầu restart sau Cell 2, restart
rồi chạy lại từ Cell 1.

### Cell 3 — train và test controlled ablation

```python
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
WORK = Path('/kaggle/working/v3_restricted_table3')
SEEDS = [2025, 2026]
SEED_ARGS = ' '.join(map(str, SEEDS))
VARIANTS = ['baseline', 'restricted']
VARIANT_ARGS = ' '.join(VARIANTS)

# Nếu CUDA OOM, đổi thành 4 cho cả hai variant.
MICRO_BATCH = 8
SEMANTIC_BUDGET = 0.8
SEMANTIC_PENALTY = 0.1

%cd /kaggle/working/PRDetect-Research
!python run_v3_ablation.py --work-dir "{WORK}" --experiment-name v3_restricted_table3 --seeds {SEED_ARGS} --variants {VARIANT_ARGS} --micro-batch-size {MICRO_BATCH} --batch-size 32 --val-batch-size 64 --eval-batch-size 4 --restricted-semantic-budget {SEMANTIC_BUDGET} --restricted-semantic-penalty {SEMANTIC_PENALTY} --max-graphs 4 --suite table3 --skip-clean-check
assert _exit_code == 0, 'Xem log lỗi phía trên; checkpoint/report hoàn tất vẫn được giữ.'
```

Lệnh này train hai variant từ đầu trên cùng HC3 train/val, cùng seed và
hyperparameter; sau đó chỉ test Table 3 HC3 → GPT-3.5 (Original + synonym
5/10/20/30). Nó không chạy Table 2/4 và không build source-domain sanity split.
Trong bất kỳ thời điểm nào workflow cũng giới hạn 4 file `.pkl`; cache train
và test do workflow tạo được xóa sau khi dùng, checkpoint/report không bị xóa.

Trong log `restricted`, cần thấy dạng:

```text
Restricted sem 0.xxx (<.5 N) struct 1.xxx
```

`sem` giảm dần về/bên dưới budget là dấu hiệu regularizer có tác động; số
`<.5` cho biết bao nhiêu semantic channels đang bị down-weight mạnh. Không chọn
budget/penalty theo GPT-3.5 test: nếu cần thử hyperparameter khác, chọn bằng
HC3 validation rồi chạy lại protocol mới.

### Cell 4 — xem/tải kết quả

```python
import csv
from IPython.display import FileLink, display

with (WORK / 'comparison.csv').open(encoding='utf-8', newline='') as stream:
    for row in csv.DictReader(stream):
        print(row['variant'], '|', row['condition'], '| Acc=', row['accuracy_mean'], '| Δ pp=', row['delta_accuracy_vs_baseline_pp'])

archive = WORK / 'v3_restricted_table3_ablation_results.zip'
assert archive.is_file(), archive
display(FileLink(str(archive)))
```

Đánh giá thành công khi `restricted` tăng Table 3 một cách nhất quán trên cả
hai seed và các noise rate, đồng thời không chỉ tăng một rate đơn lẻ. Sau khi
có tín hiệu, mới chạy Table 2/4 để kiểm tra đánh đổi in-domain và structural
noise.
