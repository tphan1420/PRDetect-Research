# V3 Class-Preserving Syntax Swap

## Mục tiêu

Thử nghiệm này giữ nguyên V3 `V1_swap` và chỉ thay đổi nơi lấy donor cho
nhánh augmentation `swap`:

| Variant | Donor của một node đã được chọn để swap |
|---|---|
| `baseline` | Một node có cùng vai trò quan hệ cú pháp trong micro-batch; có thể thuộc nhãn khác và cùng graph |
| `classswap` | Một node có cùng vai trò quan hệ cú pháp, **cùng nhãn AI/Human** và thuộc **graph khác** |

Nếu không có donor thỏa cả ba điều kiện, node đó giữ nguyên. Không fallback
sang nhãn khác. Kiến trúc R-GCN, encoder, relation vocabulary, tỷ lệ
`aug_max=0.35`, BCE clean/augmented, EMA, optimizer, lịch train và cách chọn
checkpoint đều không đổi.

Đây là ablation độc lập với CMixStyle và R-Drop. Nó kiểm tra giả thuyết rằng
swap xuyên lớp đã làm nhiễu tín hiệu phân biệt AI/Human, đặc biệt khi train
HC3 nhưng test sang GPT-3.5. Kết quả chưa được giả định là sẽ tốt hơn.

## Code đã tách

| File | Vai trò |
|---|---|
| [model/QuangV3.py](../model/QuangV3.py) | V3 gốc; chỉ truyền `y` tùy chọn vào hook augmentation, nên hành vi baseline không đổi |
| [model/V3ClassSwap.py](../model/V3ClassSwap.py) | Adapter không có tham số học, thay riêng quy tắc chọn donor |
| [train_v3.py](../train_v3.py) | Train `baseline` và `classswap`, lưu metadata checkpoint/coverage riêng |
| [evaluate_v3_quang.py](../evaluate_v3_quang.py) | Load đúng biến thể bằng `--variant classswap` |
| [run_v3_ablation.py](../run_v3_ablation.py) | Workflow build → train → dọn graph train → test từng split → ZIP |
| [tests/test_v3_classswap.py](../tests/test_v3_classswap.py) | Kiểm tra donor cùng nhãn, khác graph, không có donor thì identity, và strict-load trọng số V3 |

`classswap` không có trọng số mới: state dict của V3 baseline strict-load được
vào adapter. Dù vậy phải **train lại từ đầu** để augmentation tác động lên
quá trình học; không test một checkpoint V3 cũ với `--variant classswap`.

## Cell Kaggle

Bật GPU và Internet. Các cell giả định code này đã được push lên nhánh clone.
Workflow áp đặt tối đa 4 file `.pkl` graph và xóa riêng cache train/test do nó
tạo sau khi dùng; không xóa checkpoint hoặc dữ liệu nguồn.

### Cell 1 — clone

```python
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
if not REPO.exists():
    !git clone --branch feat/rgcn-rdrop https://github.com/tphan1420/PRDetect-Research.git "{REPO}"
    assert _exit_code == 0

%cd /kaggle/working/PRDetect-Research
assert (REPO / 'model/V3ClassSwap.py').is_file(), 'Clone chưa có Class-Preserving Syntax Swap.'
```

### Cell 2 — cài môi trường

```python
%pip install -q -r requirements-v3-eval.txt
!python -m spacy download en_core_web_sm
assert _exit_code == 0

!python -c "import torch, spacy, sklearn; assert torch.cuda.is_available(), 'Enable GPU'; spacy.load('en_core_web_sm'); print('Environment OK:', torch.__version__, spacy.__version__)"
assert _exit_code == 0
```

Nếu Kaggle yêu cầu restart sau Cell 2, restart kernel rồi chạy lại từ Cell 1.
Không cài lại `torch` của Kaggle.

### Cell 3 — train và test ablation

```python
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
WORK = Path('/kaggle/working/v3_classswap')
SEEDS = [2025, 2026]
SEED_ARGS = ' '.join(map(str, SEEDS))
VARIANTS = ['baseline', 'classswap']
VARIANT_ARGS = ' '.join(VARIANTS)

# Nếu CUDA OOM: giảm đồng thời xuống 4 cho cả baseline và classswap.
MICRO_BATCH = 8

%cd /kaggle/working/PRDetect-Research
!python run_v3_ablation.py --work-dir "{WORK}" --experiment-name v3_classswap --seeds {SEED_ARGS} --variants {VARIANT_ARGS} --micro-batch-size {MICRO_BATCH} --batch-size 32 --val-batch-size 64 --eval-batch-size 4 --max-graphs 4 --suite full
assert _exit_code == 0, 'Xem lỗi phía trên; checkpoint/report hoàn tất vẫn được giữ lại.'
```

Lệnh chạy `full` gồm Table 2 (HC3 in-domain), Table 3 (HC3 → GPT-3.5) và
Table 4 (HC3 structural noise). Baseline và classswap dùng cùng seed,
hyperparameter, cache train, split test và threshold 0.5. `comparison.csv`
chứa `delta_accuracy_vs_baseline_pp`; Table 4 Average không tính Original.

### Cell 4 — xem/tải kết quả

```python
import csv
from IPython.display import FileLink, display

with (WORK / 'comparison.csv').open(encoding='utf-8', newline='') as stream:
    for row in csv.DictReader(stream):
        print(row['variant'], '|', row['suite'], '|', row['condition'], '| acc=', row['accuracy_mean'], '| delta pp=', row['delta_accuracy_vs_baseline_pp'])

archive = WORK / 'v3_classswap_ablation_results.zip'
assert archive.is_file(), archive
display(FileLink(str(archive)))
```

Kết quả cần gửi lại là ZIP này. Để kết luận thắng benchmark, đối chiếu cùng
seed/split với GCN bạn đã tự chạy; không so trực tiếp với số một-seed trong
paper.
