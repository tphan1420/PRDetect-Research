# Thử nghiệm R-Drop trên RGCN V1

## Phạm vi

Nhánh `feat/rgcn-rdrop` được tạo từ `main`. Không dùng EMA, DeBERTa, RoBERTa aligned hoặc A1/A2/A3. Không sửa [building_graph.py](../building_graph.py), [model/RGCN.py](../model/RGCN.py), tokenizer, embedding, graph, pooling hay ngưỡng 0.5.

Mục tiêu là kiểm chứng một thay đổi trong **huấn luyện RGCN**, không khẳng định sẽ tăng khả năng chống nhiễu. R-Drop khuyến khích nhất quán dưới dropout, không trực tiếp huấn luyện chống insert/delete/repeat.

## Công thức và cách triển khai

Trong [rdrop.py](../rdrop.py), cùng batch đi qua cùng model hai lần ở chế độ train. Không reset seed giữa hai lượt, nên dropout lấy hai mask ngẫu nhiên khác nhau. Với `p1`, `p2` là xác suất human:

```text
P = [1-p1, p1]       # [AI, human]
Q = [1-p2, p2]
BCE_mean = 0.5 * (BCE(p1, y) + BCE(p2, y))
KL_sym = mean_batch(0.5 * (KL(P || Q) + KL(Q || P)))
loss = BCE_mean + rdrop_alpha * KL_sym
```

Đây là áp dụng [R-Drop](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) cho classifier sigmoid, đối chiếu với [mã tác giả](https://github.com/dropreg/R-Drop). Không dùng softmax trên tensor `[batch, 1]`, vì như vậy phân phối chỉ có một lớp và KL sẽ luôn bằng 0.

KL Bernoulli đối xứng được tính bằng biểu thức tương đương `0.5 * (p-q) * (logit(p)-logit(q))`, trung bình theo mẫu. Hai nhánh đều nhận gradient. Xác suất được clamp trong `[1e-6, 1-1e-6]` **chỉ khi tính KL** để tránh log(0); BCE giữ nguyên. Half precision được nâng lên float32 cho phép log; double được giữ nguyên cho kiểm tra gradient.

Mỗi batch chỉ có **một** backward và **một** optimizer step. Adam, learning rate, weight decay, dropout, batch size, validation accuracy và quy tắc early stopping giữ nguyên V1. Validation/test chỉ forward một lần ở `eval()`, BCE bình thường, không KL, không MC-dropout hay ensemble. Checkpoint là state_dict thuần, không đổi kiến trúc inference.

### Chế độ baseline

- `--rdrop_alpha 0` (mặc định): **một forward + BCE**, không chạy forward thứ hai, không tiêu thụ thêm RNG cho dropout.
- `--rdrop_alpha 0.1`: bật R-Drop; `0.1` chỉ là cấu hình thử ban đầu, không phải hệ số tối ưu đã chứng minh.
- Alpha phải hữu hạn và không âm. Không có scheduler/warmup hoặc loss phụ khác.

Train accuracy khi bật R-Drop dùng trung bình hai xác suất, chỉ để theo dõi; không dùng để chọn checkpoint. Loss log tách total/BCE/KL/alpha*KL. Cùng seed giữ cùng khởi tạo, nhưng hai lượt dropout làm chuỗi RNG khác baseline; không tuyên bố toàn bộ mask/shuffle giống nhau qua các epoch. CUDA/PyG cũng có thể không bitwise deterministic.

## Các thay đổi trong script

[rgcn.py](../rgcn.py) thêm `--rdrop_alpha`, `--graph_dir` và `--result_dir`. Tên checkpoint mặc định:

```text
alpha=0:    model/<dataset>_rgcn_model_<seed>.pth
alpha>0:    model/<dataset>_rdrop_a<alpha>_rgcn_model_<seed>.pth
```

Vì vậy chỉ bật R-Drop không ghi đè checkpoint V1 cùng dataset/seed. Có thể chỉ định `--save_path`; người dùng chịu trách nhiệm không dùng cùng đường dẫn cho hai điều kiện. Chạy lại cùng cấu hình/tên vẫn ghi đè, không hỗ trợ resume optimizer.

File `<checkpoint>.train.json` ghi cấu hình, phương pháp, alpha, best epoch, best validation accuracy và lịch sử train/val. Summary được ghi khi train kết thúc bình thường; checkpoint tốt nhất được lưu trong quá trình train như V1.

[test_rgcn.py](../test_rgcn.py) thêm `--checkpoint`, `--graph_dir`, `--result_dir`, `--report_json`. Report có accuracy/AUC/F1, confusion matrix (hàng nhãn thật, cột dự đoán; thứ tự AI/human), recall AI/human, nhãn thật và `p_human` từng mẫu. F1 vẫn là F1 lớp human như pipeline cũ. Khi yêu cầu report, sklearn là bắt buộc để không âm thầm ghi metric fallback bằng 0.

Train/test từ chối các cache có metadata aligned hoặc schema V2 đã biết. Tuy nhiên cache V1 cũ không có provenance đầy đủ; người dùng vẫn phải bảo đảm dùng đúng graph legacy. Không lấy graph từ thư mục `roberta_aligned/`, `deberta_aligned/` hoặc `graph_data_v2/`.

## Các cell Kaggle

Bật GPU và Internet. Cần commit/push nhánh này trước khi clone. Nếu có sẵn graph V1 đáng tin cậy, có thể dùng lại, **không cần build lại**. Hai điều kiện phải dùng chung chính xác cache train/val; không build lại giữa hai lượt train.

### Cell 1: clone

```python
%cd /kaggle/working
!git clone --branch feat/rgcn-rdrop https://github.com/tphan1420/PRDetect-Research.git
%cd /kaggle/working/PRDetect-Research
```

### Cell 2: thư viện và unit test

Nếu đang dùng môi trường V1 ổn định, giữ môi trường đó. Không cài requirements cũ chứa wheel Python/CUDA đã lỗi thời; không chủ động hạ NumPy/pandas hay thay PyTorch Kaggle. Cài bên dưới dành cho session mới. Phiên bản thực tế của môi trường vẫn phải kiểm tra bằng lệnh import.

```python
%pip install -q "transformers==4.44.2" "torch-geometric==2.5.2" spacy scikit-learn tqdm
!python -m spacy download en_core_web_sm
assert _exit_code == 0
!python -c "import torch, spacy, sklearn; from transformers import RobertaModel; from torch_geometric.nn import RGCNConv; assert torch.cuda.is_available(), 'Enable GPU'; spacy.load('en_core_web_sm'); print('Environment OK')"
assert _exit_code == 0

from pathlib import Path
assert Path('tests/test_rdrop.py').is_file(), 'Wrong branch or outdated clone'
!python -m unittest discover -s /kaggle/working/PRDetect-Research/tests -p "test_rdrop.py" -v
assert _exit_code == 0
```

### Cell 3: cấu hình và build graph V1 có giới hạn số file

Không dùng pandas hay subprocess. Truyền đường dẫn đầy đủ có `.json` để tránh nhầm tên split chứa dấu chấm như GPT3.5. Giới hạn 4 file tính đệ quy cả cache encoder khác dưới graph root; không tự xóa các file không thuộc thử nghiệm này.

```python
import json
import shutil

REPO = Path('/kaggle/working/PRDetect-Research')
GRAPH = REPO / 'graph_data'
RESULT = REPO / 'result/rdrop_check'
RESULT.mkdir(parents=True, exist_ok=True)
GRAPH.mkdir(exist_ok=True)

BASELINE = 'model/hc3_rdrop_exp_rgcn_model_2026.pth'
RDROP = 'model/hc3_rdrop_exp_rdrop_a0.1_rgcn_model_2026.pth'

SPLITS = [
    'hc3_test', 'hc3_test_insert_25', 'hc3_test_delete_25',
    'hc3_test_repeat_25', 'gpt3.5_mixed_test_split',
    'gpt3.5_mixed_test_adj_30_synonym_replace',
]

def build(split):
    target = GRAPH / f'{split}.pkl'
    if target.exists():
        print('Reusing cache (must be V1):', target.name)
        return
    assert len(list(GRAPH.rglob('*.pkl'))) < 4, 'Remove finished graph caches first'
    sources = list((REPO / 'data/original_text').rglob(split + '.json'))
    sources += list((REPO / 'data/perturbed_text').rglob(split + '.json'))
    assert len(sources) == 1, (split, sources)
    source = sources[0]
    !python building_graph.py --file "{source}"
    assert _exit_code == 0 and target.exists(), f'Build failed: {split}'
    print('Free disk GB:', round(shutil.disk_usage(REPO).free / 2**30, 2))

build('hc3_train')
build('hc3_val')
```

### Cell 4: train baseline và R-Drop tuần tự trên cùng cache

Không dùng tên checkpoint của các thử nghiệm cũ. Nếu OOM khi R-Drop, giảm batch size **ở cả hai lệnh** và chạy lại cả hai điều kiện, không so sánh hai batch size khác nhau rồi quy chênh lệch cho R-Drop.

```python
!python rgcn.py --dataset hc3_rdrop_exp --seed 2026 --rdrop_alpha 0 --no_test
assert _exit_code == 0 and Path(BASELINE).exists()

!python rgcn.py --dataset hc3_rdrop_exp --seed 2026 --rdrop_alpha 0.1 --no_test
assert _exit_code == 0 and Path(RDROP).exists()
```

Có thể chạy riêng một lệnh nếu điều kiện còn lại đã train xong. Không train lại chỉ vì test bị lỗi. Thư viện trích embedding không được nạp trong train; chỉ RGCN dùng GPU.

### Cell 5: xóa cache train/val, test từng graph cho cả hai checkpoint

Chỉ xóa đúng cache đã dùng sau khi có đủ hai checkpoint. Không xóa dữ liệu nguồn hoặc trọng số. Khi chạy lại cell này, report test đã tồn tại được giữ và không test lại nếu checkpoint ghi trong report trùng; nếu đã retrain ghi đè checkpoint cùng tên, hãy dùng RESULT mới để không tái sử dụng report cũ.

```python
assert Path(BASELINE).exists() and Path(RDROP).exists()
assert GRAPH.resolve() == (REPO / 'graph_data').resolve()
for split in ['hc3_train', 'hc3_val']:
    cache = GRAPH / f'{split}.pkl'
    if cache.exists():
        cache.unlink()

for split in SPLITS:
    jobs = []
    for mode, checkpoint in [('baseline', BASELINE), ('rdrop', RDROP)]:
        report = RESULT / f'{mode}_{split}.json'
        if report.exists():
            saved = json.loads(report.read_text())
            assert saved['checkpoint'] == checkpoint and saved['split'] == split
        else:
            jobs.append((mode, checkpoint, report))
    if jobs:
        build(split)
    for mode, checkpoint, report in jobs:
        !python test_rgcn.py --dataset hc3_rdrop_exp --seed 2026 --file {split} --checkpoint {checkpoint} --graph_dir "{GRAPH}" --result_dir "{RESULT}" --report_json "{report}" --save
        assert _exit_code == 0 and report.exists(), (mode, split)
    cache = GRAPH / f'{split}.pkl'
    if cache.exists():
        cache.unlink()
```

### Cell 6: so sánh kết quả

```python
import csv

rows = []
for split in SPLITS:
    base = json.loads((RESULT / f'baseline_{split}.json').read_text())
    trial = json.loads((RESULT / f'rdrop_{split}.json').read_text())
    assert base['y_true'] == trial['y_true'], split
    row = {'split': split, 'baseline_acc': base['accuracy'], 'rdrop_acc': trial['accuracy'],
           'delta_acc_pp': 100 * (trial['accuracy'] - base['accuracy']),
           'baseline_auc': base['auc'], 'rdrop_auc': trial['auc'],
           'baseline_f1': base['f1'], 'rdrop_f1': trial['f1'],
           'baseline_recall_ai': base['recall_ai'], 'rdrop_recall_ai': trial['recall_ai'],
           'baseline_recall_human': base['recall_human'], 'rdrop_recall_human': trial['recall_human']}
    rows.append(row)
    print(f"{split:50} baseline={base['accuracy']:.4f} R-Drop={trial['accuracy']:.4f} "
          f"delta={row['delta_acc_pp']:+.2f} pp")

with (RESULT / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
```

### Cell 7: đóng gói

```python
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display

archive = Path('/kaggle/working/rgcn_rdrop_results.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
    for file in RESULT.rglob('*'):
        if file.is_file():
            z.write(file, str(file.relative_to(REPO)))
    for checkpoint in [BASELINE, RDROP]:
        for file in [REPO / checkpoint, REPO / (checkpoint + '.train.json')]:
            assert file.exists(), file
            z.write(file, str(file.relative_to(REPO)))
display(FileLink(str(archive)))
```

## Bộ nhớ, kết luận và giới hạn

R-Drop tăng hai forward và giữ activation hai lượt đến backward; GPU memory tăng, không chỉ tăng thời gian. Không tăng số graph cache, không thêm model encoder hay optimizer thứ hai. Giới hạn số file **không bảo đảm** tổng RAM/đĩa dưới 20 GB: một graph quá lớn hoặc train+val cũng có thể vượt bộ nhớ. Workflow giữ tối đa hai cache train/val, rồi một cache test; subprocess kết thúc sẽ trả bộ nhớ GPU của bước đó.

Chọn alpha/epoch bằng validation, không dò alpha/threshold theo test. Không coi loss tổng R-Drop và BCE baseline là cùng một đại lượng để so số trực tiếp. Xem cả accuracy/AUC/F1 và recall từng lớp; nếu structural noise giảm mạnh thì không giữ chỉ vì clean/GPT3.5 tăng. Thử nghiệm ban đầu chỉ có một seed, nếu cải thiện nhỏ cần xác nhận thêm seed. Baseline cũ khác seed hoặc khác cache không phải đối chứng đủ để quy chênh lệch cho R-Drop.

Unit test trong [tests/test_rdrop.py](../tests/test_rdrop.py) kiểm tra alpha/cache, công thức KL hai lớp, gradient hai nhánh, biên xác suất, baseline một forward/RNG, R-Drop hai forward và tương thích state_dict RGCN. Các kiểm tra có tensor cần PyTorch và sẽ bị skip ở môi trường không có PyTorch; không được coi skip là đã kiểm chứng số học. Chưa có kết quả train/test Kaggle để khẳng định hiệu quả.
