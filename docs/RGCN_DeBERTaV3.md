# Thử nghiệm encoder đóng băng: DeBERTa-v3-base + RGCN

## 1. Phạm vi và giả thuyết

Nhánh `feat/rgcn-deberta-v3` được tạo từ `main`, không mang EMA hoặc A1/A2/A3 vào thử nghiệm. Mục tiêu là kiểm chứng việc thay encoder có cải thiện RGCN, không khẳng định trước kết quả.

Giữ nguyên [kiến trúc RGCN](../model/RGCN.py), hướng cạnh dependent → head và self-loop ROOT của V1, bảng quan hệ, nhãn AI=0/human=1, BCE loss, Adam, dropout và ngưỡng 0.5. DeBERTa không được fine-tune: encoder chỉ chạy khi build cache, chỉ RGCN được train. Không có nhánh CLS, gate, hybrid, noise augmentation hoặc loss mới.

Đối chứng gồm:

| Cấu hình | Mục đích |
| --- | --- |
| RoBERTa legacy + RGCN | Baseline cũ, dùng builder cũ không sửa |
| RoBERTa aligned + RGCN | Đo tác động của tokenization/alignment, eval và chia đoạn chuẩn |
| DeBERTa-v3-base aligned + RGCN | So với RoBERTa aligned để đo tác động đổi encoder |

Hai cấu hình aligned cùng chính sách gộp subword/chia đoạn nhưng tokenizer của hai model khác nhau, nên ranh giới đoạn có thể khác. Không quy mọi chênh lệch với legacy cho riêng DeBERTa. Việc chuẩn hóa alignment có liên quan đến A1 đã từng giảm structural robustness; cần đối chứng, không giả định sửa tokenization sẽ chắc chắn tăng điểm.

Con số 97.53% accuracy / 99.53% AUC trong [Update.md](./Update.md) không phải bằng chứng cho pipeline này trên RAID. [DeBERTa-Sentinel](https://arxiv.org/abs/2608.01046) báo cáo các số đó trên GLC-AIText. Tham khảo kỹ thuật: [DeBERTaV3](https://arxiv.org/abs/2111.09543), [model card Microsoft](https://huggingface.co/microsoft/deberta-v3-base), [Hugging Face tokenizer](https://huggingface.co/docs/transformers/v4.44.2/main_classes/tokenizer).

## 2. Những thay đổi trong code

### Builder riêng, không sửa baseline

[building_graph_encoder.py](../building_graph_encoder.py) hỗ trợ:

- `--encoder deberta`: `microsoft/deberta-v3-base`, 768 chiều.
- `--encoder roberta`: `roberta-base`, đối chứng alignment, 768 chiều.
- `--file`: một hoặc nhiều tên split/path JSONL.
- `--graph-root graph_data`: cache dưới `deberta_aligned/` hoặc `roberta_aligned/`.
- `--max-graphs 4`: đếm **tất cả** file `.pkl` đệ quy dưới graph root, kể cả cache legacy và encoder khác; từ chối build nếu tổng vượt giới hạn. Không tự xóa cache.
- `--max-length 512`: số subword tối đa mỗi lượt, bao gồm CLS/SEP.
- `--revision`: revision Hugging Face, nên pin commit khi cần tái lập nghiêm ngặt.
- `--seed 2026`, `--cpu`, `--spacy-model en_core_web_sm`.
- `--overwrite`: chỉ ghi đè khi yêu cầu rõ; mặc định cache tồn tại gây lỗi.

Encoder dùng fast tokenizer với `is_split_into_words=True` và `word_ids()`, RoBERTa thêm `add_prefix_space=True`. Không tra trực tiếp từ spaCy bằng `convert_tokens_to_ids`. Model gọi `requires_grad_(False)`, `eval()` và `no_grad()`.

Chuỗi subword được chia thành các cửa sổ không chồng lấn tối đa 510 subword nội dung, thêm CLS/SEP. Duyệt **toàn bộ** văn bản, không truncate đuôi. Cửa sổ có thể cắt giữa các subword của cùng một từ; cộng dồn tất cả về cùng node rồi chia tổng số subword. Mean pooling không tính CLS/SEP. Ngữ cảnh xuyên cửa sổ vẫn bị giới hạn; đây không phải encoder long-context.

Giữ nguyên một node cho mỗi token spaCy và toàn bộ cạnh như V1. Nếu tokenizer bỏ qua node chỉ chứa khoảng trắng, node đó nhận một UNK tường minh để không thay topology; thống kê riêng số node fallback. Nếu một node có nội dung bị mất alignment, builder dừng với tên file và dòng lỗi, không âm thầm bỏ mẫu. Nhãn ngoài danh mục hỗ trợ cũng gây lỗi để tránh gán nhầm.

Tích lũy embedding từng cửa sổ trên CPU, cache float32; không giữ toàn bộ activation của tài liệu trên GPU. Mỗi graph cache lưu cấu trúc dữ liệu V1 cùng `encoder_metadata`: encoder/revision, tokenizer/library/parser version, alignment, vocabulary, source SHA256, chỉ số dòng, số mẫu và thống kê UNK. File JSON đi kèm giúp kiểm tra mà không load graph lớn.

### Chặn dùng nhầm dữ liệu/checkpoint

[encoder_graph.py](../encoder_graph.py) kiểm tra fingerprint cấu hình. Train không cho trộn legacy với aligned, hoặc DeBERTa với RoBERTa. Test yêu cầu cache aligned khớp manifest checkpoint. Không dùng checkpoint RoBERTa cũ cho DeBERTa, dù cùng 768 chiều.

[rgcn.py](../rgcn.py) thêm `--graph_dir`, `--result_dir`; giữ nguyên thuật toán train và quy tắc chọn best validation accuracy/early stopping V1. Checkpoint vẫn là `state_dict` thuần; thêm:

- `<checkpoint>.encoder.json`: provenance và cấu hình lúc lưu checkpoint tốt nhất.
- `<checkpoint>.train.json`: lịch sử train/validation.

[test_rgcn.py](../test_rgcn.py) thêm `--checkpoint`, `--graph_dir`, `--result_dir`, `--report_json`. Báo cáo JSON có accuracy/AUC/F1, confusion matrix, recall AI/human và `p_human` từng mẫu. Log ghi checkpoint và graph. CLI cũ vẫn dùng `graph_data/` và tên checkpoint V1 khi không truyền tham số mới.

**Khi tải hoặc di chuyển trọng số aligned phải mang theo file `.pth.encoder.json`.** Cache metadata phải giữ nguyên; manifest không phải file tùy chọn cho các graph aligned. Chỉ load pickle/checkpoint từ nguồn tin cậy.

## 3. Các cell Kaggle

Bật GPU và Internet. Code phải được commit/push lên nhánh GitHub trước khi clone. Không cài toàn bộ requirements cũ (chứa wheel Python/CUDA cũ). Nếu môi trường đang chạy ổn, giữ lại phiên bản thư viện/parser đó ở cả hai lần chạy. Đổi phiên bản thư viện giữa build train/test sẽ bị fingerprint chặn.

### Cell 1: clone

```python
%cd /kaggle/working
!git clone --branch feat/rgcn-deberta-v3 https://github.com/tphan1420/PRDetect-Research.git
%cd /kaggle/working/PRDetect-Research
```

### Cell 2: dependency và kiểm tra

Giữ PyTorch có sẵn trên Kaggle; không chủ động ép hạ NumPy/pandas. `sentencepiece` và `protobuf` cần cho việc chuyển tokenizer DeBERTa sang fast tokenizer. Môi trường Kaggle thay đổi theo thời gian: nếu cell kiểm tra import lỗi, dừng và sửa môi trường trước, không coi log metric fallback là kết quả hợp lệ.

```python
%pip install -q "transformers==4.44.2" "torch-geometric==2.5.2" sentencepiece protobuf spacy scikit-learn tqdm
!python -m spacy download en_core_web_sm
assert _exit_code == 0
!python -c "import torch, spacy, sklearn, sentencepiece; from transformers import AutoTokenizer, AutoModel; from torch_geometric.nn import RGCNConv; assert torch.cuda.is_available(), 'Enable GPU'; spacy.load('en_core_web_sm'); print('Environment OK')"
assert _exit_code == 0
!python -m unittest discover -s tests -v
assert _exit_code == 0
```

### Cell 3: chọn encoder

Chạy cell 3–6 với `ENCODER='roberta'` trước. Sau đó đổi sang `'deberta'`, chạy lại cell 3–6. Không phải clone/cài lại. Dùng cùng seed và tham số RGCN.

```python
from pathlib import Path
import json
import shutil

REPO = Path('/kaggle/working/PRDetect-Research')
ENCODER = 'roberta'  # Lượt 2: đổi thành 'deberta'
NAME = f'hc3_{ENCODER}_aligned'
GRAPH = REPO / 'graph_data' / f'{ENCODER}_aligned'
RESULT = REPO / 'result/encoder_comparison' / ENCODER
WEIGHTS = f'model/{NAME}_rgcn_model_2026.pth'
RESULT.mkdir(parents=True, exist_ok=True)

SPLITS = [
    'hc3_test', 'hc3_test_insert_25', 'hc3_test_delete_25',
    'hc3_test_repeat_25', 'gpt3.5_mixed_test_split',
    'gpt3.5_mixed_test_adj_30_synonym_replace',
]

def build(split):
    !python building_graph_encoder.py --encoder {ENCODER} --file {split} --max-graphs 4 --seed 2026
    assert _exit_code == 0, f'Build failed: {split}'
    assert (GRAPH / f'{split}.pkl').exists()
    print('Free disk GB:', round(shutil.disk_usage(REPO).free / 2**30, 2))
```

### Cell 4: build hai graph và train

```python
build('hc3_train')
build('hc3_val')

!python rgcn.py --dataset {NAME} --seed 2026 --graph_dir "{GRAPH}" --result_dir "{RESULT}" --no_test
assert _exit_code == 0
assert Path(WEIGHTS).exists()
assert Path(WEIGHTS + '.encoder.json').exists()
```

Nếu cache đã tồn tại từ lần chạy dở, không gọi lại `build` một cách mù quáng. Kiểm tra JSON metadata trước khi tái sử dụng; mặc định builder từ chối ghi đè. Nếu cần train lại và giữ kết quả cũ, đổi `NAME` trước.

### Cell 5: giải phóng train/val và test tuần tự

Cell này xóa **đúng hai cache train/val của encoder đang chọn**, không xóa dữ liệu nguồn hay checkpoint. JSON provenance được giữ lại. Sau đó mỗi lần chỉ build một graph test, lưu report rồi mới xóa cache graph đó. Không chạy hai kernel cùng build vào một graph root.

```python
assert Path(WEIGHTS).exists() and Path(WEIGHTS + '.encoder.json').exists()
assert GRAPH.resolve().parent == (REPO / 'graph_data').resolve()
for split in ['hc3_train', 'hc3_val']:
    path = GRAPH / f'{split}.pkl'
    if path.exists():
        path.unlink()

for split in SPLITS:
    build(split)
    report = RESULT / f'{split}.json'
    !python test_rgcn.py --dataset {NAME} --seed 2026 --file {split} --checkpoint {WEIGHTS} --graph_dir "{GRAPH}" --result_dir "{RESULT}" --report_json "{report}" --save
    assert _exit_code == 0 and report.exists(), f'Test failed: {split}'
    (GRAPH / f'{split}.pkl').unlink()
```

### Cell 6: xem nhanh kết quả của encoder vừa chạy

```python
for split in SPLITS:
    report = json.loads((RESULT / f'{split}.json').read_text())
    print(f"{split:50} acc={report['accuracy']:.4f} auc={report['auc']:.4f} "
          f"f1={report['f1']:.4f} AI recall={report['recall_ai']:.4f} "
          f"human recall={report['recall_human']:.4f}")
```

### Cell 7: so sánh sau khi đã chạy cả hai encoder

Không dùng pandas. Kiểm tra hai encoder được đánh giá cùng nguồn dữ liệu, cùng số dòng và nhãn trước khi trừ điểm.

```python
import csv

ROOT = REPO / 'result/encoder_comparison'
rows = []
for split in SPLITS:
    r = json.loads((ROOT / 'roberta' / f'{split}.json').read_text())
    d = json.loads((ROOT / 'deberta' / f'{split}.json').read_text())
    rm, dm = r['encoder_metadata'], d['encoder_metadata']
    assert rm['source_sha256'] == dm['source_sha256'], split
    assert rm['source_lines'] == dm['source_lines'], split
    assert r['y_true'] == d['y_true'], split
    # Apart from encoder-specific fields, graph/parser/pooling policy must match.
    rc, dc = rm['configuration'], dm['configuration']
    for key in ['pipeline', 'hidden_size', 'max_length', 'pooling', 'chunking', 'dtype',
                'spacy_version', 'spacy_model', 'spacy_model_version',
                'dependency_vocabulary', 'whitespace_node_policy', 'edge_direction']:
        assert rc[key] == dc[key], (split, key)
    row = {'split': split, 'roberta_accuracy': r['accuracy'], 'deberta_accuracy': d['accuracy'],
           'delta_accuracy_pp': 100 * (d['accuracy'] - r['accuracy']),
           'roberta_auc': r['auc'], 'deberta_auc': d['auc'],
           'roberta_f1': r['f1'], 'deberta_f1': d['f1']}
    rows.append(row)
    print(f"{split:50} RoBERTa={r['accuracy']:.4f} DeBERTa={d['accuracy']:.4f} "
          f"delta={row['delta_accuracy_pp']:+.2f} pp")

with (ROOT / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
```

### Cell 8: tải kết quả và trọng số

```python
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display

archive = Path('/kaggle/working/rgcn_encoder_results.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
    for path in ROOT.rglob('*'):
        if path.is_file():
            z.write(path, str(path.relative_to(REPO)))
    for encoder in ['roberta', 'deberta']:
        weights = REPO / f'model/hc3_{encoder}_aligned_rgcn_model_2026.pth'
        for path in [weights, Path(str(weights) + '.encoder.json'), Path(str(weights) + '.train.json')]:
            assert path.exists(), path
            z.write(path, str(path.relative_to(REPO)))
        for path in (REPO / 'graph_data' / f'{encoder}_aligned').glob('*.json'):
            z.write(path, str(path.relative_to(REPO)))
display(FileLink(str(archive)))
```

## 4. Giới hạn bộ nhớ và tái lập

Giới hạn 4 graph là **giới hạn số file**, không bảo đảm tổng RAM/đĩa dưới 20 GB: một graph quá lớn vẫn có thể vượt ngân sách. Workflow trên giữ tối đa hai graph train/val của một encoder; khi test chỉ một graph. Builder giữ toàn bộ đặc trưng của một split trên CPU trước khi ghi, còn train giữ train+val trong RAM như V1. Không chạy đồng thời build và train; các lệnh `!python` là tiến trình riêng, kết thúc sẽ trả lại bộ nhớ encoder.

Đổi encoder bắt buộc build lại cache và train lại RGCN; không chuyển đổi được trọng số cuối từ encoder cũ. Không fine-tune encoder trong lần thử này. Không dùng test để chọn epoch, decay, ngưỡng hay biến thể alignment. Nếu cải thiện nhỏ, kiểm tra thêm seed; nếu GPT3.5 tăng mà insert/delete/repeat giảm nghiêm trọng thì không chấp nhận làm phiên bản chính.

Kiểm tra local được thực hiện bằng unit test offline cho alignment, nhãn và provenance. Test số học encoder cần PyTorch; download/forward model thật và train GPU chưa được xác minh trên local thiếu PyTorch. Các cell Kaggle là quy trình kiểm chứng tiếp theo, không phải kết quả đã chạy.
