# Kiểm chứng một ngưỡng chung cho checkpoint A1-only

## Mục tiêu

Dùng lại `hc3_a1_only_rgcn_model_2026.pth`; không train lại, không sửa trọng số,
không bật A2/A3. Chỉ chọn threshold từ validation clean/insert/delete/repeat.
Test chỉ được chạy sau khi file threshold đã được cố định.

[a1_threshold.py](../a1_threshold.py) có bốn lệnh:

- `prepare`: tạo bốn JSONL từ HC3 validation, seed 42, nhiễu 25%.
- `score`: lấy logit Human từ đúng forward pass của RGCN2, ghi JSON nhỏ.
- `fit`: chọn một threshold tối đa hóa balanced accuracy trung bình của bốn điều kiện validation.
- `evaluate`: so baseline logit 0 (probability 0,5) với threshold đã cố định.

Lấy logit trước sigmoid để tránh mất độ phân giải khi probability sát 1. AUC được
tính trên logit nên có thể khác AUC log cũ do các probability float32 bị bão hòa.
Trong cùng score file, thay threshold không thay AUC. Không đảo nhãn.

## Chuẩn bị Kaggle

Bật GPU và Internet. Dùng môi trường dependencies như lần kiểm chứng trước
(SpaCy 3.7.4 và en_core_web_sm 3.7.1), không đổi encoder. Đưa script mới lên GitHub
trước khi clone/pull, hoặc upload riêng script vào Kaggle Input và copy vào repo.
Đưa checkpoint đã train vào Kaggle Input nếu không còn trong working directory.
Không chạy lại các cell train.

### Cell 1 — Khai báo đường dẫn, copy checkpoint nếu cần

```python
import os
import shutil
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
assert REPO.is_dir(), 'Clone nhánh version2 và cài dependencies như notebook trước.'
os.chdir(REPO)

if not Path('a1_threshold.py').exists():
    scripts = list(Path('/kaggle/input').rglob('a1_threshold.py'))
    assert len(scripts) == 1, 'Upload a1_threshold.py vào Kaggle Input hoặc cập nhật code GitHub.'
    shutil.copy2(scripts[0], 'a1_threshold.py')

CHECKPOINT = Path('model/hc3_a1_only_rgcn_model_2026.pth')
if not CHECKPOINT.exists():
    matches = list(Path('/kaggle/input').rglob(CHECKPOINT.name))
    assert len(matches) == 1, 'Thêm dataset chứa đúng checkpoint A1-only vào Kaggle Input.'
    CHECKPOINT.parent.mkdir(exist_ok=True)
    shutil.copy2(matches[0], CHECKPOINT)

RESULT = Path('result/a1_threshold')
RESULT.mkdir(parents=True, exist_ok=True)
CAL_DIR = Path('data/perturbed_text/calibration_validation')
THRESHOLD = RESULT / 'threshold.json'
```

### Cell 2 — Hàm xử lý một split, tối đa ba graph

Không xóa các graph cũ ngoài phạm vi split đang xử lý. Không ghi đè graph đã có.
Nếu lỗi, dừng và giữ graph để kiểm tra; chỉ xóa khi score đã lưu thành công.

```python
def check_command():
    assert get_ipython().user_ns.get('_exit_code', 1) == 0, 'Lệnh bị lỗi; xem log trước khi tiếp tục.'

def build_and_score(split, role, condition):
    files = [Path(f'graph_data/{split}.pkl'),
             Path(f'graph_data_v2/{split}.pkl'),
             Path(f'graph_data/a1_{split}.pkl')]
    assert not any(p.exists() for p in files), 'Split đã có graph; kiểm tra trước để tránh ghi đè.'
    existing = sum(len(list(Path(d).glob('*.pkl'))) for d in ['graph_data', 'graph_data_v2'])
    assert existing + 3 <= 4, 'Cần dọn graph trung gian cũ trước.'
    used = sum(p.stat().st_size for p in Path('/kaggle/working').rglob('*')
               if p.is_file() and not p.is_symlink())
    free = min(20 * 1024**3 - used, shutil.disk_usage('/kaggle/working').free)
    assert free > 2 * 1024**3, 'Cần ít nhất 2 GB khoảng trống; graph lớn có thể cần nhiều hơn.'

    !python building_graph.py --file {split}
    check_command()
    !python building_graph_v2.py --file {split}
    check_command()
    !python export_a1_v1.py --file {split}
    check_command()

    output = RESULT / f'scores_{role}_{condition}.json'
    graph = files[2]
    !python a1_threshold.py score --file {graph} --checkpoint {CHECKPOINT} --role {role} --condition {condition} --output {output}
    check_command()
    assert output.is_file()

    for p in files:
        p.unlink()
    print('Đã lưu score và xóa 3 graph:', split)
```

Giới hạn ba graph không đảm bảo kích thước tổng dưới 20 GB hoặc đủ RAM/VRAM;
kiểm tra dung lượng thực tế trước khi chạy nếu working directory còn dữ liệu lớn.

### Cell 3 — Tạo nhiễu validation, không đọc test

```python
!python a1_threshold.py prepare --source data/original_text/benchmarks/hc3/hc3_val.json --output-dir {CAL_DIR} --seed 42 --percent 25
check_command()
```

Chỉ chạy một lần; script không ghi đè validation JSON/manifest đã tồn tại.
Từ chèn được lấy từ validation, không lấy vocabulary từ test. Cách mask và global
replace mô phỏng other_perturb.ipynb; không khẳng định bản nhiễu tái tạo chính xác
test gốc vì vocabulary và RNG ở test gốc không có provenance đầy đủ.

### Cell 4 — Thu score validation từng split, xóa graph sau mỗi lượt

```python
for condition in ['clean', 'insert', 'delete', 'repeat']:
    suffix = '' if condition == 'clean' else '_25'
    split = f'hc3_val_cal_{condition}{suffix}'
    build_and_score(split, 'validation', condition)
```

### Cell 5 — Chọn rồi cố định một threshold

```python
manifest = CAL_DIR / 'manifest.json'
clean = RESULT / 'scores_validation_clean.json'
insert = RESULT / 'scores_validation_insert.json'
delete = RESULT / 'scores_validation_delete.json'
repeat = RESULT / 'scores_validation_repeat.json'

!python a1_threshold.py fit --scores {clean} {insert} {delete} {repeat} --manifest {manifest} --output {THRESHOLD}
check_command()
```

Không sweep grid 0,5–1,0: các ngưỡng phân biệt được xét chính xác theo logit và xử
lý các score trùng nhau cùng lúc. Mỗi điều kiện có cùng trọng số; trong mỗi điều
kiện hai lớp có cùng trọng số. Nếu hòa, ưu tiên gần logit 0 hơn.
File threshold ghi checkpoint hash, vocab hash và báo cáo validation từng điều kiện.
`fit` từ chối score role=test hoặc provenance khác manifest, và không ghi đè threshold.
Nếu validation clean giảm mạnh, ghi nhận trade-off trước khi đánh giá test; không đổi
tiêu chí tối ưu sau khi xem test.

### Cell 6 — Test với threshold cố định; vẫn tối đa ba graph

```python
TEST_SPLITS = ['hc3_test', 'hc3_test_insert_25', 'hc3_test_delete_25',
               'hc3_test_repeat_25', 'gpt3.5_mixed_test_split',
               'gpt3.5_mixed_test_adj_30_synonym_replace']

assert THRESHOLD.is_file()
for split in TEST_SPLITS:
    build_and_score(split, 'test', split)
    scores = RESULT / f'scores_test_{split}.json'
    report = RESULT / f'report_test_{split}.json'
    !python a1_threshold.py evaluate --scores {scores} --threshold {THRESHOLD} --output {report}
    check_command()
```

`evaluate` không có chức năng fit; dùng nguyên threshold cho HC3 và GPT-3.5.
Score JSON đã lưu đủ nhãn/logit nên không cần graph để đọc lại báo cáo.

### Cell 7 — Bảng kết quả và file ZIP nhỏ

```python
import json
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
from IPython.display import FileLink, display

rows = []
for path in sorted(RESULT.glob('report_test_*.json')):
    report = json.loads(path.read_text())
    for mode in ['baseline', 'calibrated']:
        rows.append({'split': report['condition'], 'mode': mode, **report[mode]})
df = pd.DataFrame(rows)
display(df[['split', 'mode', 'accuracy', 'balanced_accuracy', 'auc_logit',
            'recall_ai', 'recall_human', 'predicted_human']].round(4))
df.to_csv(RESULT / 'comparison.csv', index=False)

archive = Path('/kaggle/working/a1_threshold_results.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
    for path in RESULT.glob('*'):
        if path.is_file():
            z.write(path, str(path))
    z.write(CAL_DIR / 'manifest.json', 'validation_manifest.json')
display(FileLink(str(archive)))
```

## Cách kết luận

- Nếu một threshold cải thiện structural noise nhưng clean/GPT-3.5 giảm: có trade-off,
  không tuyên bố đã giải quyết robustness toàn diện.
- Nếu cải thiện đồng thời mà AUC giữ nguyên: có bằng chứng ngưỡng cũ là một phần nguyên nhân.
- Nếu không cải thiện đủ: cần xem representation/huấn luyện, không tiếp tục chọn ngưỡng từ test.
- Validation đã dùng để chọn checkpoint trước đây; kết quả sau fit trên chính validation
  không phải ước lượng độc lập. Test với threshold cố định mới là bước kiểm chứng.
- Đây chỉ là một seed và một thiết lập perturbation, không phải bằng chứng tổng quát.

## Xác minh

Local đã kiểm tra cú pháp và diagnostics, threshold sweep so với exhaustive search
trên 100 bộ synthetic có ties, và AUROC so với phép tính cặp độc lập. Không train/test
dataset. Chưa xác minh forward trên Kaggle vì local không có Torch/PyG.
