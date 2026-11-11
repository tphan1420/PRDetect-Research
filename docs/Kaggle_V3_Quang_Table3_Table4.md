# Hướng dẫn chạy V3 Quang từ code trong repo

Model đã được port vào [model/QuangV3.py](../model/QuangV3.py), runner là [evaluate_v3_quang.py](../evaluate_v3_quang.py). Không cần file báo cáo Markdown của Quang trên Kaggle. Các phép tính model, structural features, FP16 lưu embedding và clipping quan hệ được giữ nguyên theo checkpoint; không thêm R-Drop, không sửa graph builder hiện tại.

## Chạy trực tiếp

```bash
python evaluate_v3_quang.py inspect --checkpoint-dir /kaggle/input --seeds 2021 2022 2023 2024 2025
python evaluate_v3_quang.py run --checkpoint-dir /kaggle/input --seeds 2021 2022 2023 2024 2025 --suite all --batch-size 4 --work-dir /kaggle/working/v3_quang_eval
```

- Dùng `--suite table3` để chỉ HC3 → GPT3.5; `--suite table4` để chỉ HC3 structural noise. Không có chiều GPT3.5 → HC3.
- `--device cpu` chỉ cho chẩn đoán môi trường không GPU; mặc định yêu cầu CUDA.
- `--struct-mode low` là giả định tái lập do checkpoint không lưu trường này. Không nên đổi theo điểm test. Nếu có metadata đáng tin từ lúc train, dùng giá trị đó và work-dir riêng.
- `--force` chạy lại report, không bỏ qua checkpoint/layer sai và không tự thay thế cache không rõ nguồn gốc.
- Không push các checkpoint ZIP lớn lên Git chỉ để chạy notebook. Upload/attach trọng số qua Kaggle Dataset; code và notebook là phần cần commit/push.
- Mean clean 0.9902 chỉ là sanity check theo báo cáo, không chứng minh chắc chắn pipeline đã khớp. Các thay đổi parser/vocabulary/embedding vẫn có thể ảnh hưởng.

## Kiểm tra local

```bash
python -m unittest discover -s tests -p test_v3_quang.py -v
```

Test cấu hình/discovery/tổng hợp/cache không cần PyTorch. Test số học được skip nếu chưa có PyTorch/NumPy. Kiểm tra tải checkpoint và inference thực phải chạy trong môi trường có PyTorch; không coi AST/CLI tests là bằng chứng đã chạy GPU.

## Các cell Kaggle

Có thể import [notebook](../notebooks/kaggle/v3_quang_table3_table4.ipynb), hoặc copy các cell dưới đây theo thứ tự.

# Test V3 của Quang — Table 3 & Table 4

Chỉ clone code và test checkpoint có sẵn; **không train**, không tạo script bằng %%writefile, không upload Markdown, không thay kiến trúc để ghép R-Drop.

- Bật GPU và Internet.
- Upload/attach **5 checkpoint** trong model/V3_quang vào Kaggle Input. Có seed 2021–2025, chưa có 2026.
- File .zip là checkpoint PyTorch, **không giải nén**.
- Table 3 chỉ **HC3 → GPT3.5**, các mức synonym 0/5/10/20/30%.
- Table 4: HC3 Original, Insert/Repeat/Replace/Delete 25%.
- Ngưỡng chính 0.5; ngưỡng validation trong checkpoint báo cáo riêng, không ensemble.
- Checkpoint không lưu structural mode. Mặc định giữ mode=low như code v3 trên graph mặc định; không sửa quy ước quan hệ chỉ ở bước test. CLI kiểm tra HC3 clean trước và ghi giả định vào manifest. Phiên bản parser/model/vocabulary vẫn cần tương thích lúc train.

## Cell 1 — Clone sau khi đã push code mới

```python
from pathlib import Path

REPO = Path("/kaggle/working/PRDetect-Research")
if not REPO.exists():
    !git clone --branch feat/rgcn-rdrop https://github.com/tphan1420/PRDetect-Research.git "{REPO}"
    assert _exit_code == 0

%cd /kaggle/working/PRDetect-Research
assert (REPO / "evaluate_v3_quang.py").is_file(), (
    "Clone đang cũ. Push code mới rồi dùng clone/session mới."
)
assert (REPO / "model/QuangV3.py").is_file()
```

## Cell 2 — Thư viện

Không cài requirements cũ hoặc thay PyTorch Kaggle. Nếu đã import NumPy/spaCy trước khi cài lại thư viện, restart kernel trước Cell 3. Không cần cài PyG để chạy model v3 thuần PyTorch.

```python
%pip install -q -r requirements-v3-eval.txt
!python -m spacy download en_core_web_sm
assert _exit_code == 0

!python -c "import torch, spacy, sklearn; from transformers import RobertaModel; assert torch.cuda.is_available(), 'Enable GPU'; nlp=spacy.load('en_core_web_sm'); print('Torch:', torch.__version__, 'spaCy:', spacy.__version__, 'English model:', nlp.meta['version']); assert nlp.meta['version']=='3.8.0', 'Verify parser version against Quang training environment'"
assert _exit_code == 0
```

## Cell 3 — Cấu hình và kiểm tra checkpoint

Không cần biết tên Kaggle Dataset: runner tìm đệ quy trong Kaggle Input. Nhiều bản cùng nội dung được chấp nhận; bản trùng tên nhưng khác nội dung sẽ báo lỗi. Có thể đổi SEEDS thành [2024, 2025] hoặc [2025] trước khi chạy; không có checkpoint 2026.

```python
from pathlib import Path

REPO = Path("/kaggle/working/PRDetect-Research")
%cd /kaggle/working/PRDetect-Research

CHECKPOINT_DIR = Path("/kaggle/input")
WORK = Path("/kaggle/working/v3_quang_eval")
SEEDS = [2021, 2022, 2023, 2024, 2025]
SEED_ARGS = " ".join(map(str, SEEDS))
BATCH_SIZE = 4
STRUCT_MODE = "low"
SUITE = "all"  # "table3", "table4", hoặc "all"

!python evaluate_v3_quang.py inspect --checkpoint-dir "{CHECKPOINT_DIR}" --seeds {SEED_ARGS} --struct-mode {STRUCT_MODE}
assert _exit_code == 0, "Kiểm tra đường dẫn/checkpoint trước khi build graph."
```

## Cell 4 — Chạy benchmark

Model và builder chạy ở các process riêng. Mỗi lần chỉ giữ một cache test mới ở WORK/graph_data, graph nằm ở CPU và chỉ model/batch hiện tại ở GPU. Giới hạn 4 cache tính cả các thư mục graph cũ đã biết, không tự xóa cache của thí nghiệm khác. Số file không bảo đảm tổng dung lượng dưới 20 GB.

Tự chạy HC3 Original trước để đối chiếu mean 0.9902 trong tài liệu (năm seed). Nếu lệch quá 0.5 điểm %, CLI dừng để kiểm tra pipeline; đây là cảnh báo kỹ thuật, không phải kiểm định thống kê. Không dò threshold theo test để khớp con số. Nếu chỉ chạy Table 3, HC3 clean vẫn được lưu riêng với nhãn sanity.

Chạy lại cùng cấu hình sẽ dùng lại report hợp lệ và tránh build graph không cần thiết. Nếu thay checkpoint/code/môi trường, dùng WORK mới hoặc --force cho report; cache không khớp provenance phải được kiểm tra hoặc dùng WORK mới.

```python
!python evaluate_v3_quang.py run --checkpoint-dir "{CHECKPOINT_DIR}" --seeds {SEED_ARGS} --suite {SUITE} --batch-size {BATCH_SIZE} --struct-mode {STRUCT_MODE} --max-graphs 4 --work-dir "{WORK}"
assert _exit_code == 0, "Xem lỗi phía trên; giữ lại report đã hoàn tất để chạy tiếp."
```

## Cell 5 — Tải kết quả

ZIP chứa report xác suất từng mẫu, confusion matrix, CSV từng seed, CSV mean ± sample std ở ngưỡng 0.5 và CSV ngưỡng validation. Average Table 4 chỉ tính 4 tập nhiễu, không tính Original. Metadata có hash checkpoint, tham số train, phiên bản môi trường và số cạnh bị clamp do ID vượt R. Không kèm trọng số hoặc graph để tránh ZIP quá lớn.

```python
from IPython.display import FileLink, display

archive = WORK / "v3_quang_table3_table4_results.zip"
assert archive.is_file(), "Chưa có ZIP; kiểm tra Cell 4 đã chạy xong."
display(FileLink(str(archive)))
```

