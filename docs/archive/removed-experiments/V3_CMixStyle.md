# V3-CMixStyle: triển khai và chạy ablation

## 1. Phạm vi

Triển khai phương án 3 trong [4Method.md](./4Method.md): **Class-Conditional Graph MixStyle** trên nền `V1_swap` của Quang.

Hai cấu hình train từ đầu để đối chiếu:

| Variant | Nội dung |
|---|---|
| `baseline` | Kiến trúc, swap và quy trình V3; không CMixStyle |
| `cmixstyle` | Cùng cấu hình trên, chỉ thêm CMixStyle ở nhánh augmented |

Không thêm R-Drop, class-preserving swap, length balancing, không đổi encoder/tokenizer, cạnh, relation vocabulary hoặc quy ước structural mode. EMA vốn có trong V3 vẫn giữ nguyên ở cả hai variant; không phải thử nghiệm EMA riêng trước đây.

**Phải train lại.** Bật MixStyle lúc test một checkpoint V3 cũ không tạo ra cải tiến. Không cần attach checkpoint cũ để chạy ablation này.

Đây là giả thuyết nghiên cứu về generalization, chưa có kết quả chứng minh thắng Table 3. Mục tiêu so sánh cuối cùng vẫn là các checkpoint GCN bạn tự train, không phải điểm tác giả paper.

## 2. Tổ chức code

| File | Vai trò |
|---|---|
| [model/QuangV3.py](../model/QuangV3.py) | Kiến trúc/GraphSet V3. Chỉ thêm hook `feature_transform=None` giữa `ln0` và structural embedding; mặc định không thay đổi tính toán |
| [model/v3_mixstyle.py](../model/v3_mixstyle.py) | Module augmentation độc lập, không tham số học hoặc running statistics |
| [model/V3CMixStyle.py](../model/V3CMixStyle.py) | Kế thừa V3, nối module vào hook; không sao chép forward/RGCN |
| [train_v3.py](../train_v3.py) | Trainer chung cho baseline/CMixStyle; optimizer, EMA, validation, checkpoint và lịch sử |
| [evaluate_v3_quang.py](../evaluate_v3_quang.py) | Test checkpoint cũ hoặc variant mới, nhận dạng tường minh bằng `--variant` |
| [run_v3_ablation.py](../run_v3_ablation.py) | Điều phối build → train → xóa cache train → test từng split → đóng gói |
| [tests/test_v3_mixstyle.py](../tests/test_v3_mixstyle.py) | Kiểm tra số học, gradient, identity, checkpoint và CPU smoke test |

Checkpoint cũ vẫn strict-load được vào kiến trúc nền; module mới không thêm trọng số. CLI vẫn phân biệt checkpoint baseline/CMixStyle bằng metadata để tránh test nhầm.

## 3. Cơ chế CMixStyle

Cho đặc trưng node sau `LayerNorm(proj(x))` là `H_g` với graph `g`:

1. Tính mean và **population variance** theo node, riêng từng graph và từng channel, không lấy thống kê chung cả batch.
2. Chọn donor là **graph khác, cùng nhãn AI/human**. Dùng hoán vị vòng ngẫu nhiên trong mỗi lớp để không tự chọn chính mình.
3. Rút `lambda ~ Beta(alpha, alpha)`, đặt `w = strength * (1 - lambda)`.
4. `mu_mix = (1-w)*mu_g + w*mu_donor`, tương tự cho standard deviation.
5. `H_new = ((H_g-mu_g)/sigma_g)*sigma_mix + mu_mix` với xác suất `p` riêng cho mỗi graph.

Thống kê mean/std được detach khỏi autograd; gradient vẫn đi qua đặc trưng đã chuẩn hóa. Tính toán thống kê ở FP32 kể cả khi train AMP. Graph ít hơn hai node, hoặc lớp không có hai graph hợp lệ trong micro-batch, được giữ nguyên; **không fallback sang donor khác nhãn**.

Biến đổi được đặt trước khi cộng structural embeddings, không đổi depth/children/distance hoặc cạnh. CMixStyle không yêu cầu sửa giả định relation-half hiện tại của V3; vấn đề đó được giữ tách biệt để không làm lẫn ablation.

### Hai nhánh huấn luyện

```text
Graph nguồn
  ├─ Nhánh clean: V3 bình thường → BCEWithLogits
  └─ Nhánh augmented: swap V3 → proj/LN → CMixStyle → structural/RGCN/readout → BCEWithLogits

Loss = 0.5 * BCE_clean + 0.5 * BCE_augmented
```

Không có KL/consistency loss. Swap gốc vẫn có thể lấy donor khác nhãn; ràng buộc cùng nhãn mới chỉ thuộc **CMixStyle**.

Khi `model.eval()`, MixStyle tắt hoàn toàn: không dùng nhãn, không lấy donor, không cập nhật thống kê, không tiêu thụ RNG. Validation clean và validation swap proxy đều không bật CMixStyle. Chọn checkpoint theo `val_loss + val_swap_loss`, như nền V3, không dùng test để chọn.

## 4. Mặc định và tính công bằng

| Tham số | Mặc định |
|---|---:|
| Hidden / bases / layers | 384 / 12 / 2 |
| Dropout / input dropout / edge dropout | 0.3 / 0.1 / 0.1 |
| Swap maximum | 0.35 |
| Readout / struct / gfeat | pool3 / 1 / 0 |
| Optimizer | AdamW, LR 0.0005, WD 0.01 |
| Scheduler | 5% warmup + cosine V3 |
| EMA / decay | 1 / 0.998, giữ lịch decay V3 |
| Epochs / patience | 10 / 10 |
| Effective batch / micro-batch | 32 / 8 |
| Validation batch | 64, như `predict()` trong code Quang |
| Test batch | 4 |
| MixStyle p / alpha / strength | 0.5 / 0.1 / 0.3 |
| AMP | 0 |

`strength=0.3` giới hạn tỷ lệ đóng góp thống kê donor tối đa 30%; không thay toàn bộ thống kê bằng donor.

**Micro-batch là một thay đổi thực thi cần ghi nhận:** graph chỉ nằm trên CPU, từng micro-batch lên GPU; backward nhánh clean xong trước khi chạy nhánh augmented để không giữ cả hai activation graph cùng lúc. Gradient được tích lũy có trọng số theo số graph thực tế, kể cả batch cuối không đủ 32. Optimizer, scheduler và EMA chỉ cập nhật một lần mỗi effective batch.

Donor pool của swap và CMixStyle là **micro-batch**, không phải effective batch. Vì vậy baseline mới không được mặc định là bản replay chính xác số điểm checkpoint Quang cũ. Cần so baseline mới với CMixStyle mới có cùng mọi setting; muốn donor pool train đúng 32 như code cũ, đặt `--micro-batch-size 32` nếu đủ VRAM.

Thứ tự minibatch dùng một RNG riêng theo seed để các phép rút Beta của CMixStyle không làm hai variant nhìn dữ liệu theo thứ tự khác nhau ở epoch tiếp theo. Đây cũng là một chi tiết thực thi khác với script Quang cũ, được áp dụng chung cho cả hai variant mới.

Validation batch cũng quyết định donor pool của **swap proxy**, nên không đổi nó chỉ cho một variant. Nếu cần giảm bộ nhớ validation, giảm `--val-batch-size` cho cả hai và ghi lại thay đổi.

Log `MixStyle applied/eligible` và số `graphs` được lưu từng epoch. Nếu không có donor cùng nhãn trong cả epoch, trainer báo lỗi thay vì ghi nhận một thí nghiệm CMixStyle thực ra không hoạt động.

## 5. Các cell Kaggle

Bật GPU và Internet. Chạy sau khi code mới được push lên nhánh đang dùng. Không cần upload Markdown/script hoặc checkpoint cũ.

### Cell 1 — Clone

```python
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
if not REPO.exists():
    !git clone --branch feat/rgcn-rdrop https://github.com/tphan1420/PRDetect-Research.git "{REPO}"
    assert _exit_code == 0

%cd /kaggle/working/PRDetect-Research
assert (REPO / 'run_v3_ablation.py').is_file(), 'Clone chưa có code mới.'
```

### Cell 2 — Thư viện

```python
%pip install -q -r requirements-v3-eval.txt
!python -m spacy download en_core_web_sm
assert _exit_code == 0

!python -c "import torch, spacy, sklearn; assert torch.cuda.is_available(), 'Enable GPU'; nlp=spacy.load('en_core_web_sm'); assert nlp.meta['version']=='3.8.0'; print('Environment OK:', torch.__version__, spacy.__version__)"
assert _exit_code == 0
```

Không cài lại Torch của Kaggle. Nếu đã import thư viện trước khi thay dependency, restart kernel rồi tiếp tục từ Cell 3.

### Cell 3 — Cấu hình và chạy

```python
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
WORK = Path('/kaggle/working/v3_cmixstyle')
SEEDS = [2025, 2026]  # Có thể pilot [2025], sau đó xác nhận thêm seed.
VARIANTS = ['baseline', 'cmixstyle']
SEED_ARGS = ' '.join(map(str, SEEDS))
VARIANT_ARGS = ' '.join(VARIANTS)
MICRO_BATCH = 8

%cd /kaggle/working/PRDetect-Research
!python run_v3_ablation.py --work-dir "{WORK}" --seeds {SEED_ARGS} --variants {VARIANT_ARGS} --micro-batch-size {MICRO_BATCH} --batch-size 32 --val-batch-size 64 --eval-batch-size 4 --max-graphs 4 --suite full
assert _exit_code == 0, 'Xem lỗi phía trên; không xóa checkpoint/report đã hoàn tất.'
```

Lệnh này:

1. Build `hc3_train` và `hc3_val`, không nạp encoder cùng lúc với classifier trên GPU.
2. Train cả hai variant trên cùng cache, mỗi seed được reset độc lập.
3. Sau khi train thành công, tự xóa **hai cache train/val do workflow này tạo**, không xóa nguồn hay checkpoint.
4. Build và test tuần tự từng split cho tất cả seed của từng variant, rồi xóa cache test đã dùng.
5. Tổng hợp CSV và ZIP, gồm cả checkpoint mới.

Giới hạn tối đa 4 file graph có tính cả cache dưới WORK và các thư mục graph cũ đã biết trong repo. Nếu có 4 cache cũ, chương trình dừng để bạn dọn chúng; không tự xóa dữ liệu thí nghiệm khác. Không giữ graph test trên GPU. Giới hạn số file không phải bảo đảm tuyệt đối về dung lượng byte dưới 20 GB; log sẽ báo dung lượng và dung lượng đĩa trống.

Nếu train OOM, dùng WORK mới và giảm `MICRO_BATCH=4` cho **cả hai** variant. Nếu validation OOM, giảm `--val-batch-size 16` cho cả hai. Không đổi cấu hình trong cùng một thí nghiệm rồi dùng lại kết quả cũ. Micro-batch 1 không thể làm CMixStyle nên bị từ chối.

Nếu chỉ muốn train, thêm `--stage train`. Khi train xong, chạy lại lệnh với `--stage evaluate` để chỉ test, không rebuild train graph. Chạy lại stage evaluate có thể dùng lại report đúng provenance.

Nếu chỉ muốn train bản cải tiến, đặt `VARIANTS = ['cmixstyle']`; CSV lúc đó không có chênh lệch so với baseline mới. Để kết luận tác dụng của riêng MixStyle, vẫn cần bản đối chứng train cùng setting.

### Cell 4 — Tải kết quả

```python
from pathlib import Path
from IPython.display import FileLink, display

WORK = Path('/kaggle/working/v3_cmixstyle')
archive = WORK / 'v3_cmixstyle_ablation_results.zip'
assert archive.is_file()
display(FileLink(str(archive)))
```

## 6. Checkpoint và kết quả

```text
WORK/
  model/
    v3_baseline_seed2025.pt
    v3_baseline_seed2025.train.json
    v3_cmixstyle_seed2025.pt
    v3_cmixstyle_seed2025.train.json
    ... seed 2026
  eval/baseline/result/
  eval/cmixstyle/result/
  comparison.csv
  v3_cmixstyle_ablation_results.zip
```

Checkpoint lưu kiến trúc, trọng số tốt nhất, structural mode, cấu hình MixStyle, seed, epoch tốt nhất, hash nguồn/cache/code và cờ train đã hoàn tất. Trainer chỉ bỏ qua khi `--skip-completed` thấy cả metadata và hash khớp. File best epoch của phiên train OOM không được tự coi là đã train xong; muốn chạy lại vào cùng đường dẫn dùng `train_v3.py --overwrite` hoặc chọn WORK mới. Chưa triển khai resume optimizer giữa epoch.

`comparison.csv` so với **V3 baseline mới**, không phải GCN. Sau khi tải report, cần đối chiếu tiếp với GCN tự train bằng cùng seed/split/ngưỡng. Average Table 4 không tính Original. Không ensemble.

### Test riêng một variant

```bash
python evaluate_v3_quang.py run --variant cmixstyle --checkpoint-dir /kaggle/working/v3_cmixstyle/model --seeds 2025 2026 --suite full --work-dir /kaggle/working/v3_cmixstyle/eval/cmixstyle --graph-budget-root /kaggle/working/v3_cmixstyle
```

`--variant quang` (mặc định) vẫn nhận checkpoint Quang cũ. `--variant baseline` và `--variant cmixstyle` nhận đúng tên checkpoint mới, không chọn lẫn file giữa hai phương pháp.

`--suite table2`: HC3 in-domain Original và synonym 5/10/20/30%; `table3`: HC3 → GPT-3.5; `table4`: HC3 structural noise; `full`: cả ba. `all` giữ nghĩa cũ là Table 3+4. Chưa bao gồm phần GPT-3.5 in-domain hoặc chiều train GPT-3.5 → HC3.

## 7. Kiểm chứng và diễn giải

Các kiểm tra số học bao gồm: chọn donor đúng lớp, mean/std đúng với graph kích thước khác nhau, gradient không đi qua thống kê donor, số học hữu hạn ở FP16/đặc trưng hằng, skip graph không có donor, identity khi eval/tắt module, tương thích trọng số V3, và train nhỏ trên CPU → checkpoint → strict load → inference. Không thay thế benchmark thật trên Kaggle.

Khi đánh giá thành công, cần đồng thời xem Table 3, mức giữ hiệu năng Table 2 và structural noise Table 4; không chỉ nhìn AUC hoặc một seed thuận lợi. Mọi lựa chọn strength/probability/epoch dùng validation nguồn, không chọn trên GPT-3.5 test.

Nguồn phương pháp: [Domain Generalization with MixStyle, ICLR 2021](https://arxiv.org/abs/2104.02008). Class conditioning, graph moments và vị trí chèn giữ nhánh cấu trúc là adaptation của thí nghiệm này, chưa phải một kết quả đã được paper gốc xác nhận.
