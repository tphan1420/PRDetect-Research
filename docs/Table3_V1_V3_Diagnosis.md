# Table 3: vì sao RGCN v1 >70% nhưng V3 chỉ khoảng 66%?

Ngày kiểm tra: 2026-11-11. Phạm vi: HC3 → GPT-3.5-Mixed; baseline chính là GCN tự train.

**Kết luận:** checkpoint v1 cũ thực sự đạt 70.88% trung bình Table 3. V3 hiện tại xếp hạng hai lớp tốt hơn theo AUC, nhưng nhận nhầm nhiều AI thành human tại ngưỡng 0.5. Chưa có bằng chứng rằng một thành phần riêng của V3 gây ra toàn bộ chênh lệch. Hai hướng đáng thử tiếp là **ablation readout `meanlogit`** và **sửa diễn giải quan hệ cú pháp trong một variant riêng**. Không tiếp tục CMixStyle, R-Drop hoặc A1/A2.

## 1. Số liệu đã kiểm tra lại từ dự đoán từng mẫu

Nguồn: [old/new seed 2026](../result/table3_rgcn_old_vs_new_2026.zip), [GCN/RGCN/R-Drop](../result/table3_hc3_to_gpt35_3models.zip), [V3 baseline trong thí nghiệm CMixStyle](../result/Cmix/v3_cmixstyle_ablation_results.zip).

Chạy [audit_table3.py](../audit_table3.py) để tái lập. [Dữ liệu kiểm tra](./Table3_Audit_Data.json) lưu SHA-256 nguồn, số liệu theo split/seed, độ dài và kiểm tra relation. Script đã kiểm tra lại accuracy, AUC có xử lý ties, confusion matrix và recall của **40 report**. Không train model, không dò ngưỡng trên test.

| Model | Seed | Table 3 average accuracy | Average AUC | Recall AI | Recall human |
|---|---|---:|---:|---:|---:|
| GCN tự train | 2026 | 63.40% | 86.01% | 29.28% | 97.52% |
| **RGCN v1 cũ** | **2026** | **70.88%** | **88.02%** | **44.24%** | **97.52%** |
| RGCN train lại, baseline của R-Drop | 2026 | 61.22% | 88.24% | 24.24% | 98.20% |
| V3 baseline controlled | 2026 | 68.28% | **92.05%** | 38.64% | 97.92% |
| GCN tự train | 2025 | 68.62% | 89.84% | 39.48% | 97.76% |
| V3 baseline controlled | 2025 | 64.90% | 90.96% | 30.80% | 99.00% |
| GCN tự train | mean 2025–2026 | 66.01% | 87.93% | 34.38% | 97.64% |
| V3 baseline controlled | mean 2025–2026 | 66.59% | **91.50%** | 34.72% | 98.46% |

Average Table 3 gồm Original + synonym 5/10/20/30%; các perturbation cùng văn bản không phải năm mẫu thực nghiệm độc lập. AUC ở bảng dùng thang phần trăm. Ngưỡng 0.5; `0=AI`, `1=human`.

**Đọc đúng chênh lệch:** V3 seed 2026 kém v1 cũ **2.60 điểm %**, không phải lấy 70.88 trừ trung bình hai seed 66.59 rồi gọi đó là suy giảm cùng seed. V3 thắng GCN seed 2026 nhưng thua seed 2025; mức hơn 0.58 điểm % ở trung bình chưa đủ chứng minh cải thiện ổn định. Chưa có v1 cũ seed 2025 trong gói old/new để so hai seed đầy đủ.

## 2. Phần suy giảm nào đã có bằng chứng?

### V1 cũ và RGCN train lại khác checkpoint, không khác kiến trúc

[model/RGCN.py](../model/RGCN.py) hiện tại giống `main` theo `git diff`. Nhánh `alpha=0` trong [rdrop.py](../rdrop.py) vẫn thực hiện một forward và BCE gốc. Không thể quy mức giảm 9.66 điểm % của baseline train lại cho việc “đổi sang kiến trúc R-Drop”.

Gói old/new dùng cùng evaluator, batch 8, ngưỡng 0.5 và đường dẫn graph tương ứng cho từng split; chuỗi nhãn khớp. SHA-256 của checkpoint old trùng [checkpoint v1 còn trong repo](../model/hc3_rgcn_model_2026.pth): `3bd6ecd6a8eef405b5109a5634e6828e3d9dd4d4211a08ed45be1a1eef2ec6e5`.

Ví dụ GPT-3.5 Original, 500 AI + 500 human:

| | V1 cũ | RGCN train lại |
|---|---:|---:|
| Accuracy | 72.60% | 62.50% |
| AUC | 87.82% | 88.07% |
| AI nhận đúng | 242/500 | 137/500 |
| Human nhận đúng | 484/500 | 488/500 |
| Trung vị `p(human)` của AI | 0.5235 | 0.9078 |
| Trung vị `p(human)` của human | 0.9795 | 0.9997 |

Bản train lại mất 105 AI nhận đúng và chỉ thêm 4 human nhận đúng. Đây là bằng chứng điểm số dịch về phía human; AUC gần bằng nhau **không có nghĩa dự đoán từng mẫu giống nhau** hoặc toàn bộ sai khác chỉ là một bias hằng số.

Metadata bản train lại ghi Adam, LR 1e-4, batch 32, dropout 0.5, tối đa 40 epoch, chọn validation accuracy và ưu tiên epoch muộn khi hòa; seed 2026 chọn epoch 18. README mô tả cấu hình v1 tương tự, nhưng chưa tìm thấy provenance đầy đủ của lần train checkpoint cũ: hash graph train/val, phiên bản encoder/parser/PyG và lịch sử từng epoch. Vì vậy **chưa xác định được nguyên nhân huấn luyện cụ thể của chênh lệch old/new**. Cùng seed không bảo đảm tái lập nếu cache, môi trường hoặc đường thực thi thay đổi. Gói old/new không lưu hash graph để kiểm chứng độc lập tính đồng nhất byte của cache.

### V3 tốt hơn về thứ hạng, nhưng validation HC3 không đại diện Table 3

V3 cải thiện AUC trung bình lên 91.50%, nhưng recall AI chỉ 34.72%, trong khi human là 98.46%. Điểm nghẽn thể hiện ở miền đích và quy tắc quyết định; không thể gọi V3 hoàn toàn mất khả năng phân biệt hai lớp. AUC cũng không chứng minh xác suất được calibration đúng hoặc tìm được một ngưỡng chuyển miền tốt bằng validation HC3.

V3 được chọn bằng `val_loss + val_swap_loss` trên HC3; teacher EMA, optimizer và readout đều khác v1. Proxy swap chỉ thay feature, không tạo đầy đủ khác biệt chủ đề, thể loại, độ dài và generator của GPT-3.5-Mixed. Tối ưu proxy này không bảo đảm tăng Table 3.

Ngưỡng HC3 validation của V3 controlled là 0.43 và 0.18 cho seed 2025/2026; hạ ngưỡng làm dự đoán human nhiều hơn. Không nên giải quyết bằng cách dò ngưỡng trên GPT test. Temperature scaling thuần túy với nhiệt độ dương không đổi dấu logit, nên không đổi nhãn tại ngưỡng 0.5.

## 3. Bất nhất cụ thể trong nhánh cú pháp của V3

[building_graph.py](../building_graph.py) mặc định tạo cạnh **dependent → head** cho mọi dependency, không thêm cạnh ngược. [dep_vocab.json](../dep_vocab.json) chứa các ID nhãn thông thường, không phải hai nửa forward/reverse.

Trong khi đó [model/QuangV3.py](../model/QuangV3.py) dùng `R // 2` để suy ra hướng cạnh. Metadata của cả hai V3 controlled có `R=51`, `mode=low`, tức chỉ ID `<25` được nhận là cạnh forward khi dựng `parent_array` và khi gán nhóm swap. Hash của builder, vocab và dữ liệu HC3 train/val local khớp provenance run V3 sau chuẩn hóa CRLF → LF giữa Windows và Kaggle.

Kiểm tra thực thi hàm `fwd_mask` hiện tại xác nhận các ID 25–50 bị loại, gồm `mark`, `appos`, `nummod`, `attr`, `neg`, `dep`, `npadvmod`, `nmod`… Đây đều là nhãn forward hợp lệ trong vocab hiện tại, không phải inverse relation.

Hệ quả từ code:

- Parent/depth/children/distance dùng cây thiếu một số loại cạnh. Node có cha thuộc nhóm bị loại có thể bị coi thành root.
- Swap gom các node không được gán relation vào cùng nhóm mặc định; điều kiện “cùng vai trò cú pháp” không còn chính xác cho nhóm này.
- [Class-Preserving Syntax Swap](../model/V3ClassSwap.py) kế thừa cùng phép chia nửa. Nó vẫn bảo toàn nhãn lớp, nhưng chưa khôi phục đầy đủ nhóm quan hệ thật.
- [Restricted Embeddings](../model/V3Restricted.py) nhận structural features từ cùng pipeline; hạn chế semantic features trong khi structural features sai có thể hạn chế lợi ích của hướng này.
- **RGCN message passing vẫn nhận mọi cạnh hiện có.** Đây không phải bằng chứng một nửa cạnh bị xóa khỏi toàn mô hình; số ID bị loại cũng không phải tỷ lệ cạnh thực tế bị loại.

Chưa có graph cache local để đo tần suất các nhãn bị loại hoặc tác động accuracy thực tế. `clipped_edges=0` trong report chỉ nói ID không vượt R, không xác nhận cách diễn giải quan hệ là đúng. Đổi `low` sang `high`, hoặc đổi R thành 60, đều không sửa đúng bản chất mapping.

**Đã triển khai `structmap`:** khai báo schema đồ thị một chiều; lấy mọi cạnh không self-loop làm dependent → head, gán nhóm swap bằng ID dependency thật, dành sentinel riêng cho root/unknown; xác thực mỗi node có tối đa một cha. Nếu có inverse edges phải có mapping tường minh, không suy ra từ R. Giữ nguyên message passing và embeddings. Train lại variant mới với chế độ cấu trúc lưu trong checkpoint, áp dụng nhất quán train/val/test. Không áp dụng cách tính cấu trúc mới chỉ ở inference cho checkpoint cũ.

Hai thử nghiệm đang chạy vẫn có giá trị như ablation trong cùng pipeline legacy. Không cần hủy; nếu chúng không tăng điểm cũng chưa đủ bác bỏ ý tưởng khi structural mapping được sửa.

## 4. Vì sao nên thử readout đơn giản trước?

Đếm trực tiếp theo `text.split()` trên dữ liệu local (đây là **từ phân cách whitespace**, không phải số token spaCy/BPE):

| Tập | Median AI | Median human | Tỷ lệ AI >512 từ |
|---|---:|---:|---:|
| HC3 train | 180 | 97.5 | 0.025% (1/4006) |
| HC3 validation | 170 | 91.5 | 0% |
| GPT-3.5 test | 491.5 | 533 | 40.8% |

V1 chấm logit từng node rồi lấy trung bình. V3 dùng `concat(mean, max, attention)` → MLP. Max pooling có thể phản ứng với số node và các activation hiếm; attention/MLP có thể học tín hiệu đặc trưng HC3. Đây là **giả thuyết**, chưa phải nguyên nhân đã được chứng minh.

Một kiểm tra chống kết luận quá mức: trong GPT AI, nhóm ≥512 từ lại có recall cao hơn nhóm 256–511 từ (V3 seed 2025: 39.61% so với 29.35%; seed 2026: 49.28% so với 38.23%). Vì thế dữ liệu không hỗ trợ câu đơn giản “càng dài càng bị nhận nhầm human”. Độ dài, nội dung và thể loại cùng thay đổi; phải ablate readout để kiểm tra cơ chế.

**Thí nghiệm ưu tiên:** V3 baseline `pool3` đối chiếu V3 `meanlogit`, cùng source cache, seed 2025/2026, micro-batch 8, EMA, augmentation, lịch LR và selection. Chỉ đổi readout. Đây không phải replay nguyên xi V1: backbone V3 vẫn được giữ. Khi thay head, việc tiêu thụ RNG có thể khác; cùng seed không có nghĩa dropout/shuffle hoàn toàn đồng nhất.

Đây là ablation có sẵn trong [train_v3.py](../train_v3.py), không cần thêm regularizer hoặc encoder. Nếu tăng Table 3 và giữ Table 4, có thể phát triển tiếp readout theo câu với trọng số chuẩn hóa; nếu không, bỏ giả thuyết thay vì thêm nhiều cơ chế cùng lúc. Chỉ đổi readout là một kết quả ablation, chưa phải thuật toán nghiên cứu mới.

## 5. Thứ tự và tiêu chí quyết định

1. Hoàn tất ClassSwap và Restricted đang chạy; dùng baseline đi kèm từng run.
2. Chạy `structmap` bằng Cell 6; nó kiểm tra trực tiếp bất nhất schema đã xác định, với `pool3` giữ nguyên.
3. Thử `pool3` vs `meanlogit` như Cell 7. Đây là ablation rẻ để kiểm tra một khác biệt lớn giữa v1 và V3.
4. Nếu cả hai không giúp, xem xét train trên các graph HC3 được ghép nhiều đoạn **cùng nhãn**, với phân phối số đoạn như nhau cho AI/human, nhằm giảm shortcut độ dài; cần giữ nhãn nguồn và kiểm chứng trên validation riêng. Đây là hướng nghiên cứu tiếp theo, chưa implement và không phải đề nghị chạy ngay.

Tiêu chí thực dụng đề xuất, cần chốt trước run: Table 3 tăng ít nhất **2 điểm %** so baseline cùng cấu hình, có cùng chiều cải thiện ở cả hai seed; Table 4 noise average giảm không quá **1 điểm %**, từng noise không giảm quá **2 điểm %**; HC3 clean giảm không quá **0.5 điểm %**. Đây là tiêu chí sàng lọc, không phải kiểm định thống kê. Tiếp theo xác nhận bằng seed mới và tập holdout chưa dùng phát triển; hai seed và việc xem nhiều lần test hiện tại chưa đủ cho tuyên bố tổng quát hóa chắc chắn.

Giữ threshold 0.5 cho bảng chính, báo cáo thêm AUC và recall từng lớp. Table 3 average gồm 5 điều kiện; Table 4 noise average gồm 4 nhiễu, không gồm Original. So trực tiếp với GCN tự train từng seed và ghi riêng v1 cũ. Chỉ chọn epoch bằng source validation; không chọn epoch/ngưỡng theo Table 3 hoặc Table 4. Benchmark đã dùng để định hướng ý tưởng nên kết quả tiếp theo vẫn là phát triển thực nghiệm, chưa phải test cuối độc lập.

## 6. Cell Kaggle: Corrected Structural Mapping trên Table 3–4

Variant `structmap` dùng `--struct-mode directed`. Mode này chỉ dành cho
`structmap`; trainer từ chối trộn nó với baseline legacy trong cùng run để
không vô tình ghi nhận hai schema graph khác nhau như một ablation chung.
`directed` dùng mọi cạnh khác self-loop là `dependent → head`, kiểm tra một
node chỉ có tối đa một cha và không silently clip relation ID ngoài `R`.

Chạy cell sau sau khi clone/cài dependencies. Nó train seed 2025/2026 rồi test
**Table 3 + Table 4** (`--suite all`), dùng threshold 0.5 trong CSV chính.
Kết quả được lưu riêng, không chạm checkpoint baseline/ClassSwap/Restricted.

```python
import os
from pathlib import Path

REPO = Path('/kaggle/working/PRDetect-Research')
os.chdir(REPO)

WORK = Path('/kaggle/working/v3_structmap_table3_table4')
SEEDS = '2025 2026'

!python run_v3_ablation.py \
  --work-dir "{WORK}" \
  --experiment-name v3_structmap_table3_table4 \
  --variants structmap \
  --seeds {SEEDS} \
  --struct-mode directed \
  --batch-size 32 \
  --micro-batch-size 8 \
  --val-batch-size 64 \
  --eval-batch-size 4 \
  --max-graphs 4 \
  --suite all

import pandas as pd
summary = WORK / 'eval/structmap/result/summary_threshold_0.5.csv'
display(pd.read_csv(summary))
print('Checkpoint/archive:', WORK / 'model', WORK / 'v3_structmap_table3_table4_ablation_results.zip')
```

So sánh các dòng Table 3/Table 4 với V3 baseline controlled bằng cùng seed,
ngưỡng 0.5 và cấu hình train mặc định. Không dùng `--skip-clean-check`: dòng
HC3 Original trong Table 4 là sanity check. Nếu workflow dừng với lỗi
`at most one non-self parent edge`, không bỏ qua lỗi; graph cache không theo
schema `building_graph.py` một chiều và cần được xác minh trước.

## 7. Cell Kaggle: ablation readout trên cả Table 3–4

Chạy sau cell cài dependencies và clone repo hiện tại, trong cùng môi trường đang dùng cho V3. Chỉ cần các file hiện có của repo; chưa cần sửa trainer/evaluator. Cell dùng thư mục mới, train cả hai readout với cùng cache, giải phóng cache train do workflow tạo trước khi benchmark. Các lệnh dưới **đã kiểm tra parser, chưa chạy train/GPU tại local**.

```python
import os
from pathlib import Path
import sys

REPO = Path('/kaggle/working/PRDetect-Research')
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from run_v3_ablation import child, training_cache, release_training_cache

WORK = Path('/kaggle/working/v3_readout_audit')
WORK.mkdir(parents=True, exist_ok=False)  # Dùng tên mới nếu đã chạy trước đó.
SEEDS = [2025, 2026]
caches = [training_cache(
    REPO / f'data/original_text/benchmarks/hc3/{name}.json', WORK, 4
) for name in ('hc3_train', 'hc3_val')]

for readout in ('pool3', 'meanlogit'):
    child('train_v3.py',
          '--variants', 'baseline', '--seeds', *SEEDS,
          '--graph-dir', WORK / 'training/graph_data',
          '--output-dir', WORK / readout / 'model',
          '--readout', readout,
          '--batch-size', 32, '--micro-batch-size', 8,
          '--val-batch-size', 64,
          '--hidden', 384, '--num-bases', 12, '--layers', 2,
          '--dropout', .3, '--in-dropout', .1, '--edge-drop', .1,
          '--struct', 1, '--gfeat', 0, '--struct-mode', 'low',
          '--aug', 'swap', '--aug-max', .35,
          '--ema', 1, '--ema-decay', .998,
          '--lr', .0005, '--wd', .01,
          '--epochs', 10, '--patience', 10, '--select', 'lossaug',
          '--amp', 0, '--device', 'cuda')

for cache in caches:
    release_training_cache(cache, WORK)

for readout in ('pool3', 'meanlogit'):
    child('evaluate_v3_quang.py', 'run',
          '--variant', 'baseline', '--seeds', *SEEDS,
          '--checkpoint-dir', WORK / readout / 'model',
          '--work-dir', WORK / readout / 'eval',
          '--suite', 'all',  # Table 3 + Table 4, có HC3 clean.
          '--batch-size', 4, '--struct-mode', 'low',
          '--max-graphs', 4, '--graph-budget-root', WORK,
          '--device', 'cuda')

import pandas as pd
for readout in ('pool3', 'meanlogit'):
    print('\nREADOUT:', readout)
    display(pd.read_csv(WORK / readout / 'eval/result/summary_threshold_0.5.csv'))
    print('ZIP:', WORK / readout / 'eval/v3_baseline_all_results.zip')
```

Cả hai checkpoint mang tên method `v3_baseline`, nhưng được lưu ở thư mục riêng và có `readout` trong metadata để evaluator dựng đúng model. Không gộp nhầm chúng thành một ensemble. Sau sàng lọc Table 3–4, chạy thêm suite `full` để xác nhận toàn bộ Table 2 của phương án được chọn.

## 8. Giới hạn thực hiện lần này

Môi trường Python local được cấu hình nhưng chưa có torch/numpy/spaCy/PyG và không có cache graph train/test. Vì vậy lần này chỉ thực hiện kiểm toán dự đoán đã lưu, kiểm tra source/metadata và chạy kiểm chứng hàm mask độc lập; **chưa có điểm mới của `meanlogit` hoặc `structmap`**. `structmap` đã được kiểm tra bằng unit test khi có runtime, kiểm tra parser/syntax tại local, nhưng cần chạy Kaggle để có kết quả benchmark. Không hứa một trong các hướng trên chắc chắn vượt 70%; lợi ích của chúng là kiểm tra đúng các khác biệt có căn cứ thay vì thêm cải tiến mù.
