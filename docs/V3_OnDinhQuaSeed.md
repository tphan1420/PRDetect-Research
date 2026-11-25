# Ổn định accuracy của V3 khi chuyển từ HC3 sang GPT‑3.5

## Kết quả chính

V3 dùng một checkpoint RAW cho mỗi seed. Khi tắt deterministic, đổi quy tắc quyết định bằng HC3 validation nâng accuracy GPT‑3.5 Mixed sạch từ **73,17% ± 2,86 lên 79,20% ± 1,14**. Ba seed đều vượt mức **73,2%** trong Table 3 của bài báo. GCN theo code tác giả, được project train lại trên đúng ba seed, đạt **69,73% ± 0,47**. Độ lệch chuẩn trong báo cáo là độ lệch chuẩn mẫu giữa các seed, tính theo điểm phần trăm. [PRDetect, Table 3](https://aclanthology.org/2025.findings-naacl.464.pdf#page=7)

| Seed và lần chạy | V3 RAW cũ | V3 hiệu chỉnh | V3 + A1 |
|---|---:|---:|---:|
| 2020 | 69,9% | 77,9% | 86,3% |
| 2022 | 74,4% | 79,7% | 86,2% |
| 2024 | 75,2% | 80,0% | 85,8% |
| **Trung bình ± SD** | **73,17% ± 2,86** | **79,20% ± 1,14** | **86,10% ± 0,26** |
| 2024 chạy lại | 66,6% | 80,3% | 86,7% |

Ở seed 2024, hai lần train có SHA checkpoint khác nhau. RAW lệch 8,6 điểm giữa hai lần, V3 hiệu chỉnh lệch 0,3 điểm. Dòng chạy lại được báo cáo riêng để mỗi seed có trọng số bằng nhau. Hai tệp seed 2022 trong `noDesterminsm` có cùng SHA checkpoint và chỉ đóng góp một lần.

Trên ba seed tắt deterministic, chênh lệch accuracy so với V3 gốc là **+6,03 điểm**. Khoảng bootstrap 95% theo văn bản là **từ +4,77 đến +7,30 điểm**. So với GCN train lại cùng seed, chênh lệch là **+9,47 điểm**, khoảng **từ +7,37 đến +11,53 điểm**. Phép bootstrap lấy mẫu lại 2.000 lần trong từng lớp, dùng cùng chỉ số văn bản cho cả ba seed. Khoảng này đo biến động lấy mẫu văn bản với ba checkpoint đã có.

Ở nhóm bật deterministic, seed 2020 nhận đúng **157/500 văn bản AI** và **494/500 văn bản người**. Sau hiệu chỉnh, hai số lần lượt là **295/500** và **467/500**, đưa accuracy từ **65,1% lên 76,2%**. Trên cả năm seed của nhóm này, accuracy tăng từ **69,86% ± 3,12 lên 78,06% ± 1,16**.

## Quy tắc quyết định cho một checkpoint

Sau khi chọn checkpoint bằng quy trình cũ, lấy điểm `P(human)` cho các văn bản người viết trên **HC3 validation**. Sắp xếp tăng dần và lấy phân vị 5% theo quy ước `lower`. Với 500 văn bản người, đây là phần tử thứ 25. Khi suy luận, dự đoán human nếu điểm lớn hơn hoặc bằng ngưỡng đó. Mỗi checkpoint dùng ngưỡng riêng, áp dụng cho mọi văn bản, tập và mức nhiễu. Mục tiêu thiết kế 95% là recall người viết trên validation nguồn. Trên miền mới, recall có thể thấp hơn.

Quy tắc này chỉ đổi ranh giới phân loại. Thứ tự xếp hạng và AUC của V3 giữ nguyên. Lợi ích đến từ việc giảm số văn bản AI bị gán nhãn human. Ví dụ, ở seed 2020 tắt deterministic, recall AI tăng **42,6% → 62,2%**, còn recall human giảm **97,2% → 93,6%**. Các ngưỡng được tính trước khi chấm điểm mới trên GPT‑3.5. Việc chọn hướng nghiên cứu có tham khảo các lỗi của baseline đã công bố trong project.

## Kiểm tra trên tập GPT‑3.5 validation riêng

Sau khi cố định ngưỡng bằng HC3, chạy ba checkpoint tắt deterministic trên cache `gpt3.5_mixed_val_split.pkl`. Đối chiếu văn bản sau khi chuẩn hóa khoảng trắng phát hiện hai mẫu của split này cũng xuất hiện ở GPT‑3.5 test. Loại hai mẫu đó còn **998 văn bản**. Accuracy trên tập này tăng từ **71,94% ± 3,87 lên 77,42% ± 1,73**. Theo thứ tự seed 2020, 2022, 2024, V3 hiệu chỉnh đạt 75,55%, 77,76%, 78,96%. [Dự đoán và chỉ số của phép kiểm tra](../result/Result_v3/source_calibration/gpt35_holdout.json)

Không có văn bản trùng nguyên nội dung đã chuẩn hóa giữa HC3 train/validation và hai split GPT‑3.5 đã chấm. Trong HC3, 16 dòng validation trùng nội dung với train và một nội dung validation trùng HC3 test. Phép hiệu chỉnh dùng đủ split validation chính thức để giữ cùng giao thức với các checkpoint cũ. Các graph validation được dựng lại bằng môi trường CPU cục bộ, SHA nguồn được đối chiếu với metadata huấn luyện sau khi chuẩn hóa xuống dòng.

## Độ bền trên các bảng của bài báo

Nhóm `dual_rawema` có dự đoán cho đủ năm seed 2020, 2022, 2024, 2025, 2026 và đủ các điều kiện. Các run lịch sử yêu cầu tối đa 25 epoch. Hai nhóm ở trên yêu cầu 10 epoch. Các thống kê dưới đây dùng riêng nhóm lịch sử. Table 2 và Table 3 lấy trung bình năm mức 0/5/10/20/30% trong từng seed. Table 4 lấy trung bình bốn phép gây nhiễu 25% trong từng seed, rồi mới tính độ lệch chuẩn.

| Accuracy trung bình | GCN train lại | PRDetect trong bài báo | V3 RAW cũ | V3 hiệu chỉnh | V3 + A1 |
|---|---:|---:|---:|---:|---:|
| Table 2 · HC3 | 98,44% ± 0,48 | 98,73% | 98,28% ± 0,27 | 97,53% ± 0,54 | 98,67% ± 0,27 |
| Table 3 · HC3 → GPT‑3.5 | 68,58% ± 1,03 | 72,28% | 70,95% ± 2,88 | 79,12% ± 0,50 | **86,94% ± 0,30** |
| Table 4 · HC3, bốn loại nhiễu | 82,42% ± 1,48 | 87,75% | 89,28% ± 1,63 | **93,82% ± 0,60** | 89,30% ± 1,63 |

Table 3 của V3 hiệu chỉnh đạt lần lượt **80,48%, 79,38%, 79,22%, 78,48%, 78,06%** từ sạch tới 30% nhiễu. Table 4 đạt **95,90%** khi chèn, **97,60%** khi lặp, **91,14%** khi thay và **90,64%** khi xóa. Bài báo dùng riêng seed 2024 cho Table 4. Nếu cũng chỉ lấy seed 2024, trung bình bốn nhiễu của V3 hiệu chỉnh là **93,55%**. Table 2 cho thấy chi phí khi áp dụng cùng ngưỡng cho văn bản HC3. [PRDetect, Tables 2–4](https://aclanthology.org/2025.findings-naacl.464.pdf#page=6)

## Đối chứng

**GCN tác giả với cùng quy tắc ngưỡng.** Tính lại phân vị 5% trên cùng HC3 validation cho từng checkpoint GCN đã train lại, rồi áp vào dự đoán gốc từng mẫu. Trên GPT‑3.5 sạch và ba seed 2020/2022/2024, GCN hiệu chỉnh đạt **77,70%**, còn V3 hiệu chỉnh đạt **79,20%**. Phần chênh còn lại là **+1,50 điểm**. Khoảng bootstrap 95% theo văn bản **từ −0,40 đến +3,40 điểm** chứa 0, nên riêng phép so này chưa phân định rõ phần lợi thế đến từ kiến trúc. Riêng seed 2020, GCN hiệu chỉnh đạt 78,1% so với V3 77,9%. Trên đủ năm seed, điểm trung bình từng bảng như sau. [Kết quả GCN dùng cùng ngưỡng nguồn](../result/Result_v3/source_calibration/author_human95_control.json)

| Benchmark | GCN hiệu chỉnh | V3 hiệu chỉnh |
|---|---:|---:|
| Table 2 | 96,92% ± 0,25 | 97,53% ± 0,54 |
| Table 3 | 77,28% ± 1,44 | 79,12% ± 0,50 |
| Table 4 | 89,38% ± 1,66 | 93,82% ± 0,60 |

**Giả thuyết về độ dài.** Các đồ thị HC3 validation cùng nhãn được ghép thành các nhóm 1, 2, 4, 8 và 16 văn bản. Mean, max và attention pooling được tính lại trên hợp các đồ thị. Trừ độ trôi trung điểm hai lớp theo độ dài làm accuracy GPT‑3.5 sạch ở nhóm tắt deterministic giảm từ 73,17% xuống **68,63%**. Hiệu chỉnh theo phân vị phía trên dùng cùng một ngưỡng cho mọi độ dài.

**Nhánh A1.** Một phương án dùng điểm human bằng tích của `P(human)` từ V3 và A1, quyết định ở 0,5. A1 dùng RoBERTa căn chỉnh subword với từng từ spaCy. Hai nhánh dùng cùng văn bản và cách gán nhãn. Điểm tích là điểm quyết định. A1 là một checkpoint cố định seed 2026 trong mọi lần đo, nên SD của phương án kết hợp chỉ đo ảnh hưởng của việc thay checkpoint V3. A1 chạy riêng đạt **86,9%** trên GPT‑3.5 sạch. Phương án kết hợp dùng thêm một lượt mã hóa RoBERTa và một R‑GCN.

Trên năm điều kiện Table 3, A1 chạy riêng đạt trung bình 87,58%, V3 + A1 đạt 86,94%. Lợi ích của việc kết hợp thể hiện ở Table 4, nơi A1 riêng chỉ đạt 55,98% và V3 + A1 đạt 89,30%. Trên GPT‑3.5 sạch ở ba run tắt deterministic, V3 + A1 đạt recall AI 90,73% và recall human 81,47%. V3 đơn hiệu chỉnh đạt lần lượt 66,13% và 92,27%.

Điểm A1 cho HC3 sạch, GPT‑3.5 sạch, GPT‑3.5 synonym 30% và HC3 insert/repeat/delete được lấy từ `result/a1_threshold`. Các điều kiện còn lại được tính từ cùng trọng số trên CPU. Khi chạy lại 20 mẫu cách đều của mỗi tập đã có điểm, quyết định A1 ở ngưỡng 0,5 khớp trên **119/120 mẫu**. Runtime và độ chính xác số có thể ảnh hưởng các mẫu gần ngưỡng. [Đối chiếu logit A1](../result/Result_v3/v3_a1/scorer_parity.json)

## Tái lập

Chạy các lệnh tại thư mục gốc project. Để dùng checkpoint cũ, tạo cache HC3 validation và tính ngưỡng bằng mã dưới đây. ZIP kết quả dual sẽ dùng trọng số RAW. Tệp tham số được đặt tên bằng SHA‑256 của checkpoint.

```powershell
python building_graph.py -f hc3_val
$runs = Get-ChildItem 'result/Result_v3/noDesterminsm' -Filter '*_batch1.zip'
python calibrate_v3.py --checkpoints $runs.FullName --output-dir result/Result_v3/source_calibration/parameters
```

Để phân loại trực tiếp bằng một checkpoint đã hiệu chỉnh, dùng lệnh sau. Với trọng số train mới bằng `train_v3.py`, ngưỡng được lưu trong checkpoint và có thể bỏ `--human95-dir`.

```powershell
$run = 'result/Result_v3/noDesterminsm/v3_dual_ema_raw_hc3_to_gpt35_0pct_seed2020_batch1.zip'
python v3_a1_fusion.py predict --rule human95 --v3-checkpoint $run --human95-dir result/Result_v3/source_calibration/parameters --text 'The study examined how people write about the same event.'
```

Để train mới một V3 RAW giống cấu hình batch 1 của thí nghiệm, có thể chạy lệnh dưới sau khi tạo graph `hc3_train` và `hc3_val`. Trainer chính mặc định tắt deterministic. Notebook Kaggle dual HC3 → GPT‑3.5 trong workspace cũng đã truyền `--no-deterministic` và lưu ngưỡng RAW, EMA ở epoch tốt nhất của từng nhánh.

```powershell
python train_v3.py --no-deterministic --ema 0 --batch-size 1 --micro-batch-size 1 --val-batch-size 64 --epochs 10 --seeds 2020 2022 2024 2025 2026 --output-dir result/v3_raw_human95
```

Evaluator xuất cả accuracy ở ngưỡng gốc, ở ngưỡng tối ưu accuracy validation cũ, và ở ngưỡng giữ 95% người viết. `--weights raw` nhận trực tiếp các tệp `v3_baseline_raw_seed*.pt` đã có. Với toàn bộ thư mục kết quả đi kèm workspace, lệnh cuối đọc các dự đoán lưu sẵn và ghép theo split, chỉ số mẫu, nhãn và SHA checkpoint.

```powershell
python evaluate_v3_quang.py run --variant baseline --weights raw --checkpoint-dir result/Result_v3/dual_rawema/checkpoints_batch1_raw_ema --human95-dir result/Result_v3/source_calibration/parameters --seeds 2020 2022 2024 2025 2026 --suite full --max-graphs 8 --work-dir result/v3_human95_eval
python benchmark_v3_stability.py --holdout-graph graph_data/gpt3.5_mixed_val_split.pkl --output result/Result_v3/source_calibration/gpt35_holdout.json
python benchmark_v3_stability.py --author-val-graph graph_data/hc3_val.pkl --output result/Result_v3/source_calibration/author_human95_control.json
python benchmark_v3_stability.py --output result/Result_v3/source_calibration/paired_results.json
```

Phương án hai nhánh dùng `python v3_a1_fusion.py predict --rule fusion --v3-checkpoint $run --text '...'`. Để tính điểm A1 cho cả split, lệnh `score-a1` nhận các đường dẫn JSONL qua `--sources` và lưu vào thư mục do `--output-dir` chỉ định.

Các số liệu theo từng seed, confusion matrix, recall, AUC, ngưỡng và SHA nằm trong [kết quả ghép cặp](../result/Result_v3/source_calibration/paired_results.json).
