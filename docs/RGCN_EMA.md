# RGCN + EMA

## Phạm vi

Triển khai trên nhánh `feat/rgcn-ema`, từ RGCN V1. Không đưa A1, A2, A3 vào pipeline. Giữ nguyên embedding, graph, kiến trúc, BCE loss, optimizer, dropout và ngưỡng phân loại 0.5. EMA là một thử nghiệm; không đảm bảo tăng điểm và vẫn phải kiểm tra các tập nhiễu cấu trúc.

Các thay đổi:

- [ema.py](../ema.py): lớp `ModelEMA`, giữ bản sao độc lập của model, không nhận gradient.
- [rgcn.py](../rgcn.py): bật EMA bằng `--ema`, cập nhật sau mỗi `optimizer.step()`, validation và lưu checkpoint riêng.
- [test_rgcn.py](../test_rgcn.py): thêm `--checkpoint` để chọn rõ checkpoint thường hoặc EMA. Log test ghi đường dẫn checkpoint để tránh nhầm hai kết quả.

## Cách hoạt động

Khởi tạo trọng số EMA bằng đúng trọng số RGCN lúc bắt đầu train. Sau mỗi mini-batch:

```text
theta_ema = decay * theta_ema + (1 - decay) * theta_raw
```

Mặc định `decay=0.99`, cố định từ bước đầu; không warmup, không scheduler hay bias correction. Buffer được copy trực tiếp từ model thường; RGCN hiện tại không có BatchNorm. Chỉ model thường được optimizer cập nhật. EMA không phải trung bình xác suất hay ensemble hai lượt inference.

Mỗi epoch chạy validation cho cả hai model bằng cùng dữ liệu và ngưỡng 0.5. Mỗi model lưu checkpoint có validation accuracy tốt nhất của riêng nó. Khi bằng điểm, lưu epoch mới nhất, giống quy tắc V1. Early stopping **chỉ theo validation accuracy của model thường**, giữ nguyên ngân sách train của đối chứng; EMA không kéo dài training. Vì vậy EMA có thể chưa hội tụ tối ưu khi raw dừng sớm, cần xem lịch sử validation trước khi kết luận.

Lượt validation thêm cho EMA được bao quanh bằng `torch.random.fork_rng` để không tiêu thụ RNG dùng cho shuffle/dropout ở epoch tiếp theo. Không thay đổi đường train thường khi bỏ `--ema`. Điều này không đảm bảo bitwise deterministic trên mọi CUDA/PyG backend.

## Chạy trên Kaggle

Chạy trong thư mục repository, sau khi cài dependency như pipeline RGCN V1. Không dùng graph V2/A1 để đánh giá thử nghiệm này. Nếu đã có cache graph V1 đúng pipeline thì không cần build lại. Cần có graph train và validation trước khi train.

### 1. Train một lần, lấy cả raw và EMA

```python
!python rgcn.py --dataset hc3_ema_exp --seed 2026 --ema --ema_decay 0.99 --no_test
```

`--dataset hc3_ema_exp` chỉ đặt tên checkpoint/log; train và validation mặc định vẫn là `hc3_train` và `hc3_val`. Tên mới giúp không ghi đè checkpoint HC3 cũ. Không cần train thêm một lượt không EMA để có đối chứng: cùng lần chạy này đã lưu cả hai.

Kết quả:

- `model/hc3_ema_exp_rgcn_model_2026.pth`: RGCN thường tốt nhất.
- `model/hc3_ema_exp_rgcn_model_2026_ema.pth`: EMA tốt nhất.
- `model/hc3_ema_exp_rgcn_model_2026_ema.json`: cấu hình, số lần cập nhật EMA, epoch/accuracy tốt nhất và lịch sử validation của cả hai.
- `logs/`: TensorBoard nếu có thư viện, thêm `Loss/val_ema` và `Acc/val_ema`.

Có thể đặt đường dẫn bằng `--save_path` và `--ema_save_path`. Mặc định đường dẫn EMA thêm `_ema` trước đuôi checkpoint raw. Hai đường dẫn không được trùng nhau. Mỗi lần chạy lại cùng tên sẽ ghi đè các checkpoint/summary tương ứng; dùng tên dataset khác khi đổi seed/cấu hình thử nghiệm nếu cần giữ toàn bộ kết quả.

### 2. Test đối chứng cùng một graph

Sau khi có graph test V1:

```python
!python test_rgcn.py --dataset hc3_ema_exp --seed 2026 --file hc3_test --checkpoint model/hc3_ema_exp_rgcn_model_2026.pth --save
!python test_rgcn.py --dataset hc3_ema_exp --seed 2026 --file hc3_test --checkpoint model/hc3_ema_exp_rgcn_model_2026_ema.pth --save
```

Lặp lại hai lệnh cho các tập cần đối chứng, thay `--file`:

```text
hc3_test_insert_25
hc3_test_delete_25
hc3_test_repeat_25
gpt3.5_mixed_test_split
gpt3.5_mixed_test_adj_30_synonym_replace
```

Kết quả append vào `result/test_result.txt`, có trường `checkpoint` phân biệt raw/EMA. Cần dùng cùng cấu hình kiến trúc khi train/test nếu thay các tham số `input_dim`, `hidden_dim`, `output_dim`, `num_layers`, `num_bases`.

Checkpoint EMA là `state_dict` RGCN thuần, không có tiền tố wrapper, nên inference dùng nguyên model cũ. Nếu không truyền `--checkpoint`, test vẫn nạp checkpoint thường theo tên mặc định; **không tự chuyển sang EMA**. Nếu train không có `--no_test`, script tự test cả checkpoint raw và EMA khi graph test tồn tại.

### 3. Bộ nhớ và cách kết luận

EMA giữ thêm một bản trọng số model trên cùng thiết bị, không copy tập graph, không có optimizer state/gradient riêng. Tăng thêm một lượt validation mỗi epoch, không tăng số graph cần build. Giới hạn tối đa 4 file graph vẫn cần tự duy trì như workflow trước: nên test tuần tự từng graph cho cả raw/EMA, lưu kết quả rồi mới xóa cache graph test để build tập tiếp theo. Không xóa checkpoint khi dọn graph.

Đánh giá raw và EMA của **cùng lần train** trước; không dùng chênh lệch so với checkpoint cũ khác seed để quy hết cho EMA. So sánh accuracy, AUC, F1 trên clean, structural noise và GPT3.5. Không chọn decay/epoch/threshold bằng điểm test. Nếu muốn thử decay khác, chọn bằng validation và sau đó mới đánh giá test. Nếu EMA làm structural noise giảm mạnh thì không giữ cải tiến này chỉ vì GPT3.5 tăng. Nếu chênh lệch nhỏ, chạy thêm seed trước khi khẳng định tăng hiệu năng.

Không thể phục dựng EMA đúng từ một checkpoint cuối duy nhất; cần train lại và cập nhật EMA xuyên suốt quá trình. Chưa triển khai resume optimizer/EMA: checkpoint hiện dùng cho inference, không phải trạng thái resume đầy đủ.
