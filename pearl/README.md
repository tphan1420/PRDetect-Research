# PEARL-Lite cho PRDetect

Notebook Kaggle [02 — train từ đầu](../notebooks/kaggle/pearl/02_pearl_lite_from_scratch_hc3_to_gpt35_0pct_hc3_insert25.ipynb)
train predictor rồi khởi tạo GCN ngẫu nhiên và train joint, không cần baseline checkpoint.
Config dùng `gcn_initialization="scratch"`, kiến trúc mặc định `input_dim=768`, `hidden_dim=256`, `output_dim=64`.
Chế độ cũ `gcn_initialization="checkpoint"` vẫn là mặc định để notebook 01 giữ nguyên hành vi.
Metadata checkpoint ghi rõ kiểu khởi tạo; scratch có `baseline_sha256=null`.

Pipeline chạy riêng, không sửa GCN/RGCN/LISA/PRISM baseline. Bản triển khai dùng đúng hai lớp GCN, ReLU/dropout, linear head và mean pooling của `model/GCN2.py`. Chuẩn hóa/self-loops dùng GCNConv gốc; mỗi directed arc được nhân `epsilon + (1-epsilon)*sigmoid(q_logit)`. ROOT giữ weight=1. Có adapter cho fallback của baseline, nhưng checkpoint không chuyển qua lại giữa hai backend.

## Cài đặt

Chạy từ repository root. Các lệnh `python` dưới đây phải trỏ tới cùng môi trường Python. Môi trường đã kiểm thử: Windows, Python 3.13, PyTorch CPU, PyG, spaCy 3.8.7 và `en_core_web_sm` 3.8.0.

```powershell
python -m pip install -r pearl/requirements.txt
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python -m pearl.download_roberta --output output/pearl/roberta-base
```

Nếu đã có RoBERTa local, đặt `roberta` trong config tới thư mục đó, không cần tải lại. CLI download từ chối ghi đè thư mục đã có dữ liệu. Có thể dùng model ID `roberta-base`, `local_files_only=false`, và `roberta_revision` để pin revision. Tải model cần kết nối Hugging Face.

Giữ nguyên cách tạo embeddings legacy: lấy `token.text` từ spaCy → `convert_tokens_to_ids` → chunks 512 → `last_hidden_state`; không thay bằng BPE/token alignment mới. Encoder đóng băng, không thuộc optimizer. `from_pretrained()` giữ eval mode như builder baseline. Baseline lịch sử không lưu phiên bản parser; vì vậy parity với runtime hiện tại không chứng minh parity với cache cũ chưa có tại máy.

## Chạy smoke test với dữ liệu thật

Config nhỏ là [pearl_smoke.json](../configs/pearl_smoke.json): 24 source train, 8 source validation, 5 edit families, predictor 3 epoch, GCN 2 epoch. Tất cả đường dẫn tương đối repository root. Checkpoint đầu vào là HC3 baseline có sẵn; output nằm riêng bên dưới `output/pearl/`.

```powershell
python -m pearl.prepare --config configs/pearl_smoke.json
python -m pearl.train_edge --config configs/pearl_smoke.json
python -m pearl.train_gcn --config configs/pearl_smoke.json
python -m pearl.evaluate --config configs/pearl_smoke.json --limit 4 --output output/pearl/smoke/evaluation.json
python -m pearl.infer --config configs/pearl_smoke.json --text "The careful doctor helps people."
python -m pearl.diagnostics --config configs/pearl_smoke.json --output output/pearl/smoke/parity.json
python -m pytest tests/test_pearl.py tests/test_pearl_pipeline.py -q
```

Các output đã tồn tại ở máy này từ lượt smoke thực tế. Để chạy lại, copy config và đổi ba đường dẫn `prepared`, `edge_output`, `gcn_output` sang tên run mới; không xóa hoặc ghi đè baseline. CLI từ chối ghi đè manifest/checkpoint/kết quả. Cache embeddings sạch được reuse khi chữ ký text, graph và encoder trùng; cache không hợp lệ sẽ báo lỗi.

## Chạy lớn hơn và Table 2/3/4

[pearl_hc3.json](../configs/pearl_hc3.json) bỏ giới hạn source documents, predictor 20 epoch và fine-tune 10 epoch. Chưa chạy config lớn này. Có thể đặt `train_limit`, `validation_limit` để tăng dần theo RAM/thời gian thực tế; đặt `device="cuda"` nếu môi trường có PyTorch CUDA phù hợp. Seed generation là deterministic; các kernels CUDA còn tùy runtime hỗ trợ deterministic algorithms.

```powershell
python -m pearl.prepare --config configs/pearl_hc3.json
python -m pearl.train_edge --config configs/pearl_hc3.json
python -m pearl.train_gcn --config configs/pearl_hc3.json
python -m pearl.evaluate --config configs/pearl_hc3.json --tables table2 --output output/pearl/hc3_seed2025/table2.json
python -m pearl.evaluate --config configs/pearl_hc3.json --tables table3 --output output/pearl/hc3_seed2025/table3.json
python -m pearl.evaluate --config configs/pearl_hc3.json --tables table4 --output output/pearl/hc3_seed2025/table4.json
```

Không truyền `--limit` là đánh giá toàn bộ các file benchmark. Đường dẫn mặc định lấy từ `data_root`:

| Config dataset | Table 2 | Table 3 | Table 4 |
|---|---|---|---|
| `hc3` | HC3 original + syn5/10/20/30 | HC3-trained → GPT3.5-Mixed original + synonyms | HC3 original + insert/repeat/replace/delete25 |
| `gpt35_mixed` | GPT3.5-Mixed original + synonyms | GPT-trained → HC3 original + synonyms | HC3 structural conditions, dưới checkpoint GPT |

Table 4 paper chuẩn cần checkpoint HC3-trained. Các edit benchmark đọc file có sẵn, không thay bằng generator training nhỏ. Để đổi layout, đặt `evaluation_entries` là list `{table, dataset, condition, path}`; hoặc đổi `data_root`. File thiếu được ghi `status=missing`; label order/row count của các điều kiện được kiểm tra so với original khi có.

Để train GPT3.5-Mixed, copy config HC3, đổi `dataset="gpt35_mixed"`, `source_train="data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_train_split.json"`, `source_validation="data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_val_split.json"`, baseline checkpoint GPT tương ứng và các thư mục output. Repository hiện chưa có checkpoint GCN GPT, nên chiều train GPT chưa được chạy. Không dùng checkpoint HC3 rồi gọi nó là GPT-trained.

Các tham số chính: `seed`, `edge_lr`, `gcn_lr`, `epsilon`, `lambda_edge`, `edge_epochs`, `gcn_epochs`, `edge_batch_docs`, `gcn_batch_docs`, `parser`, `roberta`, `baseline_checkpoint`. `shuffle_control=true` bật đối chứng shuffled labels tùy chọn. Constant/relation/POS priors rẻ được ghi kèm predictor metrics; không có downstream ablation tự động.

## Dữ liệu và loss

- `prepare` kiểm tra toàn bộ source partitions trước khi chọn subset. `overlap_policy="error"` dừng nếu có trùng; `quarantine_train` loại train overlap và cả nhóm parent/content liên quan, giữ validation nguyên trạng. Với HC3 hiện tại, loại 17 train rows. Không sửa file nguồn. Baseline lịch sử có thể đã thấy các văn bản trùng này; việc lọc PEARL không xóa được lịch sử đó.
- Mỗi token gốc có UID riêng, source span và destination span. Repeat/Insert có UID mới; Delete mất endpoint thì unknown. Chỉ nhận mapping 1:1 với exact character span; split/merge/resegmentation bị mask. Token index mới không được dùng làm UID.
- Source arc `(child, head, dep)` là stable nếu triple sau mapping còn nguyên, changed nếu cả hai endpoint map được nhưng triple đổi, unknown nếu endpoint mất/không chắc chắn. ROOT/self-loop không vào retention loss.
- Synonym hiện dùng lexicon pilot nhỏ, công khai trong `edit_provenance.py`; đổi bằng `synonyms` trong config. Random replacement/insert dùng `replacement_words` nếu cấu hình. Ratio tính trên eligible lexical tokens, số thực được ghi. Synonym không có từ phù hợp có thể no-op, được đếm rõ. Không gọi đây là generator WordNet official.
- Predictor chỉ dùng POS hai đầu mút, dependency, log khoảng cách, log degree và hướng trái/phải từ graph hiện tại. Không dùng Human/AI, edit family/ratio, UID, text, label retention hoặc target data làm feature.
- Predictor train bằng BCEWithLogitsLoss: trung bình valid arcs qua tất cả pairs của mỗi document, rồi trung bình documents hợp lệ trong batch. Joint fine-tune: clean classification BCE trên logits trước sigmoid cộng `lambda_edge * document-balanced edge BCE`. Edited text không vào class loss. Checkpoint chọn theo source-validation clean BCE.
- Inference nhận một text, tự parse/encode/predict q/classify. Không cần reference clean, perturbation hay provenance.

## Output và giới hạn

Prepared metadata là gzip JSONL + manifest có seed, version/hash parser, generator, mã nguồn, input, IDs và quarantine list. Chỉ lưu 2 cache embeddings sạch `.pt` (train/validation), không tạo graph `.pkl`; edited graphs chỉ có metadata. Predictor checkpoint `edge.pt`; checkpoint cuối `best.pt` lưu GCN, predictor, feature schema, epsilon và provenance. Toàn bộ output/model mới được ignore Git.

`metrics.json` có AUROC/AUPRC/prevalence, Brier/ECE/log-loss và phân phối q clean/edited; không áp nhãn `G→G'` vào cạnh `G'`. `training.json` có classification validation, epoch được chọn và kiểm tra gradient predictor. Evaluation JSON có từng condition, confusion/recall/F1/AUROC và xác suất từng mẫu.

Xem [báo cáo smoke thực tế](../docs/pearl_smoke_report.md) và [audit baseline](../docs/pearl_baseline_audit.md). Smoke xác nhận thực thi, **chưa chứng minh PEARL cải thiện PRDetect**. Chưa chạy full-data training/full benchmark/GPT-trained reverse transfer hay các controls nghiên cứu. Full preparation hiện giữ metadata pairs trong RAM; nên tăng subset dần nếu RAM hạn chế. Không thêm EditedView/Survival/kiến trúc mới để cải thiện chỉ số.
