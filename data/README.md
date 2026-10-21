# Data layout

Thư mục này chứa toàn bộ dữ liệu đầu vào cho PRDetect. Dữ liệu gốc và dữ liệu đã
được làm nhiễu được tách riêng để việc huấn luyện, đánh giá độ bền và truy vết
nguồn dữ liệu rõ ràng hơn.

```text
data/
├── original_text/                         # Dữ liệu gốc, chưa perturb
│   ├── benchmarks/                        # Benchmark dùng để huấn luyện/đánh giá
│   │   ├── hc3/
│   │   ├── gpt35_mixed/
│   │   ├── grover/
│   │   └── raid/
│   ├── generated/                         # Văn bản gốc do mô hình sinh
│   │   ├── gpt2/
│   │   └── llama2/
│   ├── derived/                           # Tập được tạo từ dữ liệu gốc
│   │   ├── matched/
│   │   └── mixed/
│   ├── converted/                         # Đầu ra mặc định của convert_datasets.py
│   └── archive/                           # Notebook và checkpoint lịch sử
│       ├── notebooks/
│       └── notebook_checkpoints/
└── perturbed_text/                        # Dữ liệu đã được biến đổi
    ├── gpt2/
    ├── gpt35_mixed/
    ├── hc3/
    ├── hc3_chatgpt/
    ├── llama2/
    └── matched/
        └── <perturbation-type>/
            ├── lexical_adjective/
            ├── lexical_synonym/
            ├── structural_noise/
            ├── rewrite/
            ├── translation/
            ├── wiki_4gram_replace/
            └── generated/
```

Không phải mọi nguồn dữ liệu đều có đủ tất cả loại nhiễu. Cây thư mục trên mô tả
quy ước tổ chức; chỉ các thư mục có dữ liệu mới xuất hiện trong Git.

## Ý nghĩa các nhóm dữ liệu

| Nhóm | Mục đích |
| --- | --- |
| `original_text/benchmarks/` | Các benchmark gốc: HC3, GPT-3.5 mixed, Grover và RAID. |
| `original_text/generated/` | Văn bản sinh trực tiếp từ GPT-2 hoặc LLaMA 2. |
| `original_text/derived/` | Tập pha trộn hoặc đã ghép cặp, được tạo từ tập gốc. |
| `original_text/converted/` | Dữ liệu được chuẩn hóa bởi [`convert_datasets.py`](../convert_datasets.py). Thư mục có thể chưa tồn tại cho đến lần chuyển đổi đầu tiên. |
| `original_text/archive/` | Tài liệu thử nghiệm cũ, không dùng làm đầu vào chính. |
| `perturbed_text/<source>/` | Biến thể nhiễu của từng nguồn, phân theo phương pháp perturb. |

## Các loại perturbation

| Thư mục | Mô tả |
| --- | --- |
| `lexical_adjective/` | Thay đổi tính từ theo một tỉ lệ xác định. |
| `lexical_synonym/` | Thay thế từ bằng từ đồng nghĩa. |
| `structural_noise/` | Chèn, xóa, lặp hoặc thay thế thành phần văn bản. |
| `rewrite/` | Viết lại văn bản bằng mô hình ngôn ngữ. |
| `translation/` | Dịch văn bản để tạo biến thể diễn đạt. |
| `wiki_4gram_replace/` | Thay thế cụm 4-gram bằng dữ liệu Wikipedia. |
| `generated/` | Biến thể được sinh lại bởi mô hình. |

## Sử dụng với các script

[`building_graph.py`](../building_graph.py) tìm đệ quy các tệp `.json` trong cả
`original_text/` và `perturbed_text/`. Vì vậy lệnh cũ vẫn dùng được khi tên tệp là
duy nhất:

```bash
python building_graph.py --file hc3_test
```

Khi có tên trùng nhau, script ưu tiên dữ liệu trong `original_text/`. Ví dụ,
`llama2_gen.json` tồn tại ở cả hai nhánh. Để chọn đúng bản perturb, truyền đường
dẫn tương đối đầy đủ:

```bash
python building_graph.py --file data/perturbed_text/llama2/generated/llama2_gen.json
```

`building_graph.py` chỉ tự tìm đệ quy tệp `.json`; với tệp `.jsonl`, hãy truyền
đường dẫn đầy đủ. Script đọc dữ liệu theo từng dòng, nên cả JSON Lines và các tệp
`.json` có một bản ghi trên mỗi dòng đều phù hợp.

- [`convert_datasets.py`](../convert_datasets.py) lưu đầu ra mặc định vào
  `data/original_text/converted/`.
- [`prepare_raid_data.py`](../prepare_raid_data.py) lưu các split RAID mặc định vào
  `data/original_text/benchmarks/raid/`.

## Quy ước thêm dữ liệu mới

1. Dữ liệu chưa perturb đặt trong `original_text/`, theo đúng nhóm chức năng.
2. Dữ liệu đã perturb đặt tại `perturbed_text/<source>/<perturbation-type>/`.
3. Giữ tên tệp mô tả được nguồn, split và mức nhiễu, ví dụ
   `hc3_test_delete_25.json`.
4. Không đặt dữ liệu dung lượng lớn vào commit thông thường nếu vượt giới hạn GitHub;
   dùng Git LFS hoặc liên kết tải dữ liệu khi cần.
