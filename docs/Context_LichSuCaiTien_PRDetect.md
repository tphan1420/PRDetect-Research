# Context nghiên cứu: lịch sử cải tiến PRDetect

Cập nhật: 12/11/2026. Mục đích: đưa cho AI khác để đề xuất hướng mới, gồm cả các kết quả âm. Đây là nhật ký thực nghiệm, **không phải danh sách đóng góp đã được công nhận**. Báo cáo đóng góp chính thức vẫn là [V3 Baseline](./BaoCao_V3_Baseline_CaiTien.md).

## 1. Mục tiêu và quy tắc so sánh

- Mục tiêu: cải thiện cả Table 2 (in-domain + synonym noise), Table 3 (cross-dataset + synonym noise), Table 4 (structural noise: insert/repeat/replace/delete 25%). Ưu tiên hiện tại: Table 3 **HC3 → GPT-3.5**, không đánh đổi bằng việc sụp Table 4.
- Baseline cuối cùng là **GCN nhóm tự train**, không phải số tác giả paper. GCN tự train khác checkpoint tác giả và không được đồng nhất hai nguồn.
- Dùng accuracy tại threshold 0.5; xem thêm AUC và recall AI/Human. Không chọn ngưỡng/hyperparameter/epoch bằng nhãn test.
- Table 3 average dưới đây = trung bình Original + synonym 5/10/20/30%; Table 4 average = bốn tập nhiễu, không gồm Original. Δ là điểm phần trăm.
- Các thí nghiệm V3 mới chủ yếu dùng seed 2025/2026; V3 lịch sử của Quang dùng 2021–2025. Baseline giữa các ZIP không hoàn toàn giống nhau: chỉ xem Δ trong cùng thí nghiệm là so sánh trực tiếp.

## 2. Toàn bộ các hướng đã triển khai/thử

| Phương pháp | Cơ chế ngắn gọn | Kết quả và quyết định |
|---|---|---|
| **GCN → RGCN V1** | Message passing phân biệt dependency relation, thay vì xem mọi cạnh giống nhau. | Checkpoint cũ seed 2026: HC3 clean **97.40 → 98.60%**; GPT clean cross **64.90 → 72.60%**. Table 3 average **63.40 → 70.88%** cùng seed. Cải thiện tốt, nhưng chưa tái lập được checkpoint cũ chỉ bằng train lại. |
| **V2 A1: aligned word/subword embedding** | Tokenize đúng RoBERTa BPE, mean-pool subword về node spaCy, xử lý văn bản dài. | Ablation cùng seed 2026: GPT clean **62.70 → 86.90%**, nhưng HC3 insert **91.80 → 51.70%**, delete **69.10 → 51.60%**, repeat **97.00 → 69.60%**. Không chấp nhận đánh đổi structural robustness; dừng hướng này. |
| **V2 A2: hierarchical bidirectional graph** | Dependency thuận/nghịch + cạnh token–sentence–document và liên kết tuần tự; schema 128 relations. | Đã chạy trong V2/A1A2; không khắc phục được sự tụt structural noise. Chưa có bằng chứng sạch rằng riêng A2 tăng điểm ổn định; không quy toàn bộ lỗi V2 cho A2. |
| **V2 A3: hybrid graph–sequence gated fusion** | Kết hợp graph representation với sequence/document embedding bằng gate, có tín hiệu quality. | GPT clean V2 đạt **89.30%**, nhưng mean sequence gate gần **1.0**, graph gate khoảng **10⁻⁶**: nhánh sequence chi phối. Không chứng minh graph tốt hơn; đã bỏ A3. |
| **Quality gating / graph-only ablation** | Kiểm tra feature chất lượng, tách graph-only khỏi hybrid để tìm nguyên nhân. | Graph-only V2 vẫn đạt GPT clean **87.20%** nhưng structural noise yếu; A1-only cũng đã tái hiện lỗi. Không thể quy lỗi chỉ cho A3/fusion; chưa chứng minh quality là cải tiến độc lập. |
| **A1 threshold calibration** | Giữ checkpoint, chọn threshold bằng validation sạch + tự tạo insert/delete/repeat, không train lại model. | Insert **51.70 → 84.10%**, delete **51.60 → 84.40%**, repeat **69.60 → 92.60%**; đổi lại HC3 clean **99.00 → 71.30%**, GPT clean **86.90 → 50.80%**. Một ngưỡng chung không cứu được mọi miền; bỏ. |
| **EMA bổ sung cho RGCN V1** | Trung bình mũ trọng số và so raw/EMA checkpoint trong cùng run. | Seed 2026: GPT clean **62.40 → 60.80%**, synonym 30% **59.90 → 58.10%**, insert **89.30 → 87.60%**. Không cải thiện, bỏ thí nghiệm EMA riêng. **Không nhầm với EMA vốn có trong V3.** |
| **Frozen encoder: RoBERTa aligned / DeBERTa-v3** | Pipeline encoder đóng băng, chunk subword, pool về word rồi train RGCN; có RoBERTa aligned làm control. | Control RoBERTa aligned: clean **99.00%**, insert **51.60%**, delete **51.40%**, repeat **69.00%**. Không vượt được rào cản structural noise. Có code DeBERTa nhưng chưa có kết quả đầy đủ chứng minh DeBERTa tốt hơn; không kết luận mọi subword/DeBERTa đều thất bại. |
| **R-Drop trên RGCN** | Hai forward dropout; BCE + symmetric KL để giảm bất nhất dự đoán. | Seed 2025/2026: Table 3 **61.03 → 65.72%** so baseline train lại, nhưng vẫn dưới GCN **66.01%**; Table 4 **81.91 → 85.10%**, trên GCN **80.21%**. Có lợi ích nhưng không giải quyết đủ Table 3; hiện không tiếp tục. |
| **V3 Quang / V1_swap** | RGCN basis decomposition + projection/LayerNorm/residual; node depth/children/distance; syntax-aware swap; mean/max/attention pooling (`pool3`); AdamW, warmup/cosine, EMA, chọn bằng clean+swap validation loss. | Hướng chính được giữ. V3 lịch sử 5 seed: HC3 Table 2 **99.02 → 98.92%** từ clean đến synonym 30%; Table 4 noise average **84.95%**, GCN 3 seed **81.08%** (khác seed, so sánh mô tả). Table 3 vẫn yếu: clean **66.80%**, synonym 30% **63.72%**. |
| **V3 train trên GPT-3.5** | Giữ V3 baseline, đổi miền train/validation sang GPT; test in-domain và cross sang HC3. | Seed 2025/2026: GPT in-domain **98.35–98.55%**; cross GPT→HC3 **84.85 → 83.45%** từ clean đến synonym 30%. Chưa có GCN tự train cùng chiều/cùng seed để kết luận thắng GCN; mốc **87.7–87.0%** là lịch sử paper, không phải baseline nhóm đã tái lập. |
| **V3-CMixStyle** | Trộn mean/std channel giữa hai graph khác nhau cùng nhãn, sau projection/LN, ở augmented branch; không thêm tham số học. | Seed 2025/2026: Table 3 **66.59 → 65.91% (−0.68)**; Table 4 **85.975 → 85.275% (−0.70)**. Không đem lại cải thiện; bỏ. |
| **V3 Class-Preserving Syntax Swap** | Donor phải cùng syntax role, cùng nhãn AI/Human và khác graph; không donor thì giữ nguyên. Giữ legacy structural mapping. | Đã có kết quả, không còn pending: Table 3 **65.89 → 64.47% (−1.42)**. Accuracy trung bình giảm ở cả 5 điều kiện; AUC cũng giảm. Không chọn làm cải tiến. |
| **V3 Structure-Guided Restricted Embeddings** | Gate channel-wise riêng cho semantic và structural representation; soft semantic budget; semantic init 0.9, structural init 1.0. | Đã có kết quả: Table 3 **66.27 → 64.69% (−1.58)**. Giảm cả 5 điều kiện, AUC cũng giảm. Không chọn. Đây là soft-gating adaptation, **không phải** tái hiện coordinate/head removal hoặc LEACE của paper Restricted Embeddings. |
| **V3 Corrected Structural Mapping (`structmap`)** | Bỏ giả định relation chia hai nửa forward/reverse; dùng mọi dependency edge dependent→head cho parent/features và swap role; sentinel root riêng. | Seed 2025/2026: Table 3 average **64.24%**, Table 4 noise average **84.675%**. ZIP chỉ chứa `structmap`, không có paired baseline: không gán Δ chính xác. Không cho thấy bước tiến so các baseline V3 đã lưu; audit code có nhưng chưa kiểm chứng đầy đủ cache của run. |
| **V3 MeanLogit readout** | Thay `pool3` + MLP bằng logit từng node rồi trung bình theo graph; giữ backbone và train protocol. | Đã implement và có cell ablation. **Chưa tìm thấy kết quả benchmark trong các artifact hiện có**, không xếp vào phương pháp thất bại hoặc thành công. |

## 3. Các bài học cần chuyển cho AI đề xuất hướng mới

1. **Điểm nghẽn là cross-domain, không phải HC3 clean.** V3 controlled đạt AUC Table 3 khoảng **91.50%** nhưng recall AI **34.72%**, human **98.46%**. Nhiều AI bị nhận thành human; AUC cao không chứng minh chỉ cần sửa threshold là đủ.
2. **Đừng đồng nhất RGCN cũ với baseline train lại.** Cùng seed 2026: Table 3 average cũ **70.88%**, baseline R-Drop train lại **61.22%**; metadata/cache/môi trường của run cũ chưa đủ để quy nguyên nhân cụ thể.
3. **Mapping legacy V3 có bất nhất.** Với R=51, `low` chỉ dùng ID 0–24 để dựng parent và swap role, trong khi builder mặc định dùng một chiều và ID là dependency labels. RGCN message passing vẫn dùng mọi cạnh. Sửa mapping cần retrain variant riêng, không đổi inference của checkpoint cũ.
4. **Không diễn giải gate là causal proof.** Restricted gate học trên source không tự biết channel nào chứa topic/style hay hữu ích OOD; layer sau có thể bù scale. Kết quả âm chỉ áp dụng cấu hình đã thử, không bác bỏ mọi phương pháp concept erasure.
5. **Không có bằng chứng đảm bảo thắng cả ba bảng.** Gợi ý mới cần giả thuyết cụ thể và ablation chỉ đổi một yếu tố; bảo toàn embedding/graph legacy nếu chưa kiểm soát được trade-off structural noise.
6. **Giới hạn thực thi:** Kaggle khoảng 20 GB đĩa, tối đa 4 graph `.pkl` cùng lúc; build/test tuần tự rồi xóa cache đã dùng. Ưu tiên thử seed 2025/2026, cùng micro-batch/donor pool và validation protocol giữa hai variant.

## 4. Nguồn để kiểm tra khi cần

- RGCN V1 và chênh lệch checkpoint: [báo cáo nâng cấp](./BaoCaoNangCap.md), [audit Table 3](./Table3_V1_V3_Diagnosis.md).
- V2 A1–A3: tài liệu lịch sử `docs/Version2_A1_A3.md` tại commit `aef2c43`; [A1A2 logs](../result/a1a2.txt), [hybrid logs](../result/Hybrid_result.txt), [V2 clean report](../result/result_V2/v2_gpt35_0_clean.json).
- Calibration / encoder / EMA: [threshold reports](../result/a1_threshold/), [encoder control](../result/ecoder_compress/test_result.txt), [EMA ZIP](../result/rgcn_ema_results.zip).
- R-Drop so GCN: [tổng hợp Table 3/4](./TongHop_Table3_Table4_GCN_RGCN_RDrop_Seeds2025_2026.md).
- V3: [báo cáo baseline](./BaoCao_V3_Baseline_CaiTien.md), [GPT-trained ZIP](../result/v3_baseline_table2_table3_results.zip).
- Ablation trực tiếp: [CMixStyle](../result/Cmix/v3_cmixstyle_ablation_results.zip), [ClassSwap](../result/v3_classswap_ablation_results.zip), [Restricted](../result/v3_restricted_table3_ablation_results.zip), [Structural Mapping](../result/v3_structmap_table3_table4_ablation_results.zip).
- Hướng đã code nhưng chưa có benchmark: [MeanLogit](./MeanLogit_Readout_Experiment.md).

**Yêu cầu dành cho AI tiếp theo:** đề xuất một hướng khác có cơ sở nghiên cứu, ít thay đổi, có thể kiểm chứng bằng paired ablation; giải thích vì sao giải quyết được thất bại đã quan sát. Không lặp lại CMixStyle, cùng-label swap hoặc soft semantic gates như thể chưa từng thử; không tuyên bố cải thiện chắc chắn hoặc chọn cấu hình bằng target test.
