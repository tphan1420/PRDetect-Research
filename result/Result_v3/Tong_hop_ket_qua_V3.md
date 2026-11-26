# Tổng hợp kết quả V3

## Phạm vi và quy ước

Báo cáo này tổng hợp 18 archive trong thư mục `Result_v3` và số liệu V3 baseline từ [Table2.md](../V3_q/Table2.md).

- Các archive đều test **HC3 → GPT-3.5 Mixed, noise 0%**, 1.000 mẫu, threshold `0.5`.
- Nhãn `1 = Human`, `0 = AI`; F1 là F1 của lớp Human.
- CorrectedStruct và Ablation 1 là các run khảo sát riêng, không được gộp vào V3 baseline.

## Table 2 — V3 baseline

Nguồn: [Table2.md](../V3_q/Table2.md). Đây là đánh giá cùng miền với 5 seed, khác với Table 3 cross-dataset phía dưới.

| Tập | 0% | 5% | 10% | 20% | 30% |
| :-- | --: | --: | --: | --: | --: |
| HC3 Accuracy | 0.9902 ± 0.0016 | 0.9904 ± 0.0015 | 0.9900 ± 0.0017 | 0.9898 ± 0.0013 | 0.9892 ± 0.0012 |
| GPT-3.5 Mixed Accuracy | 0.9835 ± 0.0035 | 0.9835 ± 0.0007 | 0.9835 ± 0.0021 | 0.9835 ± 0.0007 | 0.9855 ± 0.0007 |

## Table 3 — HC3 → GPT-3.5 Mixed, 0% noise

### Kết quả từng run

| Nhóm | Seed | Batch | Accuracy | AUC | F1 Human | Recall AI | Recall Human |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| V3 baseline | 2020 | 1 | 0.7290 | 0.8920 | 0.7830 | 0.4800 | 0.9780 |
| V3 baseline | 2022 | 1 | 0.6630 | 0.9027 | 0.7457 | 0.3380 | 0.9880 |
| V3 baseline | 2024 | 1 | 0.7330 | 0.9095 | 0.7876 | 0.4760 | 0.9900 |
| V3 baseline | 2025 | 1 | 0.7240 | 0.8890 | 0.7767 | 0.4880 | 0.9600 |
| V3 baseline | 2026 | 1 | 0.6790 | 0.8904 | 0.7548 | 0.3700 | 0.9880 |
| V3 baseline | 2024 | 64 | 0.6590 | 0.8927 | 0.7415 | 0.3400 | 0.9780 |
| V3 baseline | 2025 | 64 | 0.6640 | 0.9133 | 0.7458 | 0.3420 | 0.9860 |
| V3 baseline | 2026 | 64 | 0.7040 | 0.9010 | 0.7677 | 0.4300 | 0.9780 |
| V3 baseline | 2025 | 32 | 0.6620 | 0.9121 | 0.7455 | 0.3340 | 0.9900 |
| V3 baseline | 2026 | 32 | 0.6580 | 0.9084 | 0.7405 | 0.3400 | 0.9760 |
| V3 baseline | 2025 | 8 | 0.6760 | 0.8883 | 0.7515 | 0.3720 | 0.9800 |
| V3 baseline | 2026 | 8 | 0.7020 | 0.8644 | 0.7654 | 0.4320 | 0.9720 |
| CorrectedStruct (khảo sát) | 2022 | 1 | 0.7210 | 0.8859 | 0.7755 | 0.4780 | 0.9640 |
| CorrectedStruct (khảo sát) | 2026 | 1 | 0.6620 | 0.8603 | 0.7443 | 0.3400 | 0.9840 |
| Ablation 1: no_struct_no_swap | 2025 | 1 | 0.7370 | 0.8752 | 0.7867 | 0.5040 | 0.9700 |
| Ablation 1: no_struct_no_swap | 2026 | 1 | 0.6920 | 0.8772 | 0.7612 | 0.4020 | 0.9820 |
| Ablation 1: no_struct_no_swap | 2025 | 64 | 0.6860 | 0.8921 | 0.7573 | 0.3920 | 0.9800 |
| Ablation 1: no_struct_no_swap | 2026 | 64 | 0.6500 | 0.8944 | 0.7372 | 0.3180 | 0.9820 |

### Trung bình theo cấu hình

Độ lệch chuẩn là sample standard deviation (`n-1`).

| Nhóm | Batch | Seeds | Accuracy | AUC | F1 Human |
| :-- | --: | :-- | --: | --: | --: |
| V3 baseline | 1 | 2020, 2022, 2024, 2025, 2026 | 0.7056 ± 0.0318 | 0.8969 ± 0.0087 | 0.7695 ± 0.0180 |
| V3 baseline | 64 | 2024, 2025, 2026 | 0.6757 ± 0.0247 | 0.9023 ± 0.0104 | 0.7517 ± 0.0139 |
| V3 baseline | 32 | 2025, 2026 | 0.6600 ± 0.0028 | 0.9102 ± 0.0026 | 0.7430 ± 0.0035 |
| V3 baseline | 8 | 2025, 2026 | 0.6890 ± 0.0184 | 0.8763 ± 0.0169 | 0.7584 ± 0.0098 |
| CorrectedStruct (khảo sát) | 1 | 2022, 2026 | 0.6915 ± 0.0417 | 0.8731 ± 0.0181 | 0.7599 ± 0.0221 |
| Ablation 1: no_struct_no_swap | 1 | 2025, 2026 | 0.7145 ± 0.0318 | 0.8762 ± 0.0015 | 0.7740 ± 0.0180 |
| Ablation 1: no_struct_no_swap | 64 | 2025, 2026 | 0.6680 ± 0.0255 | 0.8932 ± 0.0016 | 0.7473 ± 0.0142 |

## Lưu ý diễn giải

- Các nhóm có seed khác nhau và batch khác nhau không phải một ablation đối chứng hoàn toàn; không dùng chúng để kết luận nguyên nhân chỉ từ chênh lệch accuracy.
- Batch 1 baseline có accuracy trung bình cao nhất trong các archive baseline hiện có. Batch 64 có AUC trung bình cao hơn trong ba seed đã lưu.
- CorrectedStruct được ghi riêng theo yêu cầu; không thay thế V3 baseline.
- Tên `Ablation 1: no_struct_no_swap` lấy từ tên archive. `config.json` bên trong vẫn ghi `model: v3_baseline`.

## Archive nguồn

- [2020 baseline batch 1](2020/v3_baseline_hc3_to_gpt35_0pct_batch1_results_2020.zip)
- [2022 baseline batch 1](2022/v3_baseline_hc3_to_gpt35_0pct_batch1_results_2022.zip)
- [2022 CorrectedStruct batch 1](2022/v3_corrected_structmap_hc3_to_gpt35_0pct_batch1_results_2022.zip)
- [2024 baseline batch 1](2024/v3_baseline_hc3_to_gpt35_0pct_batch1_results_2024.zip)
- [2024 baseline batch 64](2024/v3_baseline_hc3_to_gpt35_0pct_batch64_results_2024.zip)
- Các archive baseline seed 2025/2026 và ablation nằm trong các thư mục tương ứng.