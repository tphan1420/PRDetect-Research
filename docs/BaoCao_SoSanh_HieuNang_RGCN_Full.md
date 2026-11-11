Mình sẽ tập trung vào V3, không đưa R-Drop vào hướng thử tiếp theo. Mình sẽ phân tích lỗi cross-dataset và đối chiếu nghiên cứu về khả năng tổng quát hóa giữa các miền dữ liệu, rồi chọn cải tiến có giả thuyết rõ ràng và cách kiểm chứng, đồng thời bảo toàn lợi thế Table 2.

Ran terminal command: Get-Content -LiteralPath 'model\QuangV3.py' -Encoding UTF8 -Raw
Get-Content -LiteralPath 'docs\PRDetect_RGCN_v3_tong_hop.md' -Encoding UTF8 | Select-Object -First 105
Get-Content -LiteralPath 'building_graph.py' -Encoding UTF8 | Select-Object -First 200
Get-Content -LiteralPath 'dep_vocab.json' -Encoding UTF8 -Raw

Searched the web for MixStyle domain generalization feature statistics ICLR 2021 paper, graph neural network size generalization mean max pooling graph size shortcut graph classification paper, AI generated text detection cross domain length syntactic graph robust domain generalization

Searched the web for site.openreview.net graph size generalization pooling size shift invariant graph neural networks, site.arxiv.org "Domain Generalization with MixStyle", "Robust AI-Generated Text Detection by Restricted Embeddings" ACL

Ran terminal command: Get-Content -LiteralPath 'building_graph.py' -Encoding UTF8 | Select-Object -First 155
Get-Content -LiteralPath 'data\original_text\benchmarks\hc3\hc3_train.json' -Encoding UTF8 -TotalCount 12
Get-Content -LiteralPath 'data\original_text\benchmarks\gpt35_mixed\gpt3.5_mixed_test_split.json' -Encoding UTF8 -TotalCount 10
rg -n 'swap|mask|meanlogit|pool3|V0|V1|V2|V3|V4|V5|val_total' docs\PRDetect_RGCN_v3_tong_hop.md | Select-Object -Last 35
Get-ChildItem -LiteralPath graph_data -ErrorAction SilentlyContinue | Select-Object Name,Length

Ran terminal command: $paths=@('"'data\\original_text\\benchmarks\\hc3\\hc3_train.json','data\\original_text\\benchmarks\\hc3\\hc3_val.json','data\\original_text\\benchmarks\\hc3\\hc3_test.json','data\\original_text\\benchmarks\\gpt35_mixed\\gpt3.5_mixed_test_split.json')
foreach("'$path in $paths){
$items=@(Get-Content -LiteralPath $path -Encoding UTF8 | Where-Object {$_} | ForEach-Object {$r=$_|ConvertFrom-Json;[PSCustomObject]@{Label=$(if($r.label -match '"'human'){1}else{0});Words=[regex]::Matches("'$r.text,'"'\\S+').Count}})
foreach("'$label in 0,1){$lens=@($items|Where-Object {$_.Label -eq $label}|Select-Object -ExpandProperty Words|Sort-Object);[PSCustomObject]@{File=[IO.Path]::GetFileNameWithoutExtension($path);Class=$(if($label -eq 0){'"'AI'}else{'human'});Count="'$lens.Count;Mean=[math]::Round(($lens|Measure-Object -Average).Average,1);Median=$lens[[int][math]::Floor($lens.Count*0.5)];P90=$lens[[int][math]::Floor($lens.Count*0.9)];P95=$lens[[int][math]::Floor($lens.Count*0.95)];Over512=@($lens|Where-Object {$_ -gt 512}).Count}}
}

Searched the web for https://arxiv.org/abs/2207.07888

Ran terminal command: Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive=[IO.Compression.ZipFile]::OpenRead((Join-Path (Get-Location) '"'result\\V3_q\\v3_quang_table3_table4_results.zip'))
"'$items=@(Get-Content -LiteralPath '"'data\\original_text\\benchmarks\\gpt35_mixed\\gpt3.5_mixed_test_split.json' -Encoding UTF8 | Where-Object {"'$_} | ForEach-Object {$r=$_|ConvertFrom-Json;[PSCustomObject]@{Y=$(if($r.label -match '"'human'){1}else{0});Words=[regex]::Matches("'$r.text,'"'\\S+').Count}})
"'$reports=@(foreach($seed in 2021..2025){$reader=[IO.StreamReader]::new($archive.GetEntry('"'result/table3_'+"'$seed+'"'_gpt3.5_mixed_test_split.json').Open());"'$reader.ReadToEnd()|ConvertFrom-Json;$reader.Dispose()})
$rows=@(for($i=0;$i -lt $items.Count;$i++){if([int]$reports[0].y_true[$i] -ne $items[$i].Y){throw '"'Labels not aligned'};"'$p=($reports|ForEach-Object {[double]$_.p_human[$i]}|Measure-Object -Average).Average;$correct=0;foreach($r in $reports){if([int]([double]$r.p_human[$i] -ge 0.5) -eq $items[$i].Y){$correct++}};$n=$items[$i].Words;[PSCustomObject]@{Y=$items[$i].Y;Words=$n;LogN=[math]::Log($n+1);PHuman=$p;Recall=$correct/5;Bin=$(if($n -le 256){'"'<=256'}elseif("'$n -le 384){'"'257-384'}elseif("'$n -le 512){'"'385-512'}elseif("'$n -le 768){'"'513-768'}else{'>768'})}})
"'$rows | Group-Object Y,Bin | ForEach-Object {[PSCustomObject]@{ClassBin=$_.Name;N=$_.Count;RecallMean=[math]::Round(100*($_.Group|Measure-Object Recall -Average).Average,2);MeanPHuman=[math]::Round(($_.Group|Measure-Object PHuman -Average).Average,4)}} | Format-Table -AutoSize
foreach($label in 0,1){$classRows=@($rows|Where-Object {$_.Y -eq $label});$mx=($classRows|Measure-Object LogN -Average).Average;$my=($classRows|Measure-Object PHuman -Average).Average;$sx=0;$sy=0;$sxy=0;foreach($r in $classRows){$dx=$r.LogN-$mx;$dy=$r.PHuman-$my;$sx+=$dx*$dx;$sy+=$dy*$dy;$sxy+=$dx*$dy};'"'Class {0}: Pearson(log(words+1), mean p_human) = {1:F4}' -f "'$label,($sxy/[math]::Sqrt($sx*$sy))}
$archive.Dispose()



Mình thấy chênh lệch độ dài rất lớn: văn bản AI ở HC3 train trung bình khoảng 180 từ, còn GPT-3.5 test khoảng 504 từ. Nhưng trong GPT-3.5, nhóm AI dài hơn lại không bị nhận sai nhiều hơn, nên chưa thể kết luận độ dài là nguyên nhân chính. Mình đang tập trung thêm vào cách V3 học đặc trưng theo miền và cách swap chọn nút thay thế.

Searched the web for site.proceedings.neurips.cc "Mixup" "Graph" "Domain", site.aclanthology.org "AI-Generated" "domain" "style" "contrastive" detection generalization, "MixStyle" "graph" "class" domain generalization

Ran terminal command: Get-Content -LiteralPath 'dep_vocab.json' -Encoding UTF8 -Raw
Get-Content -LiteralPath 'evaluate_v3_quang.py' -Encoding UTF8 | Select-Object -Skip 200 -First 68
rg -n 'train_file|val_file|swap|node_rel|low|GraphSet|num_rel' docs\PRDetect_RGCN_v3_tong_hop.md | Select-Object -First 35
Get-ChildItem -LiteralPath model\V3_quang -File | Select-Object Name,Length

Configuring a Python Environment
Completed with input: {
  "resourcePath": "D:\\Tailieu\\HK7\\HT Thông minh\\PRDetect-Research"
}

Fetching Python executable information for .venv (3.13.0)
Completed with input: {
  "resourcePath": "D:\\Tailieu\\HK7\\HT Thông minh\\PRDetect-Research"
}

Ran terminal command: & '.\.venv\Scripts\python.exe' -c 'import torch; from model.QuangV3 import Net, fwd_mask; import numpy as np; net=Net(51,in_dim=4,d=8,nb=2,L=1,aug="swap",aug_max=1.0); x=torch.eye(4); src=torch.tensor([0,1,2,3]); dst=torch.tensor([1,1,3,3]); rel=torch.tensor([2,1,2,1]); bidx=torch.tensor([0,0,1,1]); hits=0; changed=0
for seed in range(20):
 g=torch.Generator().manual_seed(seed); out=net._augment(x,src,dst,rel,bidx,2,"swap",g); donor=out.argmax(1); changed+=int((donor"'!=torch.arange(4)).sum()); hits+=int((bidx[donor]!=bidx).sum())
print("Changed nodes:",changed,"; donor from other graph/class:",hits); print("fwd_mask ids 2,24,25,34,50:",fwd_mask(np.array([2,24,25,34,50]),51,"low").tolist())'"'
Tool execution failed: Exit code 1

Ran terminal command: & '.\.venv\Scripts\python.exe' -c "import torch; from model.QuangV3 import Net, fwd_mask; import numpy as np; net=Net(51,in_dim=4,d=8,nb=2,L=1,aug='swap',aug_max=1.0); x=torch.eye(4); src=torch.tensor([0,1,2,3]); dst=torch.tensor([1,1,3,3]); rel=torch.tensor([2,1,2,1]); bidx=torch.tensor([0,0,1,1]); hits=0; changed=0
for seed in range(20):
 g=torch.Generator().manual_seed(seed); out=net._augment(x,src,dst,rel,bidx,2,'swap',g); donor=out.argmax(1); changed+=int((donor"'!=torch.arange(4)).sum()); hits+=int((bidx[donor]!=bidx).sum())
print('"'Changed nodes:',changed,'; donor from other graph/class:',hits); print('fwd_mask ids 2,24,25,34,50:',fwd_mask(np.array([2,24,25,34,50]),51,'low').tolist())"
Tool execution failed: Exit code 1

Searched the web for https://arxiv.org/html/2410.08113v1

Searched the web for validation



Hướng phù hợp nhất để thử là tăng khả năng tổng quát hóa ngay trong V3 bằng biến đổi thống kê đặc trưng, giữ nguyên graph, encoder và nhánh cấu trúc. Mình cũng thấy swap hiện chọn nút trong toàn batch mà không ràng buộc nhãn AI/human; đây là một biến đáng tách riêng trong ablation, chưa đủ bằng chứng để gọi là nguyên nhân gây giảm điểm.

Ran terminal command: rg -n 'def fwd_mask|def _augment|key = node_rel|node_rel\[|self.ln0|sproj|mx =|if bidirectional|bidirectional=False|rel_id =|if self.mode' model\QuangV3.py building_graph.py
Get-Content -LiteralPath 'C:\Users\ADMIN\AppData\Roaming\Code\agentPlugins\vscode-synced-customization-agent-host-codex-217238766\203683680\skills\python-fact-grounded-coding\SKILL.md' -TotalCount 10



**Mình đề xuất chuyển V3 sang hướng “domain generalization”: giữ nguyên nền đang thắng Table 2, nhưng huấn luyện để mô hình bớt phụ thuộc vào đặc trưng riêng của HC3.** Không đưa R-Drop vào thử nghiệm tiếp theo.

Phương án ưu tiên của mình là **Class-Conditional Graph MixStyle** — biến đổi thống kê đặc trưng giữa các đồ thị cùng nhãn. Đây là một giả thuyết nghiên cứu đáng thử, không phải lời đảm bảo rằng thêm vào sẽ thắng Table 3.

## 1. Kết quả hiện tại cho thấy cần cải tiến điều gì?

V3 đã xử lý tốt **nhiễu từ vựng trong cùng miền HC3**, nhưng điều đó khác với khả năng nhận diện văn bản từ miền mới.

Ở seed chung 2025:

| Table 3 — trung bình 5 điều kiện | GCN bạn train | V3 |
|---|---:|---:|
| Accuracy | **68.62%** | 66.76% |
| AUC | 0.89843 | **0.90729** |

V3 vẫn phân biệt/xếp hạng hai lớp khá tốt, nhưng tại ngưỡng 0.5, nhiều văn bản AI từ GPT-3.5 bị nhận thành human. **Mục tiêu nên là cải thiện khả năng tổng quát hóa của đặc trưng, không tiếp tục chỉ tối ưu HC3 clean hoặc dò threshold.**

Mình kiểm tra thêm độ dài trực tiếp từ dữ liệu:

| Tập | Số từ trung bình — AI | Số từ trung bình — human |
|---|---:|---:|
| HC3 train | 179.6 | 149.6 |
| GPT-3.5 test | 504.2 | 803.1 |

Đây là số từ tách theo khoảng trắng, chưa phải số node của graph.

Có dịch chuyển độ dài rõ ràng, **nhưng không thể kết luận “văn bản dài làm V3 sai”**: trên GPT-3.5, recall AI trung bình của V3 ở nhóm 385–512 từ là khoảng **28%**, còn nhóm 513–768 từ lại khoảng **44%**.

Vì vậy, cần xét đồng thời **chủ đề, phong cách, độ dài và phân phối đặc trưng**, không chỉ sửa pooling theo độ dài.

## 2. Phương án ưu tiên: Class-Conditional Graph MixStyle

### Cơ sở nghiên cứu

[MixStyle — ICLR 2021](https://arxiv.org/abs/2104.02008) tăng khả năng tổng quát hóa bằng cách trộn thống kê đặc trưng giữa các mẫu trong quá trình train, tạo thêm biến thiên miền mà không cần dữ liệu miền đích.

Tuy nhiên, nghiên cứu gốc chủ yếu ở ảnh. **Áp dụng sang đồ thị cú pháp là một thiết kế cần kiểm chứng**, không được mặc định rằng “style” của ảnh tương đương “style” của văn bản.

Đặc biệt trong bài toán của bạn, phong cách viết có thể chính là tín hiệu phân biệt AI/human. Vì vậy mình không đề xuất trộn tùy ý giữa hai lớp.

### Cách điều chỉnh riêng cho V3

Giả sử mỗi đồ thị có ma trận đặc trưng node \(H_g\):

1. Tính trung bình và độ lệch chuẩn theo các node của từng graph.
2. Chọn một graph khác **cùng nhãn AI/human**.
3. Trộn nhẹ thống kê của hai graph.
4. Áp thống kê mới lên các node, giữ nguyên cạnh và nhãn quan hệ.
5. Chỉ sử dụng trong train; tắt khi inference.

Dạng biến đổi:

\[
\widetilde H_g=
\sigma_{\text{mix}}\odot
\frac{H_g-\mu_g}{\sigma_g+\epsilon}
+\mu_{\text{mix}}
\]

Trong đó thống kê trộn lấy từ hai graph cùng lớp.

**Vị trí phù hợp để thử:** sau phép chiếu đặc trưng ban đầu, trước khi cộng embedding cấu trúc tại `nhánh đặc trưng của V3`.

Như vậy:

- Nhánh depth/children/distance vẫn giữ nguyên.
- Không đổi tokenizer hay embedding cache.
- Không sửa cây cú pháp.
- Giữ swap augmentation và cấu hình V3 hiện tại làm nền.
- Không tăng số tham số đáng kể; khi test không cần thêm lượt dự đoán.

### Giả thuyết cần kiểm chứng

> V3 đang sử dụng một phần thống kê đặc trưng riêng của HC3. Biến đổi những thống kê này trong phạm vi cùng lớp có thể buộc mô hình dựa nhiều hơn vào dấu hiệu AI/human ổn định qua miền.

Giới hạn quan trọng: **cùng nhãn chỉ giảm nguy cơ phá tín hiệu phân loại, không bảo đảm biến đổi luôn hợp lệ**. Vì vậy phải bắt đầu nhẹ và theo dõi Table 2.

## 3. Một điểm trong V3 đáng làm ablation riêng

Trong `hàm swap hiện tại`, nút thay thế được chọn trong toàn batch theo nhóm quan hệ, **không có ràng buộc cùng nhãn AI/human**.

Điều này có nghĩa một graph AI có thể nhận embedding từ graph human.

Không nên vội gọi đây là lỗi: trộn như vậy có thể đang giúp regularization và đóng góp vào thành tích Table 2. Nhưng nó tạo ra một câu hỏi nghiên cứu rõ ràng:

> Giữ sự đa dạng chủ đề nhưng bảo toàn nhãn tác giả trong augmentation có giúp cross-dataset tốt hơn không?

Thử một biến thể nhỏ:

**Class-preserving syntax swap**

- Donor cùng nhãn AI/human.
- Cùng nhóm quan hệ cú pháp.
- Ưu tiên donor từ tài liệu khác.
- Giữ nguyên tỷ lệ swap và các cấu hình còn lại.

Đây là thí nghiệm ít tốn kém và dễ giải thích hơn việc thay nhiều thành phần cùng lúc.

**Lưu ý thêm:** cách chia quan hệ thành hai nửa trong `fwd_mask` giả định nhóm cạnh thuận/ngược, trong khi `graph builder mặc định` tạo cạnh một chiều. Cần xác minh quy ước graph lúc Quang train trước khi thiết kế augmentation mới dựa trên relation. Không nên tự sửa preprocessing rồi dùng nguyên checkpoint cũ.

## 4. Hai phương án nghiên cứu dự phòng

| Phương án | Mục đích | Đánh giá của mình |
|---|---|---|
| **Size-balanced graph composition** | Cho model gặp graph lớn hơn mà vẫn chỉ dùng HC3 train | Đáng thử sau hoặc độc lập với MixStyle |
| **Restricted embeddings / loại bỏ đặc trưng phụ thuộc miền** | Giảm các thành phần biểu diễn mang thông tin chủ đề hơn là AI/human | Có nghiên cứu sát bài toán, nhưng triển khai và chọn thành phần khó hơn |

### A. Size-balanced graph composition

Ghép 2–4 graph **khác nhau nhưng cùng nhãn** thành một graph gồm nhiều thành phần không nối nhau:

- Không tạo cạnh cú pháp giả giữa các tài liệu.
- Giữ đặc trưng node và cạnh bên trong từng thành phần.
- Áp dụng cùng phân phối số lượng thành phần cho AI và human.
- Vẫn giữ nhiều mẫu đơn nguyên bản trong train.

Mục tiêu là giảm liên hệ kiểu “graph lớn thường là human” trong dữ liệu nguồn. Chỉ lặp lại cùng một graph không đủ, vì nhiều phép pooling gần như không thay đổi khi nhân bản toàn bộ node.

[SizeShiftReg — NeurIPS 2022](https://arxiv.org/abs/2207.07888) cung cấp cơ sở nghiên cứu về sự suy giảm khi kích thước graph thay đổi. Tuy nhiên, phương pháp gốc dùng coarsening; **graph composition ở đây là biến thể mình đề xuất**, không phải tái hiện nguyên thuật toán.

Nhược điểm: ghép graph không tạo ra đầy đủ ngữ cảnh của một bài báo dài thực sự. Vì dữ liệu hiện tại cũng chưa chứng minh độ dài là nguyên nhân chính, mình xếp hướng này sau MixStyle.

### B. Restricted embeddings

[Robust AI-Generated Text Detection by Restricted Embeddings — EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.992/) nghiên cứu loại bỏ các thành phần embedding gây phụ thuộc miền.

Hướng này sát mục tiêu Table 3, nhưng không nên hiểu thành “xóa vài thành phần PCA là được”. Nghiên cứu cho thấy lựa chọn thành phần rất quan trọng; bỏ sai có thể làm khả năng transfer giảm.

Với V3, đây nên là bước sau: xác định thành phần dự đoán tốt chủ đề/miền nhưng không ổn định cho AI/human, rồi thử hạn chế chúng bằng dữ liệu nguồn. **Không chọn thành phần để xóa dựa trên nhãn GPT-3.5 test.**

## 5. Thiết kế thí nghiệm mình đề xuất

Chưa ghép tất cả cải tiến ngay. Chạy các biến thể độc lập:

| Mã | Thay đổi so với V3 |
|---|---|
| **V3-Base** | Giữ nguyên để đối chứng |
| **V3-CSwap** | Chỉ giới hạn swap cùng nhãn |
| **V3-CMixStyle** | Giữ V3 gốc, thêm class-conditional MixStyle |
| **V3-Size** | Giữ V3 gốc, thêm graph composition |

**Ưu tiên V3-CMixStyle; V3-CSwap là ablation nhỏ đi kèm.** Chỉ thử kết hợp khi từng thành phần có tín hiệu tốt.

Cách chọn và đánh giá:

- Chọn mức augmentation/checkpoint bằng validation nguồn HC3, có thêm đánh giá theo nhóm chủ đề và độ dài.
- Giữ ngưỡng 0.5 để so trực tiếp với GCN của bạn.
- Table 3 là mục tiêu cải thiện; Table 2 và Table 4 là các năng lực cần bảo toàn.
- Dùng cùng tập seed; không lấy V3 năm seed so trực tiếp với GCN hai seed.
- Không dùng dữ liệu GPT-3.5 test để trộn đặc trưng, tính thống kê train hoặc chọn hyperparameter.

Lưu ý nghiên cứu: benchmark này đã được xem nhiều lần, nên kết quả cải tiến tiếp theo vẫn cần được xác nhận trên seed và, nếu có thể, một miền đánh giá bổ sung chưa dùng để quyết định thiết kế.

**Chốt lựa chọn:** mình nghiêng về **V3 + Class-Conditional Graph MixStyle** vì nó nhắm trực tiếp vào domain shift, giữ nguyên nền cú pháp đang tốt ở Table 2, tái sử dụng graph cache và không tăng chi phí inference. Đây cũng là hướng có câu hỏi nghiên cứu và ablation rõ ràng hơn việc tiếp tục tăng layer, hidden size hoặc đổi encoder.