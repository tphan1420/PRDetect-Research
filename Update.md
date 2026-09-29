# Báo cáo Phân tích và Đề xuất Cải tiến Kiến trúc Mô hình PRDetect trong Tác vụ Nhận diện Văn bản AI trên Tập dữ liệu RAID

## 1. Khảo sát Sự Suy giảm Hiệu suất và Phân tích Đặc tính của Tập dữ liệu RAID

Mô hình PRDetect đã khẳng định được năng lực vượt trội trong việc phát hiện văn bản do Mô hình Ngôn ngữ Lớn (LLM) tạo ra trên các tập dữ liệu truyền thống như HC3 và GPT-3.5-Mixed. Bằng cách sử dụng một Mạng Tích chập Đồ thị (Graph Convolutional Network - GCN) hai lớp kết hợp với nhúng từ (word embeddings) từ RoBERTa để khai thác cấu trúc của cây cú pháp phụ thuộc (dependency syntax tree), mô hình này đạt độ chính xác lên tới 98.78% đối với văn bản gốc. Khả năng kháng nhiễu của PRDetect cũng được đánh giá cao khi hiệu suất chỉ suy giảm không đáng kể (duy trì trên 98.6%) ngay cả khi văn bản bị nhiễu loạn từ vựng thông qua việc thay thế từ đồng nghĩa ở mức 30%.

Tuy nhiên, khi triển khai mô hình này trên tập dữ liệu RAID (Robust AI Detection), hiệu suất đã sụt giảm nghiêm trọng xuống mức xấp xỉ 75%. Sự suy giảm này phản ánh một thực tế rằng các phương pháp đánh giá truyền thống chưa phản ánh đúng độ phức tạp của môi trường thực tế. Tập dữ liệu RAID hiện là chuẩn đo lường (benchmark) quy mô lớn và thử thách nhất dành cho tác vụ phát hiện văn bản AI, bao gồm hơn 10 triệu tài liệu [2]. Sự khác biệt cơ bản giữa môi trường huấn luyện ban đầu của PRDetect và môi trường kiểm thử của RAID được thể hiện chi tiết qua sự phân kỳ về quy mô và độ phức tạp.

| Đặc tính Dữ liệu | Tập dữ liệu HC3 / GPT-3.5-Mixed | Tập dữ liệu chuẩn RAID (Robust AI Detection) |
| :--- | :--- | :--- |
| **Quy mô mẫu** | Hàng chục nghìn mẫu (26.9K trên HC3) | Hơn 10 triệu tài liệu [2] |
| **Đa dạng Generators** | Chủ yếu ChatGPT và họ GPT-3/GPT-3.5 | 11 mô hình: GPT-2, GPT-3, GPT-4, ChatGPT, Cohere, MPT-30B, Mistral 7B, Llama 2 70B (bao gồm cả bản gốc và bản Chat) |
| **Đa dạng Miền (Domains)** | Hỏi-Đáp chuyên gia và Tin tức cơ bản | 8 miền đa dạng: Tóm tắt học thuật (Abstracts), Công thức nấu ăn, Sách, Reddit, Tin tức, Đánh giá (Reviews), Wikipedia, Thơ са |
| **Chiến lược Giải mã (Decoding)** | Mặc định hoặc không xác định rõ | 4 chiến lược: Greedy, Lấy mẫu ngẫu nhiên (Sampling), Greedy + Phạt lặp lại (Repetition Penalty), Sampling + Phạt lặp lại |
| **Tấn công Đối kháng (Attacks)** | Thay thế từ đồng nghĩa đơn giản (WordNet) | 11 loại hình tấn công cấp độ ký tự, từ vựng và cấu trúc (Alternative Spelling, Article Deletion, Homoglyph, Whitespace, v.v.) |

Sự sụt giảm hiệu suất của PRDetect bắt nguồn từ những giới hạn trong cấu trúc đồ thị hiện tại. Biểu diễn đồ thị đồng nhất (homogeneous graph) với ma trận kề nhị phân đã vô tình san phẳng các sắc thái ngữ pháp phức tạp. Khi đối mặt với 4 chiến lược giải mã khác nhau của 11 mô hình LLM, sự khác biệt không chỉ nằm ở việc hai từ có liên kết với nhau hay không, mà nằm ở bản chất của mối liên kết đó. Hơn nữa, việc sử dụng kiến trúc GCN tĩnh khiến mô hình mất khả năng tự bảo vệ trước các kỹ thuật tấn công đối kháng tinh vi như chèn khoảng trắng (Whitespace Addition) hay thay thế ký tự tương đồng (Homoglyph), dẫn đến sự sai lệch trong quá trình truyền thông điệp (message passing) trên toàn bộ đồ thị. Cuối cùng, độ sâu giới hạn của mạng (chỉ 2 lớp) ngăn cản PRDetect nắm bắt ngữ cảnh toàn cục của các văn bản dài trong tập RAID, trong khi việc tăng độ sâu lại gây ra hiện tượng quá mịn (over-smoothing) làm suy giảm năng lực phân loại.

Để khắc phục những hạn chế cốt lõi này và thiết lập lại độ chính xác trên tập RAID, một chiến lược tái cấu trúc toàn diện được đề xuất, bao gồm việc nâng cấp cơ chế trích xuất đặc trưng, chuyển đổi kiến trúc mạng nơ-ron đồ thị, áp dụng các kỹ thuật điều chuẩn năng lượng, và thay đổi mục tiêu tối ưu hóa.

## 2. Cải tiến Khởi tạo Nhúng Ngữ nghĩa với Kiến trúc DeBERTa-v3

Khởi tạo biểu diễn node (node embedding initialization) đóng vai trò định hướng không gian đặc trưng cho toàn bộ mạng nơ-ron đồ thị. PRDetect hiện đang tận dụng mô hình RoBERTa để trích xuất các nhúng từ này. Mặc dù RoBERTa là một mô hình ngôn ngữ mạnh, nhưng đối mặt với sự phức tạp của 11 mô hình sinh văn bản và 4 chiến lược giải mã trong tập RAID, RoBERTa bộc lộ những hạn chế trong việc nhận diện sự bất thường về cấu trúc vị trí [2].

Kiến trúc DeBERTa-v3 (Decoding-enhanced BERT with disentangled attention) được đề xuất như một giải pháp thay thế tối ưu. Các phân tích thực nghiệm trên tập RAID đã chứng minh rằng việc kết hợp các đặc trưng ngữ cảnh từ DeBERTa có thể giúp hệ thống đạt độ chính xác lên tới 97.53% và tỷ lệ diện tích dưới đường cong ROC (ROC-AUC) đạt 99.53%, vượt trội đáng kể so với các mô hình transformer tiêu chuẩn.

Điểm ưu việt cốt lõi của DeBERTa nằm ở cơ chế chú ý gỡ rối (disentangled attention). Các mô hình họ BERT truyền thống thường mã hóa vị trí bằng cách cộng trực tiếp vector vị trí vào vector nội dung, dẫn đến việc hòa lẫn các tín hiệu này. Ngược lại, DeBERTa tách biệt thông tin nội dung và thông tin vị trí tương đối thành hai luồng vector hoàn toàn độc lập. Quá trình tính toán ma trận chú ý (attention matrix) được thực hiện riêng biệt cho các luồng này trước khi tổng hợp lại thành điểm số chú ý cuối cùng. Cơ chế này được biểu diễn thông qua phương trình:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V$$

Sự phân tách này mang lại lợi thế quyết định trong tác vụ nhận diện văn bản AI. Các LLM thường xuyên để lộ "dấu vân tay" (fingerprints) thông qua các mẫu cú pháp vi tế, đặc biệt là khi áp dụng các chiến lược giải mã như Phạt lặp lại (Repetition Penalty) hoặc Lấy mẫu theo nhiệt độ (Temperature Sampling). DeBERTa-v3 có khả năng nắm bắt chính xác các sai lệch về cấu trúc và vị trí từ vựng này. Khi nhúng từ DeBERTa-v3 được sử dụng làm đầu vào cho các node của đồ thị cú pháp, mạng đồ thị phía sau sẽ được cung cấp một nền tảng biểu diễn mang tính phân loại cực kỳ cao, giúp phân biệt rõ ràng giữa sự linh hoạt tự nhiên của văn bản con người và tính quy luật nhân tạo của văn bản máy.

Hơn nữa, một hệ thống phát hiện cấu trúc lai (hybrid detector) nên được thiết lập. Thay vì chỉ đưa nhúng DeBERTa vào mạng đồ thị, hệ thống sẽ duy trì một nhánh phụ xử lý chuỗi ngữ nghĩa toàn cục (sử dụng token `[CLS]`). Sự kết hợp giữa đặc trưng đồ thị và đặc trưng ngữ nghĩa chuỗi giúp đối phó hiệu quả với các kỹ thuật viết lại (paraphrasing), nơi các dấu hiệu chuỗi có thể bị thay đổi nhưng cấu trúc đồ thị vẫn giữ lại những điểm bất thường của máy học.

## 3. Chuyển đổi Hình thái Đồ thị và Tích chập Đa Quan hệ (RGCN)

Cây cú pháp phụ thuộc (dependency parsing tree) truyền tải một lượng lớn thông tin thông qua các loại quan hệ ngữ pháp. Liên kết giữa một động từ và chủ ngữ mang một trọng số ngữ nghĩa và vai trò phân loại hoàn toàn khác so với liên kết giữa một danh từ và mạo từ của nó. Mô hình PRDetect hiện tại đang lãng phí thông tin này bằng cách sử dụng một ma trận kề nhị phân, quy giản đồ thị về dạng đồng nhất.

Để giải quyết vấn đề này, kiến trúc mạng cần được nâng cấp thành Mạng Tích chập Đồ thị Quan hệ (Relational Graph Convolutional Networks - RGCN). RGCN được thiết kế đặc biệt để hoạt động trên các đồ thị không đồng nhất (heterogeneous graphs) đa quan hệ, nơi các node và cạnh có thể mang nhiều kiểu loại khác nhau. Bằng cách xem xét các loại cạnh (như `nsubj`, `dobj`, `amod`), RGCN không bỏ sót sự phong phú trong cách thức tương tác của các từ vựng [11].

Khác với GCN truyền thống chỉ sử dụng một ma trận trọng số duy nhất, RGCN phân bổ các ma trận trọng số riêng biệt cho từng loại quan hệ. Quá trình cập nhật trạng thái ẩn (hidden state) của một node tại lớp $l+1$ được thực hiện thông qua việc tổng hợp thông tin từ các node lân cận, nhưng được điều hướng bởi tính chất của mối quan hệ [14]:

$$h_{i}^{(l+1)} = \sigma\left(\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_{i}^{r}}\frac{1}{c_{i,r}}W_{r}^{(l)}h_{j}^{(l)} + W_{0}^{(l)}h_{i}^{(l)}\right)$$

Cơ chế toán học này đảm bảo rằng đặc trưng của node $i$ sẽ được biến đổi thông qua ma trận trọng số $W_{r}^{(l)}$ tương ứng với quan hệ $r$ nối giữa node $i$ và node $j$. Trọng số tự lặp $W_{0}^{(l)}$ được bảo toàn để duy trì đặc tính cốt lõi của node trung tâm, trong khi $c_{i,r}$ đóng vai trò là hằng số chuẩn hóa cấu trúc [14].

Trên tập dữ liệu RAID, mỗi trong số 11 mô hình LLM sở hữu một khuynh hướng ngữ pháp riêng biệt, thường được biểu hiện qua việc lạm dụng hoặc tối giản hóa một số cấu trúc phụ thuộc nhất định để đảm bảo tính lưu loát. Chẳng hạn, các mô hình nhỏ như Mistral 7B hoặc MPT-30B có thể sinh ra các cụm danh từ với mật độ phụ thuộc bổ nghĩa (modifier dependencies) khác thường so với văn bản của con người. RGCN khai thác trực tiếp các mẫu hình này thông qua việc tối ưu hóa độc lập các ma trận $W_{r}$, điều mà một GCN thuần túy không bao giờ thực hiện được [16].

## 4. Chống Nhiễu Đối kháng thông qua Cơ chế Chú ý Đồ thị Động (GATv2)

Sự chuyển đổi sang RGCN giải quyết được bài toán về đa dạng ngữ pháp, nhưng nó vẫn mắc phải một điểm yếu cốt lõi của họ GCN: tính tĩnh trong việc tổng hợp thông tin. Các thuật toán gộp (aggregation) tiêu chuẩn chia đều trọng số cho các node lân cận dựa trên hằng số chuẩn hóa. Đây là một lỗ hổng chí mạng khi mô hình phải đối mặt với 11 kỹ thuật tấn công đối kháng của tập dữ liệu RAID.

| Loại Hình Tấn Công Đối Kháng trên RAID | Cơ Chế Hoạt Động | Ảnh Hưởng Tới Cây Cú Pháp |
| :--- | :--- | :--- |
| **Homoglyph** | Đánh tráo ký tự bằng các ký hiệu có hình dáng tương đồng (ví dụ: 'e' La-tinh thay bằng 'e' Cyrillic). | Gây lỗi nhận diện từ vựng (Out-of-Vocabulary), tạo ra các node nhiễu không có nhúng ngữ nghĩa chuẩn xác. |
| **Misspelling / Alternative Spelling** | Cố tình chèn các lỗi sai chính tả phổ biến hoặc chuyển đổi giữa các hệ phái chính tả (Anh-Mỹ). | Bẻ gãy các liên kết từ vựng, khiến công cụ phân tích cú pháp (parser) nối sai các node. |
| **Whitespace / Zero Width Space** | Chèn khoảng trắng hoặc ký tự tàng hình giữa các từ hoặc chữ cái. | Phá vỡ tính liền mạch của chuỗi, chia cắt một node hợp lệ thành nhiều node rác. |
| **Paraphrase / Synonym Swap** | Sử dụng LLM phụ (như T5-11B) để viết lại câu hoặc thay thế từ đồng nghĩa diện rộng. | Tái cấu trúc hoàn toàn cây cú pháp, làm vô hiệu hóa các phân phối xác suất thống kê bề mặt. |

Để ngăn chặn sự lan truyền của các tín hiệu nhiễu (error propagation) từ các node bị tấn công, mô hình cần một cơ chế đánh giá độ tin cậy của từng node lân cận. Mạng Chú ý Đồ thị (GAT) ban đầu được thiết kế để giải quyết vấn đề này, nhưng nó bị giới hạn bởi hiện tượng "chú ý tĩnh" (static attention) - nghĩa là thứ tự xếp hạng các node lân cận không thay đổi linh hoạt dựa trên đặc tính của node trung tâm truy vấn [19].

Phiên bản cải tiến GATv2 khắc phục hoàn toàn giới hạn này bằng cách cung cấp một cơ chế chú ý động thực sự (strictly dynamic attention), mang lại sức mạnh biểu diễn vượt trội cho các tương tác không gian phức tạp [19]. Khác biệt toán học cốt lõi của GATv2 nằm ở việc thay đổi thứ tự của phép biến đổi tuyến tính và hàm phi tuyến [20]:

$$e_{ij} = \mathbf{a}^{T}\text{LeakyReLU}(\mathbf{W}[h_{i} \,\vert{}\vert{}\, h_{j}])$$

Sau đó, điểm số chú ý chưa chuẩn hóa $e_{ij}$ sẽ được đưa qua hàm softmax để tạo thành trọng số chú ý $\alpha_{ij}$. Kiến trúc động này cho phép mô hình nhận diện chính xác các node bị can thiệp bởi tấn công đối kháng. Ví dụ, khi một từ bị thay thế bằng Homoglyph tạo ra một vector nhúng bất thường, GATv2 có thể tính toán động và gán cho node đó một trọng số $\alpha_{ij}$ cực kỳ thấp, cách ly hoàn toàn phần tử nhiễu khỏi quá trình tổng hợp thông tin. Nghiên cứu xác nhận GATv2 đạt hiệu suất phân loại node áp đảo so với GCN và GraphSAGE truyền thống trên các đồ thị có độ đồng nhất (homophily) biến động hoặc bị nhiễu cấu trúc [24].

Giải pháp tối ưu cho PRDetect là kiến trúc dung hợp: Mạng Chú ý Đồ thị Quan hệ (Relational Graph Attention Network - RGAT). RGAT thừa hưởng khả năng định hướng ngữ pháp của RGCN và năng lực lọc nhiễu linh hoạt của GATv2 [26]. Các nghiên cứu trong phân tích quan điểm đa khía cạnh (Aspect-Based Sentiment Analysis) đã chứng minh rằng RGAT xử lý xuất sắc các cây cú pháp phụ thuộc phức tạp, thiết lập các kết nối bền vững giữa các thực thể dù chúng nằm cách xa nhau trong câu [28]. Trong ngữ cảnh phát hiện văn bản AI, mỗi loại quan hệ sẽ được trang bị một cơ chế GATv2 độc lập, tạo ra một rào chắn đa tầng chống lại các kỹ thuật lẩn tránh tinh vi nhất của RAID.

## 5. Giải quyết Hiện tượng Quá mịn và Xây dựng Mạng Đồ thị Sâu

Độ dài và sự phức tạp của văn bản trong RAID đòi hỏi mô hình phải có tầm nhìn sâu rộng. Các miền như Tóm tắt học thuật (ArXiv Abstracts), Sách, và Tin tức thường chứa các câu ghép dài với cấu trúc phụ thuộc chằng chịt [2]. Kiến trúc 2 lớp GCN hiện tại của PRDetect [1] chỉ cho phép thông tin lan truyền tối đa 2 bước nhảy (2-hop neighborhood). Hệ quả là, mô hình không thể thu nhận ngữ cảnh toàn cục, dẫn đến việc bỏ sót các chỉ dấu phân loại quan trọng nằm rải rác trên toàn bộ cây cú pháp.

Tuy nhiên, việc gia tăng số lớp mạng đồ thị lại đối mặt với một rào cản lý thuyết nghiêm trọng: hiện tượng quá mịn (over-smoothing) [31]. Khi số lớp tăng lên, quá trình tích chập đồ thị lặp đi lặp lại khiến biểu diễn của tất cả các node hội tụ về một vector tĩnh không thể phân biệt, làm suy giảm thảm khốc khả năng của mô hình.

### 5.1. Ràng Buộc Năng Lượng Dirichlet (Dirichlet Energy Constraint)

Hiện tượng quá mịn có thể được lượng hóa và kiểm soát thông qua Năng lượng Dirichlet (Dirichlet Energy). Năng lượng Dirichlet $E(X)$ của ma trận đặc trưng node trên đồ thị đo lường mức độ mượt mà (smoothness) của các tín hiệu trải rộng qua các cạnh [31]. Nó được định nghĩa toán học như sau:

$$E(X) = \frac{1}{2}\text{tr}(X^{T}LX) = \frac{1}{2}\sum_{(i,j)\in E}a_{ij}\left\vert{}\left\vert{}\frac{x_{i}}{\sqrt{d_{i}}} - \frac{x_{j}}{\sqrt{d_{j}}}\right\vert{}\right\vert{}^{2}$$

Trong đó, $L$ đại diện cho ma trận Laplacian của đồ thị, $a_{ij}$ là trọng số liên kết giữa các node, và $d_{i}$ là bậc của node (degree) [32]. Khi đồ thị ngày càng trở nên mượt mà qua các lớp tích chập sâu, Năng lượng Dirichlet có xu hướng phân rã theo cấp số nhân và tiến về $0$ [32]. Một phát hiện đột phá gần đây trong lĩnh vực phân loại đồ thị (graph classification) chỉ ra rằng: việc mạng GNN cố gắng ghi nhớ (memorize) các nhãn nhiễu hoặc dữ liệu đối kháng không làm giảm năng lượng, mà ngược lại, làm năng lượng tăng đột biến ở các thành phần tần số cao [34]. Điều này biến Năng lượng Dirichlet không chỉ là một thước đo đo lường sự quá mịn, mà còn là một bộ phát hiện tín hiệu nhiễu cực kỳ nhạy bén.

Để giải quyết triệt để rào cản này, hệ thống RGAT cần áp dụng cơ chế Học Ràng buộc Năng lượng Dirichlet (Dirichlet Energy Constrained Learning) hoặc Tích chập Năng lượng Tăng cường (Energy Enhanced Convolution - EEConv) [31]. Kỹ thuật này can thiệp vào quy tắc cập nhật node bằng cách điều chỉnh các gia số (increments) thông qua các kết nối dư thừa có giới hạn dưới (lower-bounded residual connections). Cơ chế này buộc mô hình phải duy trì mức năng lượng trong một ngưỡng tối ưu được xác định trước, ngăn chặn sự suy giảm hội tụ về 0, đồng thời chặn đứng các đỉnh năng lượng do nhiễu gây ra [32]. Sự can thiệp hệ thống này cho phép kiến trúc RGAT mở rộng độ sâu an toàn lên 4 đến 6 lớp, đủ khả năng bao quát toàn bộ cấu trúc phân cấp (hierarchical topology) của các câu phức tạp nhất trong tập RAID mà không đánh đổi tính phân biệt của đặc trưng node.

### 5.2. Áp Dụng Chuẩn Hóa GraphNorm

Đi đôi với việc duy trì năng lượng, kỹ thuật chuẩn hóa (normalization) tại mỗi lớp đóng vai trò thiết yếu trong việc tối ưu hóa tốc độ hội tụ. Các kỹ thuật chuẩn hóa truyền thống như LayerNorm hay BatchNorm thường xử lý các node một cách độc lập hoặc chuẩn hóa trên toàn bộ batch dữ liệu, dẫn đến việc xóa sổ các thông tin cấu trúc đặc hữu của đồ thị, chẳng hạn như phân phối bậc node [36].

GraphNorm là một kỹ thuật chuẩn hóa tiên tiến được thiết kế dành riêng cho mạng nơ-ron đồ thị. GraphNorm thực hiện chuẩn hóa các đặc trưng node trong phạm vi của từng đồ thị riêng biệt (tương ứng với một văn bản độc lập trong bài toán này) [38]. GraphNorm kết hợp một tham số $\alpha$ học được (learnable parameter) để bảo toàn phương sai tự nhiên của đồ thị, qua đó triệt tiêu tác động tiêu cực của các node có bậc cao vượt trội (high-degree dominance) và hỗ trợ giảm thiểu hiện tượng quá mịn [37]. Việc tích hợp GraphNorm xen kẽ giữa các khối RGAT sẽ gia tăng tính ổn định của gradient, đảm bảo mạng sâu học được các diễn đạt ngữ pháp bền vững.

## 6. Khái quát hóa Miền và Cấu trúc Lai Phức hợp

Hiệu suất của PRDetect sụp đổ một phần lớn do sự dịch chuyển phân phối miền (domain shift). Được huấn luyện chủ yếu trên văn bản Hỏi-Đáp (HC3) và Tin tức (GPT-3.5-Mixed), mô hình nhanh chóng bộc lộ sự mong manh khi phải phân tích công thức nấu ăn, thơ ca, hoặc mã nguồn Python trong tập RAID [1]. Mô hình có xu hướng học thuộc lòng (overfit) các từ vựng đặc thù của miền thay vì nắm bắt cấu trúc cú pháp phổ quát của AI.

Để ép buộc mô hình tập trung vào các đặc trưng bất biến, phương pháp Học Đối lập Khái quát hóa Miền (Contrastive Domain Generalization) cần được đưa vào giai đoạn tinh chỉnh (fine-tuning) [40]. Trong cơ chế học đối lập tự giám sát này, mục tiêu là thu hẹp khoảng cách giữa các biểu diễn đồ thị (positive pairs) có cùng nguồn gốc sinh ra (do con người viết, hoặc do LLM tạo) dù chúng thuộc các miền văn bản khác nhau, và đẩy lùi các biểu diễn (negative pairs) thuộc các nguồn gốc khác nhau trong không gian nhúng [41]. Hàm mất mát đối lập (contrastive loss) $\mathcal{L}_{con}$ kết hợp một nhiệt độ co giãn (temperature scaling) $\tau$ sẽ đóng vai trò như một bộ điều chuẩn (regularizer) mạnh mẽ, dẫn dắt hàm Cross-Entropy Loss tiêu chuẩn [41].

Song song đó, việc ứng dụng triết lý Hỗn hợp Chuyên gia (Mixture-of-Experts - MoE) cho phân tích chuỗi thời gian hoặc chuỗi đặc trưng ngữ nghĩa mang lại một lợi thế vượt trội [43]. Bằng cách sử dụng một mạng cổng (gating network) để điều phối đầu vào, kiến trúc MoE có khả năng phân tách linh hoạt phân phối dữ liệu phức tạp của 8 miền văn bản thành các không gian con chuyên biệt. Khi đó, mỗi "chuyên gia" sẽ tinh chỉnh việc phát hiện văn bản AI cho một cụm miền cụ thể, giải quyết triệt để sự không đồng nhất về mẫu hình sinh văn bản của LLM [43].

## 7. Chiến lược Tối ưu hóa Dựa trên Điểm Vận hành Thực tế

Trong các hệ thống phân loại văn bản AI thực tế, chỉ số Độ chính xác (Accuracy) thường che đậy những nhược điểm nghiêm trọng của mô hình, đặc biệt là khi hậu quả của việc nhận diện sai văn bản do con người viết thành văn bản AI (False Positive) là không thể chấp nhận được (gây ảnh hưởng trực tiếp đến uy tín học thuật và nghề nghiệp của người dùng) [10]. Phân tích trên tập RAID cho thấy, ngay cả khi độ chính xác tổng thể có vẻ khả quan, tỷ lệ dương tính giả (False Positive Rate - FPR) của các công cụ nguồn mở khi cài đặt mặc định vẫn ở mức nguy hiểm.

Vì vậy, mục tiêu tối ưu hóa và đánh giá của mô hình cần được chuyển hướng toàn diện sang các điểm vận hành (operating points) khắt khe như `TPR@5%FPR` và `TPR@1%FPR` [18]. Chỉ số `TPR@1%FPR` (True Positive Rate at 1% False Positive Rate) yêu cầu thiết lập một ngưỡng quyết định (decision threshold) sao cho hệ thống chỉ nhận diện nhầm tối đa 1% văn bản của con người, và tại ngưỡng đó, đánh giá xem mô hình phát hiện được bao nhiêu phần trăm văn bản AI thực sự [18]. Dưới các cuộc tấn công đối kháng mạnh mẽ như viết lại tàng hình (Stealth Paraphrasing), các mô hình hiện hành trải qua sự sụp đổ thảm khốc, với chỉ số `TPR@1%FPR` trung bình rớt xuống mức 0.024 [18].

Bằng cách thiết kế lại toàn bộ quy trình học tập kết hợp DeBERTa-v3, mạng RGAT với cơ chế chú ý động GATv2, chuẩn hóa GraphNorm, và duy trì Năng lượng Dirichlet, không gian nhúng của mô hình sẽ tạo ra một biên độ phân tách rộng lớn hơn giữa phân phối văn bản AI và con người. Việc cấu hình mô hình trực tiếp trên tập kiểm định (validation set) dựa trên bách phân vị thứ 95 hoặc 99 của điểm số dự đoán sẽ bảo đảm hệ thống vận hành bền vững, đạt chỉ số AUROC xuất sắc và duy trì khả năng truy xuất cao ngay cả dưới các ràng buộc khắt khe nhất của thực tiễn [18].

Sự kết hợp đồng bộ các cải tiến kiến trúc mang tính đột phá này không chỉ giải quyết trực diện các lỗ hổng đang tồn tại của PRDetect, mà còn nâng cấp hệ thống trở thành một lá chắn toàn diện, đủ năng lực đối đầu với sự tiến hóa không ngừng của các mô hình sinh ngôn ngữ lớn trên những tập dữ liệu chuẩn khắc nghiệt nhất.

---

## Tài liệu Tham khảo

1. `2025.findings-naacl.464.pdf`
2. GitHub liamdugan/raid: RAID is the largest and most challenging, https://github.com/liamdugan/raid
3. arXiv:2501.08913v1 [cs.CL] 15 Jan 2025, https://arxiv.org/pdf/2501.08913
4. RAID: A Shared Benchmark for Robust Evaluation of Machine, https://arxiv.org/html/2405.07940v2
5. Cross-Domain Machine-Generated Text Detection Challenge arXiv, https://arxiv.org/html/2501.08913v1
6. RAID: A Shared Benchmark for Robust Evaluation of Machine, https://arxiv.org/html/2405.07940v1
7. Integrating Retrieval-Augmented Generation with Large Language, https://www.mdpi.com/2673-9585/6/3/22
8. RAID: A Shared Benchmark for Robust Evaluation of Machine, https://www.researchgate.net/publication/384214327_RAID_A_Shared_Benchmark_for_Robust_Evaluation_of_Machine-Generated_Text_Detectors
9. Detecting Multimedia Generated by Large AI Models: A Survey arXiv, https://arxiv.org/html/2402.00045v4
10. Dimension-to-Composition Evidence Routing for Mixed-Origin AI, https://arxiv.org/pdf/2608.27380
11. Relational graph convolutional networks for sentiment analysis, df
12. RELATIONAL GRAPH CONVOLUTIONAL NETWORKS FOR, https://cmde.tabrizu.ac.ir/article_19870_95b815607e52c225484be720e31980e4.pdf
13. R-GCN: Modeling Relational Data with Graph Convolution, https://arxiv.org/pdf/2404.13079, https://www.youtube.com/watch?v=Ys6VdaRguYU
14. Relational graph convolutional networks: a closer look PMC NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC9680895/
15. Temporal Relational Graph Convolutional Network Approach MDPI, https://www.mdpi.com/2504-4990/6/4/113
16. Identifying Root Cause of bugs by Capturing Changed Code Lines, arXiv, https://arxiv.org/html/2505.00990v1
17. A Theory of Link Prediction via Relational Weisfeiler-Leman on, https://proceedings.neurips.cc/paper_files/paper/2023/file/3eceb70f47690051d6769739fbf6294b-Paper-Conference.pdf
18. StealthRL: Reinforcement Learning Paraphrase Attacks for Multi, https://arxiv.org/html/2602.08934v2
19. How Attentive are Graph Attention Networks? Request PDF, https://www.researchgate.net/publication/352016176_How_Attentive_are_Graph_Attention_Networks
20. Static and Dynamic Attention: Implications for Graph Neural Networks, https://medium.com/data-science/static-and-dynamic-attention-implications-for-graph-neural-networks-eda0d9d7b60a
21. Hybrid Graph Attention-Geostatistical Methods for Spatio-temporal, https://arxiv.org/html/2603.08393v2
22. MLGT: A multimodal graph attention network for virtual screening of, https://pmc.ncbi.nlm.nih.gov/articles/PMC12962487/
23. SAT-GATv2: A Dynamic Attention-Based Graph Neural Network for, https://www.mdpi.com/2079-9292/14/3/423
24. Intro to Graph Neural Networks | PDF | Applied Mathematics Scribd, https://www.scribd.com/document/810504211/2024-Introduction-to-Graph-Neural-Networks-A-Starting
25. Feature-Enhanced Graph Neural Networks for Classification of, https://arxiv.org/abs/2512.18524
26. arXiv:2203.16369v2 [cs.CL] 23 Nov 2022, https://arxiv.org/pdf/2203.16369
27. Relational Graph Attention Network for Aspect-based Sentiment, https://aclanthology.org/2020.acl-main.295.pdf
28. arXiv:2104.04986v1 [cs.CL] 11 Apr 2021, https://arxiv.org/pdf/2104.04986
29. Relational Graph Attention Network for Aspect-based Sentiment, https://www.researchgate.net/publication/343302221_Relational_Graph_Attention_Network_for_Aspect-based_Sentiment_Analysis
30. Relational Graph Attention Network for Aspect-based Sentiment, https://www.semanticscholar.org/paper/Relational-Graph-Attention-Network-for-Aspect-based-Wang-Shen/8e85df955707cb7959299d83b6b8c8a28b2b53dd
31. Dirichlet Energy Enhancement of Graph Neural Networks by, https://arxiv.org/abs/2311.05767
32. Dirichlet Energy Constrained Learning for Deep Graph Neural, https://arxiv.org/html/2107.02392v1
33. Geometric Perspectives on Over-smoothing in Deep Graph Neural, https://openreview.net/forum?id=0e26yMOCbd
34. Energy Guided smoothness to improve Robustness in Graph arXiv, https://arxiv.org/pdf/2412.08419
35. Analysis of Dirichlet Energies as Over-smoothing Measures arXiv, https://arxiv.org/pdf/2512.09890
36. Foundations and Frontiers of Graph Learning Theory arXiv, https://arxiv.org/html/2407.03125v2
37. Normalisation and Initialisation Strategies for Graph Neural arXiv, https://arxiv.org/pdf/2602.23599
38. References Graph Neural Networks, https://graph-neural-networks.github.io/static/file/reference.pdf
39. Modality as Heterogeneity: Node Splitting and Graph Rewiring for, https://arxiv.org/html/2602.00067v1
40. Artificial Intelligence Feb 2026, https://www.arxiv.org/list/cs.AI/2026-02?skip=100&show=2000
41. LLM-Empowered Tri-Level Learning for Time Series OOD arXiv, https://arxiv.org/pdf/2410.07018
42. (PDF) An effective self-supervised framework for learning expressive, https://www.researchgate.net/publication/351297761_An_effective_self-supervised_framework_for_learning_expressive_molecular_global_representations_to_drug_discovery
43. Mixture-of-Experts for Time Series Analysis: Taxonomy, Progress, https://www.researchgate.net/publication/410643624_Mixture-of-Experts_for_Time_Series_Analysis_Taxonomy_Progress_and_Prospects
44. DETree: DEtecting Human-AI Collaborative Texts via Tree arXiv, https://arxiv.org/html/2510.17489v1
45. AI-Generated Text is Non-Stationary - arXiv, https://arxiv.org/pdf/2508.01754
46. MELD: Multi-Task Equilibrated Learning Detector for AI Bytez, https://bytez.com/docs/arxiv/2605.06903/paper
```eof

Tài liệu của bạn đã được chuyển sang định dạng Markdown thành công! Tôi đã chỉnh sửa lại toàn bộ các công thức toán học, trình bày các bảng dữ liệu cho dễ nhìn, đồng thời khắc phục các lỗi do quá trình copy/nhận dạng văn bản như các con số trích dẫn bị nhảy nhầm thành số mũ trong từ. 

Bạn xem thử file Markdown đã ưng ý chưa nhé? Nếu cần điều chỉnh hoặc làm rõ thêm phần nào, hãy báo cho tôi biết!