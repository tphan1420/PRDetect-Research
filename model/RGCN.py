r"""
Mô hình Mạng Tích chập Đồ thị Quan hệ (Relational Graph Convolutional Network - RGCN)
dành cho bài toán Nhận diện Văn bản do LLM sinh ra (PRDetect).

Dựa trên Mục 3 trong update.md:
Chuyển đổi hình thái đồ thị từ đồng nhất (homogeneous) sang đa quan hệ (multi-relational).
Công thức cập nhật trạng thái ẩn tại mỗi tầng l+1:
    h_i^{(l+1)} = \sigma( \sum_{r \in R} \sum_{j \in N_i^r} (1 / c_{i,r}) W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)} )
Trong đó:
    - W_r: Ma trận biến đổi đặc thù cho từng loại quan hệ cú pháp phụ thuộc (nsubj, dobj, amod,...).
    - W_0: Trọng số tự lặp (self-loop / root weight) bảo toàn bản sắc của node trung tâm.
    - c_{i,r}: Hệ số chuẩn hóa cấu trúc (bậc node theo quan hệ).
    - Phân rã cơ sở (Basis-decomposition num_bases): Điều chuẩn trọng số, tránh quá khớp khi số quan hệ lớn.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

    def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, size: int = None) -> torch.Tensor:
        if size is None:
            size = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        out = torch.zeros((size, x.size(-1)), device=x.device, dtype=x.dtype)
        count = torch.zeros((size, 1), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        count.index_add_(0, batch, torch.ones((batch.size(0), 1), device=x.device, dtype=x.dtype))
        return out / count.clamp(min=1.0)

    def global_add_pool(x: torch.Tensor, batch: torch.Tensor, size: int = None) -> torch.Tensor:
        if size is None:
            size = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        out = torch.zeros((size, x.size(-1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        return out

    def global_max_pool(x: torch.Tensor, batch: torch.Tensor, size: int = None) -> torch.Tensor:
        if size is None:
            size = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        out = torch.full((size, x.size(-1)), -float('inf'), device=x.device, dtype=x.dtype)
        for i in range(size):
            mask = (batch == i)
            if mask.any():
                out[i] = x[mask].max(dim=0)[0]
            else:
                out[i] = 0.0
        return out

    class RGCNConv(nn.Module):
        r"""
        Cài đặt thuần PyTorch cho RGCNConv (Fallback khi môi trường chưa cài PyTorch Geometric).
        Triển khai đầy đủ công thức:
            h_i = \sum_{r} (1 / c_{i,r}) W_r h_j + W_0 h_i
        với hỗ trợ Basis Decomposition và Root Weight.
        """
        def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                     num_bases: int = None, root_weight: bool = True, bias: bool = True):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_relations = num_relations
            self.num_bases = num_bases
            self.has_root = root_weight

            if num_bases is not None and num_bases > 0 and num_bases < num_relations:
                self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
                self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
            else:
                self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
                self.basis = None
                self.att = None

            if self.has_root:
                self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
            else:
                self.register_parameter('root', None)

            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

        def reset_parameters(self):
            if self.basis is not None:
                nn.init.xavier_uniform_(self.basis)
                nn.init.xavier_uniform_(self.att)
            else:
                nn.init.xavier_uniform_(self.weight)
            if self.root is not None:
                nn.init.xavier_uniform_(self.root)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
            num_nodes = x.size(0)
            num_edges = edge_index.size(1) if edge_index.dim() == 2 else 0

            # Tính ma trận trọng số cho từng quan hệ W_r
            if self.basis is not None:
                # W_r = \sum_b a_{r,b} V_b
                w = torch.matmul(self.att, self.basis.view(self.num_bases, -1)).view(
                    self.num_relations, self.in_channels, self.out_channels
                )
            else:
                w = self.weight

            out = torch.zeros((num_nodes, self.out_channels), device=x.device, dtype=x.dtype)

            if num_edges > 0:
                src, dst = edge_index[0], edge_index[1]
                # Duyệt qua các quan hệ xuất hiện trên cạnh
                for r in range(self.num_relations):
                    mask = (edge_type == r)
                    if not mask.any():
                        continue
                    r_src = src[mask]
                    r_dst = dst[mask]
                    w_r = w[r]  # [in_channels, out_channels]

                    # Biến đổi đặc trưng node nguồn: h_j * W_r
                    msg = torch.matmul(x[r_src], w_r)

                    # Chuẩn hóa 1 / c_{i,r} theo bậc vào của node đích trong quan hệ r
                    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
                    deg.index_add_(0, r_dst, torch.ones_like(r_dst, dtype=x.dtype))
                    deg_inv = deg[r_dst].clamp(min=1.0).unsqueeze(-1)
                    msg_norm = msg / deg_inv

                    out.index_add_(0, r_dst, msg_norm)

            # Thêm nhánh tự lặp W_0 * h_i
            if self.root is not None:
                out = out + torch.matmul(x, self.root)

            if self.bias is not None:
                out = out + self.bias

            return out


class RGCN2(nn.Module):
    """
    Kiến trúc Relational GCN 2 tầng (nâng cấp trực tiếp từ GCN2 của PRDetect).
    Khai thác các quan hệ cú pháp phụ thuộc để tối ưu hóa riêng biệt các ma trận W_r.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64,
                 num_relations: int = 60, num_bases: int = 30, dropout: float = 0.5):
        super(RGCN2, self).__init__()
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations,
                              num_bases=num_bases, root_weight=True)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations=num_relations,
                              num_bases=num_bases, root_weight=True)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, edge_index=None, edge_type=None, batch=None):
        if hasattr(data, 'x'):
            x = data.x
            edge_index = data.edge_index
            edge_type = getattr(data, 'edge_type', None)
            batch = getattr(data, 'batch', None)
        else:
            x = data

        if edge_type is None and edge_index is not None:
            # Fallback nếu đồ thị chưa có nhãn quan hệ: gán quan hệ 0 (<unk>)
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc(x)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return torch.sigmoid(x)


class RGCN4(nn.Module):
    """
    Kiến trúc Relational GCN 4 tầng (nâng cấp trực tiếp từ GCN4).
    Tăng cường độ sâu lan truyền thông điệp cú pháp cho các câu dài, cấu trúc ghép.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, hidden_dim2: int = 256,
                 hidden_dim3: int = 128, output_dim: int = 64,
                 num_relations: int = 60, num_bases: int = 30, dropout: float = 0.5):
        super(RGCN4, self).__init__()
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations,
                              num_bases=num_bases, root_weight=True)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim2, num_relations=num_relations,
                              num_bases=num_bases, root_weight=True)
        self.conv3 = RGCNConv(hidden_dim2, hidden_dim3, num_relations=num_relations,
                              num_bases=num_bases, root_weight=True)
        self.conv4 = RGCNConv(hidden_dim3, output_dim, num_relations=num_relations,
                              num_bases=num_bases, root_weight=True)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, edge_index=None, edge_type=None, batch=None):
        if hasattr(data, 'x'):
            x = data.x
            edge_index = data.edge_index
            edge_type = getattr(data, 'edge_type', None)
            batch = getattr(data, 'batch', None)
        else:
            x = data

        if edge_type is None and edge_index is not None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc(x)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return torch.sigmoid(x)


class RelationalGCN(nn.Module):
    """
    Lớp RGCN linh hoạt hỗ trợ tùy biến số tầng (num_layers), kích thước các tầng ẩn,
    tỷ lệ dropout, và phương thức pooling (mean / max / sum).
    """
    def __init__(self, input_dim: int = 768, hidden_dims: list = None, output_dim: int = 64,
                 num_relations: int = 60, num_bases: int = 30, dropout: float = 0.5,
                 pooling: str = "mean"):
        super(RelationalGCN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256]

        self.num_relations = num_relations
        self.num_bases = num_bases
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(RGCNConv(prev_dim, h_dim, num_relations=num_relations,
                                   num_bases=num_bases, root_weight=True))
            prev_dim = h_dim

        self.convs = nn.ModuleList(layers)
        self.final_conv = RGCNConv(prev_dim, output_dim, num_relations=num_relations,
                                   num_bases=num_bases, root_weight=True)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, data, edge_index=None, edge_type=None, batch=None):
        if hasattr(data, 'x'):
            x = data.x
            edge_index = data.edge_index
            edge_type = getattr(data, 'edge_type', None)
            batch = getattr(data, 'batch', None)
        else:
            x = data

        if edge_type is None and edge_index is not None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.final_conv(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc(x)

        if batch is not None:
            if self.pooling == "max":
                x = global_max_pool(x, batch)
            elif self.pooling == "sum":
                x = global_add_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)
        else:
            if self.pooling == "max":
                x, _ = torch.max(x, dim=0, keepdim=True)
            elif self.pooling == "sum":
                x = torch.sum(x, dim=0, keepdim=True)
            else:
                x = torch.mean(x, dim=0, keepdim=True)

        return torch.sigmoid(x)
