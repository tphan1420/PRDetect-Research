"""
RGAT = RGCN + GATv2 — Merge cuối cùng của Phần 3 + Phần 4.
File này chỉ hoạt động được SAU KHI Phần 3 hoàn thành
(cần edge_type trong dữ liệu .pkl).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv  # PyG >= 2.3 hỗ trợ sẵn


class RGATModel(nn.Module):
    """
    Relational Graph Attention Network.

    Kết hợp:
    - RGCN: Ma trận trọng số riêng cho từng loại quan hệ ngữ pháp (nsubj, dobj, amod, ...)
    - GATv2: Cơ chế chú ý động để lọc node nhiễu từ tấn công đối kháng

    Yêu cầu dữ liệu:
    - data.x:          Node features (token embeddings)
    - data.edge_index: Cạnh đồ thị cú pháp
    - data.edge_type:  Loại quan hệ cho mỗi cạnh (từ Phần 3)
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_relations, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = RGATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,  # Số loại quan hệ từ SpaCy
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.conv2 = RGATConv(
            in_channels=hidden_dim * heads,
            out_channels=output_dim,
            num_relations=num_relations,
            heads=1,
            concat=False,
            dropout=dropout
        )
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        x = self.conv1(x, edge_index, edge_type)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)
        return torch.sigmoid(x)
