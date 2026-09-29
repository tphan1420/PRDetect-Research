import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    class GCNConv(nn.Module):
        """Fallback GCNConv khi môi trường chưa cài PyTorch Geometric"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        def forward(self, x, edge_index):
            num_nodes = x.size(0)
            deg = torch.zeros(num_nodes, device=x.device)
            deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
            deg = deg + 1.0  # Self-loop
            deg_inv_sqrt = deg.pow(-0.5)

            # A_hat * x
            out = torch.zeros_like(x)
            src, dst = edge_index[0], edge_index[1]
            norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
            out.index_add_(0, dst, x[src] * norm.unsqueeze(-1))
            out = out + x * (deg_inv_sqrt * deg_inv_sqrt).unsqueeze(-1)  # self-loop
            return self.linear(out)

class GCN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN2, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True)  
        return torch.sigmoid(x) 