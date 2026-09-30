import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Model(nn.Module):
    """
    Mang Chu y Do thi Dong GATv2 — Cai tien Phan 4 so voi GCN2.

    4 Khac biet cot loi so voi GCN2 (model goc PRDetect):

    [1] TRONG SO DONG (Dynamic Attention)
        GCN2 : alpha_ij = 1/sqrt(deg_i * deg_j) -- co dinh theo bac node
        GATv2: alpha_ij = softmax(a^T LeakyReLU(W[h_i || h_j]))
               -> Hoc duoc trong so khac nhau cho moi cap node (i, j)

    [2] LOC NHIEU DOI KHANG (Adversarial Noise Filtering)
        Node bi tan cong (Homoglyph, Whitespace, Misspelling) tao ra
        vector embedding bat thuong. GATv2 tinh dong alpha_ij va gan
        trong so gan 0 cho node nhieu, cach ly khoi message passing.
        GCN2 khong co kha nang nay (trong so luon > 0).

    [3] MULTI-HEAD ATTENTION (Hoc da goc nhin)
        Moi head hoc mot kieu tuong quan cu phap khac nhau.
        Concat cac head cho bieu dien phong phu hon.
        GCN2: 1 phep bien doi tuyen tinh duy nhat.

    [4] ELU THAY RELU + PHAN TACH DROPOUT
        ELU xu ly dau ra am tot hon -> gradient on dinh hon.
        attn_dropout (ben trong conv): lam nhieu attention weight khi train.
        dropout (ben ngoai): lam nhieu node feature vector.
        Tach biet 2 loai dropout de dieu chinh doc lap.

    Cong thuc GATv2 (strictly dynamic attention):
        e_ij = a^T * LeakyReLU(W * [h_i || h_j])   <- khac GAT v1
        alpha_ij = softmax_j(e_ij)
        h_i' = ELU( sum_j alpha_ij * W_v * h_j )

    Tham so:
        input_dim    : Chieu nhung dau vao (768 cho RoBERTa)
        hidden_dim   : Chieu an moi attention head o layer 1
        output_dim   : Chieu ra cua layer 2 (dac trung cuoi cung)
        heads        : So attention heads o layer 1 (mac dinh 4)
        dropout      : Xac suat feature dropout cho node embedding
        attn_dropout : Xac suat dropout rieng cho attention coefficient
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 heads=4, dropout=0.5, attn_dropout=0.1):
        super(GATv2Model, self).__init__()

        # ------------------------------------------------------------------
        # Layer 1: Multi-head GATv2 (Khac biet [1] + [3] so voi GCN2)
        # GCN2.conv1: GCNConv(768, 256) -- trong so co dinh theo bac
        # GATv2.conv1: GATv2Conv(768, 64, heads=4) -- trong so DONG, 4 heads
        #
        # attn_dropout != dropout:
        #   attn_dropout (nho, ~0.1): regularize attention coefficient
        #   dropout (lon, ~0.5)     : regularize node feature vector
        # Tach biet 2 loai -> tranh double-dropout qua manh (Khac biet [4])
        # ------------------------------------------------------------------
        self.conv1 = GATv2Conv(
            in_channels=input_dim,       # 768 (RoBERTa embedding)
            out_channels=hidden_dim,     # 64 moi head
            heads=heads,                 # 4 heads -> output = 64*4 = 256
            dropout=attn_dropout,        # Attention dropout (nho)
            concat=True,                 # Concat 4 heads: [64*4 = 256]
            add_self_loops=True,         # Giu thong tin node trung tam
            share_weights=False          # Moi head hoc W rieng -> da goc nhin
        )

        # ------------------------------------------------------------------
        # Layer 2: Single-head GATv2
        # Layer cuoi dung 1 head + khong concat de ra output_dim chieu chuan.
        # Day van la dynamic attention (Khac biet [1]) -- khong giong GCN2.
        # ------------------------------------------------------------------
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * heads,  # 256 (tu layer 1)
            out_channels=output_dim,          # 64
            heads=1,                          # 1 head o layer cuoi
            dropout=attn_dropout,
            concat=False,                     # Khong concat khi 1 head
            add_self_loops=True,
            share_weights=False
        )

        # FC + Feature Dropout (Khac biet [4]: tach khoi attn_dropout)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # LayerNorm sau moi GATv2 block
        # On dinh hon BatchNorm khi moi do thi co so node khac nhau
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, data, return_attention=False):
        """
        Lan truyen thuan qua 2 lop GATv2.

        Args:
            data: torch_geometric.data.Data voi data.x, data.edge_index
            return_attention (bool): Neu True, tra them attention weights
                de phan tich node nhieu (explainability & debug)

        Returns:
            Neu return_attention=False: tensor shape [1, 1] -- xac suat nhi phan
            Neu return_attention=True : (xac suat, (alpha1, alpha2))
                                        alpha_k shape: [E, heads_k]
        """
        x, edge_index = data.x, data.edge_index

        # ==================================================================
        # Layer 1: GATv2 voi dynamic attention
        # Khac biet [1][2][3]: Moi cap node (i,j) co alpha_ij KHAC NHAU,
        # phu thuoc vao CA h_i va h_j -> node nhieu bi gan alpha gan 0
        # ==================================================================
        if return_attention:
            x, (edge_index_att1, alpha1) = self.conv1(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.conv1(x, edge_index)

        x = self.norm1(x)    # Chuan hoa truoc activation (on dinh gradient)
        x = F.elu(x)         # Khac biet [4]: ELU thay ReLU
        x = self.dropout(x)  # Feature dropout (tach biet voi attn_dropout)

        # ==================================================================
        # Layer 2: GATv2 -- van dynamic, khong phai aggregation co dinh
        # ==================================================================
        if return_attention:
            x, (edge_index_att2, alpha2) = self.conv2(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.conv2(x, edge_index)

        x = self.norm2(x)
        # Khong ap dung ELU o layer cuoi truoc fc:
        # de fc hoat dong tren khong gian tuyen tinh day du
        x = self.dropout(x)

        # ==================================================================
        # Graph-level Pooling: Mean pooling qua tat ca node cua do thi
        # (Tuong duong GCN2 de giu pipeline tuong thich voi .pkl hien co)
        # Neu dung DataLoader batch: thay bang global_mean_pool(x, data.batch)
        # ==================================================================
        x = self.fc(x)                          # [N, 1]
        x = torch.mean(x, dim=0, keepdim=True)  # [1, 1]

        if return_attention:
            return torch.sigmoid(x), (alpha1, alpha2)
        return torch.sigmoid(x)

    @torch.no_grad()
    def get_suspicious_nodes(self, data, threshold=0.05):
        """
        Tra ve chi so cac node ma GATv2 gan attention thap bat thuong.
        Cac node nay co the bi tan cong doi khang (Homoglyph, Whitespace...).

        Args:
            data      : torch_geometric.data.Data
            threshold : Nguong attention trung binh (mac dinh 0.05)

        Returns:
            List[int] : Chi so cac node dang ngo
        """
        self.eval()
        _, (alpha1, _) = self.forward(data, return_attention=True)

        # alpha1: [E, heads] -- trung binh qua cac heads
        mean_alpha = alpha1.mean(dim=-1)   # [E]
        edge_index = data.edge_index

        # Tinh attention trung binh moi node nhan duoc tu cac lang gieng
        node_attn = torch.zeros(data.x.size(0), device=data.x.device)
        node_attn.scatter_add_(0, edge_index[1], mean_alpha)
        deg = torch.bincount(edge_index[1], minlength=data.x.size(0)).float()
        deg = deg.clamp(min=1)
        node_attn = node_attn / deg  # attention trung binh moi node

        suspicious = (node_attn < threshold).nonzero(as_tuple=True)[0].tolist()
        return suspicious
