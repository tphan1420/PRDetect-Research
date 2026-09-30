import sys
import os
import argparse
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from torch_geometric.data import Data
except ImportError:
    class Data:
        def __init__(self, x=None, edge_index=None, edge_type=None, y=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.y = y

        def to(self, device):
            if self.x is not None:
                self.x = self.x.to(device)
            if self.edge_index is not None:
                self.edge_index = self.edge_index.to(device)
            if self.edge_type is not None:
                self.edge_type = self.edge_type.to(device)
            if self.y is not None:
                self.y = self.y.to(device)
            return self

from building_graph import build_graph
from model.GCN2 import GCN2
from model.RGCN import RGCN2
from dep_vocab import get_default_vocab

parser = argparse.ArgumentParser(description="Dự đoán văn bản Human hay Machine bằng PRDetect (GCN / RGCN)")
parser.add_argument('--text', type=str, required=True, help="Đoạn văn bản cần kiểm tra")
parser.add_argument('--dataset', type=str, default='hc3')
parser.add_argument('--seed', type=str, default='2024')
parser.add_argument('--model_type', choices=['rgcn', 'gcn'], default='rgcn',
                    help="Kiểu mô hình suy luận: 'rgcn' (đa quan hệ) hoặc 'gcn' (đồng nhất)")

args = parser.parse_args()


def detect(text, dataset='hc3', seed='2024', model_type='rgcn'):
    seed_int = int(seed)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 768
    hidden_dim = 256
    output_dim = 64

    # Đóng gói thành danh sách chuỗi JSON hợp lệ
    json_texts = [json.dumps({"text": text, "label": ""})]

    if model_type == 'rgcn':
        all_token_embeddings, edge_index, edge_type, y = build_graph(json_texts, return_edge_type=True)
        vocab = get_default_vocab()
        model = RGCN2(input_dim, hidden_dim, output_dim, num_relations=vocab.num_relations).to(device)
        model_path = f'./model/{dataset}_rgcn_model_{seed}.pth'

        if not os.path.exists(model_path):
            fallback = f'./model/{dataset}_gcn_model_{seed}.pth'
            if os.path.exists(fallback):
                print(f"[Lưu ý] Không tìm thấy checkpoint RGCN ({model_path}), chuyển sang dùng GCN ({fallback}).")
                model = GCN2(input_dim, hidden_dim, output_dim).to(device)
                model_path = fallback
                model_type = 'gcn'
            else:
                raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {model_path}")

        if model_type == 'rgcn':
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            data = Data(
                x=all_token_embeddings[0].to(device),
                edge_index=edge_index[0].to(device),
                edge_type=edge_type[0].to(device),
                y=y.to(device)
            )
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            data = Data(
                x=all_token_embeddings[0].to(device),
                edge_index=edge_index[0].to(device),
                y=y.to(device)
            )
    else:
        all_token_embeddings, edge_index, y = build_graph(json_texts, return_edge_type=False)
        model = GCN2(input_dim, hidden_dim, output_dim).to(device)
        model_path = f'./model/{dataset}_gcn_model_{seed}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy checkpoint GCN tại: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        data = Data(
            x=all_token_embeddings[0].to(device),
            edge_index=edge_index[0].to(device),
            y=y.to(device)
        )

    with torch.no_grad():
        outputs = model(data)
        probability = outputs.item()
        prediction = 1 if probability >= 0.5 else 0

    return probability, prediction


if __name__ == '__main__':
    prob, pred = detect(args.text, args.dataset, args.seed, args.model_type)
    label_name = "Human-written (Người viết)" if pred == 1 else "AI/LLM-generated (Máy sinh)"
    print(f"\n[KẾT QUẢ DỰ ĐOÁN]")
    print(f"  * Mô hình: {args.model_type.upper()}")
    print(f"  * Xác suất Human: {prob:.4f} ({prob*100:.2f}%)")
    print(f"  * Nhãn dự đoán: {pred} -> {label_name}")
