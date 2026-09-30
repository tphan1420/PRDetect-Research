"""
Script kiểm thử và đánh giá mô hình RGCN (Relational Graph Convolutional Network)
trên các tập dữ liệu chuẩn (HC3, GPT3.5-Mixed, RAID, DetectRL).
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from tqdm import tqdm
try:
    from sklearn.metrics import roc_auc_score, f1_score
except ImportError:
    def roc_auc_score(y_true, y_score):
        return 0.0
    def f1_score(y_true, y_pred):
        return 0.0

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data, Batch
except ImportError:
    try:
        from torch_geometric.data import DataLoader, Data, Batch
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

        class Batch:
            def __init__(self, x, edge_index, edge_type, y, batch):
                self.x = x
                self.edge_index = edge_index
                self.edge_type = edge_type
                self.y = y
                self.batch = batch
                self.num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0

            def to(self, device):
                self.x = self.x.to(device)
                self.edge_index = self.edge_index.to(device)
                if self.edge_type is not None:
                    self.edge_type = self.edge_type.to(device)
                self.y = self.y.to(device)
                self.batch = self.batch.to(device)
                return self

        def _fallback_collate(data_list):
            xs, edge_indices, edge_types, ys, batches = [], [], [], [], []
            node_offset = 0
            for graph_idx, data in enumerate(data_list):
                num_nodes = data.x.size(0)
                xs.append(data.x)
                if data.edge_index.numel() > 0:
                    edge_indices.append(data.edge_index + node_offset)
                else:
                    edge_indices.append(data.edge_index)
                if getattr(data, 'edge_type', None) is not None:
                    edge_types.append(data.edge_type)
                y_val = data.y
                if not isinstance(y_val, torch.Tensor):
                    y_val = torch.tensor([y_val])
                ys.append(y_val.view(-1))
                batches.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
                node_offset += num_nodes

            batched_x = torch.cat(xs, dim=0)
            batched_edge_index = torch.cat(edge_indices, dim=1) if len(edge_indices) > 0 else torch.zeros((2, 0), dtype=torch.long)
            batched_edge_type = torch.cat(edge_types, dim=0) if len(edge_types) > 0 else None
            batched_y = torch.cat(ys, dim=0)
            batched_batch = torch.cat(batches, dim=0)
            return Batch(batched_x, batched_edge_index, batched_edge_type, batched_y, batched_batch)

        from torch.utils.data import DataLoader as TorchDataLoader
        class DataLoader:
            def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
                self._loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_fallback_collate, **kwargs)
            def __iter__(self):
                return iter(self._loader)
            def __len__(self):
                return len(self._loader)

from model.RGCN import RGCN2, RGCN4, RelationalGCN
from dep_vocab import get_default_vocab


def create_pyg_data_list(graph_dict):
    """Chuyển đổi dữ liệu thô sang danh sách đối tượng Data để đưa vào DataLoader"""
    data_list = []
    n_samples = len(graph_dict['y'])
    all_embeddings = graph_dict['all_token_embeddings']
    all_edge_index = graph_dict['all_edge_index']
    all_edge_type = graph_dict.get('all_edge_type', None)
    all_y = graph_dict['y']

    for i in range(n_samples):
        e_type = all_edge_type[i] if all_edge_type is not None else torch.zeros(all_edge_index[i].size(1), dtype=torch.long)
        y_val = all_y[i]
        if not isinstance(y_val, torch.Tensor):
            y_tensor = torch.tensor([y_val], dtype=torch.float)
        elif y_val.dim() == 0:
            y_tensor = y_val.unsqueeze(0).float()
        else:
            y_tensor = y_val.float()

        data_list.append(Data(
            x=all_embeddings[i],
            edge_index=all_edge_index[i],
            edge_type=e_type,
            y=y_tensor
        ))
    return data_list


def test_rgcn(test_file, dataset_name, seed, num_layers=2, num_bases=30,
              input_dim=768, hidden_dim=256, output_dim=64, batch_size=32, do_save=False, cpu=False):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"\n=======================================================")
    print(f" KIỂM THỬ MÔ HÌNH RGCN TRÊN TẬP: {test_file}")
    print(f" Thiết bị: {device} | Model: {dataset_name} | Seed: {seed} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    vocab = get_default_vocab()
    num_relations = vocab.num_relations

    if num_layers == 2:
        model = RGCN2(input_dim, hidden_dim, output_dim, num_relations=num_relations, num_bases=num_bases).to(device)
    elif num_layers == 4:
        model = RGCN4(input_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4, output_dim,
                      num_relations=num_relations, num_bases=num_bases).to(device)
    else:
        hidden_dims = [max(hidden_dim // (2 ** i), output_dim * 2) for i in range(num_layers - 1)]
        model = RelationalGCN(input_dim, hidden_dims, output_dim,
                              num_relations=num_relations, num_bases=num_bases).to(device)

    # Nạp trọng số mô hình
    model_weight_path = f"./model/{dataset_name}_rgcn_model_{seed}.pth"
    if not os.path.exists(model_weight_path):
        fallback_path = f"./model/{dataset_name}_gcn_model_{seed}.pth"
        if os.path.exists(fallback_path):
            print(f"[Lưu ý] Không thấy {model_weight_path}, nhưng tìm thấy {fallback_path}.")
        raise FileNotFoundError(f"Không tìm thấy file checkpoint mô hình RGCN: {model_weight_path}")

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    clean_test_file = test_file[:-4] if test_file.endswith(".pkl") else test_file
    clean_test_file = os.path.basename(clean_test_file)
    graph_file_path = f"./graph_data/{clean_test_file}.pkl"
    if not os.path.exists(graph_file_path):
        raise FileNotFoundError(f"Không tìm thấy file đồ thị tại: {graph_file_path}")

    with open(graph_file_path, "rb") as f:
        test_data = pickle.load(f)

    if "all_edge_type" not in test_data:
        print("[Cảnh báo] File đồ thị không có 'all_edge_type', tự động tạo edge_type=0 (fallback).")
        test_data["all_edge_type"] = [
            torch.zeros(e_idx.size(1), dtype=torch.long) for e_idx in test_data["all_edge_index"]
        ]

    test_len = len(test_data['y'])
    test_dataset = create_pyg_data_list(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    test_loss = 0.0
    correct_predictions = 0
    total_test_samples = 0
    test_pres = []

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating RGCN (Batch={batch_size})"):
            batch = batch.to(device)

            outputs = model(batch)
            bsz = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
            total_test_samples += bsz
            loss = criterion(outputs, batch.y.float().view(-1, 1))
            test_loss += loss.item() * bsz

            test_pres.extend(outputs.view(-1).cpu().tolist())
            prediction = (outputs >= 0.5).long()
            correct_predictions += (prediction == batch.y.view(-1, 1)).sum().item()

    elapsed = time.time() - start_time
    test_loss /= total_test_samples
    test_acc = correct_predictions / total_test_samples
    y_pred = [1 if p >= 0.5 else 0 for p in test_pres]
    y_true = test_data['y'].view(-1, 1).cpu().numpy()

    test_f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, test_pres)
    except Exception:
        auc = 0.0

    print(f"\n[KẾT QUẢ KIỂM THỬ]")
    print(f"  * Thời gian kiểm thử: {elapsed:.2f}s ({elapsed/total_test_samples*1000:.2f} ms/mẫu)")
    print(f"  * Test Loss: {test_loss:.4f}")
    print(f"  * Test Acc:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  * ROC-AUC:   {auc:.4f}")
    print(f"  * F1-score:  {test_f1:.4f}")

    if do_save:
        os.makedirs("./result", exist_ok=True)
        log_line = f"{test_file}\tacc: {test_acc:.4f}\tauc: {auc:.4f}\tf1: {test_f1:.4f}\tseed: {seed}\tmodel: {dataset_name}_rgcn\t{datetime.now()}\n"
        with open("test_result.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
        with open("./result/test_result.txt", "a", encoding="utf-8") as f:
            f.write(log_line)
        print("-> Đã lưu kết quả vào test_result.txt và ./result/test_result.txt")

    return y_pred, test_pres


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kiểm thử mô hình RGCN trên dữ liệu đồ thị")
    parser.add_argument('--dataset', type=str, default='hc3',
                        help="Tên dataset của checkpoint mô hình (mặc định: hc3)")
    parser.add_argument('--seed', type=str, default='2024',
                        help="Random seed (VD: 2024, 2026)")
    parser.add_argument('--file', dest='test_file', type=str, default='hc3_test',
                        help="Tên file đồ thị cần test trong graph_data/ (mặc định: hc3_test)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Kích thước mini-batch khi kiểm thử (mặc định: 32)")
    parser.add_argument('--input_dim', type=int, default=768,
                        help="Số chiều vector nhúng đầu vào (mặc định: 768)")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="Số chiều tầng ẩn RGCN (mặc định: 256)")
    parser.add_argument('--output_dim', type=int, default=64,
                        help="Số chiều tầng ra (mặc định: 64)")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Số tầng RGCN của mô hình (mặc định: 2)")
    parser.add_argument('--num_bases', type=int, default=30,
                        help="Số ma trận cơ sở basis decomposition (mặc định: 30)")
    parser.add_argument('-s', '--save', dest='do_save', action='store_true',
                        help="Lưu kết quả kiểm thử vào file text")
    parser.add_argument('--cpu', action='store_true', help="Chạy trên CPU")

    args = parser.parse_args()
    test_rgcn(
        test_file=args.test_file,
        dataset_name=args.dataset,
        seed=args.seed,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        do_save=args.do_save,
        cpu=args.cpu
    )
