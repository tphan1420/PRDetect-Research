import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import pickle
import os
import argparse
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

from model.GCN2 import GCN2
from model.RGCN import RGCN2
from dep_vocab import get_default_vocab


def create_pyg_data_list(graph_dict, model_type='gcn'):
    """Chuyển đổi dữ liệu đồ thị sang danh sách đối tượng Data để đưa vào DataLoader"""
    data_list = []
    n_samples = len(graph_dict['y'])
    all_embeddings = graph_dict['all_token_embeddings']
    all_edge_index = graph_dict['all_edge_index']
    all_edge_type = graph_dict.get('all_edge_type', None)
    all_y = graph_dict['y']

    for i in range(n_samples):
        y_val = all_y[i]
        if not isinstance(y_val, torch.Tensor):
            y_tensor = torch.tensor([y_val], dtype=torch.float)
        elif y_val.dim() == 0:
            y_tensor = y_val.unsqueeze(0).float()
        else:
            y_tensor = y_val.float()

        if model_type == 'rgcn':
            e_type = all_edge_type[i] if all_edge_type is not None else torch.zeros(all_edge_index[i].size(1), dtype=torch.long)
            data_list.append(Data(
                x=all_embeddings[i],
                edge_index=all_edge_index[i],
                edge_type=e_type,
                y=y_tensor
            ))
        else:
            data_list.append(Data(
                x=all_embeddings[i],
                edge_index=all_edge_index[i],
                y=y_tensor
            ))
    return data_list


parser = argparse.ArgumentParser(description="Kiểm thử mô hình PRDetect (GCN / RGCN)")
parser.add_argument('--dataset', type=str, default='hc3')
parser.add_argument('--seed', type=str, default='2024',
                    help="Random seed (VD: 2024, 2026)")
parser.add_argument('--file', dest='test_file', type=str, default='hc3_test')
parser.add_argument('--model_type', choices=['gcn', 'rgcn'], default='gcn',
                    help="Kiểu mô hình: 'gcn' (mặc định) hoặc 'rgcn' (Mục 3 update.md)")
parser.add_argument('--batch_size', type=int, default=32,
                    help="Kích thước mini-batch khi kiểm thử (mặc định: 32)")
parser.add_argument('-s', '--save', dest='do_save', action='store_true')

args = parser.parse_args()


def test(test_file, dataset_name, seed, model_type='gcn', batch_size=32):
    seed_int = int(seed)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 768   # RoBERTa embedding
    hidden_dim = 256  # Hidden layer
    output_dim = 64   # Output dimension

    # Khởi tạo mô hình theo lựa chọn model_type
    if model_type == 'rgcn':
        vocab = get_default_vocab()
        model = RGCN2(input_dim, hidden_dim, output_dim, num_relations=vocab.num_relations).to(device)
        weight_path = f'./model/{dataset_name}_rgcn_model_{seed}.pth'
    else:
        model = GCN2(input_dim, hidden_dim, output_dim).to(device)
        weight_path = f'./model/{dataset_name}_gcn_model_{seed}.pth'

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {weight_path}")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    clean_test_file = test_file[:-4] if test_file.endswith(".pkl") else test_file
    clean_test_file = os.path.basename(clean_test_file)
    graph_file_path = f"./graph_data/{clean_test_file}.pkl"
    if not os.path.exists(graph_file_path):
        raise FileNotFoundError(f"Không tìm thấy file đồ thị: {graph_file_path}")

    with open(graph_file_path, "rb") as f:
        test_data = pickle.load(f)

    if model_type == 'rgcn' and "all_edge_type" not in test_data:
        print("[Cảnh báo] File đồ thị không có 'all_edge_type', tự động bổ sung edge_type=0.")
        test_data["all_edge_type"] = [
            torch.zeros(e_idx.size(1), dtype=torch.long) for e_idx in test_data["all_edge_index"]
        ]

    test_len = len(test_data['y'])
    test_dataset = create_pyg_data_list(test_data, model_type=model_type)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    test_loss = 0.0
    correct_predictions = 0
    total_test_samples = 0
    test_pres = list()

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Test ({model_type.upper()} Batch={batch_size})"):
            batch = batch.to(device)

            outputs = model(batch)
            bsz = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
            total_test_samples += bsz
            loss = criterion(outputs, batch.y.float().view(-1, 1))
            test_loss += loss.item() * bsz

            test_pres.extend(outputs.view(-1).cpu().tolist())
            predictions = (outputs >= 0.5).long()
            correct_predictions += (predictions == batch.y.view(-1, 1)).sum().item()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/total_test_samples*1000:.2f} ms/sample)")

    y_pred = [1 if prob >= 0.5 else 0 for prob in test_pres]
    y_true = test_data['y'].view(-1, 1).cpu().numpy()
    test_loss /= total_test_samples
    test_acc = correct_predictions / total_test_samples
    test_f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, test_pres)

    print(f"Model: {model_type.upper()} | test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}, auc: {auc:.4f}")

    if args.do_save:
        os.makedirs("./result", exist_ok=True)
        log_line = f"{test_file}\t acc: {test_acc:.4f}\t auc: {auc:.4f}\t f1: {test_f1:.4f}\t seed: {seed}\t model: {dataset_name}_{model_type}\t{datetime.now()}\n"
        with open("./result/test_result.txt", "a", encoding="utf-8") as w:
            w.write(log_line)
        with open("test_result.txt", "a", encoding="utf-8") as w:
            w.write(log_line)

    return y_pred


if __name__ == "__main__":
    print(f"Testing: file={args.test_file}, dataset={args.dataset}, seed={args.seed}, model={args.model_type}, batch_size={args.batch_size}")
    test(args.test_file, args.dataset, args.seed, args.model_type, batch_size=args.batch_size)