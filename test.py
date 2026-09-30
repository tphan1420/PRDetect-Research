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
    from torch_geometric.nn import GCNConv
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

from model.GCN2 import GCN2
from model.RGCN import RGCN2
from dep_vocab import get_default_vocab

parser = argparse.ArgumentParser(description="Kiểm thử mô hình PRDetect (GCN / RGCN)")
parser.add_argument('--dataset', type=str, default='hc3')
parser.add_argument('--seed', type=str, default='2024',
                    help="Random seed (VD: 2024, 2026)")
parser.add_argument('--file', dest='test_file', type=str, default='hc3_test')
parser.add_argument('--model_type', choices=['gcn', 'rgcn'], default='gcn',
                    help="Kiểu mô hình: 'gcn' (mặc định) hoặc 'rgcn' (Mục 3 update.md)")
parser.add_argument('-s', '--save', dest='do_save', action='store_true')

args = parser.parse_args()


def test(test_file, dataset_name, seed, model_type='gcn'):
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

    graph_file_path = f"./graph_data/{test_file}.pkl"
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
    criterion = nn.BCELoss()
    test_loss = 0.0
    correct_predictions = 0
    test_pres = list()

    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(test_len), desc=f"Test ({model_type.upper()})"):
            if model_type == 'rgcn':
                data = Data(
                    x=test_data['all_token_embeddings'][i],
                    edge_index=test_data['all_edge_index'][i],
                    edge_type=test_data['all_edge_type'][i],
                    y=test_data['y'][i]
                ).to(device)
            else:
                data = Data(
                    x=test_data['all_token_embeddings'][i],
                    edge_index=test_data['all_edge_index'][i],
                    y=test_data['y'][i]
                ).to(device)

            outputs = model(data)
            test_pres.append(outputs.item())
            loss = criterion(outputs, data.y.float().view(-1, 1))
            test_loss += loss.item()
            predictions = (outputs >= 0.5).long()
            correct_predictions += (predictions == data.y.view(-1, 1)).sum().item()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    y_pred = [1 if prob >= 0.5 else 0 for prob in test_pres]
    y_true = test_data['y'].view(-1, 1).cpu().numpy()
    test_loss /= test_len
    test_acc = correct_predictions / test_len
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
    print(f"Testing: file={args.test_file}, dataset={args.dataset}, seed={args.seed}, model={args.model_type}")
    test(args.test_file, args.dataset, args.seed, args.model_type)