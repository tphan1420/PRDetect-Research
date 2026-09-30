"""
Script đánh giá mô hình GATv2 trên các tập test.

Sử dụng:
    python test_gatv2.py --file hc3_test --dataset hc3 --seed 2024 -s
    python test_gatv2.py --file hc3_test_adj_30_synonym_replace --dataset hc3 --seed 2024 -s
    python test_gatv2.py --file raid_gpt2_test --dataset hc3 --seed 2024 -s
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import time
import os
import sys
import argparse
from datetime import datetime

from model.GATv2 import GATv2Model

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình GATv2 trên tập test"
    )
    parser.add_argument('--dataset', default='hc3',
                        help='Tên dataset đã train (hc3 hoặc gpt3.5)')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--file', dest='test_file', type=str, required=True,
                        help='Tên file test trong graph_data/ (không cần đuôi .pkl)')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('-s', '--save', dest='do_save', action='store_true',
                        help='Lưu kết quả vào result/test_result_gatv2.txt')
    return parser.parse_args()


def test():
    args = parse_args()

    # ---- Setup ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load test data ----
    test_pkl = f"./graph_data/{args.test_file}.pkl"
    if not os.path.exists(test_pkl):
        print(f"[ERROR] Không tìm thấy {test_pkl}. Hãy chạy building_graph.py trước.")
        return
    
    print(f"[Data] Loading {test_pkl} ...")
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
    test_len = len(test_data['y'])
    print(f"[Data] Test: {test_len} mẫu")

    # ---- Load model ----
    input_dim = 768
    candidate_paths = [
        f'./model/gatv2_{args.dataset}_model_{args.seed}.pth',
        f'./model/{args.dataset}_gatv2_model_{args.seed}.pth',
        f'/kaggle/working/gatv2_{args.dataset}_model_{args.seed}.pth',
        f'/kaggle/working/{args.dataset}_gatv2_model_{args.seed}.pth'
    ]
    model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if model_path is None:
        print(f"[ERROR] Không tìm thấy model checkpoint cho dataset={args.dataset}, seed={args.seed}.")
        print(f"        Đã tìm kiếm tại: {candidate_paths}")
        print(f"        Hãy chạy train_gatv2.py trước.")
        return
    
    model = GATv2Model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Model] Loaded {model_path}")

    # ---- Evaluate ----
    criterion = nn.BCELoss()
    test_loss = 0.0
    correct = 0
    test_preds = []

    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(test_len), "Testing"):
            data = Data(
                x=test_data['all_token_embeddings'][i],
                edge_index=test_data['all_edge_index'][i],
                y=test_data['y'][i]
            ).to(device)
            outputs = model(data)
            test_preds.append(outputs.item())
            loss = criterion(outputs, data.y.float().view(-1, 1))
            test_loss += loss.item()
            preds = (outputs >= 0.5).long()
            correct += (preds == data.y.view(-1, 1)).sum().item()

    elapsed = time.time() - start_time

    # ---- Metrics ----
    y_pred = [1 if p >= 0.5 else 0 for p in test_preds]
    y_true = test_data['y'].view(-1, 1)
    test_loss /= test_len
    test_acc = correct / test_len
    test_f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(test_data['y'], test_preds)

    print(f"\n{'='*60}")
    print(f"  Test file : {args.test_file}")
    print(f"  Model     : gatv2_{args.dataset} (seed={args.seed}, heads={args.heads})")
    print(f"  ────────────────────────────────")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Time      : {elapsed:.1f}s")
    print(f"{'='*60}")

    # ---- Save ----
    if args.do_save:
        os.makedirs("./result", exist_ok=True)
        result_line = (
            f"{args.test_file}\t"
            f"acc: {test_acc}\t"
            f"auc: {auc}\t"
            f"f1: {test_f1}\t"
            f"seed: {args.seed}\t"
            f"model: gatv2_{args.dataset}\t"
            f"heads: {args.heads}\t"
            f"{datetime.now()}\n"
        )
        with open("./result/test_result_gatv2.txt", "a", encoding="utf-8") as w:
            w.write(result_line)
        print(f"[Saved] → result/test_result_gatv2.txt")

    return test_preds


if __name__ == "__main__":
    test()
