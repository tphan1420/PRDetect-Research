"""
Script huấn luyện mô hình GATv2 cho PRDetect.
Thay thế GCNConv bằng GATv2Conv với cơ chế chú ý động
để chống nhiễu đối kháng (adversarial robustness).

Sử dụng:
    python train_gatv2.py --dataset hc3 --seed 2024 --heads 4
    python train_gatv2.py --dataset gpt3.5 --seed 2024 --epochs 60 --lr 0.0003
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import argparse

from model.GATv2 import GATv2Model

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình GATv2 trên dữ liệu đồ thị cú pháp PRDetect"
    )
    parser.add_argument('--dataset', default='hc3',
                        help='Tên dataset (hc3 hoặc gpt3.5)')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (mặc định 0.0001 giống gcn.py)')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--heads', type=int, default=4,
                        help='Số attention heads ở layer 1')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension mỗi head (output layer 1 = hidden_dim * heads)')
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (số epoch liên tiếp không cải thiện)')
    return parser.parse_args()


def train():
    args = parse_args()

    # ---- Seed & Device ----
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] dataset={args.dataset}, seed={seed}, device={device}")
    print(f"[Config] heads={args.heads}, hidden_dim={args.hidden_dim}, "
          f"output_dim={args.output_dim}, lr={args.lr}, dropout={args.dropout}")

    # ---- Load data (dùng lại .pkl hiện có — KHÔNG cần rebuild) ----
    dataset_name = args.dataset
    train_pkl = f"./graph_data/{dataset_name}_train.pkl"
    val_pkl = f"./graph_data/{dataset_name}_val.pkl"

    if not os.path.exists(train_pkl):
        print(f"[ERROR] Không tìm thấy {train_pkl}. Hãy chạy building_graph.py trước.")
        return
    if not os.path.exists(val_pkl):
        print(f"[ERROR] Không tìm thấy {val_pkl}. Hãy chạy building_graph.py trước.")
        return

    print(f"[Data] Loading {train_pkl} ...")
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    print(f"[Data] Loading {val_pkl} ...")
    with open(val_pkl, "rb") as f:
        val_data = pickle.load(f)

    train_len = len(train_data['y'])
    val_len = len(val_data['y'])
    print(f"[Data] Train: {train_len} mẫu, Val: {val_len} mẫu")

    # ---- Model ----
    input_dim = 768  # RoBERTa embedding dimension
    model = GATv2Model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] GATv2Model — Total params: {total_params:,}, Trainable: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # ---- TensorBoard ----
    log_dir = f'logs/gatv2_{dataset_name}_{seed}_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"[Log] TensorBoard logs → {log_dir}")

    # ---- Training Loop ----
    val_max_acc = -1
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    start_time = time.time()
    for epoch in range(args.epochs):
        # ===== Training =====
        model.train()
        epoch_loss = 0.0
        correct = 0

        for i in tqdm(range(train_len), f"Epoch {epoch+1}/{args.epochs} [Train]"):
            data = Data(
                x=train_data['all_token_embeddings'][i],
                edge_index=train_data['all_edge_index'][i],
                y=train_data['y'][i]
            ).to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (outputs >= 0.5).long()
            correct += (preds == data.y.view(-1, 1)).sum().item()

        train_loss = epoch_loss / train_len
        train_acc = correct / train_len
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)

        # ===== Validation =====
        model.eval()
        epoch_loss = 0.0
        correct = 0

        with torch.no_grad():
            for i in tqdm(range(val_len), f"Epoch {epoch+1}/{args.epochs} [Val]"):
                data = Data(
                    x=val_data['all_token_embeddings'][i],
                    edge_index=val_data['all_edge_index'][i],
                    y=val_data['y'][i]
                ).to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y.float().view(-1, 1))
                epoch_loss += loss.item()
                preds = (outputs >= 0.5).long()
                correct += (preds == data.y.view(-1, 1)).sum().item()

        val_loss = epoch_loss / val_len
        val_acc = correct / val_len
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # ===== Save best + Early Stopping =====
        if val_acc >= val_max_acc:
            val_max_acc = val_acc
            patience_counter = 0
            os.makedirs("./model", exist_ok=True)
            save_path = f'./model/gatv2_{dataset_name}_model_{seed}.pth'
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), f'./model/{dataset_name}_gatv2_model_{seed}.pth')
            print(f"  ✓ Best model saved → {save_path} (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val_acc: {val_max_acc:.4f}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Model saved: ./model/gatv2_{dataset_name}_model_{seed}.pth")
    print(f"{'='*60}")

    writer.close()


if __name__ == '__main__':
    train()
