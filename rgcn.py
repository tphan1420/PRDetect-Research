"""
Script huấn luyện mô hình Mạng Tích chập Đồ thị Quan hệ (RGCN)
dành cho bài toán Nhận diện Văn bản AI (PRDetect).

Mục 3 trong update.md:
Chuyển đổi hình thái đồ thị từ đồng nhất (homogeneous) sang đa quan hệ (RGCN),
sử dụng các ma trận W_r đặc thù cho từng loại quan hệ cú pháp phụ thuộc.
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
import torch.optim as optim
import torch.nn.functional as F
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
        # Fallback pure-PyTorch Data, Batch, DataLoader khi chưa có torch_geometric
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


def load_graph_data(file_path):
    """Nạp dữ liệu đồ thị từ file .pkl và kiểm tra tính toàn vẹn của edge_type"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu đồ thị tại: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if "all_edge_type" not in data:
        print(f"[Cảnh báo] File {file_path} không có trường 'all_edge_type'. Tự động khởi tạo quan hệ mặc định (0).")
        data["all_edge_type"] = [
            torch.zeros(e_idx.size(1), dtype=torch.long) for e_idx in data["all_edge_index"]
        ]

    return data


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


def train_rgcn(args):
    # Thiết lập Random Seed
    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\n=======================================================")
    print(f" HUẤN LUYỆN MÔ HÌNH RGCN (RELATIONAL GCN)")
    print(f" Thiết bị: {device} | Dataset: {args.dataset} | Seed: {seed}")
    print(f"=======================================================\n")

    vocab = get_default_vocab()
    num_relations = args.num_relations or vocab.num_relations
    print(f"-> Số loại quan hệ cú pháp (num_relations): {num_relations}")

    train_path = f"./graph_data/{args.train_file}.pkl"
    val_path = f"./graph_data/{args.val_file}.pkl"

    print(f"-> Đang nạp dữ liệu huấn luyện: {train_path}")
    train_data = load_graph_data(train_path)
    print(f"-> Đang nạp dữ liệu kiểm định: {val_path}")
    val_data = load_graph_data(val_path)

    train_len = len(train_data['y'])
    val_len = len(val_data['y'])
    print(f"-> Số mẫu Train: {train_len} | Số mẫu Val: {val_len}")

    print(f"-> Đang chuẩn bị DataLoader Mini-Batching (batch_size={args.batch_size}) ...")
    train_dataset = create_pyg_data_list(train_data)
    val_dataset = create_pyg_data_list(val_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"-> Số mini-batch Train: {len(train_loader)} | Số mini-batch Val: {len(val_loader)}")

    # Khởi tạo mô hình RGCN
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_bases = args.num_bases

    if args.num_layers == 2:
        model = RGCN2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            dropout=args.dropout
        ).to(device)
    elif args.num_layers == 4:
        model = RGCN4(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_dim2=hidden_dim // 2,
            hidden_dim3=hidden_dim // 4,
            output_dim=output_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            dropout=args.dropout
        ).to(device)
    else:
        # Kiến trúc nhiều tầng tùy biến
        hidden_dims = [max(hidden_dim // (2 ** i), output_dim * 2) for i in range(args.num_layers - 1)]
        model = RelationalGCN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            dropout=args.dropout
        ).to(device)

    print(f"-> Kiến trúc mô hình RGCN ({args.num_layers} tầng):")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # Khởi tạo TensorBoard nếu có thư viện
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = f"logs/{args.dataset}_rgcn_{seed}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)
        print(f"-> Tensorboard log: {log_dir}")
    except Exception:
        pass

    epochs = args.epochs
    val_max_acc = -1.0
    best_epoch = 0
    patience_counter = args.patience
    os.makedirs("./model", exist_ok=True)
    model_save_path = args.save_path or f"./model/{args.dataset}_rgcn_model_{seed}.pth"

    start_time = time.time()

    for epoch in range(epochs):
        # 1. Quá trình Huấn luyện (Training) với Mini-Batching
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            bsz = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
            total_train_samples += bsz
            loss_val = loss.item()
            epoch_loss += loss_val * bsz

            prediction = (outputs >= 0.5).long()
            correct = (prediction == batch.y.view(-1, 1)).sum().item()
            correct_predictions += correct

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{correct_predictions / total_train_samples:.4f}"})

        train_loss = epoch_loss / total_train_samples
        train_acc = correct_predictions / total_train_samples

        # 2. Quá trình Kiểm định (Validation) với Mini-Batching
        model.eval()
        val_epoch_loss = 0.0
        val_correct = 0
        total_val_samples = 0
        all_val_pres = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                outputs = model(batch)
                loss = criterion(outputs, batch.y.float().view(-1, 1))
                bsz = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
                total_val_samples += bsz
                val_epoch_loss += loss.item() * bsz

                all_val_pres.extend(outputs.view(-1).cpu().tolist())
                prediction = (outputs >= 0.5).long()
                val_correct += (prediction == batch.y.view(-1, 1)).sum().item()

        val_loss = val_epoch_loss / total_val_samples
        val_acc = val_correct / total_val_samples

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)

        # Kiểm tra lưu checkpoint tốt nhất
        if val_acc >= val_max_acc:
            val_max_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = args.patience
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> [LƯU MÔ HÌNH TỐT NHẤT] Epoch {best_epoch} - Val Acc: {val_max_acc:.4f} -> {model_save_path}")
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                print(f"[DỪNG SỚM - Early Stopping] Không cải thiện sau {args.patience} epochs.")
                break

    total_time = time.time() - start_time
    print(f"\n[Hoàn thành huấn luyện] Tổng thời gian: {total_time:.2f}s | Val Acc cao nhất: {val_max_acc:.4f} (Epoch {best_epoch})")
    if writer:
        writer.close()

    # 3. Đánh giá ngay trên tập kiểm thử Test nếu có
    if not args.no_test:
        test_path = f"./graph_data/{args.test_file}.pkl"
        if os.path.exists(test_path):
            print(f"\n--> Bắt đầu kiểm thử trên tập: {args.test_file} ...")
            evaluate_rgcn(model, model_save_path, test_path, device, args)
        else:
            print(f"[Lưu ý] Không tìm thấy file {test_path} để tự động kiểm thử.")


def evaluate_rgcn(model, model_path, test_file_path, device, args):
    """Đánh giá mô hình RGCN trên tập test với Mini-Batching và ghi kết quả vào test_result.txt"""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_data = load_graph_data(test_file_path)
    test_len = len(test_data['y'])
    test_dataset = create_pyg_data_list(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loss = 0.0
    correct_predictions = 0
    total_test_samples = 0
    test_pres = []
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing RGCN (Batch={args.batch_size})"):
            batch = batch.to(device)

            outputs = model(batch)
            bsz = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
            total_test_samples += bsz
            loss = criterion(outputs, batch.y.float().view(-1, 1))
            test_loss += loss.item() * bsz

            test_pres.extend(outputs.view(-1).cpu().tolist())
            prediction = (outputs >= 0.5).long()
            correct_predictions += (prediction == batch.y.view(-1, 1)).sum().item()

    test_loss /= total_test_samples
    test_acc = correct_predictions / total_test_samples
    y_pred = [1 if p >= 0.5 else 0 for p in test_pres]
    y_true = test_data['y'].view(-1, 1).cpu().numpy()

    test_f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, test_pres)
    except Exception:
        auc = 0.0

    print(f"\n[KẾT QUẢ KIỂM THỬ RGCN - {args.test_file}]")
    print(f"  * Loss: {test_loss:.4f}")
    print(f"  * Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  * ROC-AUC:  {auc:.4f}")
    print(f"  * F1-score: {test_f1:.4f}")

    os.makedirs("./result", exist_ok=True)
    log_line = f"{args.test_file}\tacc: {test_acc:.4f}\tauc: {auc:.4f}\tf1: {test_f1:.4f}\tseed: {args.seed}\tmodel: {args.dataset}_rgcn_{args.num_layers}layer\t{datetime.now()}\n"
    
    with open("test_result.txt", "a", encoding="utf-8") as f:
        f.write(log_line)
    with open("./result/test_result.txt", "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"-> Đã ghi nhật ký kết quả vào test_result.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Relational GCN (RGCN) cho PRDetect")
    parser.add_argument('--dataset', type=str, default='hc3',
                        help="Tên bộ dữ liệu huấn luyện (mặc định: hc3)")
    parser.add_argument('--train_file', type=str, default='hc3_train',
                        help="Tên file đồ thị train trong graph_data/ (mặc định: hc3_train)")
    parser.add_argument('--val_file', type=str, default='hc3_val',
                        help="Tên file đồ thị val trong graph_data/ (mặc định: hc3_val)")
    parser.add_argument('--test_file', type=str, default='hc3_test',
                        help="Tên file đồ thị test trong graph_data/ (mặc định: hc3_test)")
    parser.add_argument('--seed', type=str, default='2024',
                        help="Random seed (mặc định: 2024)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Kích thước mini-batch (mặc định: 32, tối ưu tốc độ huấn luyện trên GPU)")
    parser.add_argument('--epochs', type=int, default=40,
                        help="Số lượng epoch huấn luyện (mặc định: 40)")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Tốc độ học (learning rate, mặc định: 0.0001)")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Trọng số điều chuẩn L2 (mặc định: 1e-5)")
    parser.add_argument('--input_dim', type=int, default=768,
                        help="Số chiều vector nhúng đầu vào (mặc định: 768 từ RoBERTa)")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="Số chiều tầng ẩn RGCN (mặc định: 256)")
    parser.add_argument('--output_dim', type=int, default=64,
                        help="Số chiều tầng ra trước phân loại (mặc định: 64)")
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3, 4, 5, 6],
                        help="Số tầng RGCN (mặc định: 2)")
    parser.add_argument('--num_bases', type=int, default=30,
                        help="Số lượng ma trận cơ sở (basis decomposition) để tránh quá khớp (mặc định: 30)")
    parser.add_argument('--num_relations', type=int, default=None,
                        help="Số loại quan hệ (mặc định tự lấy từ dep_vocab)")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Tỷ lệ dropout (mặc định: 0.5)")
    parser.add_argument('--patience', type=int, default=5,
                        help="Số epoch kiên nhẫn trước khi dừng sớm (mặc định: 5)")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Đường dẫn lưu trọng số mô hình tốt nhất")
    parser.add_argument('--no_test', action='store_true',
                        help="Không tự động chạy test sau khi huấn luyện xong")
    parser.add_argument('--cpu', action='store_true',
                        help="Bắt buộc sử dụng CPU thay vì GPU")

    args = parser.parse_args()
    train_rgcn(args)
