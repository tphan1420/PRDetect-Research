import sys
import argparse
import json
from pathlib import Path

import torch

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

ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT_DIR / "model"
INPUT_DIM = 768
HIDDEN_DIM = 256
OUTPUT_DIM = 64


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Dự đoán văn bản Human hay Machine bằng PRDetect (GCN / RGCN)"
    )
    parser.add_argument("--text", required=True, help="Đoạn văn bản cần kiểm tra")
    parser.add_argument("--dataset", default="hc3")
    parser.add_argument("--seed", default="2024")
    parser.add_argument(
        "--model_type",
        choices=("rgcn", "gcn"),
        default="rgcn",
        help="Kiểu mô hình suy luận: rgcn hoặc gcn",
    )
    return parser.parse_args()


def _checkpoint_path(dataset, seed, model_type):
    return CHECKPOINT_DIR / f"{dataset}_{model_type}_model_{seed}.pth"


def _load_model(dataset, seed, model_type, device):
    """Nạp checkpoint và giữ fallback RGCN -> GCN như code cũ."""
    requested_type = model_type
    model_path = _checkpoint_path(dataset, seed, model_type)

    if model_type == "rgcn" and not model_path.exists():
        fallback = _checkpoint_path(dataset, seed, "gcn")
        if fallback.exists():
            print(
                f"[Lưu ý] Không tìm thấy checkpoint RGCN ({model_path}), "
                f"chuyển sang dùng GCN ({fallback})."
            )
            model_type = "gcn"
            model_path = fallback

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint {requested_type.upper()} tại: {model_path}"
        )

    if model_type == "rgcn":
        vocab = get_default_vocab()
        model = RGCN2(
            INPUT_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            num_relations=vocab.num_relations,
        )
    else:
        model = GCN2(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, model_type


def _build_input_data(text, model_type, device):
    json_texts = [json.dumps({"text": text, "label": ""})]

    if model_type == "rgcn":
        embeddings, edge_indices, edge_types, labels = build_graph(
            json_texts, return_edge_type=True
        )
    else:
        embeddings, edge_indices, labels = build_graph(
            json_texts, return_edge_type=False
        )
        edge_types = None

    if not embeddings or not edge_indices:
        raise ValueError("Không thể xây dựng đồ thị từ văn bản đầu vào.")

    data_kwargs = {
        "x": embeddings[0].to(device),
        "edge_index": edge_indices[0].to(device),
        "y": labels.to(device),
    }
    if edge_types is not None:
        data_kwargs["edge_type"] = edge_types[0].to(device)
    return Data(**data_kwargs)


def detect(text, dataset="hc3", seed="2024", model_type="rgcn"):
    if model_type not in {"gcn", "rgcn"}:
        raise ValueError("model_type phải là 'gcn' hoặc 'rgcn'.")

    seed_int = int(seed)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)
        torch.cuda.manual_seed_all(seed_int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, actual_model_type = _load_model(dataset, seed, model_type, device)
    data = _build_input_data(text, actual_model_type, device)

    with torch.no_grad():
        outputs = model(data)
        probability = float(outputs.item())
        prediction = int(probability >= 0.5)

    return probability, prediction


def main():
    args = _parse_args()
    prob, pred = detect(
        args.text,
        dataset=args.dataset,
        seed=args.seed,
        model_type=args.model_type,
    )
    label_name = (
        "Human-written (Người viết)"
        if pred == 1
        else "AI/LLM-generated (Máy sinh)"
    )
    actual_model_type = args.model_type
    if (
        args.model_type == "rgcn"
        and not _checkpoint_path(args.dataset, args.seed, "rgcn").exists()
        and _checkpoint_path(args.dataset, args.seed, "gcn").exists()
    ):
        actual_model_type = "gcn"
    print("\n[KẾT QUẢ DỰ ĐOÁN]")
    print(f"  * Mô hình: {actual_model_type.upper()}")
    print(f"  * Xác suất Human: {prob:.4f} ({prob * 100:.2f}%)")
    print(f"  * Nhãn dự đoán: {pred} -> {label_name}")


if __name__ == "__main__":
    main()
