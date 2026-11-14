"""Inference for a LISA-GCN checkpoint; no labels or GroupDRO weights required."""

import argparse
import torch

from lisa_gcn.data import build_document
from lisa_gcn.model import LisaGCN
from train_lisa_gcn import document_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--window-chunk-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    model = LisaGCN(hidden_dim=config["hidden_dim"], output_dim=config["output_dim"], dropout=config["dropout"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    document = build_document(args.text, label=0.0, document_index=0)
    with torch.no_grad():
        logit = document_logits(model, [document], 128, device, args.window_chunk_size)
        probability = torch.sigmoid(logit).item()
    print({"p_human": probability, "label": "human" if probability >= 0.5 else "ai"})


if __name__ == "__main__":
    main()
