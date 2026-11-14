"""Classify raw text with a trusted F1 checkpoint, without training."""

import argparse

import torch

from .data import to_data_list
from .graph_builder import build_records
from .model import PrismF1GCN


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if state.get("variant") != "F1":
        raise ValueError("Expected an F1 checkpoint")
    if args.cpu:
        import building_graph
        building_graph.init_models()
        building_graph.device = torch.device("cpu")
        building_graph.model.to("cpu")
    cache = build_records([{"text": args.text, "label": 0}])
    if state["preprocessing_sha256"] != cache["preprocessing_sha256"]:
        raise ValueError("Checkpoint and inference preprocessing fingerprints differ")
    config = state["config"]
    model = PrismF1GCN(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"],
                       output_dim=config["output_dim"]).to(device).eval()
    model.load_state_dict(state["model_state"])
    with torch.no_grad():
        probability = model(to_data_list(cache)[0].to(device)).item()
    report = {"p_human": probability, "label": "human" if probability >= .5 else "ai", "threshold": .5}
    print(report)
    return report


if __name__ == "__main__":
    main()
