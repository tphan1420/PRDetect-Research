"""Single text inference: no clean reference, edit log or paired input."""
from pathlib import Path
import torch
from .config import config_cli, load_config
from .embeddings import LegacyEncoder
from .evaluate import verify_runtime
from .graph import parse_graph
from .train_gcn import load_checkpoint


if __name__ == "__main__":
    import spacy
    import json
    parser = config_cli("Single-view PEARL inference")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, metadata = load_checkpoint(Path(cfg["gcn_output"]) / "best.pt", cfg["device"])
    nlp, encoder = spacy.load(cfg["parser"]), LegacyEncoder(cfg)
    verify_runtime(cfg, metadata, nlp, encoder)
    graph = parse_graph(nlp, args.text)
    with torch.no_grad():
        p, _, logits, weights = model(encoder.encode(graph).to(cfg["device"]), graph, True)
    print(json.dumps(dict(p_human=float(p.item()), prediction="Human" if p.item() >= .5 else "AI",
                          q=logits.sigmoid().cpu().tolist(), weights=weights.cpu().tolist())))
