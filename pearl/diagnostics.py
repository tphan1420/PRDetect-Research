"""Parity against actual unmodified builder/model using real cached embeddings."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .config import config_cli, load_config
from .data import file_hash, read_jsonl, write_json
from .embeddings import LegacyEncoder
from .graph import parse_graph, tensor_edges
from .train_edge import load_edge
from .weighted_gcn import PEARLGCN, baseline_from_state


def run(cfg, output):
    import spacy
    import building_graph as original
    nlp = spacy.load(cfg["parser"])
    encoder = LegacyEncoder(cfg)
    original.nlp, original.model = nlp, encoder.model
    original.tokenizer, original.device = encoder.tokenizer, encoder.model.device
    docs = read_jsonl(Path(cfg["prepared"]) / "validation_docs.jsonl.gz")[:3]
    texts = [d["text"] for d in docs] + ["Hello.", "The doctor helps the doctor.", "First sentence. Another follows."]
    rows = [json.dumps(dict(text=t, label="human")) for t in texts]
    xs, edges, _ = original.build_graph(rows, return_edge_type=False)
    base = baseline_from_state(torch.load(cfg["baseline_checkpoint"], map_location="cpu", weights_only=True))
    predictor, features, _ = load_edge(Path(cfg["edge_output"]) / "edge.pt")
    model = PEARLGCN(base, predictor, features, epsilon=1).eval()
    report = []
    with torch.no_grad():
        for text, reference_x, reference_edges in zip(texts, xs, edges):
            graph = parse_graph(nlp, text)
            x = encoder.encode(graph)
            index = torch.tensor(tensor_edges(graph)[0]).T
            torch.testing.assert_close(x, reference_x, atol=1e-6, rtol=1e-5)
            assert torch.equal(index, reference_edges)
            baseline_p = base(SimpleNamespace(x=x, edge_index=index))
            pearl_p = model(x, graph)
            torch.testing.assert_close(pearl_p, baseline_p, atol=1e-6, rtol=1e-5)
            report.append(dict(nodes=len(x), embedding_max_abs=float((x - reference_x).abs().max()),
                               probability_abs=float((pearl_p - baseline_p).abs().max()),
                               edge_index_exact=True))
    integrity_path = Path("output/pearl/baseline_integrity.json")
    integrity = None
    if integrity_path.exists():
        initial = json.loads(integrity_path.read_text())
        integrity = {p: file_hash(p) == value for p, value in initial.items()}
        assert all(integrity.values())
    result = dict(parity=report, atol=1e-6, rtol=1e-5, encoder=encoder.metadata,
                  baseline_integrity=integrity,
                  note="Parity under this installed parser/encoder; historical cache revision remains unknown.")
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = config_cli("Baseline embedding/topology/epsilon=1 parity")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.output)
