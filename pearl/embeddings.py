"""Frozen legacy PRDetect RoBERTa lookup, not a new token/BPE alignment."""
from pathlib import Path
import torch
from .data import file_hash


class LegacyEncoder:
    def __init__(self, cfg):
        from transformers import RobertaModel, RobertaTokenizer
        name = cfg.get("roberta", "roberta-base")
        kw = {"local_files_only": cfg.get("local_files_only", False)}
        if cfg.get("roberta_revision"):
            kw["revision"] = cfg["roberta_revision"]
        self.model = RobertaModel.from_pretrained(name, **kw).to(cfg["device"])
        if Path(name).is_dir():
            self.tokenizer = RobertaTokenizer(str(Path(name) / "vocab.json"), str(Path(name) / "merges.txt"), use_fast=False)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(name, do_lower_case=False, **kw)
        self.model.requires_grad_(False)
        # from_pretrained sets eval mode, as in baseline init_models().
        self.metadata = dict(name=name, revision=getattr(self.model.config, "_commit_hash", None),
                             model_training=self.model.training, hidden=self.model.config.hidden_size,
                             chunk_size=512, token_alignment="legacy-convert_tokens_to_ids")
        if Path(name).is_dir():
            self.metadata["local_hashes"] = {p.name: file_hash(p) for p in Path(name).iterdir()
                                               if p.is_file() and p.suffix in (".json", ".txt", ".bin", ".safetensors")}

    @torch.no_grad()
    def encode(self, graph):
        tokens = [t["text"] for t in graph["tokens"]]
        outputs = []
        for start in range(0, len(tokens), 512):
            ids = self.tokenizer.convert_tokens_to_ids(tokens[start:start + 512])
            tensor = torch.tensor([ids], device=self.model.device)
            outputs.append(self.model(tensor).last_hidden_state[0].cpu())
        if not outputs:
            raise ValueError("No parser nodes")
        return torch.cat(outputs)


def embedding_cache(cfg, docs, split, encoder):
    """One clean cache per partition; exact text/parser hashes checked on reuse."""
    from .data import digest
    path = Path(cfg["prepared"]) / f"{split}_embeddings.pt"
    signature = dict(texts=[d["text_hash"] for d in docs], encoder=encoder.metadata,
                     graph_hashes=[digest(str(d["graph"])) for d in docs])
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache["signature"] != signature:
            raise ValueError("Embedding cache provenance mismatch; use a new prepared directory")
        return cache["embeddings"]
    values = {}
    for i, doc in enumerate(docs):
        values[doc["doc_id"]] = encoder.encode(doc["graph"])
        print(f"{split} RoBERTa {i + 1}/{len(docs)}", flush=True)
    torch.save(dict(signature=signature, embeddings=values), path)
    return values
