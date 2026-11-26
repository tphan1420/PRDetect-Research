"""Score text with the HC3 A1 checkpoint and combine it with V3 predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from evaluate_v3_quang import REPO, score_probabilities, sha256, write_json


def source_digest(path):
    # Kaggle's LF checkout and Windows' CRLF checkout contain the same records.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def human_score(v3_p_human, a1_logit_human):
    """A soft AND. The positive class (human) requires support from both views."""
    p = np.asarray(v3_p_human, dtype=np.float64)
    z = np.asarray(a1_logit_human, dtype=np.float64)
    if (p.shape != z.shape or not np.isfinite(p).all() or not np.isfinite(z).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError("Expected aligned finite V3 probabilities and A1 human logits")
    return p * expit(z)


def read_source(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def labels_from(rows):
    labels = []
    for row in rows:
        value = row["label"]
        if isinstance(value, str):
            if "human" in value.lower():
                value = 1
            elif any(name in value.lower() for name in ("gpt", "ai", "machine")):
                value = 0
        if value not in (0, 1):
            raise ValueError(f"Unknown binary label {value!r}")
        labels.append(int(value))
    return np.asarray(labels)


def paired_scores(source, labels, v3, a1):
    if (v3["split"] != source.stem or a1["condition"] != source.stem
            or a1["source_sha256"] not in {source_digest(source), sha256(source)}
            or a1["score_space"] != "human_logit" or v3["train"] != "hc3"
            or v3["label_convention"] != {"0": "ai", "1": "human"}):
        raise ValueError(f"Score provenance differs from source {source.name}")
    records = a1["records"]
    if (v3["samples"] != len(labels) or len(records) != len(labels)
            or [r["index"] for r in records] != list(range(len(labels)))
            or not np.array_equal(labels, v3["y_true"])
            or not np.array_equal(labels, [r["label"] for r in records])):
        raise ValueError(f"Sample order or labels differ in {source.name}")
    return human_score(v3["p_human"], [r["logit_human"] for r in records])


class AlignedScorer:
    """The original A1 aligned RoBERTa and V1 dependency graph, streamed by text."""

    def __init__(self, checkpoint, device):
        import spacy
        import torch
        from transformers import AutoModel, AutoTokenizer
        from dep_vocab import get_default_vocab

        self.device = torch.device(device)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.vocab = get_default_vocab()
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False).to(device).eval()
        if checkpoint is None:
            return
        from model.RGCN import RGCN2
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        relations, bases = state["conv1.comp"].shape
        self.graph = RGCN2(input_dim=state["conv1.root"].shape[0], hidden_dim=state["conv1.root"].shape[1],
                           output_dim=state["fc.weight"].shape[1], num_relations=relations, num_bases=bases)
        self.graph.load_state_dict(state, strict=True)
        self.graph.to(device).eval()
        # At inference the same basis weights are expanded for every document.
        # Materialize them once with the identical PyG basis product.
        with torch.no_grad():
            for layer in (self.graph.conv1, self.graph.conv2):
                basis = layer.weight
                dense = (layer.comp @ basis.flatten(1)).view(relations, *basis.shape[1:])
                layer.weight = torch.nn.Parameter(dense, requires_grad=False)
                layer.num_bases = None

    def score_doc(self, doc):
        import torch
        if not len(doc):
            raise ValueError("Cannot classify an empty document")
        tokens = self.tokenizer([w.text for w in doc], is_split_into_words=True, max_length=512,
                                stride=64, truncation=True, return_overflowing_tokens=True,
                                padding=True, return_tensors="pt")
        with torch.inference_mode():
            sums = torch.zeros((len(doc), self.encoder.config.hidden_size), device=self.device)
            counts = torch.zeros(len(doc), device=self.device)
            for i in range(len(tokens["input_ids"])):
                # Padding contributes neither to attention nor to word pooling.
                size = int(tokens["attention_mask"][i].sum())
                ids = tokens["input_ids"][i:i + 1, :size].to(self.device)
                mask = tokens["attention_mask"][i:i + 1, :size].to(self.device)
                with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                    hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[0]
                word_ids = tokens.word_ids(i)[:size]
                positions = [j for j, word in enumerate(word_ids) if word is not None]
                words = torch.tensor([word_ids[j] for j in positions], device=self.device)
                sums.index_add_(0, words, hidden[positions].float())
                counts.index_add_(0, words, torch.ones_like(words, dtype=sums.dtype))
            x = sums / counts.clamp_min(1).unsqueeze(1)
            edges = torch.tensor([[w.i for w in doc], [w.head.i for w in doc]], device=self.device)
            rel = torch.tensor([self.vocab.encode(w.dep_) for w in doc], device=self.device)
            h = self.graph.conv1(x, edges, rel).relu()
            h = self.graph.conv2(h, edges, rel).relu()
            return self.graph.fc(h).mean().item()

    def v3_probability(self, doc, model):
        """Use the V3 checkpoint's original word IDs, edge schema and FP16 cache rounding."""
        import torch
        from model.QuangV3 import struct_features
        words = [w.text for w in doc]
        if not words:
            raise ValueError("Cannot classify an empty document")
        src = np.arange(len(doc))
        dst = np.array([w.head.i for w in doc])
        rel = np.array([self.vocab.encode(w.dep_) for w in doc]).clip(max=model.R - 1)
        structure, _ = struct_features(len(doc), src, dst, rel, model.R, model.mode)
        with torch.inference_mode():
            chunks = []
            for start in range(0, len(words), 512):
                ids = self.tokenizer.convert_tokens_to_ids(words[start:start + 512])
                inputs = torch.tensor([ids], device=self.device)
                chunks.append(self.encoder(inputs).last_hidden_state[0].half().float())
            batch = {"x": torch.cat(chunks), "src": torch.as_tensor(src, device=self.device),
                     "dst": torch.as_tensor(dst, device=self.device), "rel": torch.as_tensor(rel, device=self.device),
                     "st": torch.as_tensor(structure, dtype=torch.long, device=self.device),
                     "B": 1, "bidx": torch.zeros(len(words), dtype=torch.long, device=self.device)}
            return model(batch).sigmoid().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score-a1")
    score.add_argument("--sources", type=Path, nargs="+", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, default=REPO / "model/hc3_a1_only_rgcn_model_2026.pth")
    score.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict = commands.add_parser("predict", help="Classify a new English document with a single V3 seed and A1")
    predict.add_argument("--text", required=True)
    predict.add_argument("--v3-checkpoint", required=True, type=Path)
    predict.add_argument("--rule", choices=("fusion", "human95", "routed"), default="fusion")
    predict.add_argument("--router", type=Path, help="Fitted hc3_router.joblib for the routed rule")
    predict.add_argument("--human95-dir", type=Path, help="Source thresholds for checkpoints created before human95")
    predict.add_argument("--checkpoint", type=Path, default=REPO / "model/hc3_a1_only_rgcn_model_2026.pth")
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    fuse = commands.add_parser("fuse")
    for name in ("source", "a1", "v3", "output"):
        fuse.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fuse":
        labels = labels_from(read_source(args.source))
        v3, a1 = [json.loads(p.read_bytes()) for p in (args.v3, args.a1)]
        p = paired_scores(args.source, labels, v3, a1)
        write_json(args.output, {"split": args.source.stem, "threshold": 0.5, "seed": v3["seed"],
                                 "a1_checkpoint": a1["checkpoint_sha256"], "v3_checkpoint": v3["checkpoint_sha256"],
                                 **score_probabilities(labels, p, 0.5), "human_score": p.tolist()})
        return
    if args.command == "predict":
        if args.rule == "routed":
            import joblib
            from v3_a1_routing import use_v3
            if args.router is None:
                parser.error("--rule routed requires --router")
            if not use_v3(joblib.load(args.router), [args.text])[0]:
                scorer = AlignedScorer(args.checkpoint, args.device)
                logit = scorer.score_doc(scorer.nlp(args.text))
                print(json.dumps({"rule": "routed", "branch": "a1", "human_score": float(expit(logit)),
                                  "threshold": 0.5, "a1_checkpoint_sha256": sha256(args.checkpoint),
                                  "label": "human" if logit >= 0 else "ai"}))
                return
        from calibrate_v3 import load_v3_checkpoint
        model, ckpt, key = load_v3_checkpoint(args.v3_checkpoint)
        if args.rule == "routed" and ckpt.get("weight_source") == "ema":
            raise ValueError("The routed method uses the V3 RAW checkpoint")
        scorer = AlignedScorer(args.checkpoint if args.rule == "fusion" else None, args.device)
        doc = scorer.nlp(args.text)
        v3 = scorer.v3_probability(doc, model.to(scorer.device))
        if args.rule in ("human95", "routed"):
            threshold = ckpt.get("human95_threshold")
            if args.human95_dir:
                fitted = json.loads((args.human95_dir / f"{key}.json").read_bytes())
                if fitted["checkpoint_sha256"] != key:
                    raise ValueError("Mismatched source calibration")
                threshold = fitted["human95_threshold"]
            if threshold is None or not 0 <= threshold <= 1:
                raise ValueError("Fit and supply --human95-dir for this checkpoint")
            result = {"human_score": v3, "threshold": threshold}
            if args.rule == "routed":
                result["branch"] = "v3_raw"
        else:
            a1 = scorer.score_doc(doc)
            result = {"human_score": float(human_score(v3, a1)), "threshold": 0.5, "a1_p_human": float(expit(a1))}
        result.update(rule=args.rule, v3_p_human=v3, v3_checkpoint_sha256=key,
                      label="human" if result["human_score"] >= result["threshold"] else "ai")
        print(json.dumps(result))
        return
    scorer = AlignedScorer(args.checkpoint, args.device)
    digest = sha256(args.checkpoint)
    cache = {}
    for source in args.sources:
        rows = read_source(source)
        texts = [r["text"] for r in rows]
        run = {"condition": source.stem, "source_sha256": source_digest(source),
               "checkpoint_sha256": digest, "score_space": "human_logit"}
        partial = args.output_dir / f"scores_{source.stem}.partial.json"
        signature = {**run, "scorer_sha256": sha256(Path(__file__)), "device": args.device}
        completed = {}
        if partial.exists():
            saved = json.loads(partial.read_bytes())
            if saved["signature"] != signature:
                raise ValueError(f"Partial scores used a different scorer or source: {partial}")
            completed = saved["logits"]
            for i, value in completed.items():
                cache[texts[int(i)]] = value
        missing = dict.fromkeys(t for t in texts if t not in cache)
        indices = {t: i for i, t in enumerate(texts)}
        for n, doc in enumerate(scorer.nlp.pipe(missing, batch_size=16), start=1):
            cache[doc.text] = scorer.score_doc(doc)
            completed[str(indices[doc.text])] = cache[doc.text]
            if n % 32 == 0:
                write_json(partial, {"signature": signature, "logits": completed})
                print(f"{source.stem}: {n}/{len(missing)} new texts", flush=True)
        records = [{"index": i, "label": row.get("label"), "logit_human": cache[texts[i]]}
                   for i, row in enumerate(rows)]
        if all(r["label"] is not None for r in records):
            for r, label in zip(records, labels_from(rows)):
                r["label"] = int(label)
        write_json(args.output_dir / f"scores_{source.stem}.json", {**run, "records": records})
        partial.unlink(missing_ok=True)
        print(f"{source.stem}: {len(rows)} documents", flush=True)


if __name__ == "__main__":
    main()
