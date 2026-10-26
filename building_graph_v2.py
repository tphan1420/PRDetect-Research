"""Build PRDetect V2 document graphs.

A1: align SpaCy words with Transformer subwords using a fast tokenizer.
A2: add reverse dependency edges, sequential edges, sentence nodes and one
    document node so all sentence trees in a document become connected.
A3: store a sequence-level embedding produced by the same frozen encoder for
    the hybrid sequence/graph classifier.
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

from relation_vocab_v2 import (
    DOCUMENT_NODE,
    NUM_RELATIONS,
    RELATION_LABELS,
    SENTENCE_NODE,
    WORD_NODE,
    encode_dependency,
    relation_id,
)


SCHEMA_VERSION = "2.0"
DEFAULT_ENCODER = "roberta-base"


def _configure_console() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def parse_label(value: Any) -> int:
    """Keep the project's convention: Human=1 and AI=0."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 1 if "human" in str(value).lower() else 0


def resolve_source(file_name: str) -> Path:
    """Resolve an explicit JSON path or an unambiguous dataset stem."""

    requested = Path(file_name)
    if requested.is_file():
        return requested.resolve()

    clean_name = requested.stem if requested.suffix == ".json" else requested.name
    candidates: List[Path] = []
    for root in (Path("data/original_text"), Path("data/perturbed_text")):
        if root.exists():
            candidates.extend(root.rglob("{}.json".format(clean_name)))

    unique = sorted({path.resolve() for path in candidates})
    if not unique:
        raise FileNotFoundError(
            "Không tìm thấy '{}'. Hãy truyền đường dẫn JSON hoặc stem duy nhất."
            .format(file_name)
        )
    if len(unique) > 1:
        rendered = "\n".join("  - {}".format(path) for path in unique)
        raise ValueError(
            "Tên '{}' không duy nhất. Hãy truyền đường dẫn đầy đủ:\n{}"
            .format(file_name, rendered)
        )
    return unique[0]


def read_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL data while also accepting a JSON array."""

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        records = [parsed]
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    normalized: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "text" not in record:
            raise ValueError("Bản ghi #{} không có trường 'text'.".format(index))
        normalized.append(record)
    return normalized


class FrozenTextEncoder:
    """Create aligned word features and one sequence feature per document."""

    def __init__(
        self,
        encoder_name: str,
        device: torch.device,
        max_length: int = 512,
        stride: int = 64,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer_kwargs: Dict[str, Any] = {"use_fast": True}
        if "roberta" in encoder_name.lower():
            tokenizer_kwargs["add_prefix_space"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name, **tokenizer_kwargs
        )
        if not self.tokenizer.is_fast:
            raise ValueError(
                "PRDetect V2 yêu cầu fast tokenizer để dùng word_ids()."
            )

        self.model = AutoModel.from_pretrained(encoder_name).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.hidden_size = int(self.model.config.hidden_size)

    def encode_words(
        self, words: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Align subword states to SpaCy words and pool a sequence vector."""

        encoded = self.tokenizer(
            list(words),
            is_split_into_words=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            padding=True,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        amp_enabled = self.device.type == "cuda"
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                hidden = self.model(**model_inputs).last_hidden_state
        hidden = hidden.float()

        num_words = len(words)
        word_sum = torch.zeros(
            (num_words, self.hidden_size), device=self.device, dtype=torch.float32
        )
        word_count = torch.zeros(
            (num_words, 1), device=self.device, dtype=torch.float32
        )
        sequence_chunks: List[torch.Tensor] = []
        unknown_subwords = 0
        visible_subwords = 0

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"].bool()
        special_mask = encoded["special_tokens_mask"].to(self.device).bool()
        unk_token_id = self.tokenizer.unk_token_id

        for chunk_index in range(hidden.size(0)):
            word_ids = encoded.word_ids(batch_index=chunk_index)
            valid_positions = [
                position
                for position, word_id in enumerate(word_ids)
                if word_id is not None and 0 <= word_id < num_words
            ]
            if valid_positions:
                positions = torch.tensor(
                    valid_positions, device=self.device, dtype=torch.long
                )
                mapped_words = torch.tensor(
                    [int(word_ids[position]) for position in valid_positions],
                    device=self.device,
                    dtype=torch.long,
                )
                word_sum.index_add_(0, mapped_words, hidden[chunk_index, positions])
                word_count.index_add_(
                    0,
                    mapped_words,
                    torch.ones(
                        (mapped_words.size(0), 1),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )

            content_mask = attention_mask[chunk_index] & ~special_mask[chunk_index]
            if content_mask.any():
                sequence_chunks.append(hidden[chunk_index, content_mask].mean(dim=0))
                visible_subwords += int(content_mask.sum().item())
                if unk_token_id is not None:
                    unknown_subwords += int(
                        ((input_ids[chunk_index] == unk_token_id) & content_mask)
                        .sum()
                        .item()
                    )

        if not sequence_chunks:
            raise ValueError("Tokenizer không tạo được content token.")

        missing_words = int((word_count.squeeze(-1) == 0).sum().item())
        word_embeddings = word_sum / word_count.clamp(min=1.0)
        sequence_embedding = torch.stack(sequence_chunks, dim=0).mean(dim=0)

        stats = {
            "unk_subword_ratio": unknown_subwords / max(visible_subwords, 1),
            "missing_word_ratio": missing_words / max(num_words, 1),
            "num_windows": float(len(sequence_chunks)),
        }
        return word_embeddings.cpu(), sequence_embedding.cpu(), stats


def _add_edge(
    sources: List[int],
    targets: List[int],
    edge_types: List[int],
    source: int,
    target: int,
    relation: int,
) -> None:
    sources.append(source)
    targets.append(target)
    edge_types.append(relation)


def build_document_graph(
    doc: Any,
    word_embeddings: torch.Tensor,
    sequence_embedding: torch.Tensor,
    encoding_stats: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Build one connected hierarchical document graph."""

    num_words = len(doc)
    sentences = [list(sentence) for sentence in doc.sents]
    if not sentences:
        sentences = [list(doc)]

    sentence_embeddings = []
    for sentence in sentences:
        indices = torch.tensor([token.i for token in sentence], dtype=torch.long)
        sentence_embeddings.append(word_embeddings.index_select(0, indices).mean(dim=0))
    sentence_tensor = torch.stack(sentence_embeddings, dim=0)

    num_sentences = len(sentences)
    sentence_offset = num_words
    document_index = num_words + num_sentences
    document_embedding = sentence_tensor.mean(dim=0, keepdim=True)
    node_embeddings = torch.cat(
        [word_embeddings, sentence_tensor, document_embedding], dim=0
    )
    node_type = torch.tensor(
        [WORD_NODE] * num_words
        + [SENTENCE_NODE] * num_sentences
        + [DOCUMENT_NODE],
        dtype=torch.long,
    )

    sources: List[int] = []
    targets: List[int] = []
    edge_types: List[int] = []
    unknown_dependencies = 0

    for token in doc:
        forward_relation = encode_dependency(token.dep_, reverse=False)
        if RELATION_LABELS[forward_relation] == "<unk>":
            unknown_dependencies += 1
        _add_edge(
            sources,
            targets,
            edge_types,
            token.i,
            token.head.i,
            forward_relation,
        )
        if token.i != token.head.i:
            _add_edge(
                sources,
                targets,
                edge_types,
                token.head.i,
                token.i,
                encode_dependency(token.dep_, reverse=True),
            )

    for sentence_index, sentence in enumerate(sentences):
        sentence_node = sentence_offset + sentence_index
        word_indices = [token.i for token in sentence]
        for word_index in word_indices:
            _add_edge(
                sources,
                targets,
                edge_types,
                word_index,
                sentence_node,
                relation_id("word_to_sentence"),
            )
            _add_edge(
                sources,
                targets,
                edge_types,
                sentence_node,
                word_index,
                relation_id("sentence_to_word"),
            )

        for left, right in zip(word_indices, word_indices[1:]):
            _add_edge(
                sources,
                targets,
                edge_types,
                left,
                right,
                relation_id("next_token"),
            )
            _add_edge(
                sources,
                targets,
                edge_types,
                right,
                left,
                relation_id("prev_token"),
            )

        _add_edge(
            sources,
            targets,
            edge_types,
            sentence_node,
            document_index,
            relation_id("sentence_to_document"),
        )
        _add_edge(
            sources,
            targets,
            edge_types,
            document_index,
            sentence_node,
            relation_id("document_to_sentence"),
        )

    for sentence_index in range(num_sentences - 1):
        left = sentence_offset + sentence_index
        right = left + 1
        _add_edge(
            sources,
            targets,
            edge_types,
            left,
            right,
            relation_id("next_sentence"),
        )
        _add_edge(
            sources,
            targets,
            edge_types,
            right,
            left,
            relation_id("prev_sentence"),
        )

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    quality_features = torch.tensor(
        [
            math.log1p(num_words),
            math.log1p(num_sentences),
            encoding_stats["unk_subword_ratio"],
            encoding_stats["missing_word_ratio"],
            unknown_dependencies / max(num_words, 1),
        ],
        dtype=torch.float32,
    )

    return {
        "x": node_embeddings,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "node_type": node_type,
        "sequence_embedding": sequence_embedding,
        "quality_features": quality_features,
    }


def validate_graph(graph: Dict[str, torch.Tensor]) -> None:
    """Reject malformed graph tensors before they are serialized."""

    x = graph["x"]
    edge_index = graph["edge_index"]
    edge_type = graph["edge_type"]
    node_type = graph["node_type"]
    if x.dim() != 2:
        raise ValueError("x phải có shape [num_nodes, hidden_size].")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index phải có shape [2, num_edges].")
    if edge_type.numel() != edge_index.size(1):
        raise ValueError("edge_type và edge_index không cùng số cạnh.")
    if node_type.numel() != x.size(0):
        raise ValueError("node_type và x không cùng số node.")
    if edge_index.numel() and int(edge_index.max().item()) >= x.size(0):
        raise ValueError("edge_index tham chiếu node ngoài phạm vi.")
    if edge_type.numel() and int(edge_type.max().item()) >= NUM_RELATIONS:
        raise ValueError("edge_type nằm ngoài relation schema V2.")
    if int((node_type == DOCUMENT_NODE).sum().item()) != 1:
        raise ValueError("Mỗi graph V2 phải có đúng một document node.")


def build_graph_file(
    source: Path,
    output_dir: Path,
    encoder_name: str,
    spacy_model: str,
    max_length: int,
    stride: int,
    device: torch.device,
) -> Path:
    import spacy

    records = read_records(source)
    if not records:
        raise ValueError("Dataset rỗng: {}".format(source))

    print("-> SpaCy model: {}".format(spacy_model))
    nlp = spacy.load(spacy_model)
    print("-> Encoder: {} trên {}".format(encoder_name, device))
    encoder = FrozenTextEncoder(
        encoder_name=encoder_name,
        device=device,
        max_length=max_length,
        stride=stride,
    )

    graphs: List[Dict[str, torch.Tensor]] = []
    labels: List[int] = []
    source_indices: List[int] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()

    for index, record in enumerate(tqdm(records, desc="Building V2 graphs")):
        try:
            text = str(record["text"])
            doc = nlp(text)
            words = [token.text for token in doc]
            if not words:
                raise ValueError("Văn bản không tạo ra token SpaCy.")
            word_embeddings, sequence_embedding, stats = encoder.encode_words(words)
            graph = build_document_graph(
                doc, word_embeddings, sequence_embedding, stats
            )
            validate_graph(graph)
            graphs.append(graph)
            labels.append(parse_label(record.get("label", "")))
            source_indices.append(index)
        except Exception as error:
            failures.append({"index": index, "error": repr(error)})
            print("[!] Bỏ qua mẫu #{}: {}".format(index, error))

    if not graphs:
        raise RuntimeError("Không xây dựng được graph V2 hợp lệ nào.")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "encoder_name": encoder_name,
        "encoder_hidden_size": encoder.hidden_size,
        "num_relations": NUM_RELATIONS,
        "relation_labels": list(RELATION_LABELS),
        "node_type_labels": ["word", "sentence", "document"],
        "graphs": graphs,
        "y": torch.tensor(labels, dtype=torch.float32),
        "source_indices": source_indices,
        "failures": failures,
        "source": str(source),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{}.pkl".format(source.stem)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - started
    print(
        "[OK] {} graphs, {} failures, {:.2f}s -> {}".format(
            len(graphs), len(failures), elapsed, output_path
        )
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        "-f",
        nargs="+",
        default=None,
        help="Đường dẫn JSON hoặc dataset stem. Mặc định: HC3 train/val/test.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("graph_data_v2"))
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    _configure_console()
    args = parse_args()
    if args.max_length < 8:
        raise ValueError("--max-length phải >= 8.")
    if args.stride < 0 or args.stride >= args.max_length - 2:
        raise ValueError("--stride phải >= 0 và nhỏ hơn max_length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    files = args.file or ["hc3_train", "hc3_val", "hc3_test"]
    for file_name in files:
        source = resolve_source(file_name)
        print("\n[Processing V2] {}".format(source))
        build_graph_file(
            source=source,
            output_dir=args.output_dir,
            encoder_name=args.encoder,
            spacy_model=args.spacy_model,
            max_length=args.max_length,
            stride=args.stride,
            device=device,
        )


if __name__ == "__main__":
    main()
