"""Prepare isolated source partitions and parser-only retention pairs."""
import gzip
import io
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from .alignment import align_tokens
from .config import config_cli, load_config
from .data import (assert_isolated, digest, file_hash, human_label, quarantine_train_overlap,
                   select_rows, source_rows, write_json)
from .edit_provenance import DEFAULT_SYNONYMS, DEFAULT_WORDS, FAMILIES, generate_edit
from .graph import parse_graph
from .relation_retention import label_arcs


def write_records(path, records):
    # mtime=0 and empty gzip filename make byte output deterministic too.
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as out:
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(pairs, docs):
    groups = defaultdict(Counter)
    for p in pairs:
        d = docs[p["source_doc_id"]]
        keys = ["all", f"family:{p['edit_family']}", f"class:{d['label']}",
                f"ratio:{p['edit_ratio']}", f"length:{len(d['graph']['tokens']) // 128 * 128}"]
        for key in keys:
            groups[key]["pairs"] += 1
            groups[key]["source_tokens"] += len(d["graph"]["tokens"])
            groups[key]["aligned_tokens"] += len(p["alignment"])
            groups[key]["no_op_pairs"] += int(not p["edit"]["changed_text"])
        for arc in p["arc_labels"]:
            for key in keys + [f"relation:{arc['relation']}"]:
                groups[key][arc["status"]] += 1
    result = {}
    for key, count in groups.items():
        n = sum(count[s] for s in ("stable", "changed", "unknown"))
        result[key] = dict(count, token_alignment_rate=count["aligned_tokens"] / max(1, count["source_tokens"]),
                           edge_rates={s: count[s] / max(1, n) for s in ("stable", "changed", "unknown")},
                           negative_given_valid=count["changed"] / max(1, count["stable"] + count["changed"]))
    return result


def prepare(cfg):
    import spacy
    start = time.perf_counter()
    out = Path(cfg["prepared"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite prepared experiment: {out}")
    train = source_rows(cfg["source_train"], "train")
    val = source_rows(cfg["source_validation"], "validation")
    removed = []
    if cfg.get("overlap_policy", "error") == "quarantine_train":
        train, removed = quarantine_train_overlap(train, val)
    else:
        assert_isolated(train, val)
    nlp = spacy.load(cfg["parser"])
    synonyms = cfg.get("synonyms", DEFAULT_SYNONYMS)
    words = cfg.get("replacement_words", DEFAULT_WORDS)
    generator = dict(version="span-edit-v1", families=cfg.get("families", list(FAMILIES)),
                     ratios=cfg.get("ratios", [0.1]), synonyms=synonyms, words=words)
    generator_hash = digest(json.dumps(generator, sort_keys=True))
    manifest = dict(config=cfg, generator=generator, generator_hash=generator_hash,
                    spacy_version=spacy.__version__, parser_meta=nlp.meta,
                    parser_hash=digest(nlp.to_bytes().hex()), python=platform.python_version(),
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    raw_sha256={k: file_hash(cfg[k]) for k in ("source_train", "source_validation")},
                    quarantined_train=[dict(doc_id=r["doc_id"], source_row=r["source_row"], text_hash=r["text_hash"])
                                       for r in removed], splits={}, errors=[])
    # Hash actual implementation, including uncommitted files.
    manifest["code_sha256"] = {str(p): file_hash(p) for p in sorted(Path("pearl").glob("*.py"))}
    for split, rows, limit in (("train", train, cfg.get("train_limit")),
                               ("validation", val, cfg.get("validation_limit"))):
        selected = select_rows(rows, limit, cfg["seed"])
        docs, pairs = {}, []
        for row in selected:
            if row["doc_id"] in docs:
                continue  # repeated exact content IDs count once, never separate groups
            graph = parse_graph(nlp, row["text"])
            doc = dict(doc_id=row["doc_id"], parent_id=row.get("parent_id"), source_row=row["source_row"],
                       text=row["text"], text_hash=row["text_hash"], normalized_hash=row["normalized_hash"],
                       label=human_label(row), split=split, graph=graph)
            docs[doc["doc_id"]] = doc
            for family in generator["families"]:
                for ratio in generator["ratios"]:
                    seed = int(digest(f"{cfg['seed']}:{split}:{doc['doc_id']}:{family}:{ratio}")[:16], 16)
                    edit = generate_edit(row["text"], graph, family, ratio, seed, synonyms, words)
                    try:
                        edited = parse_graph(nlp, edit["text"])
                    except ValueError as exc:
                        manifest["errors"].append(dict(doc_id=doc["doc_id"], family=family, ratio=ratio, error=str(exc)))
                        continue
                    mapping, reasons = align_tokens(graph, edited, edit["provenance"])
                    pairs.append(dict(source_doc_id=doc["doc_id"], split=split, view_id="clean",
                                      next_view_id=f"{family}-{ratio}", edit_family=family, edit_ratio=ratio,
                                      edit_seed=seed, edit=edit, edited_graph=edited,
                                      alignment=mapping, alignment_reasons=reasons,
                                      arc_labels=label_arcs(graph, edited, mapping, reasons)))
        write_records(out / f"{split}_docs.jsonl.gz", docs.values())
        write_records(out / f"{split}_pairs.jsonl.gz", pairs)
        stats = summarize(pairs, docs)
        manifest["splits"][split] = dict(available_rows=len(rows), selected_documents=len(docs), pairs=len(pairs),
                                         selected_ids=list(docs), statistics=stats)
        print(f"{split}: {len(docs)} documents, {len(pairs)} pairs", flush=True)
    manifest["seconds"] = time.perf_counter() - start
    manifest["files_sha256"] = {p.name: file_hash(p) for p in sorted(out.glob("*.gz"))}
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    args = config_cli("Prepare PEARL source-only pairs").parse_args()
    prepare(load_config(args.config))
