"""Source-only I/O, split isolation and reproducibility metadata."""
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_rows(path, split):
    if split not in ("train", "validation"):
        raise ValueError("Only explicitly declared source train/validation are supported")
    rows = read_jsonl(path)
    id_hashes = {}
    for i, row in enumerate(rows):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Empty/non-text source row {i}")
        row["source_row"] = i
        row["text_hash"] = digest(row["text"])
        row["normalized_hash"] = digest(" ".join(row["text"].split()))
        row["doc_id"] = str(row.get("doc_id", row.get("id", row["text_hash"])))
        if row["doc_id"] in id_hashes and id_hashes[row["doc_id"]] != row["text_hash"]:
            raise ValueError("A document ID refers to multiple source texts")
        id_hashes[row["doc_id"]] = row["text_hash"]
        row["split"] = split
    return rows


def assert_isolated(train, validation):
    """Check complete input partitions before any subsetting or augmentation."""
    for key in ("doc_id", "parent_id", "text_hash", "normalized_hash"):
        a = {str(r[key]) for r in train if r.get(key) is not None}
        b = {str(r[key]) for r in validation if r.get(key) is not None}
        if a & b:
            raise ValueError(f"Split leakage on {key}: {len(a & b)} overlaps")
    a = {str(r[k]) for r in train for k in ("doc_id", "parent_id") if r.get(k) is not None}
    b = {str(r[k]) for r in validation for k in ("doc_id", "parent_id") if r.get(k) is not None}
    if a & b:
        raise ValueError("Split leakage across document/parent IDs")


def select_rows(rows, limit, seed):
    # Hash ordering is independent of Python hash randomization/input row ordering.
    selected = sorted(rows, key=lambda r: digest(f"{seed}:{r['doc_id']}"))
    return selected[:limit] if limit else selected


def quarantine_train_overlap(train, validation):
    """Keep validation intact; quarantine overlapping train rows in derived data only."""
    keys = ("doc_id", "parent_id", "text_hash", "normalized_hash")
    held = {str(r[k]) for r in validation for k in keys if r.get(k) is not None}
    kept, removed = list(train), []
    # Remove entire connected parent/content groups, not only the first matching row.
    while True:
        newly_removed = [r for r in kept if any(str(r[k]) in held for k in keys if r.get(k) is not None)]
        if not newly_removed:
            break
        removed_ids = {id(r) for r in newly_removed}
        kept = [r for r in kept if id(r) not in removed_ids]
        removed.extend(newly_removed)
        held.update(str(r[k]) for r in newly_removed for k in keys if r.get(k) is not None)
    assert_isolated(kept, validation)
    return kept, removed


def human_label(row):
    label = row.get("label", "")
    return int(label) if isinstance(label, (int, float)) else int("human" in str(label).lower())
