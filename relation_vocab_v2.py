"""Immutable relation schema for PRDetect V2 document graphs.

V1 may extend ``dep_vocab.json`` while building reverse edges. V2 uses a
deterministic schema so train/validation/test graphs always share relation IDs.
"""

from typing import Dict, Tuple

from dep_vocab import CANONICAL_DEP_LABELS


FORWARD_DEPENDENCIES: Tuple[str, ...] = tuple(CANONICAL_DEP_LABELS)
REVERSE_DEPENDENCIES: Tuple[str, ...] = tuple(
    "{}_rev".format(label) for label in FORWARD_DEPENDENCIES
)
STRUCTURAL_RELATIONS: Tuple[str, ...] = (
    "next_token",
    "prev_token",
    "word_to_sentence",
    "sentence_to_word",
    "next_sentence",
    "prev_sentence",
    "sentence_to_document",
    "document_to_sentence",
)

RELATION_LABELS: Tuple[str, ...] = (
    FORWARD_DEPENDENCIES + REVERSE_DEPENDENCIES + STRUCTURAL_RELATIONS
)
RELATION_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(RELATION_LABELS)
}
NUM_RELATIONS = len(RELATION_LABELS)

WORD_NODE = 0
SENTENCE_NODE = 1
DOCUMENT_NODE = 2


def encode_dependency(label: str, reverse: bool = False) -> int:
    """Return a stable ID for a SpaCy dependency label."""

    clean = label.strip()
    if clean not in RELATION_TO_ID:
        clean = clean.lower()
    if clean not in RELATION_TO_ID:
        clean = "<unk>"
    if reverse:
        clean = "{}_rev".format(clean)
    return RELATION_TO_ID[clean]


def relation_id(label: str) -> int:
    """Return the ID of a V2 structural relation."""

    return RELATION_TO_ID[label]
