"""Provenance checks for frozen-encoder graphs (no ML imports required)."""

import hashlib
import json
from pathlib import Path


PIPELINE = 'frozen_word_mean_v1'


def fingerprint(configuration):
    payload = json.dumps(configuration, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def check_graph_pair(train, validation, input_dim, num_relations):
    """Reject mixed encoders/alignment; preserve support for legacy graph pairs."""
    left, right = train.get('encoder_metadata'), validation.get('encoder_metadata')
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise ValueError('Cannot mix legacy and aligned encoder graph caches.')
    for metadata in (left, right):
        config = metadata['configuration']
        if config.get('pipeline') != PIPELINE:
            raise ValueError('Unsupported aligned encoder graph pipeline.')
        if metadata['fingerprint'] != fingerprint(config):
            raise ValueError('Invalid encoder graph metadata fingerprint.')
        if config['hidden_size'] != input_dim:
            raise ValueError('Graph hidden size does not match --input_dim.')
        if config['num_relations'] != num_relations:
            raise ValueError('Graph relation vocabulary size does not match model.')
    if left['fingerprint'] != right['fingerprint']:
        raise ValueError('Train and validation must use the same encoder and graph pipeline.')
    return left


def checkpoint_manifest_path(checkpoint):
    # Appending, rather than replacing a suffix, cannot collide with weights.
    return Path(str(checkpoint) + '.encoder.json')


def check_checkpoint_graph(checkpoint, graph, input_dim, num_relations):
    metadata = check_graph_pair(graph, graph, input_dim, num_relations)
    sidecar = checkpoint_manifest_path(checkpoint)
    if not sidecar.exists():
        if metadata is not None:
            raise ValueError('Aligned graph requires its trained checkpoint .encoder.json sidecar. '
                             'Do not reuse a legacy RoBERTa checkpoint.')
        return
    saved = json.loads(sidecar.read_text(encoding='utf-8'))
    expected = saved.get('encoder_metadata')
    if (expected is None) != (metadata is None):
        raise ValueError('Checkpoint and graph use different embedding pipelines.')
    if expected is not None and expected['fingerprint'] != metadata['fingerprint']:
        raise ValueError('Checkpoint and test graph encoder/alignment do not match.')


def pack_subwords(input_ids, word_ids, words, unk_token_id):
    """Retain every node; SentencePiece may omit whitespace-only spaCy tokens."""
    if len(input_ids) != len(word_ids):
        raise ValueError('Token IDs and word IDs have different lengths.')
    grouped = [[] for _ in words]
    previous = -1
    for token_id, word_id in zip(input_ids, word_ids):
        if word_id is None or not 0 <= word_id < len(words) or word_id < previous:
            raise ValueError('Tokenizer returned invalid/non-monotonic word alignment.')
        grouped[word_id].append(int(token_id))
        previous = word_id
    flattened, mapping = [], []
    whitespace_fallbacks = 0
    for index, (word, pieces) in enumerate(zip(words, grouped)):
        if not pieces:
            if word.strip() or unk_token_id is None:
                raise ValueError(f'No subwords for non-whitespace node {index}: {word!r}')
            pieces = [unk_token_id]
            whitespace_fallbacks += 1
        flattened.extend(pieces)
        mapping.extend([index] * len(pieces))
    return flattened, mapping, whitespace_fallbacks
