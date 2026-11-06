"""Frozen DeBERTa-v3 / aligned RoBERTa graph builder; legacy builder is untouched."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random

from encoder_graph import PIPELINE, fingerprint, pack_subwords


ENCODERS = {
    'deberta': 'microsoft/deberta-v3-base',
    'roberta': 'roberta-base',
}


def resolve_source(name):
    direct = Path(name)
    # A split name may contain a dot (gpt3.5_*). Only remove a literal
    # .json suffix; Path.stem/with_suffix would otherwise truncate at .5_*.
    json_path = direct if direct.name.endswith('.json') else Path(str(direct) + '.json')
    for candidate in (direct, json_path):
        if candidate.is_file():
            return candidate.resolve()
    stem = direct.name[:-5] if direct.name.endswith('.json') else direct.name
    found = sorted(Path('data/original_text').rglob(stem + '.json'))
    found += sorted(Path('data/perturbed_text').rglob(stem + '.json'))
    if len(found) != 1:
        raise ValueError(f'Expected one source for {name!r}, found {len(found)}; pass its full path.')
    return found[0].resolve()


def parse_label(value):
    # Same class convention as V1: AI=0, human=1, but reject unknown labels.
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    label = str(value).strip().lower()
    if 'human' in label:
        return 1
    if label in {'0', 'ai', 'machine', 'chatgpt', 'hc3chatgpt', 'gpt', 'generated', 'ai-generated'}:
        return 0
    if label == '1':
        return 1
    raise ValueError(f'Unsupported binary label: {value!r}')


class FrozenWordEncoder:
    def __init__(self, encoder, revision, max_length, device):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.name = ENCODERS[encoder]
        options = {'use_fast': True, 'revision': revision}
        if encoder == 'roberta':
            options['add_prefix_space'] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **options)
        if not self.tokenizer.is_fast:
            raise ValueError('A fast tokenizer with word_ids() is required.')
        self.model = AutoModel.from_pretrained(self.name, revision=revision).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.max_length = min(max_length, self.tokenizer.model_max_length, 512)
        if self.max_length <= 2:
            raise ValueError('max_length must leave room for CLS, SEP and content.')
        self.configuration = {
            'pipeline': PIPELINE, 'encoder': self.name,
            'revision': getattr(self.model.config, '_commit_hash', None) or revision,
            'tokenizer_revision': self.tokenizer.init_kwargs.get('_commit_hash', revision),
            'tokenizer_class': type(self.tokenizer).__name__,
            'transformers_version': transformers.__version__,
            'hidden_size': self.model.config.hidden_size,
            'max_length': self.max_length, 'pooling': 'mean_subwords',
            'chunking': 'nonoverlapping_subwords_full_coverage',
            'dtype': 'float32', 'frozen': True, 'eval': True,
            'whitespace_node_policy': 'unk_if_tokenizer_omits',
            'edge_direction': 'dependent_to_head_with_root_self_loop',
        }

    def encode(self, words):
        torch = self.torch
        if not words:
            raise ValueError('Empty document; no graph nodes.')
        encoded = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=False,
            truncation=False, return_attention_mask=False, return_token_type_ids=False,
        )
        ids, node_ids, fallback_count = pack_subwords(
            encoded['input_ids'], encoded.word_ids(), words, self.tokenizer.unk_token_id,
        )
        # Accumulate on CPU; never retain all document activations on GPU.
        sums = torch.zeros((len(words), self.model.config.hidden_size), dtype=torch.float32)
        counts = torch.zeros(len(words), dtype=torch.float32)
        capacity = self.max_length - 2
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(ids), capacity):
                content = ids[start:start + capacity]
                model_ids = self.tokenizer.build_inputs_with_special_tokens(content)
                # Both supported checkpoints use a single leading CLS/trailing SEP.
                if len(model_ids) != len(content) + 2 or model_ids[1:-1] != content:
                    raise ValueError('Unexpected special-token template; alignment would be invalid.')
                inputs = torch.tensor([model_ids], dtype=torch.long, device=self.device)
                hidden = self.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).last_hidden_state
                vectors = hidden[0, 1:-1].float().cpu()
                indices = torch.tensor(node_ids[start:start + capacity], dtype=torch.long)
                sums.index_add_(0, indices, vectors)
                counts.index_add_(0, indices, torch.ones(len(content)))
                del hidden, vectors, inputs
        if (counts == 0).any():
            raise ValueError('Some graph nodes have no embedding.')
        embeddings = sums / counts.unsqueeze(1)
        if not torch.isfinite(embeddings).all():
            raise ValueError('Encoder produced non-finite embeddings.')
        return embeddings, {
            'subwords': len(ids), 'unknown_subwords': sum(t == self.tokenizer.unk_token_id for t in ids),
            'whitespace_fallback_nodes': fallback_count,
        }


def build_one(source, target, encoder, nlp, vocab, args):
    import torch
    from tqdm import tqdm

    features, edges, relations, labels, source_lines = [], [], [], [], []
    totals = {'nodes': 0, 'subwords': 0, 'unknown_subwords': 0, 'whitespace_fallback_nodes': 0}
    source_digest = hashlib.sha256()
    with source.open('rb') as stream:
        for line_number, raw_line in enumerate(tqdm(stream, desc=source.stem), 1):
            source_digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                text = record['text']
                if not isinstance(text, str):
                    raise ValueError('text must be a string')
                label = parse_label(record['label'])
                doc = nlp(text)
                embeddings, stats = encoder.encode([token.text for token in doc])
                edge_index = torch.tensor([[token.i for token in doc],
                                           [token.head.i for token in doc]], dtype=torch.long)
                edge_type = torch.tensor([vocab.encode(token.dep_) for token in doc], dtype=torch.long)
                if edge_index.shape != (2, len(doc)) or embeddings.size(0) != len(doc):
                    raise ValueError('Node/edge alignment mismatch')
            except Exception as error:
                # Never silently drop different samples for different encoders.
                raise RuntimeError(f'{source}:{line_number}: {error}') from error
            features.append(embeddings)
            edges.append(edge_index)
            relations.append(edge_type)
            labels.append(label)
            source_lines.append(line_number)
            totals['nodes'] += len(doc)
            for key, value in stats.items():
                totals[key] += value
    if not labels:
        raise ValueError(f'No samples in {source}')
    configuration = dict(encoder.configuration)
    configuration.update({
        'spacy_version': __import__('spacy').__version__,
        'spacy_model': args.spacy_model, 'spacy_model_version': nlp.meta.get('version'),
        'num_relations': vocab.num_relations, 'dependency_vocabulary': dict(vocab.dep2id),
    })
    metadata = {
        'configuration': configuration, 'fingerprint': fingerprint(configuration),
        'source': str(source), 'source_sha256': source_digest.hexdigest(),
        'source_lines': source_lines, 'samples': len(labels), 'statistics': totals,
        'build_seed': args.seed,
    }
    data = {'all_token_embeddings': features, 'all_edge_index': edges,
            'all_edge_type': relations, 'y': torch.tensor(labels, dtype=torch.float32),
            'encoder_metadata': metadata}
    temporary = target.with_suffix('.pkl.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    target.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {target}: {len(labels)} samples; {totals}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', nargs='+', required=True)
    parser.add_argument('--encoder', choices=ENCODERS, default='deberta')
    parser.add_argument('--revision', default='main', help='HF revision; pin a commit for reproducibility')
    parser.add_argument('--graph-root', type=Path, default=Path('graph_data'))
    parser.add_argument('--max-graphs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--max-length', type=int, default=512, help='Total subword window length, 3..512')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not 3 <= args.max_length <= 512:
        parser.error('--max-length must be between 3 and 512')
    root = args.graph_root.resolve()
    output = root / (args.encoder + '_aligned')
    output.mkdir(parents=True, exist_ok=True)
    sources = [resolve_source(name) for name in args.file]
    jobs = [(source, output / (source.stem + '.pkl')) for source in sources]
    if len({target for _, target in jobs}) != len(jobs):
        raise ValueError('Duplicate graph output names.')
    for _, target in jobs:
        if target.exists() and not args.overwrite:
            raise FileExistsError(f'{target}: use --overwrite explicitly or remove the cache yourself.')
    existing = set(root.rglob('*.pkl'))
    pending = {target for _, target in jobs} - existing
    if len(existing) + len(pending) > args.max_graphs:
        raise RuntimeError(f'Graph limit {args.max_graphs}: {len(existing)} existing + '
                           f'{len(pending)} new. Remove finished caches before building more.')
    import torch
    import spacy
    from dep_vocab import get_default_vocab

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    nlp = spacy.load(args.spacy_model)
    encoder = FrozenWordEncoder(args.encoder, args.revision, args.max_length,
                                'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    vocab = get_default_vocab()
    for source, target in jobs:
        build_one(source, target, encoder, nlp, vocab, args)


if __name__ == '__main__':
    main()
