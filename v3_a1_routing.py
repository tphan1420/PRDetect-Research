"""Select A1 or V3 for one English text using a corruption detector fitted on HC3."""

from collections import Counter
import hashlib
import math
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE | re.ASCII)
_FEATURES = ("mean_word_surprisal", "unknown_fraction", "mean_bigram_association", "unseen_pair_fraction",
             "mean_context_surprisal", "open_class_fraction")


def _contexts(tokens, common):
    classes = ["<boundary>", *(w if w in common else "<other>" for w in tokens), "<boundary>"]
    return zip(classes, classes[1:], classes[2:])


def _tokens(text):
    return _WORD.findall(text.lower().replace("’", "'"))


def _text_key(text):
    return hashlib.sha256(" ".join(text.split()).encode()).digest()


def _features(tokens, reference):
    words, pairs, total, common, centers, windows, triples = reference
    if not tokens:
        return (math.log(total + 1), 1.0, 0.0, 1.0, 0.0, 1.0)
    counts = [words[w] for w in tokens]
    rarity = math.log(total + 1) - sum(math.log1p(c) for c in counts) / len(tokens)
    unknown = counts.count(0) / len(tokens)
    association = 0.0
    unseen = 0
    # Compare each transition to its unigram background with bigram backoff.
    for i, pair in enumerate(zip(tokens, tokens[1:])):
        hits = pairs[pair]
        unseen += hits == 0
        left, right = counts[i], counts[i + 1]
        for context, target in ((left, right), (right, left)):
            association += math.log1p(hits * total / (10 * (target + 1))) - math.log1p(context / 10)
    n_pairs = max(1, len(tokens) - 1)
    surprise = 0.0
    for left, center, right in _contexts(tokens, common):
        prior = (centers[center] + 1) / (total + len(centers) + 1)
        likelihood = (triples[left, center, right] + 10 * prior) / (windows[left, right] + 10)
        surprise -= math.log(likelihood)
    other = sum(w not in common for w in tokens)
    return rarity, unknown, association / (2 * n_pairs), unseen / n_pairs, surprise / len(tokens), other / len(tokens)


def _replace(tokens, vocabulary, rng):
    changed = list(tokens)
    if changed:
        n = max(1, round(len(tokens) * rng.uniform(0.10, 0.35)))
        positions = rng.choice(len(tokens), size=n, replace=False)
        for idx, donor in zip(positions, rng.integers(len(vocabulary), size=n)):
            changed[idx] = vocabulary[donor]
    return changed


def fit_router(train_texts, validation_texts):
    """Training text supplies reference counts and synthetic edits. Validation sets the cutoff."""
    train_texts = list(train_texts)
    keys = {_text_key(text) for text in train_texts}
    words, pairs = Counter(), Counter()
    examples, reference_docs = [], []
    for text in train_texts:
        tokens = _tokens(text)
        if int.from_bytes(_text_key(text)[:4], "big") % 5 == 0:
            examples.append(tokens)
        else:
            words.update(tokens)
            pairs.update(zip(tokens, tokens[1:]))
            reference_docs.append(tokens)
    common = {w for w, _ in words.most_common(256)}
    centers, windows, triples = Counter(), Counter(), Counter()
    for tokens in reference_docs:
        for left, center, right in _contexts(tokens, common):
            centers[center] += 1
            windows[left, right] += 1
            triples[left, center, right] += 1
    reference = (words, pairs, sum(words.values()), common, centers, windows, triples)
    donors = sorted({word for tokens in examples for word in tokens})
    if not donors or not reference[2]:
        raise ValueError("Need documents in both parts of the source training split")
    rng = np.random.default_rng(42)
    pristine = [_features(tokens, reference) for tokens in examples]
    edited = [_features(_replace(tokens, donors, rng), reference) for tokens in examples]
    detector = make_pipeline(StandardScaler(), LogisticRegression())
    detector.fit(pristine + edited, [0] * len(pristine) + [1] * len(edited))
    validation = [text for text in validation_texts if _text_key(text) not in keys]
    if not validation:
        raise ValueError("Need clean source validation documents separate from training")
    clean_scores = detector.decision_function([_features(_tokens(t), reference) for t in validation])
    cutoff = float(np.quantile(clean_scores, 0.95, method="higher"))
    edited_scores = detector.decision_function([
        _features(_replace(_tokens(t), donors, rng), reference) for t in validation])
    return {"reference": reference, "detector": detector, "cutoff": cutoff,
            "source_validation": {"documents": len(validation), "features": list(_FEATURES),
                                  "clean_routed_to_v3": float(np.mean(clean_scores > cutoff)),
                                  "synthetic_routed_to_v3": float(np.mean(edited_scores > cutoff))}}


def use_v3(router, texts):
    features = [_features(_tokens(t), router["reference"]) for t in texts]
    if not features:
        return np.empty(0, dtype=bool)
    return router["detector"].decision_function(features) > router["cutoff"]


def predict_human(route_to_v3, v3_p_human, a1_logit_human, human95_threshold):
    mask = np.asarray(route_to_v3, dtype=bool)
    p, z = np.asarray(v3_p_human, dtype=float), np.asarray(a1_logit_human, dtype=float)
    if (mask.shape != p.shape or z.shape != p.shape or not np.isfinite(z).all()
            or not np.all((p >= 0) & (p <= 1)) or not 0 <= human95_threshold <= 1):
        raise ValueError("Expected aligned decisions, probabilities, logits and a valid V3 threshold")
    return np.where(mask, p >= human95_threshold, z >= 0)
