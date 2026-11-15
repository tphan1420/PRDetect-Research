"""Edits preserve character gaps and log source/destination spans and token UIDs."""
import math
import random
from .data import digest

FAMILIES = ("synonym", "replace", "insert", "repeat", "delete")
# Small, explicit pilot lexicon. Not the official WordNet benchmark generator.
DEFAULT_SYNONYMS = {
    "good": ["fine"], "bad": ["poor"], "big": ["large"], "small": ["little"],
    "important": ["significant"], "different": ["distinct"], "many": ["numerous"],
    "new": ["novel"], "old": ["aged"], "happy": ["glad"], "quick": ["rapid"],
    "large": ["big"], "difficult": ["hard"], "easy": ["simple"], "help": ["assist"],
    "use": ["utilize"], "make": ["create"], "people": ["persons"], "also": ["additionally"],
    "often": ["frequently"], "very": ["extremely"], "possible": ["feasible"],
    "likely": ["probable"], "careful": ["cautious"], "doctor": ["physician"],
}
DEFAULT_WORDS = ["however", "stone", "quietly", "blue", "because", "table", "perhaps", "walk"]


def generate_edit(text, graph, family, ratio, seed, synonyms=None, words=None):
    if family not in (*FAMILIES, "identity") or not 0 <= ratio <= 1:
        raise ValueError("Invalid family or ratio")
    synonyms = DEFAULT_SYNONYMS if synonyms is None else synonyms
    words = DEFAULT_WORDS if words is None else words
    if not words or any(not isinstance(w, str) or not w.strip() for w in words):
        raise ValueError("Replacement/insertion vocabulary must be nonempty")
    rng = random.Random(seed)
    eligible = [t for t in graph["tokens"] if t["text"].isalpha()
                and (family != "synonym" or synonyms.get(t["text"].lower()))]
    # Ratio is relative to eligible lexical tokens; actual count is recorded.
    count = min(len(eligible), math.ceil(ratio * len(eligible))) if family != "identity" else 0
    selected = {t["i"] for t in rng.sample(eligible, count)}
    parts, log, cursor, length = [], [], 0, 0

    def append(value):
        nonlocal length
        start = length
        parts.append(value)
        length += len(value)
        return [start, length]

    for token in graph["tokens"]:
        append(text[cursor:token["start"]])
        uid = f"t{token['i']}"
        hit = token["i"] in selected
        op = family if hit else "keep"
        value = token["text"]
        if hit and family == "insert":
            span = append(rng.choice(words))
            log.append(dict(uid=f"new-{uid}", source_uid=None, parent_uid=None,
                            source_span=None, dst_span=span, op="insert"))
            append(" ")
        if hit and family in ("replace", "synonym"):
            choices = synonyms[value.lower()] if family == "synonym" else [w for w in words if w.lower() != value.lower()]
            if not choices:
                raise ValueError("No non-identity replacement available")
            value = rng.choice(choices)
            if token["text"].istitle():
                value = value[:1].upper() + value[1:]
        span = None if hit and family == "delete" else append(value)
        log.append(dict(uid=uid, source_uid=uid, parent_uid=None,
                        source_span=[token["start"], token["end"]], dst_span=span, op=op))
        if hit and family == "repeat":
            append(" ")
            span = append(token["text"])
            log.append(dict(uid=f"clone-{uid}", source_uid=None, parent_uid=uid,
                            source_span=None, dst_span=span, op="repeat"))
        cursor = token["end"]
    append(text[cursor:])
    edited = "".join(parts)
    return dict(text=edited, provenance=log, selected_tokens=sorted(selected),
                requested_ratio=ratio, actual_selected=count, eligible_tokens=len(eligible),
                seed=seed, family=family, changed_text=edited != text,
                source_text_hash=digest(text), next_text_hash=digest(edited))
