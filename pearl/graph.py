"""Parser-only sidecars, preserving baseline token and tensor-edge order."""


def parse_graph(nlp, text):
    doc = nlp(text)
    if not doc or not doc.has_annotation("DEP") or not doc.has_annotation("POS"):
        raise ValueError("A trained POS/dependency parser is required; blank spaCy is invalid")
    sentence = {t.i: s for s, sent in enumerate(doc.sents) for t in sent}
    return {"tokens": [dict(i=t.i, text=t.text, start=t.idx, end=t.idx + len(t),
                            head=t.head.i, dep=t.dep_, pos=t.pos_, sent=sentence[t.i]) for t in doc]}


def arcs(graph):
    return [(t["i"], t["head"], t["dep"]) for t in graph["tokens"] if t["i"] != t["head"]]


def tensor_edges(graph, bidirectional=False):
    result = []
    mapping = {}
    for t in graph["tokens"]:
        mapping[t["i"]] = [len(result)]
        result.append((t["i"], t["head"]))
        if bidirectional and t["i"] != t["head"]:
            mapping[t["i"]].append(len(result))
            result.append((t["head"], t["i"]))
    return result, mapping
