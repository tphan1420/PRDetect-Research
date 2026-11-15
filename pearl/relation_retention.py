"""Relation retention conditioned on both uniquely aligned endpoints."""
from .graph import arcs


def label_arcs(source, edited, mapping, reasons):
    triples = set(arcs(edited))
    labels = []
    for child, head, relation in arcs(source):
        valid = child in mapping and head in mapping
        y = int((mapping[child], mapping[head], relation) in triples) if valid else None
        labels.append(dict(src_arc_id=child, child=child, head=head, relation=relation,
                           relation_valid=valid, relation_retained=y,
                           status="unknown" if not valid else "stable" if y else "changed",
                           unknown_reasons=[] if valid else [reasons[k] for k in (child, head) if k not in mapping]))
    return labels
