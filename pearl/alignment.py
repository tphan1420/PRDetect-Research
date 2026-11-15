"""Conservative exact-span 1:1 provenance alignment, never text matching."""
from collections import Counter


def align_tokens(source, edited, provenance):
    destination = {}
    for t in edited["tokens"]:
        destination.setdefault((t["start"], t["end"]), []).append(t["i"])
    mapping, reasons = {}, {}
    for t in source["tokens"]:
        uid = f"t{t['i']}"
        records = [p for p in provenance if p["source_uid"] == uid]
        reason = "ambiguous_provenance"
        if len(records) == 1:
            p = records[0]
            span = p["dst_span"]
            if p["source_span"] != [t["start"], t["end"]]:
                reason = "source_span_mismatch"
            elif span is None:
                reason = "deleted"
            else:
                matches = destination.get(tuple(span), [])
                overlap = [r for r in provenance if r["dst_span"] is not None
                           and r["dst_span"][0] < span[1] and r["dst_span"][1] > span[0]]
                if len(matches) == 1 and len(overlap) == 1:
                    mapping[t["i"]] = matches[0]
                    reason = "unique"
                else:
                    reason = "split_merge_or_resegmentation"
        reasons[t["i"]] = reason
    duplicates = Counter(mapping.values())
    for src, dst in list(mapping.items()):
        if duplicates[dst] != 1:
            del mapping[src]
            reasons[src] = "many_to_one"
    return mapping, reasons
