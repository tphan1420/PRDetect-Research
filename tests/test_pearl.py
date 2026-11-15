import copy
from types import SimpleNamespace
import pytest
import torch

from pearl.alignment import align_tokens
from pearl.data import assert_isolated, quarantine_train_overlap
from pearl.edge_features import EdgeFeatures
from pearl.edge_predictor import EdgeRetentionPredictor, document_edge_loss
from pearl.edit_provenance import generate_edit
from pearl.graph import parse_graph, tensor_edges
from pearl.relation_retention import label_arcs
from pearl.train_gcn import load_checkpoint, save_checkpoint
from pearl.weighted_gcn import PEARLGCN, expand_arc_weights
from model.GCN2 import GCN2


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def pair(nlp, text, family, **kwargs):
    g = parse_graph(nlp, text)
    edit = generate_edit(text, g, family, 1, 2025, **kwargs)
    target = parse_graph(nlp, edit["text"])
    mapping, reasons = align_tokens(g, target, edit["provenance"])
    return g, target, edit, mapping, label_arcs(g, target, mapping, reasons)


def test_identity_and_deterministic(nlp):
    text = "The careful doctor saw the doctor.\n\nShe smiled."
    g, target, edit, mapping, labels = pair(nlp, text, "identity")
    assert edit["text"] == text
    assert len(mapping) == len(g["tokens"])
    assert all(a["relation_retained"] == 1 for a in labels)
    assert generate_edit(text, g, "repeat", .3, 9) == generate_edit(text, g, "repeat", .3, 9)


def test_synonym_uid(nlp):
    g, target, edit, mapping, _ = pair(nlp, "The careful doctor smiled.", "synonym",
                                      synonyms={"careful": ["cautious"]})
    child = next(t["i"] for t in g["tokens"] if t["text"] == "careful")
    assert target["tokens"][mapping[child]]["text"] == "cautious"
    assert len(mapping) == len(g["tokens"])


@pytest.mark.parametrize("family", ["insert", "repeat", "replace"])
def test_repeated_tokens_and_insert(nlp, family):
    g, target, edit, mapping, labels = pair(nlp, "Doctors help doctors and doctors help people.", family)
    assert len(mapping) == len(g["tokens"])
    assert len(set(mapping.values())) == len(mapping)
    clones = [p for p in edit["provenance"] if p["source_uid"] is None]
    if family in ("insert", "repeat"):
        assert clones
        for p in clones:
            assert p["uid"].startswith(("clone-", "new-"))
            assert p["dst_span"] not in [[t["start"], t["end"]] for t in target["tokens"] if t["i"] in mapping.values()]


def test_delete_masks_endpoints(nlp):
    _, _, edit, mapping, labels = pair(nlp, "The careful doctor smiles.", "delete")
    deleted = {int(p["uid"][1:]) for p in edit["provenance"] if p["op"] == "delete"}
    assert deleted.isdisjoint(mapping)
    for a in labels:
        if a["child"] in deleted or a["head"] in deleted:
            assert not a["relation_valid"] and a["relation_retained"] is None


def test_split_unknown(nlp):
    g, _, _, mapping, labels = pair(nlp, "The doctor smiled.", "synonym", synonyms={"doctor": ["medical worker"]})
    i = next(t["i"] for t in g["tokens"] if t["text"] == "doctor")
    assert i not in mapping
    assert any(a["status"] == "unknown" for a in labels)


def test_merge_unknown():
    source = {"tokens": [dict(i=0, start=0, end=1), dict(i=1, start=2, end=3)]}
    target = {"tokens": [dict(i=0, start=0, end=2)]}
    provenance = [dict(source_uid="t0", source_span=[0, 1], dst_span=[0, 1]),
                  dict(source_uid="t1", source_span=[2, 3], dst_span=[1, 2])]
    mapping, reasons = align_tokens(source, target, provenance)
    assert not mapping
    assert set(reasons.values()) == {"split_merge_or_resegmentation"}


@pytest.mark.parametrize("change", ["head", "dep"])
def test_arc_change(nlp, change):
    g = parse_graph(nlp, "The doctor helps people.")
    target = copy.deepcopy(g)
    token = next(t for t in target["tokens"] if t["i"] != t["head"])
    if change == "head":
        token["head"] = token["i"]
    else:
        token["dep"] = "DIFFERENT"
    mapping = {t["i"]: t["i"] for t in g["tokens"]}
    labels = label_arcs(g, target, mapping, {})
    assert next(a for a in labels if a["child"] == token["i"])["relation_retained"] == 0


def test_split_isolation_and_quarantine():
    a = [{"doc_id": "a", "normalized_hash": "same"}, {"doc_id": "safe"}]
    b = [{"doc_id": "b", "normalized_hash": "same"}]
    with pytest.raises(ValueError, match="leakage"):
        assert_isolated(a, b)
    kept, removed = quarantine_train_overlap(a, b)
    assert kept == [a[1]] and removed == [a[0]]
    with pytest.raises(ValueError):
        assert_isolated([{"doc_id": "parent"}], [{"doc_id": "child", "parent_id": "parent"}])


def test_weight_order_root_reverse_and_gradient(nlp):
    g = parse_graph(nlp, "The doctor helps people.")
    q = torch.linspace(.1, .9, sum(t["i"] != t["head"] for t in g["tokens"]), requires_grad=True)
    for reverse in (False, True):
        index, weights = expand_arc_weights(g, q, .8, reverse)
        edges, mapping = tensor_edges(g, reverse)
        assert index.T.tolist() == [list(e) for e in edges]
        j = 0
        for token in g["tokens"]:
            for column in mapping[token["i"]]:
                expected = torch.tensor(1.) if token["i"] == token["head"] else .8 + .2 * q[j]
                torch.testing.assert_close(weights[column], expected)
            j += int(token["i"] != token["head"])
        weights.sum().backward()
    assert q.grad.abs().sum() > 0


def test_mask_and_document_balance():
    logits = torch.tensor([0., 5.], requires_grad=True)
    pair = {"arc_labels": [{"relation_valid": True, "relation_retained": 1},
                           {"relation_valid": False, "relation_retained": None}]}
    one = document_edge_loss(logits, [pair])
    repeated = document_edge_loss(logits, [pair] * 7)
    torch.testing.assert_close(one, repeated)
    one.backward()
    assert logits.grad[0] != 0 and logits.grad[1] == 0


@pytest.mark.parametrize("text", ["Hello.", "The careful doctor helps people.", "A. B. C."])
def test_model_parity_backward_checkpoint(nlp, text, tmp_path):
    torch.manual_seed(5)
    g = parse_graph(nlp, text)
    encoder = EdgeFeatures.fit([g])
    base = GCN2(4, 8, 6)
    model = PEARLGCN(base, EdgeRetentionPredictor(encoder.dim, 8), encoder, epsilon=1)
    x = torch.randn(len(g["tokens"]), 4)
    edges, _ = tensor_edges(g)
    model.eval()
    original = base(SimpleNamespace(x=x, edge_index=torch.tensor(edges).T))
    torch.testing.assert_close(model(x, g), original, atol=1e-6, rtol=1e-5)
    model.epsilon = .8
    _, logits, _, weights = model(x, g, True)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    assert sum(p.grad.abs().sum() for p in model.predictor.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in base.parameters() if p.grad is not None) > 0
    save_checkpoint(tmp_path / "model.pt", model, {"test": True})
    loaded, metadata = load_checkpoint(tmp_path / "model.pt")
    torch.testing.assert_close(loaded(x, g), model(x, g))
    assert metadata == {"test": True}


def test_feature_inputs_ignore_provenance(nlp):
    g = parse_graph(nlp, "The doctor helps people.")
    features = EdgeFeatures.fit([g])
    other = copy.deepcopy(g)
    other.update(label=1, edit_family="delete", uid="secret", edit_ratio=.9)
    torch.testing.assert_close(features.encode(g), features.encode(other))


def test_real_baseline_checkpoint_parity(nlp):
    from pathlib import Path
    from pearl.weighted_gcn import baseline_from_state
    path = Path("model/hc3_gcn_model_2025.pth")
    if not path.exists():
        pytest.skip("Local historical checkpoint not available")
    base = baseline_from_state(torch.load(path, weights_only=True, map_location="cpu"))
    g = parse_graph(nlp, "The doctor helps people. Another sentence follows.")
    features = EdgeFeatures.fit([g])
    model = PEARLGCN(base, EdgeRetentionPredictor(features.dim), features, 1).eval()
    x = torch.randn(len(g["tokens"]), 768)
    edges, _ = tensor_edges(g)
    with torch.no_grad():
        torch.testing.assert_close(model(x, g), base(SimpleNamespace(x=x, edge_index=torch.tensor(edges).T)), atol=1e-6, rtol=1e-5)
