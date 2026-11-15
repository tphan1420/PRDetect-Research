"""Tiny integration uses explicit synthetic embeddings; CLI smoke uses real RoBERTa."""
import json
from pathlib import Path
import torch
import pytest
from pearl.config import seed_all
from pearl.data import write_json
from pearl.prepare import prepare
from pearl.train_edge import train as train_edge
from pearl.train_gcn import train as train_gcn, load_checkpoint
from model.GCN2 import GCN2


class FixtureEncoder:
    def __init__(self, cfg):
        self.metadata = {"test_fixture": True, "hidden": 4}

    def encode(self, graph):
        return torch.randn(len(graph["tokens"]), 4, generator=torch.Generator().manual_seed(99))


@pytest.mark.parametrize("initialization", ["checkpoint", "scratch"])
def test_prepare_predictor_joint_checkpoint(tmp_path, monkeypatch, initialization):
    import pearl.train_gcn as module
    monkeypatch.setattr(module, "LegacyEncoder", FixtureEncoder)
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train_file.write_text(json.dumps({"text": "The careful doctor helps people.", "label": "human"}) + "\n" +
                          json.dumps({"text": "A blue stone falls quickly.", "label": "machine"}) + "\n", encoding="utf-8")
    val_file.write_text(json.dumps({"text": "People help the new doctor.", "label": "human"}) + "\n" +
                        json.dumps({"text": "An old tree grows slowly.", "label": "machine"}) + "\n", encoding="utf-8")
    seed_all(2025)
    torch.set_num_threads(2)
    baseline_path = tmp_path / "baseline.pth"
    if initialization == "checkpoint":
        torch.save(GCN2(4, 8, 6).state_dict(), baseline_path)
    cfg = dict(seed=2025, source_train=str(train_file), source_validation=str(val_file),
               parser="en_core_web_sm", prepared=str(tmp_path / "prepared"),
               edge_output=str(tmp_path / "edge"), gcn_output=str(tmp_path / "gcn"),
               device="cpu", edge_epochs=1, gcn_epochs=1, edge_batch_docs=2, gcn_batch_docs=2,
               epsilon=.8, lambda_edge=.1, gcn_initialization=initialization,
               input_dim=4, hidden_dim=8, output_dim=6)
    if initialization == "checkpoint":
        cfg["baseline_checkpoint"] = str(baseline_path)
    manifest = prepare(cfg)
    assert manifest["splits"]["train"]["pairs"] == 10
    # Binary records deterministic across independent output directories.
    second_cfg = dict(cfg, prepared=str(tmp_path / "second"))
    prepare(second_cfg)
    for name in ("train_docs.jsonl.gz", "train_pairs.jsonl.gz", "validation_docs.jsonl.gz", "validation_pairs.jsonl.gz"):
        assert (Path(cfg["prepared"]) / name).read_bytes() == (Path(second_cfg["prepared"]) / name).read_bytes()
    report = train_edge(cfg)
    assert report["metrics"]["mlp"]["n"] > 0
    joint = train_gcn(cfg)
    assert joint["max_predictor_gradient_l1"] > 0
    loaded, metadata = load_checkpoint(Path(cfg["gcn_output"]) / "best.pt")
    assert metadata["epoch"] == 1
    assert metadata["gcn_initialization"] == initialization
    if initialization == "scratch":
        assert metadata["baseline_sha256"] is None and not baseline_path.exists()
    assert loaded.epsilon == .8
    assert (Path(cfg["prepared"]) / "train_embeddings.pt").exists()
    assert not list(tmp_path.rglob("*.pkl"))
