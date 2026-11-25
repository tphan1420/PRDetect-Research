"""Fit a decision threshold for a frozen V3 using HC3 validation only.

The default rule retains at least 95% of human validation documents. The
optional length experiment also measures logit drift on disjoint graph unions.
"""

import argparse
from argparse import Namespace
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scipy.special import expit, logit

from evaluate_v3_quang import REPO, TRAINING_DOMAINS, environment, human95_threshold, positive_int, sha256, write_json
from train_v3 import build_model, load_graph


def load_v3_checkpoint(path):
    """Accept a completed V3 checkpoint or the project's dual RAW/EMA run zip."""
    source, digest = path, None
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".pt") and "_raw_seed" in n]
            if len(names) == 1:
                data = archive.read(names[0])
                digest = hashlib.sha256(data).hexdigest()
                source = io.BytesIO(data)
            elif not any(n.endswith("/data.pkl") for n in archive.namelist()):
                raise ValueError(f"Expected one RAW checkpoint inside {path}")
    if digest is None:
        digest = sha256(path)
    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    args = Namespace(**ckpt["args"])
    if (ckpt.get("method") != "v3_baseline" or not ckpt.get("training_complete") or args.train_file != "hc3_train"
            or args.val_file != "hc3_val" or args.readout != "pool3" or args.gfeat):
        raise ValueError("Expected a completed HC3 pool3 checkpoint without graph-level features")
    model = build_model(args, "baseline", ckpt["R"])
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt, digest


@torch.inference_mode()
def document_summaries(model, dataset, device, batch_size):
    """Collect the exact sufficient statistics of each eval-mode readout."""
    captures = {}

    def capture_gate(module, inputs, output):
        captures["gate"] = output.flatten()

    def capture_pool(module, inputs):
        captures["pool"] = inputs[0]

    gate_hook = model.gate.register_forward_hook(capture_gate)
    pool_hook = model.mlp.register_forward_pre_hook(capture_pool)
    pools, log_masses = [], []
    try:
        for start in range(0, dataset.G, batch_size):
            ids = torch.arange(start, min(start + batch_size, dataset.G))
            batch = dataset.batch(ids, device)
            model(batch)
            gates = captures["gate"].split(dataset.nn[ids].tolist())
            log_masses.append(torch.stack([g.logsumexp(0) for g in gates]).cpu())
            pools.append(captures["pool"].cpu())
    finally:
        gate_hook.remove()
        pool_hook.remove()
    return torch.cat(pools), torch.cat(log_masses)


def union_pools(pools, log_masses, counts, groups):
    means, maxima, attention = pools.chunk(3, dim=-1)
    lengths = counts[groups]
    mean = (means[groups] * (lengths / lengths.sum(1, keepdim=True)).unsqueeze(2)).sum(1)
    maximum = maxima[groups].amax(1)
    weights = log_masses[groups].softmax(1).unsqueeze(2)
    return torch.cat((mean, maximum, (attention[groups] * weights).sum(1)), dim=1)


@torch.inference_mode()
def probe_length(model, dataset, text_lengths, device, batch_size):
    pools, masses = document_summaries(model, dataset, device, batch_size)
    head = model.mlp.cpu()
    labels = dataset.y.numpy()
    # Fixed proxy designs across checkpoints. Each permutation samples without
    # replacement inside a union. No seed controls the fitted model or its run.
    rng = np.random.default_rng(31)
    class_curves = []
    for label in (0, 1):
        ids = np.flatnonzero(labels == label)
        lengths, locations = [], []
        for size in (1, 2, 4, 8, 16):
            if len(ids) < size:
                break
            if size == 1:
                groups = ids[:, None]
            else:
                groups = np.stack([rng.choice(ids, size, replace=False) for _ in range(len(ids))])
            vectors = union_pools(pools, masses, dataset.nn, torch.from_numpy(groups))
            logits = head(vectors).squeeze(-1).numpy()
            lengths.append(float(np.median(text_lengths[groups].sum(1))))
            locations.append(float(np.median(logits)))
        class_curves.append((lengths, locations))
    # Keep the original boundary at typical source lengths. Subtract only the
    # estimated drift of the balanced class midpoint on longer documents.
    reference = float(np.median(text_lengths))
    knots = np.unique([reference] + [n for lengths, _ in class_curves for n in lengths if n > reference])
    midpoint = np.mean([np.interp(np.log(knots), np.log(lengths), logits)
                        for lengths, logits in class_curves], axis=0)
    return {"lengths": knots.tolist(), "offsets": (midpoint - midpoint[0]).tolist(),
            "source_class_medians": class_curves}


def calibrated_probabilities(probabilities, lengths, calibration):
    p, lengths = np.asarray(probabilities, dtype=float), np.asarray(lengths)
    if (p.ndim != 1 or p.shape != lengths.shape or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.isfinite(lengths).all() or np.any(lengths < 1)):
        raise ValueError("Expected aligned probabilities in [0,1] and positive token counts")
    offset = np.interp(np.log(lengths), np.log(calibration["lengths"]), calibration["offsets"])
    return expit(logit(p) - offset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--val-graph", type=Path, default=REPO / "graph_data/hc3_val.pkl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--probe-length", action="store_true", help="Also reproduce the graph union diagnostic")
    args = parser.parse_args()
    source = REPO / TRAINING_DOMAINS["hc3"]["val_source"]
    source_lf = source.read_bytes().replace(b"\r\n", b"\n")
    normalized_hash = hashlib.sha256(source_lf).hexdigest()
    provenance = {"val_source_sha256": sha256(source), "val_source_lf_sha256": normalized_hash,
                  "val_graph_sha256": sha256(args.val_graph), "environment": environment()}
    source_hashes = {provenance["val_source_sha256"], normalized_hash,
                     hashlib.sha256(source_lf.replace(b"\n", b"\r\n")).hexdigest()}
    device = torch.device(args.device)
    dataset = None
    for path in args.checkpoints:
        model, checkpoint, digest = load_v3_checkpoint(path)
        training_files = checkpoint.get("provenance", {}).get("data", {})
        for recorded_path, recorded_hash in training_files.items():
            if Path(recorded_path.replace("\\", "/")).name == source.name and recorded_hash not in source_hashes:
                raise ValueError(f"HC3 validation source differs from checkpoint provenance: {path}")
        output = args.output_dir / f"{digest}.json"
        mode, relations = checkpoint["struct_mode"], checkpoint["R"]
        if dataset is None or (dataset.mode, dataset.R) != (mode, relations):
            dataset = load_graph(args.val_graph, source, relations, mode)
        model.to(device)
        logits = []
        with torch.inference_mode():
            for start in range(0, dataset.G, args.batch_size):
                ids = torch.arange(start, min(start + args.batch_size, dataset.G))
                logits.append(model(dataset.batch(ids, device)).cpu())
        floor = human95_threshold(dataset.y.numpy(), torch.cat(logits).numpy())
        fitted = {"human_validation_coverage": 0.95, "human_logit_floor": floor,
                  "human95_threshold": torch.tensor(floor, dtype=torch.float32).sigmoid().item()}
        if args.probe_length:
            with source.open(encoding="utf-8") as stream:
                lengths = np.array([max(1, len(json.loads(line)["text"].split())) for line in stream if line.strip()])
            fitted.update(probe_length(model, dataset, lengths, device, args.batch_size))
        write_json(output, {**fitted, **provenance, "checkpoint_sha256": digest,
                            "seed": checkpoint["args"]["seeds"][0], "checkpoint_file": path.name})
        print(f"{path.name} -> {output.name} | human95 threshold {fitted['human95_threshold']:.6f}", flush=True)


if __name__ == "__main__":
    main()
