"""Train the two HC3 branches used by V4 with the notebook 07 recipe."""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoConfig

from evaluate_v3_quang import REPO, TRAINING_DOMAINS, sha256, write_json
from model.RGCN import RGCN2
from train_v3 import atomic_save
from v3_a1_fusion import AlignedScorer, labels_from, read_source, source_digest


def prepare_graphs(cache, sources, revision, device="cuda", *, prepare_a1=True):
    local_encoder = REPO / "roberta-base"
    versions = {k: version(k) for k in ("torch", "transformers", "spacy", "en-core-web-sm",
                                         "tokenizers", "torch-geometric")}
    recipe = [revision, device, versions, [source_digest(s) for s in sources],
              [sha256(REPO / n) for n in ("dep_vocab.json", "dep_vocab.py", "building_graph.py",
                                          "v3_a1_fusion.py", "train_v4.py")]]
    if (local_encoder / "vocab.json").is_file():
        recipe.append({p.name: sha256(p) for p in local_encoder.iterdir() if p.is_file()})
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    dest = cache / key
    dest.mkdir(parents=True, exist_ok=True)
    missing = [s for s in sources if not (dest / f"{s.stem}.pkl").exists()]
    if missing:
        with TemporaryDirectory(dir=dest) as tmp:
            if (local_encoder / "vocab.json").is_file():
                (Path(tmp) / "roberta-base").symlink_to(local_encoder, target_is_directory=True)
            subprocess.run([sys.executable, str(REPO / "building_graph.py"), "--file", *map(str, missing)],
                           cwd=tmp, check=True)
            for s in missing:
                name = f"{s.stem}.pkl"
                (Path(tmp) / "graph_data" / name).replace(dest / name)
    if not prepare_a1:
        return dest
    # Instantiate the frozen encoder even on cache hits to preserve notebook 07's RNG sequence.
    scorer = AlignedScorer(None, device, revision=revision)
    for s in sources:
        path = dest / f"{s.stem}.pt"
        if path.exists():
            continue
        rows = read_source(s)
        graphs = []
        docs = scorer.nlp.pipe((r["text"] for r in rows), batch_size=16)
        for doc, y in zip(docs, labels_from(rows)):
            x, edge, rel = scorer.aligned_graph(doc)
            if int(rel.max()) >= 60:
                raise ValueError("A1 relation exceeds the training vocabulary")
            graphs.append(Data(x=x.cpu().half(), edge_index=edge.cpu(), edge_type=rel.cpu(),
                               y=torch.tensor([y], dtype=torch.float)))
        with TemporaryDirectory(dir=dest) as tmp:
            staged = Path(tmp) / path.name
            torch.save(dict(source_sha256=source_digest(s), graphs=graphs), staged)
            staged.replace(path)
    return dest


def fit_a1(args, graph_dir, sources, out, device="cuda"):
    loaders = []
    for i, s in enumerate(sources):
        cached = torch.load(graph_dir / f"{s.stem}.pt", weights_only=False, map_location="cpu")
        if cached["source_sha256"] != source_digest(s):
            raise ValueError(f"Wrong cached A1 graphs for {s}")
        loaders.append(DataLoader(cached["graphs"], batch_size=args.a1_batch, shuffle=i == 0))
    train, val = loaders
    model = RGCN2(input_dim=768, hidden_dim=256, output_dim=64, num_relations=60, num_bases=30).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.a1_lr)
    best = float("inf")
    history = []
    for epoch in range(1, args.a1_epochs + 1):
        model.train()
        for batch in train:
            batch = batch.to(device)
            batch.x = batch.x.float()
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy(model(batch).flatten(), batch.y).backward()
            opt.step()
        model.eval()
        losses = []
        with torch.inference_mode():
            for batch in val:
                batch = batch.to(device)
                batch.x = batch.x.float()
                losses.append(torch.nn.functional.binary_cross_entropy(model(batch).flatten(), batch.y,
                                                                       reduction="sum"))
        loss = sum(torch.stack(losses).tolist()) / len(val.dataset)
        history.append(dict(epoch=epoch, validation_bce=loss))
        if loss < best:
            best = loss
            atomic_save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                        out / f"a1_seed{args.seed}.pt")
        print(f"A1 epoch {epoch}: validation BCE {loss:.5f}", flush=True)
    write_json(out / "a1_history.json", history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--v3-epochs", type=int, default=10)
    parser.add_argument("--v3-struct", type=int, choices=(0, 1), default=1,
                        help="Enable V3 node structural features.")
    parser.add_argument("--v3-aug", choices=("swap", "none"), default="swap",
                        help="V3 training augmentation. Checkpoint selection uses the same validation proxy.")
    parser.add_argument("--a1-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--a1-batch", type=int, default=4)
    parser.add_argument("--a1-lr", type=float, default=0.0001)
    parser.add_argument("--a1-checkpoint", type=Path,
                        help="Reuse the reference V4 A1 checkpoint for this seed instead of training A1.")
    parser.add_argument("--serial", action="store_true", help="Use one GPU even when two are available")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Training requires a CUDA GPU")
    if min(args.v3_epochs, args.a1_epochs, args.batch_size, args.micro_batch_size,
           args.val_batch_size, args.a1_batch, args.a1_lr) <= 0 or args.micro_batch_size > args.batch_size:
        parser.error("Check the epoch counts, learning rate and batch sizes")
    if args.a1_checkpoint is not None:
        args.a1_checkpoint = args.a1_checkpoint.resolve()
        if not args.a1_checkpoint.is_file() or args.a1_checkpoint.name != f"a1_seed{args.seed}.pt":
            parser.error(f"Expected an existing A1 checkpoint named a1_seed{args.seed}.pt")
        AlignedScorer.load_a1(args.a1_checkpoint, "cpu")
    dest = args.output_dir.resolve() / f"seed{args.seed}"
    if dest.exists():
        parser.error(f"Completed output already exists: {dest}. Use a fresh output directory for a new run.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sources = [REPO / TRAINING_DOMAINS["hc3"][k] for k in ("train_source", "val_source")]
    revision = AutoConfig.from_pretrained("roberta-base")._commit_hash
    if not revision:
        raise ValueError("Cannot identify the frozen encoder revision")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    graphs = prepare_graphs(args.feature_cache.resolve(), sources, revision,
                            prepare_a1=args.a1_checkpoint is None)
    torch.cuda.empty_cache()
    with TemporaryDirectory(dir=dest.parent) as tmp:
        out = Path(tmp) / "models"
        out.mkdir()
        cmd = [sys.executable, str(REPO / "train_v3_dual_best.py"), "--seeds", str(args.seed),
               "--device", "cuda", "--no-deterministic", "--ema", "1", "--epochs", str(args.v3_epochs),
               "--struct", str(args.v3_struct), "--aug", args.v3_aug,
               "--batch-size", str(args.batch_size), "--micro-batch-size", str(args.micro_batch_size),
               "--val-batch-size", str(args.val_batch_size), "--graph-dir", str(graphs), "--output-dir", str(out)]
        if args.a1_checkpoint is not None:
            shutil.copyfile(args.a1_checkpoint, out / args.a1_checkpoint.name)
            subprocess.run(cmd, cwd=REPO, check=True)
        elif not args.serial and torch.cuda.device_count() >= 2:
            second = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[1].strip()
            with subprocess.Popen(cmd, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": second}) as proc:
                try:
                    fit_a1(args, graphs, sources, out)
                    if proc.wait():
                        raise subprocess.CalledProcessError(proc.returncode, cmd)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
        else:
            subprocess.run(cmd, cwd=REPO, check=True)
            fit_a1(args, graphs, sources, out)
        paths = [out / f"v3_baseline_ema_seed{args.seed}.pt", out / f"a1_seed{args.seed}.pt"]
        record = json.loads((out / f"v3_baseline_dual_seed{args.seed}.train.json").read_bytes())
        if not record["complete"] or record["checkpoint_sha256"]["ema"] != sha256(paths[0]):
            raise ValueError("Incomplete EMA training record")
        config = record["provenance"]["config"]
        if (config["struct"], config["aug"], config["ema"]) != (args.v3_struct, args.v3_aug, 1):
            raise ValueError("Unexpected structural features, augmentation or EMA configuration")
        manifest = out / "manifest.json"
        write_json(manifest, [dict(seed=args.seed, file=p.name, sha256=sha256(p)) for p in paths])
        model_name = {(1, "swap"): "v4", (0, "swap"): "v4_abla2",
                      (1, "none"): "v4_abla3", (0, "none"): "v4_abla1"}[args.v3_struct, args.v3_aug]
        write_json(out / "v4_train.json", dict(model=model_name, threshold=0.5, encoder_revision=revision,
                   args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
        with ZipFile(out / f"{model_name}_hc3_seed{args.seed}.zip", "w", ZIP_DEFLATED) as z:
            for p in [*paths, manifest]:
                z.write(p, p.name)
        out.rename(dest)
    print(f"V4 checkpoints and training records: {dest}", flush=True)


if __name__ == "__main__":
    main()
