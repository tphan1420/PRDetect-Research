"""Train V3 augmentation ablations from the same legacy-domain graphs.

No test files are read during training or checkpoint selection. No R-Drop,
relation remapping, or length balancing is introduced.
Model definitions, augmentation, and training are separate for future ablations.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from evaluate_v3_quang import (REPO, TRAINING_DOMAINS, environment, positive_int,
                               sha256, write_json)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=("baseline", "meanlogit", "classswap", "restricted", "cmixstyle", "structmap"),
                        default=["baseline", "cmixstyle"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--graph-dir", type=Path, default=REPO / "graph_data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "result/v3_cmixstyle/model")
    parser.add_argument("--train-file", default="hc3_train")
    parser.add_argument("--val-file", default="hc3_val")
    parser.add_argument("--struct-mode", choices=("low", "high", "directed"), default="low")
    parser.add_argument("--num-rel", type=int, default=0, help="0: infer from training graph only")
    parser.add_argument("--hidden", type=positive_int, default=384)
    parser.add_argument("--num-bases", type=positive_int, default=12)
    parser.add_argument("--layers", type=positive_int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--in-dropout", type=float, default=0.1)
    parser.add_argument("--edge-drop", type=float, default=0.1)
    parser.add_argument("--struct", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gfeat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--readout", choices=("pool3", "meanlogit"), default="pool3")
    parser.add_argument("--aug", choices=("none", "swap", "mask", "both"), default="swap")
    parser.add_argument("--aug-max", type=float, default=0.35)
    parser.add_argument("--ema", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=32,
                        help="Effective graphs per optimizer/EMA/scheduler step")
    parser.add_argument("--micro-batch-size", type=positive_int, default=8,
                        help="Graph donor pool size; use the same value in both variants")
    parser.add_argument("--val-batch-size", "--eval-batch-size", dest="val_batch_size",
                        type=positive_int, default=64,
                        help="Also controls the donor pool of the fixed validation swap proxy")
    parser.add_argument("--select", choices=("lossaug", "loss", "acc", "auc"), default="lossaug")
    parser.add_argument("--mixstyle-p", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--mixstyle-strength", type=float, default=0.3)
    parser.add_argument("--restricted-semantic-init", type=float, default=0.9)
    parser.add_argument("--restricted-structural-init", type=float, default=1.0)
    parser.add_argument("--restricted-semantic-budget", type=float, default=0.8)
    parser.add_argument("--restricted-semantic-penalty", type=float, default=0.1)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args):
    pairs = {(item["train_file"], item["val_file"]) for item in TRAINING_DOMAINS.values()}
    if (args.train_file, args.val_file) not in pairs:
        raise ValueError("Use a supported train/validation pair from TRAINING_DOMAINS")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise ValueError("Seeds/variants must not contain duplicates")
    for name in ("dropout", "in_dropout", "edge_drop", "aug_max", "mixstyle_p", "mixstyle_strength"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    for name in ("lr", "mixstyle_alpha"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.wd) or args.wd < 0 or not 0 <= args.ema_decay < 1 or args.num_rel < 0:
        raise ValueError("Invalid weight decay, EMA decay or relation count")
    if args.micro_batch_size > args.batch_size:
        raise ValueError("micro-batch-size must not exceed batch-size")
    if "structmap" in args.variants and args.struct_mode != "directed":
        raise ValueError("Corrected Structural Mapping requires --struct-mode directed")
    if args.struct_mode == "directed" and set(args.variants) != {"structmap"}:
        raise ValueError("--struct-mode directed is reserved for the structmap variant")
    if ("baseline" in args.variants and "meanlogit" in args.variants
            and args.readout != "pool3"):
        raise ValueError("The paired baseline/meanlogit ablation requires --readout pool3")
    if "cmixstyle" in args.variants:
        if args.aug == "none":
            raise ValueError("CMixStyle is applied to the existing augmented branch; --aug cannot be none")
        if args.micro_batch_size < 2 or args.mixstyle_p == 0 or args.mixstyle_strength == 0:
            raise ValueError("CMixStyle needs a donor pool >=2, nonzero probability and strength")
    if "classswap" in args.variants and args.aug not in ("swap", "both"):
        raise ValueError("Class-preserving Syntax Swap requires --aug swap or --aug both")
    if "restricted" in args.variants:
        if not args.struct:
            raise ValueError("Structure-Guided Restricted Embeddings requires --struct 1")
        if not 0 < args.restricted_semantic_init < 1:
            raise ValueError("restricted-semantic-init must be in (0,1)")
        if not 0 < args.restricted_structural_init < 2:
            raise ValueError("restricted-structural-init must be in (0,2)")
        if not 0 <= args.restricted_semantic_budget <= 1:
            raise ValueError("restricted-semantic-budget must be in [0,1]")
        if not math.isfinite(args.restricted_semantic_penalty) or args.restricted_semantic_penalty < 0:
            raise ValueError("restricted-semantic-penalty must be finite and nonnegative")


def effective_readout(args, variant):
    """Resolve a named architecture variant before construction and serialization."""
    return "meanlogit" if variant == "meanlogit" else args.readout


def build_model(args, variant, R):
    from model.QuangV3 import Net
    from model.V3ClassSwap import ClassPreservingSyntaxSwapNet
    from model.V3CMixStyle import CMixStyleNet
    from model.V3Restricted import StructureGuidedRestrictedNet
    from model.V3StructuralMap import CorrectedStructuralMappingNet

    kwargs = dict(d=args.hidden, nb=args.num_bases, L=args.layers, p=args.dropout,
                  p_in=args.in_dropout, edge_drop=args.edge_drop,
                  use_struct=bool(args.struct), use_gfeat=bool(args.gfeat),
                  readout=effective_readout(args, variant), aug=args.aug,
                  aug_max=args.aug_max, mode=args.struct_mode)
    if variant in ("baseline", "meanlogit"):
        return Net(R, **kwargs)
    if variant == "structmap":
        return CorrectedStructuralMappingNet(R, **kwargs)
    if variant == "classswap":
        return ClassPreservingSyntaxSwapNet(R, **kwargs)
    if variant == "restricted":
        return StructureGuidedRestrictedNet(
            R, **kwargs, semantic_init=args.restricted_semantic_init,
            structural_init=args.restricted_structural_init,
            semantic_budget=args.restricted_semantic_budget,
            semantic_penalty=args.restricted_semantic_penalty,
        )
    if variant != "cmixstyle":
        raise ValueError(f"Unknown variant: {variant}")
    return CMixStyleNet(R, **kwargs, mixstyle_p=args.mixstyle_p,
                       mixstyle_alpha=args.mixstyle_alpha, mixstyle_strength=args.mixstyle_strength)


def load_graph(path, source, R, mode):
    """Validate original cache schema and exact source label order, then keep on CPU."""
    import numpy as np
    import torch
    from model import QuangV3 as v3

    original_load = v3._load_pickle

    def checked_load(filename):
        raw = original_load(filename)
        if not isinstance(raw, dict):
            raise ValueError("Use dictionary caches from building_graph.py")
        if any(key in raw for key in ("encoder_metadata", "schema_version", "all_cls_embeddings",
                                      "all_quality", "quality_features")):
            raise ValueError("Use legacy V1 graphs, not aligned/V2 caches")
        for x, edge, rel in zip(raw["all_token_embeddings"], raw["all_edge_index"], raw["all_edge_type"]):
            edge, rel = torch.as_tensor(edge), torch.as_tensor(rel)
            if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 768:
                raise ValueError("Invalid legacy node embeddings; do not use aligned/V2 graphs")
            if edge.shape != (2, rel.numel()) or rel.numel() == 0:
                raise ValueError("Invalid relation/edge shapes")
            if int(rel.min()) < 0 or int(edge.min()) < 0 or int(edge.max()) >= x.shape[0]:
                raise ValueError("Invalid relation/edge indices")
            if R and int(rel.max()) >= R:
                raise ValueError("Validation relation IDs exceed training R; refusing silent clipping")
        sizes = [len(raw[k]) for k in ("all_token_embeddings", "all_edge_index", "all_edge_type", "y")]
        if len(set(sizes)) != 1:
            raise ValueError("Cache arrays have different numbers of samples")
        return raw

    v3._load_pickle = checked_load
    try:
        ds = v3.GraphSet(str(path), R=R or None, mode=mode, store_dev="cpu")
    finally:
        v3._load_pickle = original_load
    expected = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                label = json.loads(line)["label"]
                expected.append(int(label) if isinstance(label, (int, float)) else int("human" in str(label).lower()))
    if not np.array_equal(ds.y.numpy(), expected) or set(expected) != {0, 1}:
        raise ValueError("Cache/source labels or counts differ; rebuild with building_graph.py")
    return ds


def validation_loss(probabilities, truth):
    import numpy as np

    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(-(truth * np.log(p) + (1 - truth) * np.log(1 - p)).mean())


def atomic_save(payload, path):
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def signature(args, variant, seed, data_hashes):
    config = vars(args).copy()
    for key in ("variants", "seeds", "output_dir", "graph_dir", "skip_completed", "overwrite"):
        config.pop(key)
    config["effective_readout"] = effective_readout(args, variant)
    if variant in ("baseline", "meanlogit", "classswap", "restricted", "structmap"):
        for key in ("mixstyle_p", "mixstyle_alpha", "mixstyle_strength"):
            config.pop(key)
    code = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), REPO / "model/QuangV3.py", REPO / "model/V3ClassSwap.py", REPO / "model/V3Restricted.py", REPO / "model/V3CMixStyle.py", REPO / "model/V3StructuralMap.py",
        REPO / "model/v3_mixstyle.py", REPO / "building_graph.py", REPO / "dep_vocab.py",
        REPO / "dep_vocab.json",
    )}
    record = dict(config=config, variant=variant, seed=seed, data=data_hashes,
                  code_sha256=code, environment=environment())
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return digest, record


def completed_run(path, record_path, run_signature):
    if not path.is_file() or not record_path.is_file():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return (record.get("complete") is True and record.get("run_signature") == run_signature
            and record.get("checkpoint_sha256") == sha256(path))


def train_one(args, variant, seed, train, val, run_signature, provenance):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from model.QuangV3 import predict, set_seed
    from evaluate_v3_quang import score_probabilities

    device = torch.device(args.device)
    set_seed(seed)
    model = build_model(args, variant, train.R).to(device)
    teacher = copy.deepcopy(model).eval() if args.ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(train.G / args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(total * 0.05))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.95 + 0.05,
    )
    amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    gm = train.GF.float().mean(0).cpu()
    gs = train.GF.float().std(0).cpu().clamp_min(1e-6)
    checkpoint_path = args.output_dir / f"v3_{variant}_seed{seed}.pt"
    record_path = checkpoint_path.with_suffix(".train.json")
    saved_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    saved_args.update(seeds=[seed], cons=0.0, readout=effective_readout(args, variant))
    history, best_score, best_epoch, bad, step = [], float("inf"), 0, 0, 0
    start = time.time()
    # MixStyle samples Beta coefficients on CPU; do not let those extra draws
    # change the source minibatch order in the next epoch of the ablation.
    sampler_generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"{variant} seed={seed} | R={train.R} mode={args.struct_mode} | "
          f"effective batch={args.batch_size}, micro={args.micro_batch_size}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train.G, generator=sampler_generator)
        epoch_loss = 0.0
        mixstyle_coverage = {"graphs": 0, "eligible": 0, "applied": 0}
        classswap_coverage = {"selected": 0, "eligible": 0, "swapped": 0}
        for offset in range(0, train.G, args.batch_size):
            indices = permutation[offset:offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for micro in indices.split(args.micro_batch_size):
                batch = train.batch(micro, device, gm, gs)
                weight = batch["B"] / indices.numel()
                branch_weight = 0.5 if args.aug != "none" else 1.0
                # Release clean activations BEFORE the augmented forward. The
                # effective step still averages all examples/branches exactly.
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    clean = model(batch, mixstyle=False) if variant == "cmixstyle" else model(batch)
                    clean_loss = F.binary_cross_entropy_with_logits(clean.float(), batch["y"])
                scaler.scale(weight * branch_weight * clean_loss).backward()
                epoch_loss += branch_weight * float(clean_loss.detach()) * batch["B"]
                del clean, clean_loss
                if args.aug != "none":
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        noisy = model(batch, aug_mode=args.aug)
                        noisy_loss = F.binary_cross_entropy_with_logits(noisy.float(), batch["y"])
                    scaler.scale(weight * 0.5 * noisy_loss).backward()
                    epoch_loss += 0.5 * float(noisy_loss.detach()) * batch["B"]
                    if variant == "cmixstyle":
                        for key in mixstyle_coverage:
                            mixstyle_coverage[key] += model.mixstyle.last_stats[key]
                    if variant == "classswap":
                        for key in classswap_coverage:
                            classswap_coverage[key] += model.last_swap_stats[key]
                    del noisy, noisy_loss
                if variant == "restricted":
                    restriction_loss = model.restriction_loss()
                    scaler.scale(weight * restriction_loss).backward()
                    epoch_loss += float(restriction_loss.detach()) * batch["B"]
                    del restriction_loss
                del batch
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                schedule.step()
                step += 1
                if teacher is not None:
                    decay = min(args.ema_decay, (1 + step) / (10 + step))
                    with torch.no_grad():
                        for target, current in zip(teacher.parameters(), model.parameters()):
                            target.mul_(decay).add_(current.detach(), alpha=1 - decay)
        evaluation = teacher if teacher is not None else model
        p = predict(evaluation, val, device, gm, gs, bs=args.val_batch_size, amp=amp)
        p_swap = predict(evaluation, val, device, gm, gs, bs=args.val_batch_size, amp=amp, aug_mode="swap")
        truth = val.y.numpy().astype(int)
        metrics = score_probabilities(truth, p, 0.5)
        clean_loss, swap_loss = validation_loss(p, truth), validation_loss(p_swap, truth)
        restricted_stats = model.restriction_stats() if variant == "restricted" else None
        score = {"lossaug": clean_loss + swap_loss, "loss": clean_loss,
                 "acc": -metrics["accuracy"], "auc": -metrics["auc"]}[args.select]
        if not math.isfinite(epoch_loss) or not math.isfinite(score):
            raise FloatingPointError("Non-finite loss; no completed checkpoint will be marked")
        history.append(dict(epoch=epoch, train_loss=epoch_loss / train.G, val_loss=clean_loss,
                            val_swap_loss=swap_loss, selection_score=score, **metrics,
                            mixstyle=mixstyle_coverage, classswap=classswap_coverage,
                            restricted=restricted_stats))
        if variant == "cmixstyle" and mixstyle_coverage["eligible"] == 0:
            raise RuntimeError("No eligible same-class MixStyle donors: increase micro-batch-size")
        improved = score < best_score - 1e-6
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            thresholds = np.linspace(0.05, 0.95, 91)
            accuracies = np.array([((p >= threshold) == truth).mean() for threshold in thresholds])
            threshold = float(np.median(thresholds[accuracies >= accuracies.max() - 1e-9]))
            state = {k: v.detach().cpu().clone() for k, v in evaluation.state_dict().items()}
            atomic_save(dict(
                state=state, args=saved_args, R=train.R, gm=gm, gs=gs, thr=threshold,
                method=f"v3_{variant}", struct_mode=args.struct_mode, best_epoch=best_epoch,
                training_complete=False,
                mixstyle=dict(p=args.mixstyle_p, alpha=args.mixstyle_alpha,
                              strength=args.mixstyle_strength) if variant == "cmixstyle" else None,
                classswap=dict(same_label=True, different_graph=True, same_relation=True)
                if variant == "classswap" else None,
                restricted=dict(semantic_init=args.restricted_semantic_init,
                                structural_init=args.restricted_structural_init,
                                semantic_budget=args.restricted_semantic_budget,
                                semantic_penalty=args.restricted_semantic_penalty)
                if variant == "restricted" else None,
                run_signature=run_signature, provenance=provenance,
            ), checkpoint_path)
            del state
        else:
            bad += 1
        write_json(record_path, dict(complete=False, run_signature=run_signature,
                                    provenance=provenance, best_epoch=best_epoch, history=history))
        restricted_log = ""
        if restricted_stats:
            restricted_log = (f" | Restricted sem {restricted_stats['semantic_mean']:.3f} "
                              f"(<.5 {restricted_stats['semantic_below_0_5']}) "
                              f"struct {restricted_stats['structural_mean']:.3f}")
        print(f"Epoch {epoch:02d} | loss {epoch_loss/train.G:.4f} | val {clean_loss:.4f} "
              f"swap {swap_loss:.4f} | acc {metrics['accuracy']:.4f} | "
              f"MixStyle {mixstyle_coverage['applied']}/{mixstyle_coverage['eligible']} eligible | "
              f"ClassSwap {classswap_coverage['swapped']}/{classswap_coverage['selected']} selected "
              f"{restricted_log} ({'best' if improved else 'keep'})", flush=True)
        if bad >= args.patience:
            break
    finished = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    finished["training_complete"] = True
    atomic_save(finished, checkpoint_path)
    del finished
    write_json(record_path, dict(complete=True, run_signature=run_signature,
                                checkpoint_sha256=sha256(checkpoint_path), provenance=provenance,
                                best_epoch=best_epoch, selection=args.select, history=history,
                                train_seconds=time.time() - start))
    print(f"Saved: {checkpoint_path}", flush=True)
    del model, teacher, optimizer, scaler, schedule
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU, or explicitly use --device cpu")
    args.graph_dir, args.output_dir = args.graph_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain = next(item for item in TRAINING_DOMAINS.values()
                  if (item["train_file"], item["val_file"]) == (args.train_file, args.val_file))
    sources = [REPO / domain["train_source"], REPO / domain["val_source"]]
    graphs = [args.graph_dir / f"{name}.pkl" for name in (args.train_file, args.val_file)]
    data_hashes = {str(p): sha256(p) for p in sources + graphs}
    pending = []
    for variant in args.variants:
        for seed in args.seeds:
            path = args.output_dir / f"v3_{variant}_seed{seed}.pt"
            record_path = path.with_suffix(".train.json")
            sig, provenance = signature(args, variant, seed, data_hashes)
            if args.skip_completed and completed_run(path, record_path, sig):
                print(f"Skip completed: {path.name}", flush=True)
            elif (path.exists() or record_path.exists()) and not args.overwrite:
                raise FileExistsError(f"Incomplete/different run: {path}. Use a new output directory "
                                      "or --overwrite; a saved best epoch is not proof training finished.")
            else:
                pending.append((variant, seed, sig, provenance))
    if not pending:
        return
    train = load_graph(graphs[0], sources[0], args.num_rel, args.struct_mode)
    val = load_graph(graphs[1], sources[1], train.R, args.struct_mode)
    for variant, seed, sig, provenance in pending:
        train_one(args, variant, seed, train, val, sig, provenance)


if __name__ == "__main__":
    main()
