"""Train V3 once and select RAW and EMA independently on HC3 validation."""
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from evaluate_v3_quang import REPO, TRAINING_DOMAINS, environment, sha256, write_json
from train_v3 import (atomic_save, build_model, build_parser, configure_determinism,
                      effective_readout, load_graph, validate_args, validation_loss)


def signature(args, variant, seed, data_hashes):
    config = vars(args).copy()
    for key in ("variants", "seeds", "output_dir", "graph_dir", "skip_completed", "overwrite"):
        config.pop(key)
    config["effective_readout"] = effective_readout(args, variant)
    code = {str(p.relative_to(REPO)): sha256(p) for p in (
        Path(__file__), REPO / "train_v3.py", REPO / "model/QuangV3.py", REPO / "building_graph.py", REPO / "dep_vocab.py",
        REPO / "dep_vocab.json",
    )}
    record = dict(config=config, variant=variant, seed=seed, data=data_hashes,
                  code_sha256=code, environment=environment())
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return digest, record


def completed_run(paths, record_path, run_signature):
    if not record_path.is_file() or not all(path.is_file() for path in paths.values()):
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return (record.get("complete") is True and record.get("run_signature") == run_signature
            and record.get("checkpoint_sha256") == {w: sha256(p) for w, p in paths.items()})


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
    ema_checkpoint_path = args.output_dir / f"v3_{variant}_ema_seed{seed}.pt"
    raw_checkpoint_path = args.output_dir / f"v3_{variant}_raw_seed{seed}.pt"
    record_path = args.output_dir / f"v3_{variant}_dual_seed{seed}.train.json"
    saved_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    saved_args.update(seeds=[seed], cons=0.0, readout=effective_readout(args, variant))
    history = []
    best_ema_score = best_raw_score = float("inf")
    best_ema_epoch = best_raw_epoch = 0
    bad = step = 0
    start = time.time()
    sampler_generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"{variant} seed={seed} | R={train.R} mode={args.struct_mode} | "
          f"effective batch={args.batch_size}, micro={args.micro_batch_size}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train.G, generator=sampler_generator)
        epoch_loss = 0.0
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
                    clean = model(batch)
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
                    del noisy, noisy_loss
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
        # Each weight family chooses its own best validation epoch.  The raw
        # model and EMA teacher therefore receive identical training updates,
        # but neither is selected using the other model's validation scores.
        ema_model = teacher if teacher is not None else model
        raw_model = model
        truth = val.y.numpy().astype(int)

        def validate(candidate):
            probabilities = predict(candidate, val, device, gm, gs, bs=args.val_batch_size, amp=amp)
            swapped = predict(candidate, val, device, gm, gs, bs=args.val_batch_size, amp=amp, aug_mode="swap")
            metrics = score_probabilities(truth, probabilities, 0.5)
            clean_loss = validation_loss(probabilities, truth)
            swap_loss = validation_loss(swapped, truth)
            score = {"lossaug": clean_loss + swap_loss, "loss": clean_loss,
                     "acc": -metrics["accuracy"], "auc": -metrics["auc"]}[args.select]
            thresholds = np.linspace(0.05, 0.95, 91)
            accuracies = np.array([((probabilities >= threshold) == truth).mean() for threshold in thresholds])
            threshold = float(np.median(thresholds[accuracies >= accuracies.max() - 1e-9]))
            return probabilities, clean_loss, swap_loss, score, metrics, threshold

        ema_p, ema_clean, ema_swap, ema_score, ema_metrics, ema_threshold = validate(ema_model)
        raw_p, raw_clean, raw_swap, raw_score, raw_metrics, raw_threshold = validate(raw_model)
        if (not math.isfinite(epoch_loss) or not math.isfinite(ema_score)
                or not math.isfinite(raw_score)):
            raise FloatingPointError("Non-finite loss; no completed checkpoint will be marked")

        ema_improved = ema_score < best_ema_score - 1e-6
        raw_improved = raw_score < best_raw_score - 1e-6
        history.append(dict(
            epoch=epoch, train_loss=epoch_loss / train.G,
            ema_val_loss=ema_clean, ema_val_swap_loss=ema_swap,
            ema_selection_score=ema_score, ema_metrics=ema_metrics,
            raw_val_loss=raw_clean, raw_val_swap_loss=raw_swap,
            raw_selection_score=raw_score, raw_metrics=raw_metrics,
            ema_improved=ema_improved, raw_improved=raw_improved,
        ))

        # Clone both states before saving.  The counterpart state is retained
        # only as provenance; `state` is exactly the weight family being tested.
        current_raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        current_ema = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        common = dict(args=saved_args, R=train.R, gm=gm, gs=gs,
                      method=f"v3_{variant}", struct_mode=args.struct_mode,
                      training_complete=False, run_signature=run_signature,
                      provenance=provenance, dual_selection=True)
        if ema_improved:
            best_ema_score, best_ema_epoch = ema_score, epoch
            atomic_save(dict(
                state=current_ema, raw_state=current_raw, ema_state=current_ema,
                human95_threshold=float(np.quantile(ema_p[truth == 1], 0.05, method="lower")), thr=ema_threshold, best_epoch=epoch, weight_source="ema",
                selection_model="ema", selection_score=ema_score, **common,
            ), ema_checkpoint_path)
        if raw_improved:
            best_raw_score, best_raw_epoch = raw_score, epoch
            atomic_save(dict(
                state=current_raw, raw_state=current_raw, ema_state=current_ema,
                human95_threshold=float(np.quantile(raw_p[truth == 1], 0.05, method="lower")), thr=raw_threshold, best_epoch=epoch, weight_source="raw",
                selection_model="raw", selection_score=raw_score, **common,
            ), raw_checkpoint_path)
        bad = 0 if (ema_improved or raw_improved) else bad + 1
        write_json(record_path, dict(
            complete=False, run_signature=run_signature, provenance=provenance,
            best_ema_epoch=best_ema_epoch, best_raw_epoch=best_raw_epoch,
            best_ema_score=best_ema_score, best_raw_score=best_raw_score,
            history=history,
        ))
        print(f"Epoch {epoch:02d} | loss {epoch_loss/train.G:.4f} | "
              f"EMA val {ema_clean:.4f} ({'best' if ema_improved else 'keep'}) | "
              f"Raw val {raw_clean:.4f} ({'best' if raw_improved else 'keep'})", flush=True)
        if bad >= args.patience:
            break

    for checkpoint_path in (ema_checkpoint_path, raw_checkpoint_path):
        finished = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        finished["training_complete"] = True
        atomic_save(finished, checkpoint_path)
    write_json(record_path, dict(
        complete=True, run_signature=run_signature, provenance=provenance,
        checkpoint_sha256={"ema": sha256(ema_checkpoint_path), "raw": sha256(raw_checkpoint_path)},
        best_ema_epoch=best_ema_epoch, best_raw_epoch=best_raw_epoch,
        best_ema_score=best_ema_score, best_raw_score=best_raw_score,
        selection=args.select, history=history, train_seconds=time.time() - start,
    ))
    print(f"Saved EMA: {ema_checkpoint_path}", flush=True)
    print(f"Saved Raw: {raw_checkpoint_path}", flush=True)
    del model, teacher, optimizer, scaler, schedule
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.determinism = configure_determinism(args.deterministic)
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
            paths = {w: args.output_dir / f"v3_{variant}_{w}_seed{seed}.pt" for w in ("raw", "ema")}
            record_path = args.output_dir / f"v3_{variant}_dual_seed{seed}.train.json"
            sig, provenance = signature(args, variant, seed, data_hashes)
            if args.skip_completed and completed_run(paths, record_path, sig):
                print(f"Reuse completed RAW and EMA checkpoints for seed {seed}", flush=True)
            elif (any(path.exists() for path in paths.values()) or record_path.exists()) and not args.overwrite:
                raise FileExistsError(f"Incomplete/different dual run: {record_path}. Use a new output directory "
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
