"""Train the standalone source-only LISA-GCN Core on raw JSONL documents."""

import argparse
import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from lisa_gcn.checkpoint import checkpoint_config, load_lisa_checkpoint
from lisa_gcn.data import build_documents, length_quantile_groups, read_jsonl
from lisa_gcn.model import LisaGCN, mean_document_logits
from lisa_gcn.objective import GroupDRO, classification_losses, scale_consistency


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def document_logits(model, documents, scale, device, window_chunk_size):
    """Encode windows in chunks without detaching their gradients."""
    graphs, ids = [], []
    for local_id, document in enumerate(documents):
        view = document.graphs_by_scale[scale]
        graphs.extend(view)
        ids.extend([local_id] * len(view))
    if not graphs:
        raise ValueError("a document batch must contain windows")
    logits = []
    for start in range(0, len(graphs), window_chunk_size):
        batch = Batch.from_data_list(graphs[start:start + window_chunk_size]).to(device)
        logits.append(model(batch))
    return mean_document_logits(torch.cat(logits), torch.tensor(ids, device=device), len(documents))


def paired_document_logits(model, documents, alternate_scale, device, micro_batch_size, window_chunk_size):
    """Accumulate document logits across micro-batches before one optimizer step."""
    primary, alternate = [], []
    for start in range(0, len(documents), micro_batch_size):
        micro_documents = documents[start:start + micro_batch_size]
        primary.append(document_logits(model, micro_documents, 128, device, window_chunk_size))
        alternate.append(document_logits(model, micro_documents, alternate_scale, device, window_chunk_size))
    return torch.cat(primary), torch.cat(alternate)


def validation_loss(model, documents, device, window_chunk_size):
    model.eval()
    losses = []
    with torch.no_grad():
        for document in documents:
            logit = document_logits(model, [document], 128, device, window_chunk_size)
            losses.append(F.binary_cross_entropy_with_logits(logit, torch.tensor([document.label], device=device)))
    return torch.stack(losses).mean().item()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--output", type=Path, default=Path("model/lisa_gcn"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--effective-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1,
                        help="Documents forwarded at once; gradients accumulate into one effective-batch loss.")
    parser.add_argument("--window-chunk-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--scale-warmup-epochs", type=int, default=2)
    parser.add_argument("--group-dro-eta", type=float, default=0.01)
    parser.add_argument("--resume")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.effective_batch_size < 1 or args.micro_batch_size < 1 or args.window_chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    if args.micro_batch_size > args.effective_batch_size:
        raise ValueError("micro-batch-size must not exceed effective-batch-size")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train = build_documents(read_jsonl(args.train_json))
    validation = build_documents(read_jsonl(args.val_json))
    boundaries, train_groups = length_quantile_groups([item.token_count for item in train])
    model = LisaGCN(hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dro = GroupDRO(groups=4, eta=args.group_dro_eta).to(device)
    start_epoch, best, bad = 1, math.inf, 0
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / f"lisa_gcn_seed{args.seed}.pt"
    if args.resume:
        state = load_lisa_checkpoint(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        dro.log_q = state["group_dro_log_q"].to(device)
        start_epoch, best, bad = state["epoch"] + 1, state["best_val_loss"], state["bad_epochs"]

    history = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        batch_losses = []
        for start in range(0, len(order), args.effective_batch_size):
            indices = order[start:start + args.effective_batch_size]
            documents = [train[index] for index in indices]
            labels = torch.tensor([item.label for item in documents], device=device)
            groups = torch.tensor([train_groups[index] for index in indices], device=device)
            alternate_scale = (96, 160)[torch.randint(0, 2, (), generator=generator).item()]
            first, second = paired_document_logits(model, documents, alternate_scale, device,
                                                    args.micro_batch_size, args.window_chunk_size)
            group_loss, group_means, present = dro.loss(classification_losses(first, second, labels), groups)
            alpha = args.alpha if epoch > args.scale_warmup_epochs else 0.0
            total = group_loss + alpha * scale_consistency(first, second)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            batch_losses.append(float(total.detach()))

        val_loss = validation_loss(model, validation, device, args.window_chunk_size)
        row = {"epoch": epoch, "train_loss": sum(batch_losses) / len(batch_losses), "val_loss": val_loss,
               "group_weights": dro.weights.detach().cpu().tolist(), "alpha": alpha}
        history.append(row)
        improved = val_loss < best
        if improved:
            best, bad = val_loss, 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "best_val_loss": best, "bad_epochs": bad, "config": checkpoint_config(args),
                        "group_boundaries": boundaries, "group_dro_log_q": dro.log_q.cpu(),
                        "preprocessing_version": "lisa-sentence-window-v1", "history": history}, checkpoint_path)
        else:
            bad += 1
        print(json.dumps({**row, "best": improved}), flush=True)
        if bad >= args.patience:
            break
    print(f"Saved best source-validation checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
