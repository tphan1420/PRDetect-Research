"""PRDetect V4: same-seed HC3 V3 EMA and A1, with a fixed product decision."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import evaluate_v3_quang as ev
from v3_a1_fusion import AlignedScorer, human_score, labels_from, paired_scores, read_source, source_digest


THRESHOLD = 0.5
SEEDS = [2020, 2022, 2024, 2025, 2026]


@dataclass
class CheckpointPair:
    seed: int
    config: dict
    a1: Path
    a1_sha256: str

    @property
    def ema_sha256(self):
        return self.config["checkpoint_sha256"][str(self.seed)]

    @classmethod
    def load(cls, directory, seed, device, batch_size):
        args = SimpleNamespace(variant="baseline", weights="ema", train_domain="hc3",
                               checkpoint_dir=directory, seeds=[seed], device=device,
                               batch_size=batch_size, struct_mode="low", graph_budget_root=None)
        config = ev.make_config(args)
        ema = Path(config["checkpoint_paths"][str(seed)])
        a1 = ema.with_name(f"a1_seed{seed}.pt")
        pair = cls(seed, config, a1, ev.sha256(a1))
        manifest = ema.parent / "manifest.json"
        if manifest.is_file():
            records = ev.read_json(manifest)
            for path, digest in ((ema, pair.ema_sha256), (a1, pair.a1_sha256)):
                matches = [r["sha256"] for r in records if r["seed"] == seed and r["file"] == path.name]
                if matches != [digest]:
                    raise ValueError(f"Checkpoint differs from manifest: {path}")
        return pair

    def fuse(self, source, truth, v3, a1):
        if (v3["seed"] != self.seed or v3["checkpoint_sha256"] != self.ema_sha256
                or a1["checkpoint_sha256"] != self.a1_sha256):
            raise ValueError(f"Expected V4 EMA and A1 reports for seed {self.seed}")
        p = paired_scores(source, truth, v3, a1)
        return dict(model="v4", seed=self.seed, source_sha256=a1["source_sha256"], samples=len(truth),
                    threshold=THRESHOLD, metrics=ev.score_probabilities(truth, p, THRESHOLD),
                    v3_sha256=self.ema_sha256, a1_sha256=self.a1_sha256,
                    y_true=truth.tolist(), p_human_fusion=p.tolist())


def predict(pair, text):
    import torch
    device = torch.device(pair.config["device"])
    _, checkpoint, ema = ev._load_model(pair.config, pair.seed, device)
    scorer = AlignedScorer(pair.a1, device)
    doc = scorer.nlp(text)
    ema_p = scorer.v3_probability(doc, ema, amp=bool(checkpoint["args"].get("amp")))
    logit = scorer.score_doc(doc)
    p = float(human_score(ema_p, logit))
    return dict(model="v4", seed=pair.seed, v3_sha256=pair.ema_sha256, a1_sha256=pair.a1_sha256,
                ema_p_human=ema_p, a1_logit_human=logit, p_human=p, threshold=THRESHOLD,
                label="human" if p >= THRESHOLD else "ai")


def score_a1_split(scorer, heads, rows):
    """Encode each distinct text once and use its full-precision graph for every A1 head."""
    texts = [row["text"] for row in rows]
    unique = dict.fromkeys(texts)
    logits = {seed: {} for seed in heads}
    for i, doc in enumerate(scorer.nlp.pipe(unique, batch_size=16), start=1):
        inputs = scorer.aligned_graph(doc)
        for seed, head in heads.items():
            logits[seed][doc.text] = scorer.a1_logit(head, inputs)
        if i % 100 == 0:
            print(f"A1: {i}/{len(unique)} texts, {len(heads)} seeds", flush=True)
    return {seed: [scores[text] for text in texts] for seed, scores in logits.items()}


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def evaluate(pairs, cases, out):
    out.mkdir(parents=True, exist_ok=True)
    device = pairs[0].config["device"]
    scorer = AlignedScorer(None, device)
    heads = {p.seed: scorer.load_a1(p.a1, device) for p in pairs}
    sources = defaultdict(list)
    for case in cases:
        sources[case.source].append(case)
    results = []
    for relative, matching in sources.items():
        source = ev.REPO / relative
        rows = read_source(source)
        truth = labels_from(rows)
        digest = source_digest(source)
        print(f"Evaluating {source.stem}", flush=True)
        logits = score_a1_split(scorer, heads, rows)
        with TemporaryDirectory(dir=out) as tmp:
            # building_graph writes relative to cwd; isolate each split and clean it on exit.
            local_encoder = ev.REPO / "roberta-base"
            if (local_encoder / "vocab.json").is_file():
                (Path(tmp) / "roberta-base").symlink_to(local_encoder, target_is_directory=True)
            subprocess.run([sys.executable, str(ev.REPO / "building_graph.py"), "--file", str(source)],
                           cwd=tmp, check=True)
            graph = Path(tmp) / "graph_data" / f"{source.stem}.pkl"
            for pair in pairs:
                dest = out / f"seed{pair.seed}"
                a1 = dict(condition=source.stem, source_sha256=digest, checkpoint_sha256=pair.a1_sha256,
                          score_space="human_logit", records=[
                              dict(index=i, label=int(y), logit_human=z)
                              for i, (y, z) in enumerate(zip(truth, logits[pair.seed]))])
                v3 = ev.infer_checkpoint(pair.config, pair.seed, graph, len(rows), digest)
                report = pair.fuse(source, truth, v3, a1)
                ev.write_json(dest / f"scores_{source.stem}.json", a1)
                ev.write_json(dest / f"{source.stem}_v3.json", v3)
                ev.write_json(dest / f"{source.stem}_fusion.json", {**report, "source": relative})
                for case in matching:
                    results.append(dict(table=case.suite, condition=case.condition, noise_percent=case.noise_percent,
                                        seed=pair.seed, source=relative, samples=len(rows),
                                        **{m: report["metrics"][m] for m in ("accuracy", "f1", "auc")}))
        write_csv(out / "results_by_seed.csv", results)
    summary = []
    for case in cases:
        vals = [r["accuracy"] for r in results if r["source"] == case.source and r["table"] == case.suite]
        summary.append(dict(table=case.suite, condition=case.condition, noise_percent=case.noise_percent,
                            source=case.source, seeds=len(vals), accuracy_mean=statistics.mean(vals),
                            accuracy_std=statistics.stdev(vals) if len(vals) > 1 else None,
                            accuracy_min=min(vals), accuracy_max=max(vals)))
    write_csv(out / "summary.csv", summary)
    ev.write_json(out / "manifest.json", dict(model="v4", threshold=THRESHOLD,
                  seeds=[p.seed for p in pairs], cases=[vars(c) for c in cases],
                  checkpoints=[dict(seed=p.seed, ema_sha256=p.ema_sha256, a1_sha256=p.a1_sha256)
                               for p in pairs]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("predict")
    single.add_argument("--seed", required=True, type=int)
    single.add_argument("--text", required=True)
    tables = commands.add_parser("evaluate")
    tables.add_argument("--suite", choices=["three", "full", "table2", "table3", "table4"], default="full")
    tables.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    tables.add_argument("--batch-size", type=int, default=4, help="V3 inference batch size")
    tables.add_argument("--output-dir", type=Path, required=True)
    for sub in (single, tables):
        sub.add_argument("--checkpoint-dir", type=Path, required=True)
        sub.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.command == "predict":
        pair = CheckpointPair.load(args.checkpoint_dir, args.seed, args.device, 1)
        print(json.dumps(predict(pair, args.text)))
        return
    if len(set(args.seeds)) != len(args.seeds) or args.batch_size < 1:
        parser.error("Use unique seeds and a positive batch size")
    pairs = [CheckpointPair.load(args.checkpoint_dir, s, args.device, args.batch_size) for s in args.seeds]
    cases = ev.benchmark_cases("full" if args.suite == "three" else args.suite, "hc3")
    if args.suite == "three":
        cases = [c for c in cases if (c.suite, c.condition) in
                 {("table2", "Original"), ("table3", "Original"), ("table4", "Replace 25%")}]
    evaluate(pairs, cases, args.output_dir)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
