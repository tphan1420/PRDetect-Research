import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def config_cli(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    return parser


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("seed", 2025)
    cfg.setdefault("parser", "en_core_web_sm")
    cfg.setdefault("device", "cpu")
    cfg.setdefault("threads", 2)
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    return cfg
