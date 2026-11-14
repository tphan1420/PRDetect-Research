"""Checkpoint helpers that keep LISA-GCN checkpoints safe and portable."""

from pathlib import Path, PosixPath, WindowsPath

import torch


def load_lisa_checkpoint(path, map_location):
    """Load a trusted LISA-GCN checkpoint without enabling arbitrary pickle.

    Early checkpoints stored ``argparse`` values verbatim, including a
    ``pathlib.Path`` under ``config['output']``.  PyTorch 2.6 blocks that
    harmless class under ``weights_only=True`` unless it is explicitly
    allowlisted.  New checkpoints avoid the issue through ``checkpoint_config``.
    """
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        # ``weights_only`` was introduced after older supported PyTorch builds.
        # LISA-GCN checkpoints are local artifacts created by this project.
        return torch.load(path, map_location=map_location)
    with safe_globals([Path, PosixPath, WindowsPath]):
        return torch.load(path, map_location=map_location, weights_only=True)


def checkpoint_config(args):
    """Convert parsed arguments to checkpoint metadata accepted by weights-only load."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
