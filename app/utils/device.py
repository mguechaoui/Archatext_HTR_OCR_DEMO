"""Resolve the compute device once, consistently, across all model loaders."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """
    Resolve a device string the way every model loader in this codebase
    should: respect an explicit choice, otherwise pick the best available
    backend, falling back to CPU.
    """
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        logger.warning("PyTorch not importable while resolving device; defaulting to CPU")

    return "cpu"
