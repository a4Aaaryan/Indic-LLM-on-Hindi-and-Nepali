"""Generic helper utilities for logging, reproducibility, and I/O."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List

from common.config import ensure_dir, resolve_path


def setup_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            resolved_log = resolve_path(log_file)
            ensure_dir(resolved_log.parent)
            file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard libraries for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save an object to a JSON file."""
    resolved_path = resolve_path(path)
    ensure_dir(resolved_path.parent)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load an object from a JSON file."""
    resolved_path = resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict[str, Any]], path: str | Path) -> None:
    """Save a list of dictionaries to a JSONL file."""
    resolved_path = resolve_path(path)
    ensure_dir(resolved_path.parent)
    with open(resolved_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield records line-by-line from a JSONL file."""
    resolved_path = resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
