"""Configuration and path management utilities.

All filesystem paths should be resolved relative to the repository root
using the functions defined here to avoid hardcoded absolute paths.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


def resolve_path(relative_or_absolute_path: str | Path) -> Path:
    """Resolve a path relative to the repository root if it is not already absolute."""
    path = Path(relative_or_absolute_path)
    if path.is_absolute():
        return path
    return get_repo_root() / path


def ensure_dir(dir_path: str | Path) -> Path:
    """Ensure a directory exists, creating parent directories if needed."""
    resolved = resolve_path(dir_path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file with path resolution."""
    resolved_path = resolve_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}
