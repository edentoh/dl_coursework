# config_utils.py
from __future__ import annotations
from pathlib import Path
import tomllib


def load_toml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with path.open("rb") as f:
        return tomllib.load(f)


def deep_get(d: dict, key_path: str, default=None):
    """
    deep_get(cfg, "detector.train.epochs", 15)
    """
    cur = d
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
