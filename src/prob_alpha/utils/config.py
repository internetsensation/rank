from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and merge included config files.

    The top-level config may contain:

    ```yaml
    include:
      data: configs/data.yml
      features: configs/features.yml
    ```

    Included keys are shallow-merged into the top-level dictionary.
    """
    path = Path(path)
    cfg = load_yaml(path)
    includes = cfg.get("include", {}) or {}
    merged: dict[str, Any] = {}
    for _, include_path in includes.items():
        include_file = Path(include_path)
        if not include_file.is_absolute():
            include_file = Path.cwd() / include_file
        merged.update(load_yaml(include_file))
    merged.update({k: v for k, v in cfg.items() if k != "include"})
    return merged


def ensure_parent_dir(path: str | Path) -> Path:
    """Create a file's parent directory and return the normalized Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
