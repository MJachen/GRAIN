"""Configuration loading with explicit inheritance and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates the official schema."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, resolve one local parent config and validate invariants."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ConfigurationError("The configuration root must be a mapping.")

    parent_name = current.pop("inherits", None)
    if parent_name:
        parent_path = (config_path.parent / str(parent_name)).resolve()
        parent = load_config(parent_path)
        current = _deep_merge(parent, current)

    current["_meta"] = {"config_path": str(config_path)}
    validate_config(current)
    return current


def validate_config(config: dict[str, Any]) -> None:
    required = {"experiment", "data", "split", "training", "evaluation"}
    missing = required.difference(config)
    if missing:
        raise ConfigurationError(f"Missing top-level sections: {sorted(missing)}")

    data = config["data"]
    modalities = data.get("modalities", {})
    if tuple(modalities) != ("plain", "ce"):
        raise ConfigurationError("Modalities must be ordered exactly as plain, ce.")
    for name in ("plain", "ce"):
        item = modalities[name]
        for field in ("available_column", "path_column", "dimension"):
            if field not in item:
                raise ConfigurationError(f"data.modalities.{name}.{field} is required")
        if item["dimension"] is not None and int(item["dimension"]) <= 0:
            raise ConfigurationError(f"{name} dimension must be positive")

    split = config["split"]
    if int(split["n_outer_folds"]) < 2:
        raise ConfigurationError("n_outer_folds must be at least 2")
    fraction = float(split["validation_fraction"])
    if not 0.0 < fraction < 1.0:
        raise ConfigurationError("validation_fraction must be between 0 and 1")
    if list(split["k_candidates"]) != list(range(2, 11)):
        raise ConfigurationError("K candidates must be exactly 2 through 10")

    if config["training"].get("checkpoint_metric") != "validation_auc":
        raise ConfigurationError("Checkpoint selection must use validation_auc")
    if int(config["evaluation"].get("positive_class", -1)) != 1:
        raise ConfigurationError("The official positive class must be 1")
