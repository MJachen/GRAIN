"""Standard-library JSON configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


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
    """Load JSON, resolve one local parent config and validate invariants."""

    config_path = Path(path).resolve()
    if config_path.suffix.lower() != ".json":
        raise ConfigurationError(
            "Official configs use JSON so the legacy irae environment needs no new package."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        current = json.load(handle)
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
    for field in ("enforce_fingerprint", "require_inventory_registration"):
        if field in data and not isinstance(data[field], bool):
            raise ConfigurationError(f"data.{field} must be a boolean")
    if data.get("enforce_fingerprint", False) and not data.get("expected_sha256"):
        raise ConfigurationError("data.expected_sha256 is required when fingerprint enforcement is enabled")
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
    model_dimension = config.get("model", {}).get("feature_dimension")
    modality_dimensions = [modalities[name]["dimension"] for name in ("plain", "ce")]
    if model_dimension is not None and any(
        value is not None and int(value) != int(model_dimension)
        for value in modality_dimensions
    ):
        raise ConfigurationError(
            "model.feature_dimension must match each configured within-cohort modality dimension"
        )

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
    k_selection = config["training"].get("k_selection", {})
    if k_selection.get("mode") not in {"fixed", "validation"}:
        raise ConfigurationError("training.k_selection.mode must be fixed or validation")
    if k_selection.get("mode") == "fixed" and k_selection.get("fixed_k") not in range(2, 11):
        raise ConfigurationError("fixed_k must be between 2 and 10")
    if int(config["evaluation"].get("positive_class", -1)) != 1:
        raise ConfigurationError("The official positive class must be 1")
