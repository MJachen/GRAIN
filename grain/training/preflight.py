"""Read-only/full-CV readiness checks for one formal dataset configuration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from grain.data import load_legacy_feature_table, generate_nested_manifests_from_arrays
from grain.data.paths import RepositoryPathPolicy
from grain.evaluation import compute_metrics
from grain.models import GRAIN

from .cross_validation import build_fold_runtime, build_model


ARCHITECTURE_FIELDS = (
    "feature_dimension",
    "patient_dimension",
    "relation_dimension",
    "gat_hidden_dimension",
    "transformer_dimension",
    "fusion_dimension",
    "classifier_hidden_dimension",
    "attention_tokens",
    "transformer_ffn_dimension",
    "relation_activation",
    "relation_alpha",
    "relation_gamma",
    "graph_attention_heads",
    "graph_layers",
    "fusion_attention_heads",
    "gat_dropout",
    "transformer_dropout",
)


def run_preflight(config: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    data = config["data"]
    policy = RepositoryPathPolicy.create(repository_root, data["source_root"])
    try:
        source = policy.resolve_source(data["feature_source"])
        record("dataset_path_exists", source.is_file(), source.as_posix())
    except Exception as error:
        source = Path(data["source_root"]) / str(data.get("feature_source"))
        record("dataset_path_exists", False, str(error))

    inventory_path = repository_root / "artifacts" / "data_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected_sha = data.get("expected_sha256")
    inventory_matches = [
        item for item in inventory["datasets"] if item["sha256"] == expected_sha
    ]
    record(
        "data_sha256_recorded",
        bool(expected_sha and inventory_matches),
        expected_sha or "missing",
    )
    layout = data.get("legacy_layout", {})
    layout_ok = all(
        layout.get(key) is not None
        for key in ("skip_leading_columns", "plain_dimension", "ce_dimension")
    )
    record("modality_layout_explicit", layout_ok, json.dumps(layout, sort_keys=True))
    input_ok = config["model"].get("feature_dimension") is not None
    record("input_dimension_explicit", input_ok, str(config["model"].get("feature_dimension")))

    unresolved = [key for key in ARCHITECTURE_FIELDS if config["model"].get(key) is None]
    record("architecture_fully_resolved", not unresolved, str(unresolved))
    training = config["training"]
    optimizer_ok = (
        training.get("optimizer") == "adam"
        and training.get("learning_rate") is not None
        and training.get("weight_decay") is not None
    )
    record("optimizer_resolved", optimizer_ok, str(training.get("optimizer")))
    k_config = training.get("k_selection", {})
    k_ok = k_config.get("mode") in {"fixed", "validation"} and (
        k_config.get("mode") != "validation" or int(k_config.get("search_epochs") or 0) > 0
    )
    record("k_protocol_resolved", k_ok, json.dumps(k_config, sort_keys=True))
    record("max_epochs_resolved", int(training.get("epochs") or 0) > 0, str(training.get("epochs")))

    cohort = None
    try:
        cohort = load_legacy_feature_table(
            source,
            sample_id_prefix=data["sample_id_prefix"],
            skip_leading_columns=int(layout["skip_leading_columns"]),
            plain_dimension=int(layout["plain_dimension"]),
            ce_dimension=int(layout["ce_dimension"]),
            label_column=data["label_column"],
            expected_sha256=expected_sha,
        )
        record("explicit_mask", cohort.modality_mask.dtype == torch.bool, str(cohort.modality_mask.dtype))
    except Exception as error:
        record("explicit_mask", False, f"{type(error).__name__}: {error}")

    record(
        "label_free_inference",
        "labels" not in inspect.signature(GRAIN.forward).parameters,
        str(tuple(inspect.signature(GRAIN.forward).parameters)),
    )
    record(
        "validation_only_checkpoint",
        training.get("checkpoint_metric") == "validation_auc",
        str(training.get("checkpoint_metric")),
    )
    try:
        auc = compute_metrics(
            np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.4, 0.35, 0.3])
        ).auc
        record("probability_auc", abs(auc - 0.75) < 1e-12, str(auc))
    except Exception as error:
        record("probability_auc", False, str(error))

    external = bool(config["evaluation"].get("external_only", False))
    if cohort is not None and not external:
        try:
            split = generate_nested_manifests_from_arrays(
                cohort.sample_ids,
                cohort.labels.tolist(),
                config["experiment"]["name"],
                int(config["split"]["n_outer_folds"]),
                float(config["split"]["validation_fraction"]),
                int(config["split"]["seed"]),
            )[0]
            runtime = build_fold_runtime(
                config=config,
                cohort=cohort,
                split=split,
                selected_k=2,
                device=torch.device("cpu"),
                seed=int(config["experiment"]["seed"]),
            )
            runtime.anchor_bank.assert_excludes(tuple(split.validation) + tuple(split.test))
            record("training_only_anchors", True, f"fold0 anchors={len(runtime.anchor_bank.patient_ids)}")
            validation_features, validation_masks, _, validation_ids = cohort.select(
                split.validation[:2]
            )
            runtime.model.eval()
            with torch.no_grad():
                output = runtime.model(
                    validation_features.to(torch.device("cpu")),
                    validation_masks.to(torch.device("cpu")),
                    runtime.anchor_bank,
                    validation_ids,
                )
            finite = bool(torch.isfinite(output.probabilities).all())
            record(
                "formal_model_forward",
                finite and output.probabilities.shape == (len(validation_ids),),
                f"shape={tuple(output.probabilities.shape)}, finite={finite}",
            )
        except Exception as error:
            record("training_only_anchors", False, f"{type(error).__name__}: {error}")
            record("formal_model_forward", False, f"{type(error).__name__}: {error}")
    elif external:
        record("training_only_anchors", True, "external target uses source-training anchors only")
        try:
            build_model(config, selected_k=2)
            record("formal_model_construction", True, "external profile constructs on CPU")
        except Exception as error:
            record("formal_model_construction", False, f"{type(error).__name__}: {error}")
    else:
        record("training_only_anchors", False, "dataset did not pass explicit-mask loading")
        try:
            build_model(config, selected_k=2)
            record("formal_model_construction", True, "profile constructs on CPU")
        except Exception as error:
            record("formal_model_construction", False, f"{type(error).__name__}: {error}")

    try:
        output_dir = repository_root / str(config["experiment"]["output_root"])
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".formal_preflight_write_probe"
        try:
            with probe.open("x", encoding="utf-8") as handle:
                handle.write("write-probe\n")
        finally:
            if probe.is_file():
                probe.unlink()
        record("output_directory_writable", True, output_dir.as_posix())
    except Exception as error:
        record("output_directory_writable", False, str(error))

    try:
        policy.resolve_write(source)
        record("legacy_dataset_read_only_policy", False, "source accepted as write target")
    except ValueError:
        record("legacy_dataset_read_only_policy", True, source.as_posix())

    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository_root, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    record("git_worktree_clean", not status, status or "clean")
    record("git_commit_recorded", bool(commit), commit)

    alignment = config.get("external_alignment")
    if alignment:
        alignment_ok = alignment.get("status") == "READY"
        if external:
            record(
                "external_feature_alignment",
                alignment_ok,
                alignment.get("reason", alignment.get("status")),
            )
        else:
            record(
                "external_feature_alignment_not_required_for_internal_cv",
                True,
                alignment.get("reason", alignment.get("status")),
            )

    declared_readiness = config.get("formal_readiness")
    if declared_readiness:
        record(
            "declared_dataset_readiness",
            declared_readiness.get("status") == "READY",
            declared_readiness.get("reason", declared_readiness.get("status")),
        )

    blockers = [name for name, value in checks.items() if not value["pass"]]
    return {
        "dataset": config["experiment"]["name"],
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "git_commit": commit,
    }
