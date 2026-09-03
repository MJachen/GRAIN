"""Independent integrity audit for a completed formal GRAIN CV run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .metrics import compute_metrics


OFFICIAL_METRICS = ("acc", "f1", "rec", "auc", "pre", "spec", "npv")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_formal_run(output_root: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    repository = Path(repository_root).resolve()
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    manifest = _read_json(root / "run_manifest.json")
    expected_count = int(manifest["expected_samples"])
    expected_sha = manifest["data_sha256"]
    expected_commit = manifest["git_commit"]
    record("run_complete", manifest.get("status") == "COMPLETE", manifest.get("status"))
    fold_dirs = sorted(path for path in root.glob("fold_*" ) if path.is_dir())
    record("fold_completeness", len(fold_dirs) == 10, [path.name for path in fold_dirs])

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    record("git_commit_consistency", current_commit == expected_commit, current_commit)
    source = Path(manifest["data_source"])
    record(
        "data_fingerprint_current",
        source.is_file() and _sha256(source) == expected_sha,
        source.as_posix(),
    )

    all_test_rows: list[dict[str, str]] = []
    reference_universe: set[str] | None = None
    fold_failures: dict[str, list[str]] = {}
    for fold in range(10):
        fold_name = f"fold_{fold:02d}"
        fold_dir = root / fold_name
        failures: list[str] = []
        required = (
            "resolved_config.json", "split.json", "data_fingerprint.json", "k_search.csv",
            "train_history.csv", "best.pt", "checkpoint_metadata.json",
            "validation_predictions.csv", "test_predictions.csv", "metrics.json",
        )
        missing = [name for name in required if not (fold_dir / name).is_file()]
        if missing:
            failures.append(f"missing:{missing}")
            fold_failures[fold_name] = failures
            continue

        split = _read_json(fold_dir / "split.json")
        train, validation, test = map(set, (split["train"], split["validation"], split["test"]))
        if train & validation or train & test or validation & test:
            failures.append("split_overlap")
        universe = train | validation | test
        if reference_universe is None:
            reference_universe = universe
        elif universe != reference_universe:
            failures.append("fold_universe_mismatch")

        fingerprint = _read_json(fold_dir / "data_fingerprint.json")
        if fingerprint.get("sha256") != expected_sha:
            failures.append("data_fingerprint_mismatch")
        config = _read_json(fold_dir / "resolved_config.json")
        if config["training"]["checkpoint_metric"] != "validation_auc":
            failures.append("checkpoint_metric_not_validation_auc")
        if config["training"]["k_selection"]["mode"] != "validation":
            failures.append("k_selection_not_validation")

        k_rows = _read_csv(fold_dir / "k_search.csv")
        try:
            candidate_values = [int(row["candidate_k"]) for row in k_rows]
            selected_rows = [row for row in k_rows if row["selected"].lower() == "true"]
            selected_by_rule = max(
                k_rows, key=lambda row: (float(row["validation_auc"]), -int(row["candidate_k"]))
            )
            if candidate_values != list(range(2, 11)):
                failures.append("k_candidates_invalid")
            if len(selected_rows) != 1 or selected_rows[0]["candidate_k"] != selected_by_rule["candidate_k"]:
                failures.append("k_selection_rule_invalid")
        except (KeyError, ValueError):
            failures.append("k_search_unparseable")

        history = _read_csv(fold_dir / "train_history.csv")
        if len(history) != 100 or [int(row["epoch"]) for row in history] != list(range(100)):
            failures.append("training_history_incomplete")
        numeric_history_fields = (
            "loss", "classification", "auxiliary", "contrastive", "balance",
            "gradient_norm", "validation_loss", "validation_auc",
        )
        try:
            if not all(
                np.isfinite(float(row[field])) for row in history for field in numeric_history_fields
            ):
                failures.append("nonfinite_training_history")
        except (KeyError, ValueError):
            failures.append("training_history_unparseable")

        metadata = _read_json(fold_dir / "checkpoint_metadata.json")
        if metadata.get("selection_split") != "validation" or metadata.get("selection_metric") != "validation_auc":
            failures.append("checkpoint_provenance_invalid")
        if int(metadata.get("test_evaluation_count", -1)) != 1:
            failures.append("test_evaluation_count_not_one")
        if metadata.get("git_commit") != expected_commit:
            failures.append("fold_git_commit_mismatch")
        if metadata.get("data_sha256") != expected_sha:
            failures.append("fold_data_sha_mismatch")
        if metadata.get("checkpoint_sha256") != _sha256(fold_dir / "best.pt"):
            failures.append("checkpoint_sha_mismatch")
        anchors = set(metadata.get("training_anchor_ids", []))
        if not anchors.issubset(train) or anchors & (validation | test):
            failures.append("anchor_leakage")
        if int(metadata.get("selected_k", -1)) != int(split.get("selected_k", -2)):
            failures.append("selected_k_mismatch")
        if history and int(metadata.get("epoch", -1)) != int(
            max(history, key=lambda row: (float(row["validation_auc"]), -int(row["epoch"])))["epoch"]
        ):
            failures.append("best_epoch_not_validation_selected")

        test_rows = _read_csv(fold_dir / "test_predictions.csv")
        validation_rows = _read_csv(fold_dir / "validation_predictions.csv")
        if {row.get("sample_id") for row in test_rows} != test:
            failures.append("test_prediction_ids_mismatch")
        if {row.get("sample_id") for row in validation_rows} != validation:
            failures.append("validation_prediction_ids_mismatch")
        try:
            probabilities = np.asarray([float(row["probability"]) for row in test_rows])
            labels = np.asarray([int(row["label"]) for row in test_rows])
            predictions = np.asarray([int(row["prediction"]) for row in test_rows])
            if not np.isfinite(probabilities).all() or bool(((probabilities < 0) | (probabilities > 1)).any()):
                failures.append("invalid_probability")
            if not np.array_equal(predictions, (probabilities >= 0.5).astype(int)):
                failures.append("prediction_threshold_mismatch")
            recomputed = compute_metrics(labels, probabilities, threshold=0.5)
            stored = _read_json(fold_dir / "metrics.json")["test_metrics"]
            if any(abs(getattr(recomputed, name) - float(stored[name])) > 1e-12 for name in OFFICIAL_METRICS):
                failures.append("metric_recomputation_mismatch")
        except (KeyError, ValueError):
            failures.append("test_predictions_unparseable")
        all_test_rows.extend(test_rows)
        if failures:
            fold_failures[fold_name] = failures

    record("fold_integrity", not fold_failures, fold_failures or "all folds passed")
    sample_ids = [row.get("sample_id") for row in all_test_rows]
    record(
        "prediction_completeness",
        len(sample_ids) == expected_count
        and len(set(sample_ids)) == expected_count
        and reference_universe is not None
        and set(sample_ids) == reference_universe,
        {"rows": len(sample_ids), "unique": len(set(sample_ids)), "expected": expected_count},
    )
    oof_rows = _read_csv(root / "oof_predictions.csv") if (root / "oof_predictions.csv").is_file() else []
    record(
        "oof_matches_fold_predictions",
        sorted((row.get("sample_id"), row.get("probability")) for row in oof_rows)
        == sorted((row.get("sample_id"), row.get("probability")) for row in all_test_rows),
        len(oof_rows),
    )
    blockers = [name for name, value in checks.items() if not value["pass"]]
    report = {
        "status": "PASS" if not blockers else "FAIL",
        "output_root": root.as_posix(),
        "checks": checks,
        "blockers": blockers,
    }
    (root / "integrity_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
