"""Formal leakage-safe outer-CV orchestration and artifact generation."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from grain.data import (
    generate_nested_manifests_from_arrays,
    load_legacy_feature_table,
    resolve_data_fingerprint,
)
from grain.data.paths import RepositoryPathPolicy
from grain.evaluation import EvaluationResult, compute_metrics

from .checkpoint import ValidationCheckpointManager, load_checkpoint
from .cross_validation import build_fold_runtime, select_k_on_validation
from .trainer import test, train_one_epoch, validate


METRIC_NAMES = ("acc", "f1", "rec", "auc", "pre", "spec", "npv")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_git_commit(repository_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository_root, text=True
    ).strip()
    if status:
        raise RuntimeError(f"Formal run requires a clean worktree:\n{status}")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def load_formal_cohort(config: dict[str, Any], repository_root: Path):
    data = config["data"]
    policy = RepositoryPathPolicy.create(repository_root, data["source_root"])
    layout = data["legacy_layout"]
    return load_legacy_feature_table(
        policy.resolve_source(data["feature_source"]),
        sample_id_prefix=data["sample_id_prefix"],
        skip_leading_columns=int(layout["skip_leading_columns"]),
        plain_dimension=int(layout["plain_dimension"]),
        ce_dimension=int(layout["ce_dimension"]),
        label_column=data["label_column"],
        expected_sha256=data.get("expected_sha256"),
        enforce_fingerprint=bool(data.get("enforce_fingerprint", False)),
    )


def _distribution(cohort, sample_ids) -> dict[str, int]:
    mask = cohort.select(sample_ids)[1]
    return {
        "plain_only": int((mask[:, 0] & ~mask[:, 1]).sum()),
        "ce_only": int((~mask[:, 0] & mask[:, 1]).sum()),
        "both": int(mask.all(dim=1).sum()),
    }


def _effective_k(selected_k: int, anchor_ids: tuple[str, ...], query_ids) -> dict[str, float]:
    anchors = set(anchor_ids)
    values = [min(selected_k, len(anchors) - int(query in anchors)) for query in query_ids]
    if not values or min(values) < 1:
        raise RuntimeError("At least one eligible training anchor is required for every query")
    return {"min": int(min(values)), "max": int(max(values)), "mean": float(np.mean(values))}


def _prediction_diagnostics(result: EvaluationResult) -> tuple[dict[str, Any], list[str]]:
    probability = result.probabilities
    prediction = probability >= result.metrics.threshold
    standard_deviation = float(np.std(probability))
    warnings = []
    if bool(np.all(prediction)):
        warnings.append("ALL_POSITIVE")
    if bool(np.all(~prediction)):
        warnings.append("ALL_NEGATIVE")
    if standard_deviation < 1e-6:
        warnings.append("NEAR_CONSTANT_PROBABILITY")
    return {
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
        "probability_mean": float(np.mean(probability)),
        "probability_sd": standard_deviation,
        "predicted_positive": int(prediction.sum()),
        "predicted_negative": int((~prediction).sum()),
    }, warnings


def _peak_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _write_history_header(path: Path) -> tuple[Any, csv.DictWriter]:
    fields = [
        "epoch", "loss", "classification", "auxiliary", "contrastive", "balance",
        "gradient_norm", "batches", "validation_loss", "validation_auc", "checkpoint_saved",
    ]
    handle = path.open("x", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    handle.flush()
    return handle, writer


def _write_k_search(path: Path, fold: int, candidates, selected_k: int) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "candidate_k", "validation_auc", "validation_loss", "selected"],
        )
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "fold": fold,
                    "candidate_k": item.k,
                    "validation_auc": item.validation_auc,
                    "validation_loss": item.validation_loss,
                    "selected": item.k == selected_k,
                }
            )


def _write_fold_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_oof(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["sample_id", "patient_id_or_stable_sample_id", "fold", "split", "label", "probability", "prediction"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: item["sample_id"]))


def run_formal_cv(
    *,
    config: dict[str, Any],
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> Path:
    """Run one authorized Center A/B formal 10-fold experiment."""

    if config["experiment"]["name"] not in {"grain_center_a_formal", "grain_center_b_formal"}:
        raise ValueError("Phase 5A authorizes only formal Center A and Center B configs")
    if int(config["split"]["n_outer_folds"]) != 10:
        raise ValueError("Formal Phase 5A requires exactly 10 outer folds")
    if int(config["training"]["epochs"]) != 100:
        raise ValueError("Formal Phase 5A requires 100 final-training epochs")
    if int(config["training"]["k_selection"]["search_epochs"]) != 100:
        raise ValueError("Formal Phase 5A requires 100 epochs for every K candidate")
    if config["split"]["k_candidates"] != list(range(2, 11)):
        raise ValueError("Formal K candidates must be 2 through 10")

    commit = clean_git_commit(repository_root)
    cohort = load_formal_cohort(config, repository_root)
    if cohort.feature_dimension != int(config["model"]["feature_dimension"]):
        raise ValueError("Bound feature dimension does not match the frozen architecture")
    fingerprint = resolve_data_fingerprint(
        source_path=cohort.source_path,
        source_sha256=cohort.source_sha256,
        inventory_path=repository_root / "artifacts" / "data_inventory.json",
        require_inventory_registration=bool(
            config["data"].get("require_inventory_registration", False)
        ),
    )
    manifests = generate_nested_manifests_from_arrays(
        cohort.sample_ids,
        cohort.labels.tolist(),
        config["experiment"]["name"],
        n_outer_folds=10,
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )

    output_subdir = config["experiment"].get("output_subdir")
    if not output_subdir:
        raise ValueError("Formal config must declare experiment.output_subdir")
    output_root = repository_root / config["experiment"]["output_root"] / output_subdir
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite formal output: {output_root}")
    output_root.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started = time.perf_counter()
    run_manifest = {
        "designation": "FORMAL GRAIN_OFFICIAL RESULT",
        "status": "RUNNING",
        "dataset": config["experiment"]["name"],
        "sample_id_kind": cohort.id_kind,
        "expected_samples": len(cohort.sample_ids),
        "expected_outer_folds": 10,
        "git_commit": commit,
        "data_sha256": cohort.source_sha256,
        "data_source": cohort.source_path,
        "data_registration_status": fingerprint["registration_status"],
        "inventory_registered": fingerprint["inventory_registered"],
        "architecture_config": config["experiment"]["architecture_manifest"],
        "resolved_config": "resolved_config.json",
        "test_policy": "exactly_once_per_fold_after_best_validation_checkpoint_reload",
        "started_at_utc": utc_now(),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _write_json(output_root / "resolved_config.json", config)
    _write_json(output_root / "data_fingerprint.json", fingerprint)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    retained_final_objects: list[tuple[torch.nn.Module, torch.optim.Optimizer]] = []
    try:
        for fold, split in enumerate(manifests):
            fold_started = time.perf_counter()
            fold_seed = int(config["experiment"]["seed"]) + fold
            split.validate(cohort.sample_ids)
            if set(split.train) & set(split.validation) or set(split.train) & set(split.test) or set(split.validation) & set(split.test):
                raise AssertionError("Fold split overlap")
            fold_dir = output_root / f"fold_{fold:02d}"
            fold_dir.mkdir()
            _write_json(fold_dir / "resolved_config.json", config)
            _write_json(fold_dir / "data_fingerprint.json", fingerprint)
            progress(f"FOLD {fold:02d}/09 START seed={fold_seed}")

            k_started = time.perf_counter()
            def k_epoch(candidate: int, epoch: int, result) -> None:
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    progress(
                        f"FOLD {fold:02d} K={candidate} epoch={epoch + 1:03d}/100 "
                        f"loss={result.loss:.6f}"
                    )

            def k_result(candidate: int, result: EvaluationResult) -> None:
                progress(
                    f"FOLD {fold:02d} K={candidate} validation_auc={result.metrics.auc:.6f}"
                )

            selected_k, candidates = select_k_on_validation(
                config=config,
                cohort=cohort,
                split=split,
                device=device,
                seed=fold_seed,
                epochs=100,
                observer=k_result,
                epoch_observer=k_epoch,
            )
            k_runtime = time.perf_counter() - k_started
            split = replace(split, selected_k=selected_k)
            split.to_json(fold_dir / "split.json")
            _write_k_search(fold_dir / "k_search.csv", fold, candidates, selected_k)
            progress(f"FOLD {fold:02d} SELECTED_K={selected_k}")

            runtime = build_fold_runtime(
                config=config,
                cohort=cohort,
                split=split,
                selected_k=selected_k,
                device=device,
                seed=fold_seed,
            )
            if any(runtime.model is old_model or runtime.optimizer is old_optimizer for old_model, old_optimizer in retained_final_objects):
                raise AssertionError("Model or optimizer object was reused across folds")
            retained_final_objects.append((runtime.model, runtime.optimizer))
            anchors = set(runtime.anchor_bank.patient_ids)
            if not anchors.issubset(split.train):
                raise AssertionError("Anchor outside outer training split")
            if anchors & (set(split.validation) | set(split.test)):
                raise AssertionError("Validation/test patient entered anchor bank")

            checkpoint = ValidationCheckpointManager(fold_dir / "best.pt")
            history_handle, history_writer = _write_history_header(fold_dir / "train_history.csv")
            peak_rss = _peak_rss_bytes()
            training_started = time.perf_counter()
            try:
                for epoch in range(100):
                    training_result = train_one_epoch(
                        model=runtime.model,
                        optimizer=runtime.optimizer,
                        cohort=cohort,
                        training_ids=split.train,
                        anchor_bank=runtime.anchor_bank,
                        batch_size=int(config["training"]["batch_size"]),
                        device=device,
                        epoch_seed=fold_seed + epoch,
                        contrastive_temperature=float(config["training"]["contrastive_temperature"]),
                        balance_margin=float(config["training"]["balance_margin"]),
                        contrastive_weight=float(config["training"]["loss_weights"]["contrastive"]),
                        balance_weight=float(config["training"]["loss_weights"]["balance"]),
                        auxiliary_weight=float(config["training"]["loss_weights"]["auxiliary"]),
                    )
                    validation_result = validate(
                        model=runtime.model,
                        cohort=cohort,
                        sample_ids=split.validation,
                        anchor_bank=runtime.anchor_bank,
                        fold=fold,
                        batch_size=int(config["training"]["batch_size"]),
                        device=device,
                        threshold=float(config["evaluation"]["threshold"]),
                    )
                    saved = checkpoint.consider(
                        validation_result,
                        model=runtime.model,
                        optimizer=runtime.optimizer,
                        scheduler=runtime.scheduler,
                        epoch=epoch,
                        fold=fold,
                        seed=fold_seed,
                        selected_k=selected_k,
                        resolved_config=config,
                        data_fingerprint=fingerprint,
                        git_commit=commit,
                    )
                    history_writer.writerow(
                        {
                            "epoch": epoch,
                            **asdict(training_result),
                            "validation_loss": validation_result.loss,
                            "validation_auc": validation_result.metrics.auc,
                            "checkpoint_saved": saved,
                        }
                    )
                    history_handle.flush()
                    if runtime.scheduler is not None:
                        runtime.scheduler.step()
                    current_rss = _peak_rss_bytes()
                    if current_rss is not None:
                        peak_rss = max(peak_rss or 0, current_rss)
                    if epoch == 0 or (epoch + 1) % 10 == 0:
                        progress(
                            f"FOLD {fold:02d} FINAL epoch={epoch + 1:03d}/100 "
                            f"train_loss={training_result.loss:.6f} "
                            f"val_auc={validation_result.metrics.auc:.6f}"
                        )
            finally:
                history_handle.close()
            training_runtime = time.perf_counter() - training_started

            payload = load_checkpoint(fold_dir / "best.pt", runtime.model, map_location=device)
            best_validation = validate(
                model=runtime.model,
                cohort=cohort,
                sample_ids=split.validation,
                anchor_bank=runtime.anchor_bank,
                fold=fold,
                batch_size=int(config["training"]["batch_size"]),
                device=device,
                threshold=float(config["evaluation"]["threshold"]),
            )
            best_validation.write_csv(
                fold_dir / "validation_predictions.csv", float(config["evaluation"]["threshold"])
            )

            test_evaluation_count = 0
            held_out_test = test(
                model=runtime.model,
                cohort=cohort,
                sample_ids=split.test,
                anchor_bank=runtime.anchor_bank,
                fold=fold,
                batch_size=int(config["training"]["batch_size"]),
                device=device,
                threshold=float(config["evaluation"]["threshold"]),
            )
            test_evaluation_count += 1
            if test_evaluation_count != 1:
                raise AssertionError("Outer test must be evaluated exactly once")
            held_out_test.write_csv(
                fold_dir / "test_predictions.csv", float(config["evaluation"]["threshold"])
            )
            diagnostics, warnings = _prediction_diagnostics(held_out_test)
            if payload["epoch"] in {0, 99}:
                warnings.append("BEST_EPOCH_BOUNDARY")
            if selected_k in {2, 10}:
                warnings.append("EXTREME_K_SELECTION")
            effective_k = {
                "train": _effective_k(selected_k, runtime.anchor_bank.patient_ids, split.train),
                "validation": _effective_k(selected_k, runtime.anchor_bank.patient_ids, split.validation),
                "test": _effective_k(selected_k, runtime.anchor_bank.patient_ids, split.test),
            }
            all_k = [value for item in effective_k.values() for value in (item["min"], item["max"])]
            effective_k["overall"] = {
                "min": int(min(all_k)),
                "max": int(max(all_k)),
                "mean": float(np.mean([
                    *[min(selected_k, len(anchors) - int(value in anchors)) for value in split.train],
                    *[min(selected_k, len(anchors)) for _ in split.validation],
                    *[min(selected_k, len(anchors)) for _ in split.test],
                ])),
            }
            checkpoint_metadata = {
                "selection_split": "validation",
                "selection_metric": "validation_auc",
                "epoch": int(payload["epoch"]),
                "best_val_auc": float(payload["best_val_auc"]),
                "fold": fold,
                "seed": fold_seed,
                "selected_k": selected_k,
                "git_commit": commit,
                "data_sha256": cohort.source_sha256,
                "checkpoint_sha256": _sha256(fold_dir / "best.pt"),
                "test_evaluation_count": test_evaluation_count,
                "training_anchor_ids": list(runtime.anchor_bank.patient_ids),
                "number_of_training_anchors": len(runtime.anchor_bank.patient_ids),
                "effective_k": effective_k,
            }
            _write_json(fold_dir / "checkpoint_metadata.json", checkpoint_metadata)
            fold_metrics = {
                "fold": fold,
                "seed": fold_seed,
                "selected_k": selected_k,
                "best_epoch": int(payload["epoch"]),
                "validation_auc": float(payload["best_val_auc"]),
                "training_anchors": len(runtime.anchor_bank.patient_ids),
                "effective_k_min": effective_k["overall"]["min"],
                "effective_k_max": effective_k["overall"]["max"],
                "effective_k_mean": effective_k["overall"]["mean"],
                **{name: getattr(held_out_test.metrics, name) for name in METRIC_NAMES},
                **diagnostics,
                "k_search_runtime_seconds": k_runtime,
                "training_runtime_seconds": training_runtime,
                "fold_runtime_seconds": time.perf_counter() - fold_started,
                "peak_rss_bytes": peak_rss,
                "warnings": "|".join(warnings),
            }
            _write_json(
                fold_dir / "metrics.json",
                {
                    "designation": "FORMAL GRAIN_OFFICIAL RESULT",
                    "fold_metrics": fold_metrics,
                    "test_metrics": held_out_test.metrics.to_dict(),
                    "validation_metrics": best_validation.metrics.to_dict(),
                    "split_sizes": {
                        "train": len(split.train),
                        "validation": len(split.validation),
                        "test": len(split.test),
                    },
                    "modality_distribution": {
                        "train": _distribution(cohort, split.train),
                        "validation": _distribution(cohort, split.validation),
                        "test": _distribution(cohort, split.test),
                    },
                    "diagnostics": diagnostics,
                    "warnings": warnings,
                },
            )
            fold_rows.append(fold_metrics)
            for sample_id, label, probability in zip(
                held_out_test.sample_ids, held_out_test.labels, held_out_test.probabilities
            ):
                oof_rows.append(
                    {
                        "sample_id": sample_id,
                        "patient_id_or_stable_sample_id": sample_id,
                        "fold": fold,
                        "split": "test",
                        "label": int(label),
                        "probability": float(probability),
                        "prediction": int(probability >= float(config["evaluation"]["threshold"])),
                    }
                )
            progress(
                f"FOLD {fold:02d}/09 COMPLETE test_auc={held_out_test.metrics.auc:.6f} "
                f"best_epoch={payload['epoch']}"
            )

        ids = [row["sample_id"] for row in oof_rows]
        if len(ids) != len(cohort.sample_ids) or len(set(ids)) != len(ids) or set(ids) != set(cohort.sample_ids):
            raise AssertionError("OOF predictions are incomplete or duplicated")
        _write_oof(output_root / "oof_predictions.csv", oof_rows)
        _write_fold_metrics(output_root / "fold_metrics.csv", fold_rows)
        pooled = compute_metrics(
            np.asarray([row["label"] for row in oof_rows]),
            np.asarray([row["probability"] for row in oof_rows]),
            float(config["evaluation"]["threshold"]),
        )
        fold_values = {name: np.asarray([row[name] for row in fold_rows]) for name in METRIC_NAMES}
        summary = {
            "designation": "FORMAL GRAIN_OFFICIAL RESULT",
            "fold_mean": {name: float(values.mean()) for name, values in fold_values.items()},
            "fold_sd": {name: float(values.std(ddof=1)) for name, values in fold_values.items()},
            "pooled_oof": pooled.to_dict(),
            "mean_fold_auc": float(fold_values["auc"].mean()),
            "sd_fold_auc": float(fold_values["auc"].std(ddof=1)),
            "pooled_oof_auc": pooled.auc,
            "oof_count": len(oof_rows),
        }
        _write_json(output_root / "summary_metrics.json", summary)
        run_manifest["status"] = "COMPLETE"
        run_manifest["completed_at_utc"] = utc_now()
        run_manifest["runtime_seconds"] = time.perf_counter() - run_started
        run_manifest["completed_folds"] = 10
        _write_json(output_root / "run_manifest.json", run_manifest)
        progress(f"FORMAL CV COMPLETE output={output_root}")
        return output_root
    except BaseException as error:
        run_manifest["status"] = "FAILED"
        run_manifest["failed_at_utc"] = utc_now()
        run_manifest["failure"] = f"{type(error).__name__}: {error}"
        run_manifest["runtime_seconds"] = time.perf_counter() - run_started
        _write_json(output_root / "run_manifest.json", run_manifest)
        raise
